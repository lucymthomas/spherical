# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Rotating and evaluating mode weights without building the full 𝔇 matrix.

Mode weights use the same flat layout as `spherical.Modes` and `sYlm`:

    [f(ℓ, m) for ℓ in range(ell_min, ell_max+1) for m in range(-ℓ, ℓ+1)]

along the last axis, indexed by `spherical.Yindex(ℓ, m, ell_min)`.  `ell_max`
is inferred from the length of that axis.

Unlike `spherical.Wigner.rotate` and `.evaluate`, which apply one set of
rotors to every set of modes, these functions **broadcast** the batch
dimensions of the modes against those of the rotors.  So modes of shape
(T, n_modes) with rotors of shape (T, 4) give one rotation per time sample —
the case of a precessing waveform — while modes of shape (n_modes,) with
rotors of shape (N, 4) evaluate one function at N points.  For every
combination, add axes of length 1 explicitly, e.g. `modes[:, None, :]` with
`R[None, :, :]`.

"""

from functools import partial, lru_cache
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from .phases import euler_phases
from .recursion import H_wedge, _wedge_pieces
from .matrices import _powers, _check_spin_weight, _sYlm, _Y_tables
from .generators import omega, apply_right_generator, mode_layout, spin_mixing_coefficients


def ell_max_from_size(size, ell_min=0):
    """ell_max such that `size` = Σ_{ℓ=ell_min}^{ell_max} (2ℓ+1), or raise ValueError."""
    ell_max = math.isqrt(size + ell_min**2) - 1
    if ell_max < ell_min or (ell_max + 1) ** 2 - ell_min**2 != size:
        raise ValueError(
            f"A last axis of length {size} does not hold a whole number of ℓ values "
            f"starting from ell_min={ell_min}"
        )
    return ell_max


def _check_ell_min(ell_min):
    if not isinstance(ell_min, int) or ell_min < 0:
        raise ValueError(f"ell_min={ell_min!r} must be a non-negative Python int")


# Consecutive ℓ blocks are grouped into one block-diagonal matrix product as
# long as the group has at most this many modes.  One larger product is faster
# than several small ones while the wasted zeros are few; for ℓ ≤ 8 (81 modes)
# a single product is fastest on the CPU.
_ROTATE_GROUP_SIZE = 81


@lru_cache(maxsize=None)
def _rotation_tables(ell_min, ell_max, group_size):
    """Static tables for `rotate`.

    Returns (m, groups).  m is the m of each mode in the flat layout.  groups
    is a tuple of (mode_slice, Δ), one per group of consecutive ℓ, where Δ is
    the block-diagonal matrix of Δˡ = dˡ(π/2) for the ℓ in the group, computed
    in float64 by the numba implementation.
    """
    from ..wigner import Wigner

    wigner = Wigner(ell_max)
    d = wigner.d(np.exp(0.5j * np.pi))

    def Δ(ell):
        ms = range(-ell, ell + 1)
        return np.array([[d[wigner.dindex(ell, mp, m_)] for m_ in ms] for mp in ms])

    m = np.concatenate([np.arange(-ell, ell + 1) for ell in range(ell_min, ell_max + 1)])
    groups, ells, start = [], [], 0

    def close():
        nonlocal start
        size = sum(2 * ell + 1 for ell in ells)
        block = np.zeros((size, size))
        i = 0
        for ell in ells:
            block[i: i + 2 * ell + 1, i: i + 2 * ell + 1] = Δ(ell)
            i += 2 * ell + 1
        groups.append((slice(start, start + size), block))
        start += size

    for ell in range(ell_min, ell_max + 1):
        if ells and sum(2 * e + 1 for e in ells) + 2 * ell + 1 > group_size:
            close()
            ells = []
        ells.append(ell)
    close()
    return m, tuple(groups)


def _times_real(x, A):
    """x @ A for complex x and a real constant matrix A, as one real product."""
    y = jnp.stack([x.real, x.imag], axis=-2) @ jnp.asarray(A, dtype=x.real.dtype)
    return jax.lax.complex(y[..., 0, :], y[..., 1, :])


@partial(jax.jit, static_argnames=("ell_min",))
def rotate(modes, R, ell_min=0):
    """Mode weights of a function in a frame rotated by R.

    Computes  f′ₗₘ = Σₘ′ fₗₘ′ 𝔇ˡ_{m′,m}(R),  as `spherical.Wigner.rotate`
    does, but with one rotor per set of modes (broadcasting; see the module
    docstring), and without building 𝔇.

    Parameters
    ----------
    modes : complex array_like, shape (..., Ysize(ell_min, ell_max))
        Mode weights fₗₘ.  The rotation does not depend on the spin weight.
    R : array_like, shape (..., 4)
        Rotors; their batch shape must broadcast against that of `modes`.
    ell_min : int
        Smallest ℓ present in `modes` (static).

    Returns
    -------
    rotated : complex array of shape broadcast(modes.shape[:-1], R.shape[:-1]) + (n_modes,)

    Notes
    -----
    Writing 𝔇ˡ_{m′,m} = dˡ_{m′,m}(β) zₐ^{m′} zᵧ^{m}, and factorizing

        dˡ_{m′,m}(β) = i^{m-m′} Σₖ Δˡ_{k,m′} e^{ikβ} Δˡ_{k,m},   Δˡ = dˡ(π/2),

    this computes

        f′ₗₘ = (i zᵧ)ᵐ Σₖ Δˡ_{k,m} zᵦᵏ Σₘ′ Δˡ_{k,m′} fₗₘ′ (zₐ/i)^{m′}.

    Δ is a constant, so apart from multiplying by phases this is two products
    with fixed matrices, which run efficiently for large batches, and the
    recursion for dˡ(β) is not needed at all.

    """
    _check_ell_min(ell_min)
    modes = jnp.asarray(modes)
    ell_max_from_size(modes.shape[-1], ell_min)  # validate
    return _rotate(modes, jnp.asarray(R), ell_min)


def _rotate_impl(modes, R, ell_min):
    """`rotate` itself, without validation or the custom derivative."""
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    m, groups = _rotation_tables(ell_min, ell_max, _ROTATE_GROUP_SIZE)
    zₐ, zᵦ, zᵧ = euler_phases(R)
    i_m = 1j ** (m % 4)
    index = m + ell_max
    u = modes * (_powers(zₐ, ell_max)[..., index] * jnp.asarray(np.conj(i_m), zₐ.dtype))
    zᵦᵐ = _powers(zᵦ, ell_max)[..., index]
    rotated = []
    for modes_slice, Δ in groups:
        h = _times_real(u[..., modes_slice], Δ.T) * zᵦᵐ[..., modes_slice]
        rotated.append(_times_real(h, Δ))
    rotated = rotated[0] if len(rotated) == 1 else jnp.concatenate(rotated, axis=-1)
    return rotated * (_powers(zᵧ, ell_max)[..., index] * jnp.asarray(i_m, zₐ.dtype))


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _rotate(modes, R, ell_min):
    return _rotate_impl(modes, R, ell_min)


def _rotate_jvp(ell_min, primals, tangents):
    # rotate is linear in the modes, and δ(f 𝔇) = (f 𝔇) G for a change of rotor.
    # Symbolic zeros skip whichever part is not needed (e.g. rotating the mode
    # tangents when differentiating only with respect to the rotor).
    (modes, R), (δmodes, δR) = primals, tangents
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    rotated = _rotate(modes, R, ell_min)
    δrotated = jnp.zeros_like(rotated)
    if not isinstance(δR, SymbolicZero):
        δrotated = δrotated + apply_right_generator(rotated, omega(R, δR), mode_layout(ell_min, ell_max))
    if not isinstance(δmodes, SymbolicZero):
        δrotated = δrotated + _rotate(δmodes, R, ell_min)
    return rotated, δrotated


_rotate.defjvp(_rotate_jvp, symbolic_zeros=True)


@partial(jax.jit, static_argnames=("spin_weight", "ell_min"))
def evaluate(modes, R, spin_weight, ell_min=0):
    """Values of the spin-weighted function with these modes, at rotors R.

    Computes  f(R) = Σₗₘ fₗₘ ₛYₗₘ(R),  as `spherical.Wigner.evaluate` does,
    but broadcasting the batch shapes of `modes` and `R` (see the module
    docstring).  For a point (θ, ϕ) on the sphere, use the rotor
    `from_spherical_coordinates(θ, ϕ)`.

    Parameters
    ----------
    modes : complex array_like, shape (..., Ysize(ell_min, ell_max))
        Mode weights fₗₘ.  Weights with ℓ < |spin_weight| are ignored, since
        the corresponding harmonics vanish.
    R : array_like, shape (..., 4)
        Rotors; their batch shape must broadcast against that of `modes`.
    spin_weight : int
        Spin weight s of the function (static).  For gravitational-wave strain
        h = h₊ - i hₓ, s = -2.
    ell_min : int
        Smallest ℓ present in `modes` (static).

    Returns
    -------
    f : complex array of shape broadcast(modes.shape[:-1], R.shape[:-1])

    Notes
    -----
    Rotating modes and then evaluating them never needs to be done in two
    steps: since 𝔇(R₁R₂) = 𝔇(R₁)𝔇(R₂),

        evaluate(rotate(f, R₁), R₂) = evaluate(f, R₁ R₂),

    which needs only Σ(2ℓ+1) harmonics per rotor instead of Σ(2ℓ+1)² elements
    of 𝔇.  See `multiply` for composing rotors.

    """
    _check_ell_min(ell_min)
    _check_spin_weight(spin_weight)
    modes = jnp.asarray(modes)
    ell_max_from_size(modes.shape[-1], ell_min)  # validate
    return _evaluate(modes, jnp.asarray(R), spin_weight, ell_min)


def _evaluate_sums(modes_list, R, terms, ell_min):
    """Σₗₘ wₗₘ fₗₘ ₛYₗₘ(R) for each term (k, s, w), with f = modes_list[k] and w a
    static tuple of per-mode weights (or None for all ones).  All terms share
    one H recursion and one set of phases.

    Each sum is taken piece by piece straight from the rows of the H wedge: each
    piece is a run of consecutive modes whose d values come from one row (see
    `gather_wedge`), so the harmonics are never stored as a whole array.  The
    weights are constants, folded into the constant factors of each piece.
    """
    ell_max = ell_max_from_size(modes_list[0].shape[-1], ell_min)
    s_max = max(abs(s) for _, s, _ in terms)
    zₐ, zᵦ, zᵧ = euler_phases(R)
    H = H_wedge(zᵦ, ell_max, mp_max=s_max)  # ₛY reads only rows |m′| ≤ |s|
    zₐᵏ = _powers(zₐ, ell_max)
    zᵧᵏ = _powers(zᵧ, s_max)
    real = zₐᵏ.real.dtype
    results = []
    for k, s, weights in terms:
        modes = modes_list[k]
        keys, factor, m_index = _Y_tables(s, ell_min, ell_max)
        if weights is not None:
            factor = factor * np.asarray(weights)
        total = jnp.zeros(jnp.broadcast_shapes(modes.shape[:-1], R.shape[:-1]), dtype=zₐᵏ.dtype)
        position = 0
        for row, first, last, step in _wedge_pieces(keys):
            length = abs(last - first) + 1
            here = slice(position, position + length)
            position += length
            if not np.any(factor[here]):
                continue  # e.g. harmonics with ℓ < |s|, which vanish
            x = H[row]
            d = x[..., first: last + 1] if step == 1 else jnp.flip(x[..., last: first + 1], axis=-1)
            j = m_index[here]  # consecutive powers of zₐ (m increases along a piece)
            total = total + jnp.sum(
                modes[..., here] * (d * factor[here].astype(real)) * zₐᵏ[..., j[0]: j[-1] + 1], axis=-1
            )
        results.append(total * zᵧᵏ[..., s_max - s])  # zᵧ^{-s}
    return tuple(results)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _evaluate_many(modes_list, R, terms, ell_min):
    return _evaluate_sums(modes_list, R, terms, ell_min)


def _evaluate_many_jvp(terms, ell_min, primals, tangents):
    # f = Σ w f ₛY, so δf = Σ w δf ₛY + Σ w f δ(ₛY), and by `sYlm_tangent`
    #   Σ w f δ(ₛY) = 2[-i s ωz Σ w f ₛY - (iωx + ωy) Σ w c₊ f ₛ₊₁Y - (iωx - ωy) Σ w c₋ f ₛ₋₁Y],
    # with the constants c₊, c₋ of `spin_mixing_coefficients`.  Every sum is
    # itself a term of this function, so the rotor part needs three numbers per
    # rotor, all from one call (one H recursion), and never the harmonics.  That
    # call is to this same function, so higher derivatives are exact too.
    (modes_list, R), (δmodes_list, δR) = primals, tangents
    T = len(terms)
    ell_max = ell_max_from_size(modes_list[0].shape[-1], ell_min)

    if isinstance(δR, SymbolicZero):
        values = _evaluate_many(modes_list, R, terms, ell_min)
        δvalues = [jnp.zeros_like(v) for v in values]
    else:
        raised, lowered = [], []
        for k, s, weights in terms:
            w = np.ones(modes_list[k].shape[-1]) if weights is None else np.asarray(weights)
            c_raise, c_lower = spin_mixing_coefficients(s, ell_min, ell_max)
            raised.append((k, s + 1, tuple((w * c_raise).tolist())))
            lowered.append((k, s - 1, tuple((w * c_lower).tolist())))
        sums = _evaluate_many(modes_list, R, terms + tuple(raised) + tuple(lowered), ell_min)
        values = sums[:T]
        ωx, ωy, ωz = omega(R, δR)
        δvalues = [
            2 * (-1j * s * ωz * values[t] - (1j * ωx + ωy) * sums[T + t] - (1j * ωx - ωy) * sums[2 * T + t])
            for t, (_, s, _) in enumerate(terms)
        ]

    # The part linear in the modes, Σ w δf ₛY.  The harmonics here come from
    # `_sYlm`, whose own derivative rule is exact, rather than from a call to this
    # function on the tangents δf: JAX drops custom rules from calls on tangents
    # when it builds reverse-mode derivatives, and the plain implementation
    # differentiates wrongly at the poles, which would spoil mixed second
    # derivatives (in modes and rotor) there.  This stores the harmonics, but
    # only when differentiating with respect to the modes.
    harmonics = {}
    for t, (k, s, weights) in enumerate(terms):
        δm = δmodes_list[k]
        if isinstance(δm, SymbolicZero):
            continue
        if s not in harmonics:
            harmonics[s] = _sYlm(s, R, ell_max, ell_min)
        Y = harmonics[s]
        if weights is not None:
            Y = Y * np.asarray(weights).astype(jnp.real(Y).dtype)
        δvalues[t] = δvalues[t] + jnp.sum(δm * Y, axis=-1)
    return tuple(values), tuple(δvalues)


_evaluate_many.defjvp(_evaluate_many_jvp, symbolic_zeros=True)


def _evaluate(modes, R, s, ell_min):
    return _evaluate_many((modes,), R, ((0, s, None),), ell_min)[0]
