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

from .phases import euler_phases
from .recursion import H_wedge, flatten_wedge, wedge_layout, wedge_key
from .matrices import ϵ, _powers, _check_spin_weight, sYlm_from_phases, _sYlm
from .generators import omega, apply_right_generator, mode_layout, sYlm_tangent


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


@lru_cache(maxsize=None)
def _rotation_tables(ell_min, ell_max):
    """Static tables for `rotate`.

    Returns (m_index, blocks), where m_index maps each mode to m + ell_max (an
    index into `_powers(·, ell_max)`), and blocks is a tuple with one entry per
    ℓ of (ℓ, mode_slice, wedge_index, sign), where wedge_index and sign have
    shape (2ℓ+1, 2ℓ+1) and give dˡ_{m′,m} = sign · W[wedge_index] in the
    flattened wedge W.
    """
    _, offset, _ = wedge_layout(ell_max)
    m_index = np.array([m + ell_max for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)])
    blocks = []
    start = 0
    for ell in range(ell_min, ell_max + 1):
        size = 2 * ell + 1
        ms = range(-ell, ell + 1)
        wedge_index = np.array([[offset[wedge_key(ell, mp, m)] for m in ms] for mp in ms])
        sign = np.array([[ϵ(mp) * ϵ(-m) for m in ms] for mp in ms], dtype=float)
        blocks.append((ell, slice(start, start + size), wedge_index, sign))
        start += size
    return m_index, tuple(blocks)


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
    Writing 𝔇ˡ_{m′,m} = dˡ_{m′,m}(β) zₐ^{m′} zᵧ^{m}, this computes

        f′ₗₘ = zᵧᵐ Σₘ′ (fₗₘ′ zₐ^{m′}) dˡ_{m′,m}(β),

    with dˡ read directly from the H wedge.  Each ℓ block is written as a
    broadcast product and a sum, rather than a matrix product, so that XLA can
    fuse the gather from the wedge into the reduction.

    """
    _check_ell_min(ell_min)
    modes = jnp.asarray(modes)
    ell_max_from_size(modes.shape[-1], ell_min)  # validate
    return _rotate(modes, jnp.asarray(R), ell_min)


def _rotate_impl(modes, R, ell_min):
    """`rotate` itself, without validation or the custom derivative."""
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    m_index, blocks = _rotation_tables(ell_min, ell_max)

    zₐ, zᵦ, zᵧ = euler_phases(R)
    W = flatten_wedge(H_wedge(zᵦ, ell_max), ell_max)
    zₐᵏ = _powers(zₐ, ell_max)
    zᵧᵏ = _powers(zᵧ, ell_max)

    g = modes * zₐᵏ[..., m_index]  # fₗₘ′ zₐ^{m′}
    rotated = []
    for ell, modes_slice, wedge_index, sign in blocks:
        dˡ = W[..., wedge_index] * sign.astype(W.dtype)  # (..., m′, m)
        sum_over_mp = jnp.sum(g[..., modes_slice, None] * dˡ, axis=-2)
        rotated.append(sum_over_mp * zᵧᵏ[..., ell_max - ell: ell_max + ell + 1])
    return jnp.concatenate(rotated, axis=-1)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _rotate(modes, R, ell_min):
    return _rotate_impl(modes, R, ell_min)


@_rotate.defjvp
def _rotate_jvp(ell_min, primals, tangents):
    # rotate is linear in the modes, and δ(f 𝔇) = (f 𝔇) G for a change of rotor
    (modes, R), (δmodes, δR) = primals, tangents
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    rotated = _rotate(modes, R, ell_min)
    δrotated = _rotate(δmodes, R, ell_min) + apply_right_generator(
        rotated, omega(R, δR), mode_layout(ell_min, ell_max)
    )
    return rotated, δrotated


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


def _evaluate_with(modes, Y):
    return jnp.sum(modes * Y, axis=-1)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _evaluate(modes, R, s, ell_min):
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    return _evaluate_with(modes, sYlm_from_phases(s, *euler_phases(R), ell_max, ell_min))


@_evaluate.defjvp
def _evaluate_jvp(s, ell_min, primals, tangents):
    # f = Σ fₗₘ ₛYₗₘ, so δf = Σ δfₗₘ ₛYₗₘ + Σ fₗₘ δ(ₛYₗₘ), with δ(ₛYₗₘ) mixing in s±1
    (modes, R), (δmodes, δR) = primals, tangents
    ell_max = ell_max_from_size(modes.shape[-1], ell_min)
    Y = _sYlm(s, R, ell_max, ell_min)
    δY = sYlm_tangent(
        s, Y, _sYlm(s + 1, R, ell_max, ell_min), _sYlm(s - 1, R, ell_max, ell_min),
        omega(R, δR), ell_min, ell_max,
    )
    return _evaluate_with(modes, Y), _evaluate_with(δmodes, Y) + _evaluate_with(modes, δY)
