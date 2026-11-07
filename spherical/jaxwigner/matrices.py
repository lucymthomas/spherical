# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Wigner d and 𝔇 matrices, and spin-weighted spherical harmonics, in JAX.

The outputs use exactly the same flat layouts as `spherical.Wigner`:

    d, 𝔇 : [X(ℓ, m′, m) for ℓ in range(ell_min, ell_max+1)
                       for m′ in range(-ℓ, ℓ+1)
                       for m in range(-ℓ, ℓ+1)]
    sYlm : [Y(ℓ, m) for ℓ in range(ell_min, ell_max+1)
                    for m in range(-ℓ, ℓ+1)]

so they can be indexed with `spherical.WignerDindex(ℓ, m′, m, ell_min)` and
`spherical.Yindex(ℓ, m, ell_min)`, with any leading batch dimensions given by
the input.

Each output is built from the flattened H wedge by one gather, using index and
sign tables computed once (in NumPy) for each (ell_min, ell_max, s).

"""

from functools import partial, lru_cache
import math

import numpy as np
import jax
import jax.numpy as jnp

from .phases import euler_phases
from .recursion import H_wedge, flatten_wedge, wedge_layout, wedge_key
from .generators import omega, apply_right_generator, D_layout, sYlm_tangent


def ϵ(k):
    """1 for k ≤ 0, and (-1)^k for k > 0"""
    return -1 if (k > 0 and k % 2) else 1


def _check_ells(ell_min, ell_max):
    if not (isinstance(ell_min, int) and isinstance(ell_max, int)):
        raise TypeError(f"ell_min={ell_min!r} and ell_max={ell_max!r} must be Python ints")
    if not 0 <= ell_min <= ell_max:
        raise ValueError(f"Need 0 <= ell_min <= ell_max; got ell_min={ell_min}, ell_max={ell_max}")


def _powers(z, k_max):
    """Stack of zᵏ for k = -k_max, ..., k_max along a new last axis, for |z| = 1.

    Positive powers are built by repeated multiplication; negative powers are
    their complex conjugates.  For k ≤ 8 the accumulated rounding is a few ulp.

    """
    positive = [jnp.ones_like(z)]
    for _ in range(k_max):
        positive.append(positive[-1] * z)
    negative = [jnp.conj(p) for p in positive[:0:-1]]
    return jnp.stack(negative + positive, axis=-1)


@lru_cache(maxsize=None)
def _D_tables(ell_min, ell_max):
    """Static tables mapping each (ℓ, m′, m) of the 𝔇 layout to the wedge.

    Returns (wedge_index, sign, mp_index, m_index), where the last two index the
    output of `_powers(·, ell_max)`.
    """
    _, offset, _ = wedge_layout(ell_max)
    wedge_index, sign, mp_index, m_index = [], [], [], []
    for ell in range(ell_min, ell_max + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                wedge_index.append(offset[wedge_key(ell, mp, m)])
                sign.append(ϵ(mp) * ϵ(-m))
                mp_index.append(mp + ell_max)
                m_index.append(m + ell_max)
    return tuple(np.array(x) for x in (wedge_index, sign, mp_index, m_index))


@lru_cache(maxsize=None)
def _Y_tables(s, ell_min, ell_max):
    """Static tables for ₛYₗₘ = (-1)ˢ √((2ℓ+1)/4π) 𝔇ˡ_{m,-s}.

    Returns (wedge_index, factor, m_index).  Entries with ℓ < |s| get factor 0.
    """
    _, offset, _ = wedge_layout(ell_max)
    wedge_index, factor, m_index = [], [], []
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            if ell < abs(s):
                wedge_index.append(0)
                factor.append(0.0)
            else:
                wedge_index.append(offset[wedge_key(ell, m, -s)])
                factor.append(
                    (-1) ** (s % 2) * ϵ(m) * ϵ(s) * math.sqrt((2 * ell + 1) / (4 * math.pi))
                )
            m_index.append(m + ell_max)
    return np.array(wedge_index), np.array(factor), np.array(m_index)


@partial(jax.jit, static_argnames=("ell_max", "ell_min"))
def wigner_d(expiβ, ell_max, ell_min=0):
    """Wigner's (small) d matrix dˡ_{m′,m}(β) for every ℓ in [ell_min, ell_max].

    Parameters
    ----------
    expiβ : complex array_like
        e^{iβ}, with any batch shape.
    ell_max, ell_min : int
        Range of ℓ values to return.  These are static: each distinct value
        compiles a new function.

    Returns
    -------
    d : real array of shape expiβ.shape + (WignerDsize(ell_min, ell_max),)

    """
    _check_ells(ell_min, ell_max)
    W = flatten_wedge(H_wedge(expiβ, ell_max), ell_max)
    wedge_index, sign, _, _ = _D_tables(ell_min, ell_max)
    return W[..., wedge_index] * sign.astype(W.dtype)


@partial(jax.jit, static_argnames=("ell_max", "ell_min"))
def wigner_D(R, ell_max, ell_min=0):
    """Wigner's 𝔇 matrix 𝔇ˡ_{m′,m}(R) for every ℓ in [ell_min, ell_max].

    Uses 𝔇ˡ_{m′,m}(R) = dˡ_{m′,m}(β) zₐ^{m′} zᵧ^{m}, as in `spherical.Wigner.D`.

    Parameters
    ----------
    R : array_like, shape (..., 4)
        Quaternions `[w, x, y, z]` (need not be normalized).
    ell_max, ell_min : int
        Range of ℓ values to return (static).

    Returns
    -------
    𝔇 : complex array of shape R.shape[:-1] + (WignerDsize(ell_min, ell_max),)

    Notes
    -----
    For rotating or evaluating mode weights, it is usually much cheaper not to
    build this full array at all; it is provided mainly as a reference and for
    cases where one rotation is applied to many sets of modes.

    Derivatives with respect to R are exact everywhere, including at the poles
    β = 0, π (see `spherical.jaxwigner.generators`).

    """
    _check_ells(ell_min, ell_max)
    return _wigner_D(jnp.asarray(R), ell_max, ell_min)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _wigner_D(R, ell_max, ell_min):
    zₐ, zᵦ, zᵧ = euler_phases(R)
    W = flatten_wedge(H_wedge(zᵦ, ell_max), ell_max)
    wedge_index, sign, mp_index, m_index = _D_tables(ell_min, ell_max)
    zₐᵏ = _powers(zₐ, ell_max)
    zᵧᵏ = _powers(zᵧ, ell_max)
    return (W[..., wedge_index] * sign.astype(W.dtype)) * (zₐᵏ[..., mp_index] * zᵧᵏ[..., m_index])


@_wigner_D.defjvp
def _wigner_D_jvp(ell_max, ell_min, primals, tangents):
    (R,), (δR,) = primals, tangents
    D = _wigner_D(R, ell_max, ell_min)
    δD = apply_right_generator(D, omega(R, δR), D_layout(ell_min, ell_max))
    return D, δD


@partial(jax.jit, static_argnames=("s", "ell_max", "ell_min"))
def sYlm(s, R, ell_max, ell_min=0):
    """Spin-weighted spherical harmonics ₛYₗₘ(R) for every ℓ in [ell_min, ell_max].

    Uses ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡ_{m,-s}(R), as in `spherical.Wigner.sYlm`.
    Entries with ℓ < |s| are zero.  Only the one column of 𝔇 that is needed is
    computed.

    Parameters
    ----------
    s : int
        Spin weight (static).
    R : array_like, shape (..., 4)
        Quaternions `[w, x, y, z]`.  For a point on the sphere with spherical
        coordinates (θ, ϕ), use `quaternionic.array.from_spherical_coordinates`.
    ell_max, ell_min : int
        Range of ℓ values to return (static).

    Returns
    -------
    Y : complex array of shape R.shape[:-1] + (Ysize(ell_min, ell_max),)

    Notes
    -----
    Derivatives with respect to R are exact everywhere, including at the poles;
    they involve the harmonics of spin weight s±1 (see
    `spherical.jaxwigner.generators`).

    """
    _check_ells(ell_min, ell_max)
    _check_spin_weight(s)
    return _sYlm(s, jnp.asarray(R), ell_max, ell_min)


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
def _sYlm(s, R, ell_max, ell_min):
    return sYlm_from_phases(s, *euler_phases(R), ell_max, ell_min)


@_sYlm.defjvp
def _sYlm_jvp(s, ell_max, ell_min, primals, tangents):
    (R,), (δR,) = primals, tangents
    Y = _sYlm(s, R, ell_max, ell_min)
    δY = sYlm_tangent(
        s, Y, _sYlm(s + 1, R, ell_max, ell_min), _sYlm(s - 1, R, ell_max, ell_min),
        omega(R, δR), ell_min, ell_max,
    )
    return Y, δY


def _check_spin_weight(s):
    if not isinstance(s, int):
        raise TypeError(f"s={s!r} must be a Python int")


def sYlm_from_phases(s, zₐ, zᵦ, zᵧ, ell_max, ell_min=0):
    """ₛYₗₘ from the phases returned by `euler_phases`; the core of `sYlm`.

    Not jitted by itself: it is meant to be called inside other jitted
    functions, so that XLA can fuse it with whatever consumes the harmonics.
    """
    W = flatten_wedge(H_wedge(zᵦ, ell_max), ell_max)
    wedge_index, factor, m_index = _Y_tables(s, ell_min, ell_max)
    zₐᵏ = _powers(zₐ, ell_max)
    zᵧ_to_minus_s = _powers(zᵧ, abs(s))[..., abs(s) - s]
    return (W[..., wedge_index] * factor.astype(W.dtype)) * (zₐᵏ[..., m_index] * zᵧ_to_minus_s[..., None])
