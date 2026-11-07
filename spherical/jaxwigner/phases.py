# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Conversion from quaternions to the complex phases used by the 𝔇 recursion."""

import jax.numpy as jnp
from jax import lax


def euler_phases(R):
    """Return (zₐ, zᵦ, zᵧ) = (e^{iα}, e^{iβ}, e^{iγ}) for a batch of rotors.

    This is a JAX translation of `quaternionic.converters.to_euler_phases`, and
    gives the same values, in the same order as that function's output `z`.

    Parameters
    ----------
    R : array_like, shape (..., 4)
        Quaternions `[w, x, y, z]`.  They need not be normalized: only their
        direction matters.  They must not be zero.

    Returns
    -------
    zₐ, zᵦ, zᵧ : complex arrays, each of shape R.shape[:-1]

    Notes
    -----
    With a = w² + z² and b = x² + y²,

        zᵦ = e^{iβ}       = [(a - b) + 2i √a √b] / (a + b)
        zₚ = e^{i(α+γ)/2} = (w + iz) / √a
        zₘ = e^{i(α-γ)/2} = (y - ix) / √b
        zₐ = zₚ zₘ,   zᵧ = zₚ z̄ₘ

    At the poles (a = 0 or b = 0) one of zₚ, zₘ is undefined; as in
    `quaternionic`, it is then set to 1.  The square roots are guarded with
    `jnp.where` so that no NaN is produced there, even in gradients of the
    values that do not depend on the undefined phase.

    """
    R = jnp.asarray(R)
    w, x, y, z = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    a = w * w + z * z
    b = x * x + y * y
    a_positive = a > 0
    b_positive = b > 0
    # "Double where": feed a harmless value to sqrt where the result is discarded,
    # so its derivative (which is infinite at 0) never reaches the gradient.
    sqrt_a = jnp.sqrt(jnp.where(a_positive, a, 1))
    sqrt_b = jnp.sqrt(jnp.where(b_positive, b, 1))
    sqrt_ab = jnp.where(a_positive & b_positive, sqrt_a * sqrt_b, 0)

    norm = a + b
    zᵦ = lax.complex((a - b) / norm, 2 * sqrt_ab / norm)
    one = jnp.ones_like(zᵦ)
    zₚ = jnp.where(a_positive, lax.complex(w, z) / sqrt_a, one)
    zₘ = jnp.where(b_positive, lax.complex(y, -x) / sqrt_b, one)
    zₐ = zₚ * zₘ
    zᵧ = zₚ * jnp.conj(zₘ)
    return zₐ, zᵦ, zᵧ
