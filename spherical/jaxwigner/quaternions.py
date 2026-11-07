# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Minimal quaternion helpers in JAX, matching the conventions of `quaternionic`.

Quaternions are arrays whose last axis holds `[w, x, y, z]`.  These functions
exist so that rotors can be built, composed and differentiated inside
`jax.jit`, where `quaternionic` (a NumPy/numba package) cannot be used.

"""

import jax.numpy as jnp


def multiply(q1, q2):
    """Quaternion product q1 q2, broadcasting over leading axes.

    With the conventions used throughout `spherical`, 𝔇(q1 q2) = 𝔇(q1) 𝔇(q2).
    """
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)
    a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    ], axis=-1)


def conjugate(q):
    """Quaternion conjugate [w, -x, -y, -z]; the inverse of a unit quaternion."""
    q = jnp.asarray(q)
    return q * jnp.asarray([1, -1, -1, -1], dtype=q.dtype)


def inverse(q):
    """Quaternion inverse q̄ / |q|²."""
    q = jnp.asarray(q)
    return conjugate(q) / jnp.sum(q * q, axis=-1, keepdims=True)


def _exp_z(angle):
    """exp(angle z / 2)"""
    zero = jnp.zeros_like(angle)
    return jnp.stack([jnp.cos(angle / 2), zero, zero, jnp.sin(angle / 2)], axis=-1)


def _exp_y(angle):
    """exp(angle y / 2)"""
    zero = jnp.zeros_like(angle)
    return jnp.stack([jnp.cos(angle / 2), zero, jnp.sin(angle / 2), zero], axis=-1)


def from_euler_angles(α, β, γ):
    """Rotor R = exp(α z/2) exp(β y/2) exp(γ z/2), as `quaternionic.array.from_euler_angles`.

    The arguments broadcast against each other.  The result has shape
    broadcast(α, β, γ).shape + (4,).
    """
    α, β, γ = jnp.broadcast_arrays(*(jnp.asarray(x) for x in (α, β, γ)))
    return multiply(multiply(_exp_z(α), _exp_y(β)), _exp_z(γ))


def from_spherical_coordinates(θ, ϕ):
    """Rotor R = exp(ϕ z/2) exp(θ y/2), as `quaternionic.array.from_spherical_coordinates`.

    This rotates z onto the point (θ, ϕ) on the sphere, and x and y onto the
    unit vectors θ̂ and ϕ̂ there.  Spin-weighted harmonics evaluated on this
    rotor are the usual ₛYₗₘ(θ, ϕ).
    """
    θ, ϕ = jnp.broadcast_arrays(jnp.asarray(θ), jnp.asarray(ϕ))
    return multiply(_exp_z(ϕ), _exp_y(θ))
