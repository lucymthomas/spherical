# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Tests of derivatives in `spherical.jaxwigner`.

`wigner_D`, `sYlm`, `rotate` and `evaluate` have custom derivative rules
(`spherical.jaxwigner.generators`), because plain automatic differentiation of
their implementations is wrong at the poles β = 0, π.  These tests check the
rules in three independent ways:

1. Away from the poles, plain automatic differentiation of the underlying
   implementation is valid, and must agree with the custom rules.
2. Everywhere, including at the poles, derivatives must agree with finite
   differences, which only use function values.
3. JAX's `check_grads` compares forward and reverse mode, to second order,
   against finite differences.

"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from spherical import jaxwigner as jw  # noqa: E402
from spherical.jaxwigner import matrices, modes as modes_module  # noqa: E402

from .jaxwigner_helpers import random_rotors, special_rotors, random_modes  # noqa: E402

ELL_MAX = 4  # derivatives are tested at a smaller ℓ to keep compile times down
S = -2


def split(f):
    """Wrap a complex-valued function to return stacked real and imaginary parts."""
    def wrapped(*args):
        value = f(*args)
        return jnp.stack([jnp.real(value), jnp.imag(value)])
    return wrapped


def central_difference(f, x, direction, h=1e-6):
    return (f(x + h * direction) - f(x - h * direction)) / (2 * h)


def functions_of_rotor():
    """(name, f(R), plain implementation of f(R)) for each differentiable function."""
    f = jnp.asarray(random_modes((), ELL_MAX, spin_weight=S))
    return [
        ("wigner_D",
         lambda R: jw.wigner_D(R, ELL_MAX),
         lambda R: matrices._wigner_D.fun(R, ELL_MAX, 0)),
        ("sYlm",
         lambda R: jw.sYlm(S, R, ELL_MAX),
         lambda R: matrices._sYlm.fun(S, R, ELL_MAX, 0)),
        ("rotate",
         lambda R: jw.rotate(f, R),
         lambda R: modes_module._rotate_impl(f, R, 0)),
        ("evaluate",
         lambda R: jw.evaluate(f, R, S),
         lambda R: modes_module._evaluate.fun(f, R, S, 0)),
    ]


FUNCTIONS = functions_of_rotor()
NAMES = [name for name, _, _ in FUNCTIONS]


@pytest.mark.parametrize("index", range(len(FUNCTIONS)), ids=NAMES)
def test_custom_rule_matches_plain_autodiff_away_from_poles(index):
    _, f, f_plain = FUNCTIONS[index]
    for R in random_rotors(6, seed=31):
        R = jnp.asarray(R)
        expected = jax.jacfwd(split(f_plain))(R)
        np.testing.assert_allclose(np.asarray(jax.jacfwd(split(f))(R)), np.asarray(expected), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(jax.jacrev(split(f))(R)), np.asarray(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("index", range(len(FUNCTIONS)), ids=NAMES)
def test_derivatives_match_finite_differences_everywhere(index):
    # Including at and near the poles, where plain autodiff is wrong.
    _, f, _ = FUNCTIONS[index]
    for R in np.concatenate([special_rotors(), random_rotors(4, seed=32)]):
        R = jnp.asarray(R)
        J = jax.jacfwd(split(f))(R)
        J_reverse = jax.jacrev(split(f))(R)
        for k in range(4):
            direction = jnp.zeros(4).at[k].set(1.0)
            expected = central_difference(split(f), R, direction)
            np.testing.assert_allclose(np.asarray(J[..., k]), np.asarray(expected), rtol=0, atol=1e-7)
            np.testing.assert_allclose(np.asarray(J_reverse[..., k]), np.asarray(expected), rtol=0, atol=1e-7)


@pytest.mark.parametrize("index", range(len(FUNCTIONS)), ids=NAMES)
def test_check_grads_second_order(index):
    # A step of 1e-6 keeps the finite-difference error (∝ step²) well inside
    # check_grads' tolerances; its default step is too large for these functions.
    _, f, _ = FUNCTIONS[index]
    for R in (special_rotors()[[0, 2, 4, 7]], random_rotors(3, seed=33)):
        check_grads(f, (jnp.asarray(R),), order=2, modes=["fwd", "rev"], eps=1e-6)


def test_derivative_ignores_normalization():
    # 𝔇 depends only on the direction of R, so its derivative along R itself vanishes.
    for R in np.concatenate([special_rotors(), random_rotors(4)]):
        R = jnp.asarray(R)
        _, δD = jax.jvp(lambda q: jw.wigner_D(q, ELL_MAX), (R,), (R,))
        np.testing.assert_allclose(np.asarray(δD), 0, atol=1e-13)


def test_gradient_with_respect_to_sky_position_at_poles():
    # Face-on (θ = 0) and face-off (θ = π) observers are common and lie on the poles.
    f = jnp.asarray(random_modes((), ELL_MAX, spin_weight=S))
    strain = split(lambda θ, ϕ: jw.evaluate(f, jw.from_spherical_coordinates(θ, ϕ), S))
    for θ in (0.0, 1e-9, 0.8, np.pi - 1e-9, np.pi):
        for ϕ in (0.0, 1.3):
            for argnum in (0, 1):
                derivative = jax.jacfwd(strain, argnums=argnum)(θ, ϕ)
                if argnum == 0:
                    expected = central_difference(lambda t: strain(t, ϕ), θ, 1.0)
                else:
                    expected = central_difference(lambda p: strain(θ, p), ϕ, 1.0)
                np.testing.assert_allclose(np.asarray(derivative), np.asarray(expected), rtol=0, atol=1e-7)


def test_gradient_with_respect_to_euler_angles_at_poles():
    f = jnp.asarray(random_modes((), ELL_MAX))
    rotated = split(lambda angles: jw.rotate(f, jw.from_euler_angles(*angles)))
    for angles in ([0.3, 0.0, -0.5], [0.3, np.pi, -0.5], [0.3, 1.2, -0.5]):
        angles = jnp.asarray(angles)
        J = jax.jacfwd(rotated)(angles)
        for k in range(3):
            expected = central_difference(rotated, angles, jnp.zeros(3).at[k].set(1.0))
            np.testing.assert_allclose(np.asarray(J[..., k]), np.asarray(expected), rtol=0, atol=1e-7)


def test_second_derivatives_at_pole():
    # Hessian of a real loss with respect to the observer inclination, at θ = 0:
    # all orders of forward/reverse mode agree, and match finite differences of
    # the (exact) gradient.
    f = jnp.asarray(random_modes((), ELL_MAX, spin_weight=S))

    def loss(θ):
        return jnp.abs(jw.evaluate(f, jw.from_spherical_coordinates(θ, 0.4), S)) ** 2

    θ = 0.0
    h_ff = jax.jacfwd(jax.jacfwd(loss))(θ)
    h_rr = jax.jacrev(jax.jacrev(loss))(θ)
    h_fr = jax.jacfwd(jax.grad(loss))(θ)
    expected = central_difference(jax.grad(loss), θ, 1.0)
    for h in (h_ff, h_rr, h_fr):
        np.testing.assert_allclose(float(h), float(expected), rtol=1e-8)


def test_derivative_with_respect_to_modes():
    # evaluate and rotate are linear in the modes.
    R = jnp.asarray(random_rotors(1)[0])
    f = jnp.asarray(random_modes((), ELL_MAX, spin_weight=S))
    J = jax.jacfwd(lambda f: jw.evaluate(f, R, S), holomorphic=True)(f)
    np.testing.assert_allclose(np.asarray(J), np.asarray(jw.sYlm(S, R, ELL_MAX)), rtol=0, atol=1e-14)
    J = jax.jacfwd(lambda f: jw.rotate(f, R), holomorphic=True)(f)
    identity_rotated = jw.rotate(jnp.eye(f.shape[-1], dtype=f.dtype), R)
    np.testing.assert_allclose(np.asarray(J), np.asarray(identity_rotated).T, rtol=0, atol=1e-14)


def test_gradients_finite_at_special_rotors():
    f = jnp.asarray(random_modes((len(special_rotors()),), ELL_MAX, spin_weight=S))

    def loss(R):
        return jnp.sum(jnp.abs(jw.evaluate(jw.rotate(f, R), R, S)) ** 2)

    g = jax.grad(loss)(jnp.asarray(special_rotors()))
    assert np.all(np.isfinite(np.asarray(g)))


def test_grad_composes_with_jit_and_vmap():
    f = jnp.asarray(random_modes((), ELL_MAX, spin_weight=S))

    def loss(θ, ϕ):
        return jnp.abs(jw.evaluate(f, jw.from_spherical_coordinates(θ, ϕ), S)) ** 2

    θ = jnp.array([0.0, 0.5, np.pi])
    ϕ = jnp.array([0.1, 0.2, 0.3])
    batched = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))(θ, ϕ)
    # (At θ = 0 and π, |h|² does not depend on ϕ, so ∂/∂ϕ is zero up to rounding;
    # hence an absolute tolerance.)
    for i in range(3):
        single = jax.grad(loss, argnums=(0, 1))(θ[i], ϕ[i])
        np.testing.assert_allclose(np.asarray(batched[0][i]), np.asarray(single[0]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(batched[1][i]), np.asarray(single[1]), rtol=1e-12, atol=1e-12)


def test_float32_gradients():
    f = random_modes((), ELL_MAX, spin_weight=S)

    def loss(θ, f):
        return jnp.abs(jw.evaluate(f, jw.from_spherical_coordinates(θ, 0.4), S)) ** 2

    for θ in (0.0, 0.9):
        g32 = jax.grad(loss)(jnp.float32(θ), f.astype(np.complex64))
        g64 = jax.grad(loss)(θ, f)
        assert g32.dtype == jnp.float32
        np.testing.assert_allclose(float(g32), float(g64), rtol=1e-4, atol=1e-4)


def test_wigner_d_derivative_in_beta():
    # wigner_d takes e^{iβ} directly; it is smooth in β, including at 0 and π.
    def d_of_β(β):
        return jw.wigner_d(jnp.exp(1j * β), ELL_MAX)

    for β in (0.0, 0.7, np.pi):
        np.testing.assert_allclose(
            np.asarray(jax.jacfwd(d_of_β)(β)), np.asarray(central_difference(d_of_β, β, 1.0)),
            rtol=0, atol=1e-7,
        )
