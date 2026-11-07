# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Tests of `rotate`, `evaluate` and the quaternion helpers in `spherical.jaxwigner`.

These are checked three ways: against the numba `spherical.Wigner.rotate` and
`.evaluate`; against the JAX 𝔇 matrices and harmonics (already tested against
numba in test_jaxwigner.py), which confirms that skipping the full 𝔇 changes
nothing; and through identities such as composition and norm preservation.

"""

import numpy as np
import pytest
import quaternionic

import spherical

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from spherical import jaxwigner as jw  # noqa: E402

from .jaxwigner_helpers import (  # noqa: E402
    ELL_MAX, EPS, tolerance, random_rotors, all_rotors, random_modes, D_block, mode_block,
)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def test_multiply_vs_quaternionic():
    q1, q2 = random_rotors(20, seed=1), random_rotors(20, seed=2)
    expected = (quaternionic.array(q1) * quaternionic.array(q2)).ndarray
    np.testing.assert_allclose(np.asarray(jw.multiply(q1, q2)), expected, rtol=0, atol=4 * EPS)


def test_inverse_and_conjugate():
    q = 2.5 * random_rotors(20)
    identity = np.broadcast_to([1.0, 0, 0, 0], q.shape)
    np.testing.assert_allclose(np.asarray(jw.multiply(q, jw.inverse(q))), identity, rtol=0, atol=10 * EPS)
    unit = random_rotors(20)
    np.testing.assert_allclose(np.asarray(jw.multiply(unit, jw.conjugate(unit))), identity, rtol=0, atol=10 * EPS)


def test_from_angles_vs_quaternionic():
    rng = np.random.default_rng(7)
    α, β, γ = rng.uniform(-2 * np.pi, 2 * np.pi, size=(3, 30))
    np.testing.assert_allclose(
        np.asarray(jw.from_euler_angles(α, β, γ)),
        quaternionic.array.from_euler_angles(α, β, γ).ndarray, rtol=0, atol=4 * EPS,
    )
    θ, ϕ = rng.uniform(0, np.pi, 30), rng.uniform(0, 2 * np.pi, 30)
    np.testing.assert_allclose(
        np.asarray(jw.from_spherical_coordinates(θ, ϕ)),
        quaternionic.array.from_spherical_coordinates(θ, ϕ).ndarray, rtol=0, atol=4 * EPS,
    )


def test_from_spherical_coordinates_broadcasts():
    θ = np.linspace(0, np.pi, 5)[:, None]
    ϕ = np.linspace(0, 2 * np.pi, 7)[None, :]
    assert jw.from_spherical_coordinates(θ, ϕ).shape == (5, 7, 4)


def test_ell_max_from_size():
    for ell_min in range(4):
        for ell_max in range(ell_min, 10):
            assert jw.ell_max_from_size(spherical.Ysize(ell_min, ell_max), ell_min) == ell_max
    with pytest.raises(ValueError):
        jw.ell_max_from_size(10, 0)
    with pytest.raises(ValueError):
        jw.ell_max_from_size(0, 0)


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ell_max", [0, 1, 2, 5, ELL_MAX])
def test_rotate_vs_spherical(ell_max):
    R = all_rotors()
    f = random_modes((len(R),), ell_max)
    wigner = spherical.Wigner(ell_max)
    expected = np.array([
        wigner.rotate(spherical.Modes(f[i], spin_weight=0), R[i]).ndarray for i in range(len(R))
    ])
    np.testing.assert_allclose(np.asarray(jw.rotate(f, R)), expected, rtol=0, atol=tolerance(ell_max))


@pytest.mark.parametrize("ell_min", [0, 2, 5])
def test_rotate_vs_wigner_D(ell_min):
    # f′ₗₘ = Σₘ′ fₗₘ′ 𝔇ˡ_{m′,m}, using the JAX 𝔇 for each ℓ block
    R = all_rotors()
    f = random_modes((len(R),), ELL_MAX, ell_min)
    rotated = np.asarray(jw.rotate(f, R, ell_min))
    D = np.asarray(jw.wigner_D(R, ELL_MAX, ell_min))
    for ell in range(ell_min, ELL_MAX + 1):
        expected = np.einsum("...i,...ij->...j", mode_block(f, ell, ell_min), D_block(D, ell, ell_min))
        np.testing.assert_allclose(mode_block(rotated, ell, ell_min), expected, rtol=0, atol=tolerance(ell))


def test_rotate_by_identity():
    f = random_modes((3,), ELL_MAX)
    for q in ([1.0, 0, 0, 0], [-1.0, 0, 0, 0]):
        np.testing.assert_allclose(np.asarray(jw.rotate(f, np.array(q))), f, rtol=0, atol=tolerance(ELL_MAX))


def test_rotate_composition_and_inverse():
    R1, R2 = all_rotors(), random_rotors(len(all_rotors()), seed=11)
    f = random_modes((len(R1),), ELL_MAX)
    twice = jw.rotate(jw.rotate(f, R1), R2)
    once = jw.rotate(f, jw.multiply(R1, R2))
    np.testing.assert_allclose(np.asarray(twice), np.asarray(once), rtol=0, atol=tolerance(ELL_MAX))
    back = jw.rotate(jw.rotate(f, R1), jw.conjugate(R1))
    np.testing.assert_allclose(np.asarray(back), f, rtol=0, atol=tolerance(ELL_MAX))


def test_rotate_preserves_norm_per_ell():
    R = all_rotors()
    f = random_modes((len(R),), ELL_MAX)
    rotated = np.asarray(jw.rotate(f, R))
    for ell in range(ELL_MAX + 1):
        np.testing.assert_allclose(
            np.linalg.norm(mode_block(rotated, ell), axis=-1),
            np.linalg.norm(mode_block(f, ell), axis=-1),
            rtol=tolerance(ell), atol=0,
        )


def test_rotate_broadcasting():
    T, N = 6, 5
    f_T = random_modes((T,), 4)
    R_T = random_rotors(T)
    R_N = random_rotors(N, seed=3)
    # one rotor per set of modes
    assert jw.rotate(f_T, R_T).shape == (T, f_T.shape[-1])
    # one set of modes, many rotors
    many = np.asarray(jw.rotate(f_T[0], R_N))
    assert many.shape == (N, f_T.shape[-1])
    np.testing.assert_allclose(many[2], np.asarray(jw.rotate(f_T[0], R_N[2])), rtol=0, atol=tolerance(4))
    # every combination, via explicit axes
    grid = np.asarray(jw.rotate(f_T[:, None, :], R_N[None, :, :]))
    assert grid.shape == (T, N, f_T.shape[-1])
    np.testing.assert_allclose(grid[4, 1], np.asarray(jw.rotate(f_T[4], R_N[1])), rtol=0, atol=tolerance(4))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [-2, 0, 1])
def test_evaluate_vs_spherical(s):
    R = all_rotors()
    f = random_modes((len(R),), ELL_MAX, spin_weight=s)
    wigner = spherical.Wigner(ELL_MAX)
    expected = np.array([
        wigner.evaluate(spherical.Modes(f[i], spin_weight=s), R[i]) for i in range(len(R))
    ]).ravel()
    np.testing.assert_allclose(np.asarray(jw.evaluate(f, R, s)), expected, rtol=0, atol=tolerance(ELL_MAX))


@pytest.mark.parametrize("s", [-2, 0, 3])
@pytest.mark.parametrize("ell_min", [0, 2])
def test_evaluate_vs_sYlm(s, ell_min):
    R = all_rotors()
    f = random_modes((len(R),), ELL_MAX, ell_min, spin_weight=s)
    expected = np.sum(f * np.asarray(jw.sYlm(s, R, ELL_MAX, ell_min)), axis=-1)
    np.testing.assert_allclose(np.asarray(jw.evaluate(f, R, s, ell_min)), expected, rtol=0, atol=tolerance(ELL_MAX))


def test_evaluate_ignores_modes_below_spin_weight():
    # ₛYₗₘ = 0 for ℓ < |s|, so whatever is stored there must not matter.
    R = random_rotors(10)
    f = random_modes((10,), ELL_MAX, spin_weight=-2)
    g = f.copy()
    g[..., : spherical.Yindex(2, -2)] = 1e3
    np.testing.assert_array_equal(np.asarray(jw.evaluate(f, R, -2)), np.asarray(jw.evaluate(g, R, -2)))


def test_evaluate_on_grid_shape():
    f = random_modes((), ELL_MAX, spin_weight=-2)
    θ = np.linspace(0, np.pi, 9)[:, None]
    ϕ = np.linspace(0, 2 * np.pi, 13)[None, :]
    values = jw.evaluate(f, jw.from_spherical_coordinates(θ, ϕ), -2)
    assert values.shape == (9, 13)


def test_evaluate_of_rotated_modes_is_evaluate_at_composed_rotor():
    # The identity that makes rotating unnecessary for a waveform at one point:
    # evaluate(rotate(f, R₁), R₂) = evaluate(f, R₁ R₂)
    s = -2
    R1 = all_rotors()
    R2 = random_rotors(len(R1), seed=21)
    f = random_modes((len(R1),), ELL_MAX, spin_weight=s)
    two_steps = jw.evaluate(jw.rotate(f, R1), R2, s)
    one_step = jw.evaluate(f, jw.multiply(R1, R2), s)
    np.testing.assert_allclose(np.asarray(two_steps), np.asarray(one_step), rtol=0, atol=tolerance(ELL_MAX))


# ---------------------------------------------------------------------------
# JAX behavior
# ---------------------------------------------------------------------------

def test_float32():
    R = all_rotors()
    f = random_modes((len(R),), ELL_MAX, spin_weight=-2)
    rotated32 = jw.rotate(f.astype(np.complex64), R.astype(np.float32))
    values32 = jw.evaluate(f.astype(np.complex64), R.astype(np.float32), -2)
    assert rotated32.dtype == jnp.complex64
    assert values32.dtype == jnp.complex64
    scale = np.max(np.abs(f))
    atol = 50 * (ELL_MAX + 1) * 1.2e-7 * scale * f.shape[-1] ** 0.5
    np.testing.assert_allclose(np.asarray(rotated32), np.asarray(jw.rotate(f, R)), rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(values32), np.asarray(jw.evaluate(f, R, -2)), rtol=0, atol=atol)


def test_inside_user_jit_and_vmap():
    R = all_rotors()
    f = random_modes((len(R),), 4, spin_weight=-2)

    @jax.jit
    def strain(f, R):
        return jw.evaluate(jw.rotate(f, R), R, -2)

    expected = np.asarray(jw.evaluate(jw.rotate(f, R), R, -2))
    np.testing.assert_allclose(np.asarray(strain(f, R)), expected, rtol=0, atol=tolerance(4))
    vmapped = jax.vmap(lambda f, q: jw.evaluate(f, q, -2))(f, R)
    np.testing.assert_allclose(np.asarray(vmapped), np.asarray(jw.evaluate(f, R, -2)), rtol=0, atol=tolerance(4))


def test_invalid_arguments():
    R = random_rotors(2)
    with pytest.raises(ValueError):
        jw.rotate(np.zeros((2, 10), dtype=complex), R)  # 10 is not a valid number of modes
    with pytest.raises(ValueError):
        jw.rotate(np.zeros((2, 9), dtype=complex), R, ell_min=-1)
    with pytest.raises(TypeError):
        jw.evaluate(np.zeros((2, 9), dtype=complex), R, -2.0)
