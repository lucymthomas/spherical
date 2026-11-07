# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Tests of the JAX implementation in `spherical.jaxwigner`.

The numba implementation in `spherical.Wigner` is the reference.  On top of
direct comparisons, these tests check mathematical identities that do not
depend on any reference implementation, numerical edge cases (poles,
unnormalized and negated quaternions), and JAX-specific behavior (batch
shapes, vmap, single precision).

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
    ELL_MAX, EPS, tolerance, random_rotors, special_rotors, all_rotors, D_block,
)


# ---------------------------------------------------------------------------
# Comparison with the numba implementation
# ---------------------------------------------------------------------------

def test_euler_phases_vs_quaternionic():
    R = all_rotors()
    zₐ, zᵦ, zᵧ = (np.asarray(z) for z in jw.euler_phases(R))
    expected = quaternionic.array(R).to_euler_phases
    np.testing.assert_allclose(zₐ, expected[:, 0], rtol=0, atol=10 * EPS)
    np.testing.assert_allclose(zᵦ, expected[:, 1], rtol=0, atol=10 * EPS)
    np.testing.assert_allclose(zᵧ, expected[:, 2], rtol=0, atol=10 * EPS)


def test_euler_phases_ignore_normalization():
    R = all_rotors()
    for scale in [1e-3, 2.5, 1e3]:
        for z, z_scaled in zip(jw.euler_phases(R), jw.euler_phases(scale * R)):
            np.testing.assert_allclose(z_scaled, z, rtol=0, atol=10 * EPS)


@pytest.mark.parametrize("ell_max", range(ELL_MAX + 1))
def test_d_vs_spherical(ell_max):
    β = np.concatenate([[0.0, np.pi, 1e-8, np.pi - 1e-8, np.pi / 2],
                        np.random.default_rng(5).uniform(0, np.pi, 20)])
    expiβ = np.exp(1j * β)
    for ell_min in sorted({0, ell_max // 2, ell_max}):
        wigner = spherical.Wigner(ell_max, ell_min=ell_min)
        expected = np.array([wigner.d(z) for z in expiβ])
        result = np.asarray(jw.wigner_d(expiβ, ell_max, ell_min))
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=0, atol=tolerance(ell_max))


@pytest.mark.parametrize("ell_max", range(ELL_MAX + 1))
def test_D_vs_spherical(ell_max):
    R = all_rotors()
    for ell_min in sorted({0, ell_max // 2, ell_max}):
        expected = spherical.Wigner(ell_max, ell_min=ell_min).D(R)
        result = np.asarray(jw.wigner_D(R, ell_max, ell_min))
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=0, atol=tolerance(ell_max))


@pytest.mark.parametrize("s", range(-3, 4))
def test_sYlm_vs_spherical(s):
    # Each (s, ell_min, ell_max) compiles a new function, so test a representative
    # subset rather than every combination.
    R = all_rotors()
    for ell_max in sorted({abs(s), abs(s) + 1, ELL_MAX}):
        for ell_min in sorted({0, ell_max // 2, ell_max}):
            expected = spherical.Wigner(ell_max, ell_min=ell_min).sYlm(s, R)
            result = np.asarray(jw.sYlm(s, R, ell_max, ell_min))
            assert result.shape == expected.shape
            np.testing.assert_allclose(result, expected, rtol=0, atol=tolerance(ell_max))


def test_sYlm_zero_below_spin_weight():
    R = random_rotors(5)
    Y = np.asarray(jw.sYlm(-2, R, 4))
    assert np.all(Y[:, : spherical.Yindex(2, -2)] == 0)


# ---------------------------------------------------------------------------
# Identities that do not depend on a reference implementation
# ---------------------------------------------------------------------------

def test_d_closed_forms_ell_1():
    # Convention-independent elements of d¹: d¹₀₀ = cosβ, d¹₁₁ = d¹₋₁₋₁ = (1+cosβ)/2,
    # and d¹₁₋₁ = d¹₋₁₁ = (1-cosβ)/2.
    β = np.linspace(0, np.pi, 17)
    d = np.asarray(jw.wigner_d(np.exp(1j * β), 1))
    index = lambda mp, m: spherical.WignerDindex(1, mp, m)
    atol = 10 * EPS
    np.testing.assert_allclose(d[:, index(0, 0)], np.cos(β), atol=atol)
    np.testing.assert_allclose(d[:, index(1, 1)], (1 + np.cos(β)) / 2, atol=atol)
    np.testing.assert_allclose(d[:, index(-1, -1)], (1 + np.cos(β)) / 2, atol=atol)
    np.testing.assert_allclose(d[:, index(1, -1)], (1 - np.cos(β)) / 2, atol=atol)
    np.testing.assert_allclose(d[:, index(-1, 1)], (1 - np.cos(β)) / 2, atol=atol)


def test_D_identity_rotation():
    D = np.asarray(jw.wigner_D(np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]]), ELL_MAX))
    for ell in range(ELL_MAX + 1):
        np.testing.assert_allclose(
            D_block(D, ell), np.broadcast_to(np.eye(2 * ell + 1), (2, 2 * ell + 1, 2 * ell + 1)),
            rtol=0, atol=10 * EPS,
        )


def test_D_unitary():
    D = np.asarray(jw.wigner_D(all_rotors(), ELL_MAX))
    for ell in range(ELL_MAX + 1):
        Dˡ = D_block(D, ell)
        product = Dˡ @ np.conj(np.swapaxes(Dˡ, -1, -2))
        np.testing.assert_allclose(
            product, np.broadcast_to(np.eye(2 * ell + 1), product.shape),
            rtol=0, atol=tolerance(ell),
        )


def test_D_composition():
    # 𝔇ˡ(R₁R₂) = 𝔇ˡ(R₁) 𝔇ˡ(R₂), the representation property
    R1 = quaternionic.array(all_rotors())
    R2 = quaternionic.array(random_rotors(len(R1), seed=99))
    D1, D2, D12 = (np.asarray(jw.wigner_D(R.ndarray, ELL_MAX)) for R in (R1, R2, R1 * R2))
    for ell in range(ELL_MAX + 1):
        np.testing.assert_allclose(
            D_block(D1, ell) @ D_block(D2, ell), D_block(D12, ell),
            rtol=0, atol=tolerance(ell),
        )


def test_D_inverse():
    # 𝔇ˡ(R⁻¹) = 𝔇ˡ(R)†
    R = quaternionic.array(all_rotors())
    D, D_inverse = (np.asarray(jw.wigner_D(q.ndarray, ELL_MAX)) for q in (R, R.inverse))
    for ell in range(ELL_MAX + 1):
        np.testing.assert_allclose(
            D_block(D_inverse, ell), np.conj(np.swapaxes(D_block(D, ell), -1, -2)),
            rtol=0, atol=tolerance(ell),
        )


def test_D_invariant_under_sign_and_scale():
    # R and -R are the same rotation, and only the direction of R matters.
    R = all_rotors()
    D = np.asarray(jw.wigner_D(R, ELL_MAX))
    for R2 in (-R, 3.7 * R, 1e-3 * R):
        np.testing.assert_allclose(np.asarray(jw.wigner_D(R2, ELL_MAX)), D, rtol=0, atol=tolerance(ELL_MAX))


def test_sYlm_orthonormal():
    # ∫ ₛYₗₘ ₛȲₗ′ₘ′ dΩ = δₗₗ′ δₘₘ′, by Gauss–Legendre quadrature in cosθ and
    # uniform quadrature in ϕ, which is exact for these band-limited products.
    s, ell_max = -2, ELL_MAX
    n_θ, n_ϕ = ell_max + 2, 2 * ell_max + 2
    x, w = np.polynomial.legendre.leggauss(n_θ)
    θ = np.arccos(x)
    ϕ = np.arange(n_ϕ) * 2 * np.pi / n_ϕ
    Θ, Φ = np.meshgrid(θ, ϕ, indexing="ij")
    R = quaternionic.array.from_spherical_coordinates(Θ, Φ).ndarray
    Y = np.asarray(jw.sYlm(s, R, ell_max, abs(s)))  # shape (n_θ, n_ϕ, modes)
    weights = (w[:, None] * np.full(n_ϕ, 2 * np.pi / n_ϕ)[None, :])[..., None, None]
    gram = np.sum(weights * Y[..., :, None] * np.conj(Y[..., None, :]), axis=(0, 1))
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), rtol=0, atol=tolerance(ell_max))


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------

def test_no_nonfinite_values_at_poles():
    R = special_rotors()
    for f in (lambda: jw.wigner_D(R, ELL_MAX), lambda: jw.sYlm(-2, R, ELL_MAX)):
        assert np.all(np.isfinite(np.asarray(f())))


def test_d_continuous_at_poles():
    # Values just inside [0, π] should be close to the values at the endpoints.
    for β0 in (0.0, np.pi):
        β = np.array([β0, β0 + 1e-9 if β0 == 0 else β0 - 1e-9])
        d = np.asarray(jw.wigner_d(np.exp(1j * β), ELL_MAX))
        np.testing.assert_allclose(d[1], d[0], rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# JAX behavior
# ---------------------------------------------------------------------------

# XLA compiles a separate program for each input shape, and may vectorize or
# fuse multiply-adds differently in each, so results for the same rotor computed
# in batches of different shapes agree to rounding error, not bit for bit.

def test_arbitrary_batch_shapes():
    R = random_rotors(3 * 5).reshape(3, 5, 4)
    D = np.asarray(jw.wigner_D(R, 4))
    assert D.shape == (3, 5, spherical.WignerDsize(0, 4))
    np.testing.assert_allclose(
        D.reshape(15, -1), np.asarray(jw.wigner_D(R.reshape(15, 4), 4)), rtol=0, atol=tolerance(4)
    )
    D_single = np.asarray(jw.wigner_D(R[0, 0], 4))
    assert D_single.shape == (spherical.WignerDsize(0, 4),)


def test_single_rotor_matches_batch():
    R = all_rotors()
    D = np.asarray(jw.wigner_D(R, ELL_MAX))
    for i in [0, 3, 7, len(R) - 1]:
        np.testing.assert_allclose(
            np.asarray(jw.wigner_D(R[i], ELL_MAX)), D[i], rtol=0, atol=tolerance(ELL_MAX)
        )


def test_vmap_matches_batch():
    R = all_rotors()
    vmapped = jax.vmap(lambda q: jw.wigner_D(q, 4))(R)
    np.testing.assert_allclose(
        np.asarray(vmapped), np.asarray(jw.wigner_D(R, 4)), rtol=0, atol=tolerance(4)
    )


def test_float32():
    R = all_rotors()
    D32 = jw.wigner_D(R.astype(np.float32), ELL_MAX)
    Y32 = jw.sYlm(-2, R.astype(np.float32), ELL_MAX)
    assert D32.dtype == jnp.complex64
    assert Y32.dtype == jnp.complex64
    D64 = np.asarray(jw.wigner_D(R, ELL_MAX))
    # Single precision has eps ≈ 1.2e-7; allow growth with ℓ as for float64.
    np.testing.assert_allclose(np.asarray(D32), D64, rtol=0, atol=50 * (ELL_MAX + 1) * 1.2e-7)


def test_invalid_arguments():
    R = random_rotors(2)
    with pytest.raises(ValueError):
        jw.wigner_D(R, 2, 3)
    with pytest.raises(ValueError):
        jw.wigner_D(R, -1)
    with pytest.raises(TypeError):
        jw.sYlm(-2.0, R, 4)
