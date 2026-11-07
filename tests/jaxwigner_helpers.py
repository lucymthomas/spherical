# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Shared rotors, tolerances and helpers for the `spherical.jaxwigner` tests."""

import math

import numpy as np
import quaternionic

import spherical

ELL_MAX = 8  # the largest ℓ this implementation is designed for
EPS = np.finfo(float).eps


def tolerance(ell_max):
    """Absolute tolerance for comparing float64 results up to ell_max."""
    return 50 * (ell_max + 1) * EPS


def random_rotors(n, seed=1234):
    rng = np.random.default_rng(seed)
    return quaternionic.array(rng.normal(size=(n, 4))).normalized.ndarray


def special_rotors():
    """Rotors at and near the poles β = 0 and β = π, and other awkward cases."""
    δ = 1e-8
    q = [
        [1, 0, 0, 0],  # identity
        [-1, 0, 0, 0],  # identity, other sign
        [0, 0, 0, 1],  # rotation about z by π: β = 0 with α + γ = π
        [math.cos(0.3), 0, 0, math.sin(0.3)],  # β = 0
        [0, 1, 0, 0],  # rotation about x by π: β = π
        [0, 0, 1, 0],  # rotation about y by π: β = π
        [0, math.cos(0.7), math.sin(0.7), 0],  # β = π
        [1, δ, 0, 0],  # β just above 0
        [1, 0, δ, δ],
        [δ, 1, 0, 0],  # β just below π
        [0, 1, δ, δ],
        [1, 1, 1, 1],  # β = π/2
    ]
    return quaternionic.array(np.array(q, dtype=float)).normalized.ndarray


def all_rotors():
    return np.concatenate([special_rotors(), random_rotors(40)])


def random_modes(shape, ell_max, ell_min=0, spin_weight=0, seed=4321):
    """Random complex mode weights, zero for ℓ < |spin_weight|."""
    rng = np.random.default_rng(seed)
    n = spherical.Ysize(ell_min, ell_max)
    f = rng.normal(size=shape + (n,)) + 1j * rng.normal(size=shape + (n,))
    for ell in range(ell_min, min(abs(spin_weight), ell_max + 1)):
        f[..., spherical.Yindex(ell, -ell, ell_min): spherical.Yindex(ell, ell, ell_min) + 1] = 0
    return f


def D_block(D, ell, ell_min=0):
    """The (2ℓ+1)×(2ℓ+1) matrix 𝔇ˡ_{m′,m} from a flat 𝔇 array, with batch dims."""
    i1 = spherical.WignerDindex(ell, -ell, -ell, ell_min)
    i2 = spherical.WignerDindex(ell, ell, ell, ell_min) + 1
    return D[..., i1:i2].reshape(D.shape[:-1] + (2 * ell + 1, 2 * ell + 1))


def mode_block(f, ell, ell_min=0):
    """The 2ℓ+1 mode weights fₗₘ for one ℓ, with batch dims."""
    return f[..., spherical.Yindex(ell, -ell, ell_min): spherical.Yindex(ell, ell, ell_min) + 1]
