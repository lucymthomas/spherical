# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Batched, GPU-friendly Wigner 𝔇 matrices and spin-weighted harmonics in JAX.

This subpackage reimplements the core of `spherical.Wigner` in JAX, aimed at
modest ℓ (the design target is ℓ_max ≤ 8, as used in gravitational-waveform
models) evaluated for very large batches of rotations.

Design
------
The recursion of Gumerov & Duraiswami (arXiv:1403.7698), as implemented in
`spherical.recursions.wignerH`, is unrolled at trace time for a fixed
`ell_max`.  Every intermediate quantity is an array with the batch shape of the
input rotors, and every operation is elementwise across that batch.  That means
XLA can fuse the recursion into a few kernels in which each GPU thread handles
one rotor.

Conventions are identical to `spherical`: quaternions are `[w, x, y, z]`, the
𝔇 and d matrices are returned as flat arrays indexed by `spherical.WignerDindex`,
and harmonics and mode weights by `spherical.Yindex`.

`rotate` and `evaluate` work directly from the H wedge, without building 𝔇, and
broadcast the batch shapes of the modes and rotors against each other.

`wigner_D`, `sYlm`, `rotate` and `evaluate` can be differentiated with respect
to the rotor (and the modes), in forward and reverse mode and to any order.
Their derivative rules come from the Lie algebra of rotations rather than from
differentiating the implementation, so they are exact everywhere, including at
the poles β = 0, π where plain automatic differentiation would fail; see
`spherical.jaxwigner.generators`.

JAX defaults to 32-bit floats.  For results comparable to `spherical`, enable
64-bit mode before creating any arrays:

    import jax
    jax.config.update("jax_enable_x64", True)

This module never changes that setting itself.

"""

from .phases import euler_phases
from .quaternions import multiply, conjugate, inverse, from_euler_angles, from_spherical_coordinates
from .recursion import H_wedge, H_element, flatten_wedge, wedge_layout
from .matrices import wigner_d, wigner_D, sYlm
from .modes import rotate, evaluate, ell_max_from_size

__all__ = [
    "euler_phases",
    "multiply", "conjugate", "inverse", "from_euler_angles", "from_spherical_coordinates",
    "H_wedge", "H_element", "flatten_wedge", "wedge_layout",
    "wigner_d", "wigner_D", "sYlm",
    "rotate", "evaluate", "ell_max_from_size",
]
