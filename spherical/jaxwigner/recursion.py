# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""The H recursion of Gumerov & Duraiswami (arXiv:1403.7698), unrolled for JAX.

This follows `spherical.recursions.wignerH` step by step, with three changes:

1. Each row of the wedge — fixed n and m′, all m ≥ |m′| — is computed as one
   vector operation over m, instead of element by element.  That keeps the
   number of operations XLA has to compile at O(ℓ²) rather than O(ℓ³), which
   is what keeps compile times short.

2. Rows are kept in a Python dict keyed by (n, m′) instead of a flat array with
   hand-computed offsets.  The dict exists only while tracing.

3. The auxiliary `Hv` array of the original is not needed.  It holds elements
   H^{m′+1, m′}_n just below the diagonal, which by symmetry equal elements of
   the wedge that the recursion has already computed.

H is related to Wigner's d by  dˡ_{m′,m} = ϵ(m′) ϵ(-m) Hˡ_{m′,m},  with
ϵ(k) = 1 for k ≤ 0 and (-1)^k for k > 0, and has the symmetries

    H^{m′,m}_n = H^{m,m′}_n = H^{-m′,-m}_n,

so only the "wedge" m ≥ |m′| is computed.

"""

from functools import lru_cache
import math

import numpy as np
import jax.numpy as jnp


# Recursion coefficients, matching `Wigner._a`, `_b`, `_d`, `_g` and `_h` in
# spherical/wigner.py.  They are evaluated in float64 at trace time and enter the
# compiled code as constants.

def _a(n, m):
    m = abs(m)
    return math.sqrt((n + 1 + m) * (n + 1 - m) / ((2 * n + 1) * (2 * n + 3)))


def _b(n, m):
    value = math.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1)))
    return -value if m < 0 else value


def _d(n, m):
    value = 0.5 * math.sqrt((n - m) * (n + m + 1))
    return -value if m < 0 else value


def _g(n, m):
    return 2 * (m + 1) / math.sqrt((n - m) * (n + m + 1))


def _h(n, m):
    return math.sqrt((n + m + 2) * (n - m - 1) / ((n - m) * (n + m + 1)))


def wedge_key(n, mp, m):
    """Map any (n, m′, m) to the equivalent (n, m′, m) with m ≥ |m′|, by symmetry."""
    if abs(m) >= abs(mp):
        return (n, mp, m) if m >= 0 else (n, -mp, -m)
    return (n, m, mp) if mp >= 0 else (n, -m, -mp)


def H_element(H, n, mp, m):
    """Look up Hⁿ_{m′,m} in the rows returned by `H_wedge`, using symmetry."""
    n, mp, m = wedge_key(n, mp, m)
    return H[n, mp][..., m - abs(mp)]


@lru_cache(maxsize=None)
def wedge_layout(ell_max):
    """Order in which `flatten_wedge` concatenates rows, and where each starts.

    Returns
    -------
    rows : tuple of (n, m′)
    offset : dict mapping (n, m′, m) in the wedge to its flat index
    size : int

    """
    rows = tuple((n, mp) for n in range(ell_max + 1) for mp in range(-n, n + 1))
    offset = {}
    i = 0
    for n, mp in rows:
        for m in range(abs(mp), n + 1):
            offset[n, mp, m] = i
            i += 1
    return rows, offset, i


def flatten_wedge(H, ell_max):
    """Concatenate the rows of `H_wedge` into one array, in `wedge_layout` order.

    Kept for inspection and tests.  The functions in this package do not use
    it: they read their outputs straight from the rows with `gather_wedge`,
    which avoids storing a second copy of the whole wedge.
    """
    rows, _, _ = wedge_layout(ell_max)
    return jnp.concatenate([H[row] for row in rows], axis=-1)


@lru_cache(maxsize=None)
def _wedge_pieces(keys):
    """Split a tuple of wedge keys (n, m′, m) into runs read from one row.

    Returns a tuple of (row, first, last, step): elements first, first+step, ...,
    last of row `row`, with step ±1.  Consecutive keys from the same row with
    neighbouring m form one run, so each run becomes one static slice.
    """
    pieces = []
    for n, mp, m in keys:
        row, i = (n, mp), m - abs(mp)
        if pieces:
            prow, first, last, step = pieces[-1]
            if prow == row and (i - last) in (1, -1) and (first == last or i - last == step):
                pieces[-1] = (row, first, i, i - last)
                continue
        pieces.append((row, i, i, 1))
    return tuple(pieces)


def gather_wedge(H, keys):
    """Array of the wedge elements `keys` (a tuple of (n, m′, m), each with m ≥ |m′|).

    Built from static slices of the rows of `H_wedge`, so XLA does not need the
    whole wedge flattened into one array first.
    """
    parts = []
    for row, first, last, step in _wedge_pieces(keys):
        x = H[row]
        if step == 1:
            parts.append(x[..., first: last + 1])
        else:
            parts.append(jnp.flip(x[..., last: first + 1], axis=-1))
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)


def _shift_left(x):
    """x[1:], padded with one trailing copy of x[-1] so the length is unchanged.

    Used for the H^{m′,m+1} term, whose coefficient dᵐ_n is exactly 0 at the one
    place (m = n) where the padding is read.
    """
    return jnp.concatenate([x[..., 1:], x[..., -1:]], axis=-1)


def H_wedge(expiβ, ell_max, mp_max=None):
    """Compute the wedge of H for all n ≤ ell_max.

    Parameters
    ----------
    expiβ : complex array
        e^{iβ} for each rotor; any batch shape.
    ell_max : int
        Largest ℓ required.  This must be a Python int: the recursion is
        unrolled for it at trace time.
    mp_max : int, optional
        Compute only the rows with |m′| ≤ mp_max (default: all of them).  The
        recursion for a row only uses rows with smaller |m′|, so these rows are
        exact, and the rest are never computed.  ₛYₗₘ, for example, needs only
        |m′| ≤ |s|.

    Returns
    -------
    H : dict
        Maps (n, m′) with |m′| ≤ min(n, mp_max), n ≤ ell_max, to a real array of shape
        expiβ.shape + (n - |m′| + 1,), holding H^{m′,m}_n for m = |m′|, ..., n.
        Use `H_element` to look up any element, including by symmetry.

    """
    ell_max = int(ell_max)
    if ell_max < 0:
        raise ValueError(f"ell_max={ell_max} must be non-negative")
    mp_max = ell_max if mp_max is None else min(int(mp_max), ell_max)
    if mp_max < 0:
        raise ValueError(f"mp_max={mp_max} must be non-negative")

    expiβ = jnp.asarray(expiβ)
    cosβ = jnp.real(expiβ)
    sinβ = jnp.imag(expiβ)
    dtype = cosβ.dtype
    sinβ_squared = sinβ * sinβ

    def const(values):
        """Coefficient vector as a constant of the working precision."""
        return jnp.asarray(np.asarray(values, dtype=float), dtype=dtype)

    def column(x):
        return x[..., None]

    # Step 1: H⁰₀₀ = 1
    H = {(0, 0): column(jnp.ones_like(cosβ))}
    if ell_max == 0:
        return H

    # Step 2: the m′=0 row, H^{0,m}_n for 0 ≤ m ≤ n and 1 ≤ n ≤ ell_max+1.  These
    # are normalized associated Legendre functions, computed as in Holmes &
    # Featherstone (2002): first without the factors of sin(β)^m, which are
    # applied afterwards.  Row ell_max+1 is needed only by step 3.  This
    # recursion runs in m within each n, so it is done one element at a time;
    # that is O(ℓ²) operations in total.
    sinβ_powers = [jnp.ones_like(sinβ)]
    for _ in range(ell_max + 1):
        sinβ_powers.append(sinβ_powers[-1] * sinβ)
    H0 = {}  # n -> array of H^{0,m}_n for m = 0, ..., n
    diagonal = {1: math.sqrt(3)}  # n -> H^{0,n}_n without its factor sin(β)^n / √(4n+2)
    H0[1] = jnp.stack(
        [(_g(1, 0) * cosβ) / math.sqrt(2), diagonal[1] * sinβ / math.sqrt(6)], axis=-1
    )
    for n in range(2, ell_max + (2 if mp_max >= 1 else 1)):
        row = {}
        row[n] = math.sqrt(1 + 0.5 / n) * diagonal[n - 1]
        diagonal[n] = row[n]
        row[n - 1] = _g(n, n - 1) * cosβ * row[n]
        for m in range(n - 2, 0, -1):
            row[m] = _g(n, m) * cosβ * row[m + 1] - _h(n, m) * sinβ_squared * row[m + 2]
        row[0] = _g(n, 0) * cosβ * row[1] - _h(n, 0) * sinβ_squared * row[2]
        unnormalized = jnp.stack([row[m] * jnp.ones_like(cosβ) for m in range(n + 1)], axis=-1)
        sinβᵐ = jnp.stack(sinβ_powers[: n + 1], axis=-1)
        H0[n] = unnormalized * sinβᵐ / math.sqrt(4 * n + 2)
    for n in range(1, ell_max + 1):
        H[n, 0] = H0[n]

    # Step 3: the m′=1 row, H^{1,m}_n for 1 ≤ m ≤ n, from row 0 at n+1:
    #   b⁰_{n+1} H^{1,m}_n = ½[b^{-m-1}_{n+1}(1-cosβ) H^{0,m+1}_{n+1}
    #                          - b^{m-1}_{n+1}(1+cosβ) H^{0,m-1}_{n+1}]
    #                        - aᵐ_n sinβ H^{0,m}_{n+1}
    if mp_max == 0:
        return H
    one_minus_cosβ = column(1 - cosβ)
    one_plus_cosβ = column(1 + cosβ)
    sinβ_column = column(sinβ)
    for n in range(1, ell_max + 1):
        ms = range(1, n + 1)
        row0 = H0[n + 1]  # m = 0, ..., n+1
        H[n, 1] = (1 / _b(n + 1, 0)) * (
            0.5 * (
                const([_b(n + 1, -m - 1) for m in ms]) * one_minus_cosβ * row0[..., 2:]
                - const([_b(n + 1, m - 1) for m in ms]) * one_plus_cosβ * row0[..., :-2]
            )
            - const([_a(n, m) for m in ms]) * sinβ_column * row0[..., 1:-1]
        )

    # Step 4: rows m′+1 = 2, ..., n, each for m = m′+1, ..., n:
    #   d^{m′}_n H^{m′+1,m}_n = d^{m′-1}_n H^{m′-1,m}_n - d^{m-1}_n H^{m′,m-1}_n
    #                           + d^{m}_n H^{m′,m+1}_n
    # (dⁿ_n = 0, so the last term vanishes at m = n, as it must.)
    for n in range(2, ell_max + 1):
        for mp in range(1, min(n, mp_max)):
            ms = range(mp + 1, n + 1)
            H[n, mp + 1] = (1 / _d(n, mp)) * (
                _d(n, mp - 1) * H[n, mp - 1][..., 2:]  # m from m′+1; row m′-1 starts at m′-1
                - const([_d(n, m - 1) for m in ms]) * H[n, mp][..., :-1]
                + const([_d(n, m) for m in ms]) * _shift_left(H[n, mp])[..., 1:]
            )

    # Step 5: rows m′-1 = -1, ..., -n, each for m = 1-m′, ..., n:
    #   d^{m′-1}_n H^{m′-1,m}_n = d^{m′}_n H^{m′+1,m}_n + d^{m-1}_n H^{m′,m-1}_n
    #                             - d^{m}_n H^{m′,m+1}_n
    for n in range(1, ell_max + 1):
        for mp in range(0, -min(n, mp_max), -1):
            ms = range(1 - mp, n + 1)
            start = (1 - mp) - abs(mp + 1)  # where m = 1-m′ sits in row m′+1
            H[n, mp - 1] = (1 / _d(n, mp - 1)) * (
                _d(n, mp) * H[n, mp + 1][..., start:]
                + const([_d(n, m - 1) for m in ms]) * H[n, mp][..., :-1]
                - const([_d(n, m) for m in ms]) * _shift_left(H[n, mp])[..., 1:]
            )

    return H
