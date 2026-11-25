#!/usr/bin/env python
"""Time candidate implementations of `rotate` against each other on this device.

All compute  f′ₗₘ = (i zᵧ)ᵐ Σₖ Δˡ_{k,m} zᵦᵏ Σₘ′ Δˡ_{k,m′} fₗₘ′ (zₐ/i)^{m′}  with
Δˡ = dˡ(π/2); they differ in how the two products with Δ are done, and how the
powers zᵏ are computed:

dense-G      Δ as block-diagonal products of at most G modes (`rotate` today, G = 36)
parity-G     Δ split by its symmetries Δ_{k,-m} = (-1)^{ℓ+k} Δ_{k,m} and
             Δ_{-k,m} = (-1)^{ℓ+m} Δ_{k,m} into four blocks per ℓ (about 1/4 of the
             arithmetic), applied in the basis f_m ± f_{-m}; blocks grouped into
             products of at most G
parity-fused the same blocks, written out as elementwise multiply-adds (no matrix
             products), so that XLA can fuse everything into few kernels

Each comes in two versions: "chain" builds zᵏ by repeated multiplication (as the
package does today), "binary" from z, z², z⁴, … in one broadcast expression,
which XLA fuses into far fewer kernels.

    python benchmarks/rotate_variants.py [--ell-max 8] [--n 1000000] [--output-dir DIR]
"""

import argparse
import json
import os
import time
from functools import lru_cache, partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spherical.jaxwigner import modes as M  # noqa: E402
from spherical.jaxwigner.phases import euler_phases  # noqa: E402
from spherical.jaxwigner.matrices import _powers as powers_chain  # noqa: E402


def powers_binary(z, k_max):
    """zᵏ for k = -k_max..k_max (|z| = 1), from z, z², z⁴, … as one broadcast product."""
    k = np.arange(-k_max, k_max + 1)
    a = np.abs(k)
    square = z[..., None]
    result = jnp.ones_like(square)
    bit = 1
    while bit <= k_max:
        result = result * jnp.where(jnp.asarray((a & bit) != 0), square, jnp.ones_like(square))
        bit <<= 1
        if bit <= k_max:
            square = square * square
    return jnp.where(jnp.asarray(k < 0), jnp.conj(result), result)


def _phases(modes, R, ell_min, powers):
    ell_max = M.ell_max_from_size(modes.shape[-1], ell_min)
    m = np.concatenate([np.arange(-ell, ell + 1) for ell in range(ell_min, ell_max + 1)])
    zₐ, zᵦ, zᵧ = euler_phases(R)
    i_m = 1j ** (m % 4)
    index = m + ell_max
    u = modes * (powers(zₐ, ell_max)[..., index] * jnp.asarray(np.conj(i_m), zₐ.dtype))
    pb = powers(zᵦ, ell_max)[..., index]
    pg = powers(zᵧ, ell_max)[..., index] * jnp.asarray(i_m, zₐ.dtype)
    return ell_max, u, pb, pg


def rotate_dense(modes, R, ell_min, powers, group):
    ell_max, u, pb, pg = _phases(modes, R, ell_min, powers)
    _, groups = M._rotation_tables(ell_min, ell_max, group)
    out = []
    for sl, Δ in groups:
        h = M._times_real(u[..., sl], Δ.T) * pb[..., sl]
        out.append(M._times_real(h, Δ))
    out = out[0] if len(out) == 1 else jnp.concatenate(out, axis=-1)
    return out * pg


@lru_cache(maxsize=None)
def parity_plan(ell_min, ell_max):
    """Δ = P B Pᵀ, with P the orthonormal basis (e_m ± e_{-m})/√2 and B in dense blocks.

    Returns plans for Δ and Δᵀ, each (to_basis, from_basis, blocks):
    to_basis/from_basis are (a, ca, b, cb) giving y = x[a] ca + x[b] cb, and
    blocks is a tuple of (in_slice, out_slice, block).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    _, groups = M._rotation_tables(ell_min, ell_max, 10**9)  # one block-diagonal Δ
    Δ = groups[0][1]
    n = len(Δ)
    P = np.zeros((n, n))
    start = 0
    for ell in range(ell_min, ell_max + 1):
        centre = start + ell
        P[centre, centre] = 1
        free = iter(j for j in range(start, start + 2 * ell + 1) if j != centre)
        for m in range(1, ell + 1):
            j = next(free)
            P[centre + m, j] = P[centre - m, j] = 1 / np.sqrt(2)
            j = next(free)
            P[centre + m, j], P[centre - m, j] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        start += 2 * ell + 1
    B = P.T @ Δ @ P
    B[np.abs(B) < 1e-12] = 0
    nonzero = B != 0
    graph = np.block([[np.zeros((n, n)), nonzero], [nonzero.T, np.zeros((n, n))]])
    count, label = connected_components(csr_matrix(graph), directed=False)
    rows, cols, blocks = [], [], []
    for c in range(count):
        r, q = np.flatnonzero(label[:n] == c), np.flatnonzero(label[n:] == c)
        blocks.append((slice(len(cols), len(cols) + len(q)), slice(len(rows), len(rows) + len(r)), B[np.ix_(r, q)]))
        rows += list(r)
        cols += list(q)

    def pairs(matrix):
        a, ca, b, cb = [], [], [], []
        for row in matrix:
            j = np.flatnonzero(row)
            a.append(j[0]); ca.append(row[j[0]])
            b.append(j[-1]); cb.append(row[j[-1]] if len(j) > 1 else 0.0)
        return tuple(np.array(x) for x in (a, ca, b, cb))

    plan = (pairs(P.T[cols]), pairs(P[:, rows]), tuple(blocks))
    plan_T = (pairs(P.T[rows]), pairs(P[:, cols]), tuple((o, i, b.T) for i, o, b in blocks))
    return plan, plan_T


def _pairs(x, t):
    a, ca, b, cb = t
    return x[..., a] * jnp.asarray(ca, x.dtype) + x[..., b] * jnp.asarray(cb, x.dtype)


def _blocks_fused(x, blocks):
    out = []
    for ins, _, block in blocks:
        acc = None
        for j in range(block.shape[1]):
            term = x[..., ins.start + j: ins.start + j + 1] * jnp.asarray(block[:, j], x.dtype)
            acc = term if acc is None else acc + term
        out.append(acc)
    return jnp.concatenate(out, axis=-1)


def _blocks_products(x, blocks, group):
    out, current, size = [], [], 0

    def flush():
        n_in = sum(b.shape[1] for _, _, b in current)
        n_out = sum(b.shape[0] for _, _, b in current)
        A = np.zeros((n_in, n_out))
        i = o = 0
        for _, _, b in current:
            A[i: i + b.shape[1], o: o + b.shape[0]] = b.T
            i += b.shape[1]
            o += b.shape[0]
        s = current[0][0].start
        out.append(x[..., s: s + n_in] @ jnp.asarray(A, x.dtype))

    for block in blocks:
        if current and size + block[2].shape[1] > group:
            flush()
            current, size = [], 0
        current.append(block)
        size += block[2].shape[1]
    flush()
    return jnp.concatenate(out, axis=-1)


def _apply(X, plan, group):
    to_basis, from_basis, blocks = plan
    y = _pairs(X, to_basis)
    y = _blocks_fused(y, blocks) if group == "fused" else _blocks_products(y, blocks, group)
    return _pairs(y, from_basis)


def rotate_parity(modes, R, ell_min, powers, group):
    ell_max, u, pb, pg = _phases(modes, R, ell_min, powers)
    plan, plan_T = parity_plan(ell_min, ell_max)
    H = _apply(jnp.stack([u.real, u.imag], axis=-2), plan, group)
    h = jax.lax.complex(H[..., 0, :], H[..., 1, :]) * pb
    F = _apply(jnp.stack([h.real, h.imag], axis=-2), plan_T, group)
    return jax.lax.complex(F[..., 0, :], F[..., 1, :]) * pg


def variants():
    out = {}
    for pname, powers in [("chain", powers_chain), ("binary", powers_binary)]:
        for group in (36, 81):
            out[f"dense-{group}/{pname}"] = partial(rotate_dense, powers=powers, group=group)
        for group in (16, 36, 81):
            out[f"parity-{group}/{pname}"] = partial(rotate_parity, powers=powers, group=group)
        out[f"parity-fused/{pname}"] = partial(rotate_parity, powers=powers, group="fused")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ell-max", type=int, default=8)
    parser.add_argument("--n", type=int, default=1_000_000, help="number of rotors")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default=None, help="where to save a JSON file of the results")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    ell_max, N = args.ell_max, args.n
    n_modes = (ell_max + 1) ** 2
    device = jax.devices()[0]
    print(f"device {device.device_kind}, ℓ_max {ell_max}, N {N}")
    rng = np.random.default_rng(0)
    rows = []
    for real, complex_ in [(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)]:
        R = jnp.asarray(rng.normal(size=(N, 4)), real)
        f = jnp.asarray(rng.normal(size=(N, n_modes)) + 1j * rng.normal(size=(N, n_modes)), complex_)
        reference = jax.block_until_ready(jax.jit(lambda f, R: M._rotate_impl(f, R, 0))(f, R))
        for name, fn in variants().items():
            jitted = jax.jit(lambda f, R, fn=fn: fn(f, R, 0))
            try:
                start = time.perf_counter()
                compiled = jitted.lower(f, R).compile()
                compile_s = time.perf_counter() - start
                text = compiled.as_text()
                kernels = text[text.index("ENTRY"):].count(" fusion(")
                out = jax.block_until_ready(compiled(f, R))
                difference = float(jnp.max(jnp.abs(out - reference)))
                del out
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(f, R))
                    times.append(time.perf_counter() - start)
                ns = float(np.median(times)) / N * 1e9
                row = dict(precision=real.__name__, variant=name, ns_per_rotor=ns, compile_s=compile_s,
                           kernels=kernels, max_difference=difference, status="ok")
                print(f"{real.__name__:8} {name:22} {ns:8.1f} ns/rotor   {kernels:4} kernels"
                      f"   compile {compile_s:4.1f} s   max difference {difference:.1e}", flush=True)
            except Exception as error:  # e.g. out of memory
                row = dict(precision=real.__name__, variant=name, status="error", error=str(error)[:300])
                print(f"{real.__name__:8} {name:22} failed: {str(error)[:120]}", flush=True)
            rows.append(row)
        del R, f, reference

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{args.tag}_" if args.tag else ""
        path = os.path.join(args.output_dir, f"rotate_variants_{tag}l{ell_max}_{stamp}.json")
        with open(path, "w") as file:
            json.dump(dict(device=device.device_kind, ell_max=ell_max, N=N, results=rows), file, indent=1)
        print("saved", path)


if __name__ == "__main__":
    main()
