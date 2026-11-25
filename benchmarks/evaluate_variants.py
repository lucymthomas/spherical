#!/usr/bin/env python
"""Time candidate implementations of the final sum in `evaluate` on this device.

`evaluate` computes  Σₗₘ fₗₘ ₛYₗₘ(R) = zᵧ^{-s} Σₗₘ fₗₘ cₗₘ dˡ_{m,-s}(β) zₐᵐ,  with the
d values read from the rows of the H wedge and constant factors cₗₘ.  On a T4
the H recursion is cheap in float32; most of the time goes on this sum, which
XLA compiles into one kernel where each thread loops over one rotor's terms.
The variants differ only in how the sum is written:

current      as in the package: a sum per run of modes from one row of H
one-reduce   the harmonics (without zᵧ) as one array, then a single reduction
by-m         Σ over ℓ first, giving one coefficient per m (17 at ℓ_max = 8),
             then Σₘ gₘ zₐᵐ: about half the arithmetic
by-m-horner  the same, with Σₘ gₘ zₐᵐ by Horner's rule (no powers of zₐ)
*-T          the same, with the batch moved to the last axis before the sum, so
             neighbouring GPU threads read neighbouring memory

    python benchmarks/evaluate_variants.py [--ell-max 8] [--n 1000000] [--output-dir DIR]
"""

import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spherical import jaxwigner as jw  # noqa: E402
from spherical.jaxwigner.phases import euler_phases  # noqa: E402
from spherical.jaxwigner.recursion import H_wedge, _wedge_pieces  # noqa: E402
from spherical.jaxwigner.matrices import _powers, _Y_tables  # noqa: E402

SPIN_WEIGHT = -2


def _harmonic_parts(modes, R, s):
    """(ell_max, Y, kept, m, zₐ, zᵧ^{-s}): Y holds cₗₘ dˡ_{m,-s} for the modes `kept`."""
    ell_max = int(round(np.sqrt(modes.shape[-1]))) - 1
    zₐ, zᵦ, zᵧ = euler_phases(R)
    H = H_wedge(zᵦ, ell_max, mp_max=abs(s))
    keys, factor, m_index = _Y_tables(s, 0, ell_max)
    real = zₐ.real.dtype
    parts, kept, position = [], [], 0
    for row, first, last, step in _wedge_pieces(keys):
        length = abs(last - first) + 1
        here = np.arange(position, position + length)
        position += length
        if not np.any(factor[here]):
            continue
        x = H[row]
        d = x[..., first: last + 1] if step == 1 else jnp.flip(x[..., last: first + 1], axis=-1)
        parts.append(d * jnp.asarray(factor[here], real))
        kept.append(here)
    kept = np.concatenate(kept)
    Y = jnp.concatenate(parts, axis=-1)
    zᵧ_s = _powers(zᵧ, abs(s))[..., abs(s) - s]
    return ell_max, Y, kept, m_index[kept] - ell_max, zₐ, zᵧ_s


def _contiguous(kept):
    return slice(kept[0], kept[-1] + 1) if np.all(np.diff(kept) == 1) else kept


def one_reduce(modes, R, transposed=False):
    ell_max, Y, kept, m, zₐ, zᵧ_s = _harmonic_parts(modes, R, SPIN_WEIGHT)
    terms = modes[..., _contiguous(kept)] * Y * _powers(zₐ, ell_max)[..., m + ell_max]
    total = jnp.sum(terms.T, axis=0) if transposed else jnp.sum(terms, axis=-1)
    return total * zᵧ_s


def _by_m_coefficients(modes, Y, kept, m, ell_max, transposed):
    """gₘ = Σₗ fₗₘ cₗₘ dˡ_{m,-s}, as an array with m = -ell_max..ell_max on one axis."""
    x = modes[..., _contiguous(kept)] * Y
    if transposed:
        x = x.T
    width = 2 * ell_max + 1
    g = None
    ells = np.floor(np.sqrt(kept)).astype(int)  # mode index ℓ² + ℓ + m
    for ell in np.unique(ells):
        columns = np.flatnonzero(ells == ell)
        block = x[columns] if transposed else x[..., columns]
        pad = [(ell_max - ell, ell_max - ell)]
        block = jnp.pad(block, pad + [(0, 0)] if transposed else [(0, 0)] * (block.ndim - 1) + pad)
        g = block if g is None else g + block
    assert g.shape[0 if transposed else -1] == width
    return g


def by_m(modes, R, transposed=False, horner=False):
    ell_max, Y, kept, m, zₐ, zᵧ_s = _harmonic_parts(modes, R, SPIN_WEIGHT)
    g = _by_m_coefficients(modes, Y, kept, m, ell_max, transposed)

    def column(k):  # g for m = k
        return g[k + ell_max] if transposed else g[..., k + ell_max]

    if horner:
        positive = column(ell_max)
        for k in range(ell_max - 1, -1, -1):
            positive = positive * zₐ + column(k)
        z̄ = jnp.conj(zₐ)
        negative = column(-ell_max)
        for k in range(-ell_max + 1, 0):
            negative = negative * z̄ + column(k)
        total = positive + negative * z̄
    else:
        powers = _powers(zₐ, ell_max)
        total = jnp.sum(g * powers.T, axis=0) if transposed else jnp.sum(g * powers, axis=-1)
    return total * zᵧ_s


def variants():
    return {
        "current": lambda f, R: jw.evaluate(f, R, SPIN_WEIGHT),
        "one-reduce": lambda f, R: one_reduce(f, R),
        "one-reduce-T": lambda f, R: one_reduce(f, R, transposed=True),
        "by-m": lambda f, R: by_m(f, R),
        "by-m-T": lambda f, R: by_m(f, R, transposed=True),
        "by-m-horner": lambda f, R: by_m(f, R, horner=True),
        "by-m-horner-T": lambda f, R: by_m(f, R, transposed=True, horner=True),
    }


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
    print(f"device {device.device_kind}, ℓ_max {ell_max}, N {N}, spin weight {SPIN_WEIGHT}")
    rng = np.random.default_rng(0)
    rows = []
    for real, complex_ in [(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)]:
        R = jnp.asarray(rng.normal(size=(N, 4)), real)
        f = jnp.asarray(rng.normal(size=(N, n_modes)) + 1j * rng.normal(size=(N, n_modes)), complex_)
        reference = jax.block_until_ready(jax.jit(variants()["current"])(f, R))
        scale = float(jnp.max(jnp.abs(reference)))
        for name, fn in variants().items():
            try:
                start = time.perf_counter()
                compiled = jax.jit(fn).lower(f, R).compile()
                compile_s = time.perf_counter() - start
                text = compiled.as_text()
                kernels = text[text.index("ENTRY"):].count(" fusion(")
                out = jax.block_until_ready(compiled(f, R))
                difference = float(jnp.max(jnp.abs(out - reference))) / scale
                del out
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    jax.block_until_ready(compiled(f, R))
                    times.append(time.perf_counter() - start)
                ns = float(np.median(times)) / N * 1e9
                row = dict(precision=real.__name__, variant=name, ns_per_rotor=ns, compile_s=compile_s,
                           kernels=kernels, relative_difference=difference, status="ok")
                print(f"{real.__name__:8} {name:15} {ns:8.1f} ns/rotor   {kernels:4} kernels"
                      f"   compile {compile_s:4.1f} s   relative difference {difference:.1e}", flush=True)
            except Exception as error:  # e.g. out of memory
                row = dict(precision=real.__name__, variant=name, status="error", error=str(error)[:300])
                print(f"{real.__name__:8} {name:15} failed: {str(error)[:120]}", flush=True)
            rows.append(row)
        del R, f, reference

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{args.tag}_" if args.tag else ""
        path = os.path.join(args.output_dir, f"evaluate_variants_{tag}l{ell_max}_{stamp}.json")
        with open(path, "w") as file:
            json.dump(dict(device=device.device_kind, ell_max=ell_max, N=N, spin_weight=SPIN_WEIGHT,
                           results=rows), file, indent=1)
        print("saved", path)


if __name__ == "__main__":
    main()
