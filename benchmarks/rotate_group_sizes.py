#!/usr/bin/env python
"""Time `rotate` with different groupings of its ℓ blocks, to choose the default.

`rotate` multiplies by the block-diagonal matrix of Δˡ = dˡ(π/2), in groups of
consecutive ℓ.  One big product wastes arithmetic on zeros but runs as one
efficient kernel; separate products per ℓ do no wasted work.  Which is faster
depends on the device and precision (GPUs other than data-centre ones are slow
at float64 arithmetic), so this times each grouping.

    python benchmarks/rotate_group_sizes.py [--ell-max 8] [--n 1000000] [--output-dir DIR]

The group size is the largest number of modes in one product: 1 means one
product per ℓ, and (ell_max+1)² means a single product.
"""

import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from spherical.jaxwigner import modes as modes_module  # noqa: E402


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
    sizes = sorted({1, 16, 36, 81, n_modes} & set(range(1, n_modes + 1)))
    device = jax.devices()[0]
    print(f"device {device.device_kind}, ℓ_max {ell_max}, N {N}")
    rng = np.random.default_rng(0)
    rows = []
    for real, complex_ in [(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)]:
        R = jnp.asarray(rng.normal(size=(N, 4)), real)
        f = jnp.asarray(rng.normal(size=(N, n_modes)) + 1j * rng.normal(size=(N, n_modes)), complex_)
        reference = None
        for size in sizes:
            modes_module._ROTATE_GROUP_SIZE = size
            groups = len(modes_module._rotation_tables(0, ell_max, size)[1])
            # A new function each time, so that jit traces it with this group size
            fn = jax.jit(lambda f, R: modes_module._rotate_impl(f, R, 0))
            try:
                start = time.perf_counter()
                out = jax.block_until_ready(fn(f, R))
                compile_s = time.perf_counter() - start
                if reference is None:
                    reference = out
                difference = float(jnp.max(jnp.abs(out - reference)))
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    jax.block_until_ready(fn(f, R))
                    times.append(time.perf_counter() - start)
                ns = float(np.median(times)) / N * 1e9
                del out
                row = dict(precision=real.__name__, group_size=size, groups=groups, ns_per_rotor=ns,
                           compile_s=compile_s, max_difference=difference, status="ok")
                print(f"{real.__name__:8} group size {size:4} ({groups:2} products): {ns:8.1f} ns/rotor"
                      f"   (compile {compile_s:4.1f} s, max difference {difference:.1e})", flush=True)
            except Exception as error:  # e.g. out of memory
                row = dict(precision=real.__name__, group_size=size, groups=groups, status="error",
                           error=str(error)[:300])
                print(f"{real.__name__:8} group size {size:4}: failed: {str(error)[:120]}", flush=True)
            rows.append(row)
        del R, f, reference

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{args.tag}_" if args.tag else ""
        path = os.path.join(args.output_dir, f"rotate_group_sizes_{tag}l{ell_max}_{stamp}.json")
        with open(path, "w") as file:
            json.dump(dict(device=device.device_kind, ell_max=ell_max, N=N, results=rows), file, indent=1)
        print("saved", path)


if __name__ == "__main__":
    main()
