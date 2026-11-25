#!/usr/bin/env python
"""Profile benchmark operations: where does the device time go, kernel by kernel?

For each operation and precision this compiles the function, runs it a few times
under the JAX profiler, and adds up the time of every compiled kernel (XLA
fusion) in the trace.  It prints the kernels that take the most time, with their
output shapes and the JAX operations they came from, and saves:

    <tag>_<op>_<precision>.hlo.txt   the optimized HLO, to look kernels up in
    profile_<tag>_l<ell_max>_<time>.json   the per-kernel summary

    python benchmarks/profile_ops.py [--ops evaluate waveform ...] [--n 1000000] [--output-dir DIR]
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import spherical  # noqa: E402
from spherical import jaxwigner as jw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_benchmark as gb  # noqa: E402

DEFAULT_OPS = ["evaluate", "waveform", "sYlm", "rotate", "waveform_value_and_grad", "rotate_value_and_grad"]


def hlo_instructions(text):
    """Map each instruction name in the ENTRY computation to (shape, kind, op_name)."""
    entry = text[text.index("ENTRY"):]
    out = {}
    for line in entry.splitlines():
        m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = (.+?) ([a-z][\w\-]*)\(", line)
        if not m:
            continue
        name, shape, kind = m.groups()
        op = re.search(r'op_name="([^"]*)"', line)
        out[name] = (shape.split("{")[0][:60], kind, op.group(1)[-80:] if op else "")
    return out


def kernel_times(trace_dir, names, runs):
    """Total time per HLO instruction name, per run, from a perfetto trace."""
    files = glob.glob(os.path.join(trace_dir, "**", "*.trace.json.gz"), recursive=True)
    files += glob.glob(os.path.join(trace_dir, "**", "perfetto_trace.json.gz"), recursive=True)
    if not files:
        raise RuntimeError(f"no trace file found under {trace_dir}")
    with gzip.open(sorted(files, key=os.path.getmtime)[-1], "rt") as file:
        trace = json.load(file)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    device_pids = {
        e["pid"] for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and "/device:" in str(e.get("args", {}).get("name", ""))
    }
    totals = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if device_pids and e.get("pid") not in device_pids:
            continue
        name = e.get("name", "")
        # GPU kernel names can carry suffixes; match them to HLO instruction names
        key = name if name in names else next((n for n in (name.split(":")[0], name.replace("_", ".")) if n in names), None)
        if key is None and device_pids:
            key = f"[{name[:50]}]"
        if key is not None:
            totals[key] += e["dur"] / runs  # microseconds per run
    return dict(totals), bool(device_pids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", default=DEFAULT_OPS)
    parser.add_argument("--ell-max", type=int, default=8)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--precision", choices=["float32", "float64", "both"], default="both")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    precisions = ["float32", "float64"] if args.precision == "both" else [args.precision]
    device = jax.devices()[0]
    tag = args.tag or device.device_kind.replace(" ", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"device {device.device_kind}, ℓ_max {args.ell_max}, N {args.n}")
    summary = []
    for precision in precisions:
        dtype = jnp.float32 if precision == "float32" else jnp.float64
        for op in args.ops:
            fn, make_inputs = gb.make_operation(op, args.ell_max, jax, jnp, jw, spherical)
            try:
                inputs = jax.block_until_ready(make_inputs(jax.random.PRNGKey(0), args.n, dtype))
                compiled = jax.jit(fn).lower(*inputs).compile()
                text = compiled.as_text()
                with open(os.path.join(args.output_dir, f"{tag}_{op}_{precision}.hlo.txt"), "w") as file:
                    file.write(text)
                names = hlo_instructions(text)
                jax.block_until_ready(compiled(*inputs))
                start = time.perf_counter()
                for _ in range(args.runs):
                    jax.block_until_ready(compiled(*inputs))
                wall = (time.perf_counter() - start) / args.runs * 1e6
                with tempfile.TemporaryDirectory() as trace_dir:
                    with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
                        for _ in range(args.runs):
                            jax.block_until_ready(compiled(*inputs))
                    totals, on_device = kernel_times(trace_dir, names, args.runs)
                del inputs
            except Exception as error:  # e.g. out of memory
                print(f"\n{precision} {op}: failed: {str(error)[:200]}")
                summary.append(dict(op=op, precision=precision, status="error", error=str(error)[:300]))
                continue
            kernel_total = sum(totals.values())
            print(f"\n{precision} {op}: {wall / args.n * 1e3:.1f} ns/rotor wall clock; "
                  f"{len(totals)} kernels, {kernel_total / args.n * 1e3:.1f} ns/rotor in kernels"
                  f"{'' if on_device else ' (no device timeline found; host events)'}")
            rows = []
            for name, us in sorted(totals.items(), key=lambda kv: -kv[1]):
                shape, kind, op_name = names.get(name, ("", "", ""))
                rows.append(dict(kernel=name, us_per_run=us, ns_per_rotor=us / args.n * 1e3,
                                 share=us / kernel_total if kernel_total else 0, shape=shape, kind=kind, op_name=op_name))
            for r in rows[: args.top]:
                print(f"  {r['share']:6.1%} {r['ns_per_rotor']:7.2f} ns  {r['kernel'][:34]:34} {r['shape'][:28]:28} {r['op_name'][-50:]}")
            summary.append(dict(op=op, precision=precision, status="ok", wall_ns_per_rotor=wall / args.n * 1e3,
                                kernel_ns_per_rotor=kernel_total / args.n * 1e3, n_kernels=len(totals),
                                on_device=on_device, kernels=rows))

    path = os.path.join(args.output_dir, f"profile_{tag}_l{args.ell_max}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as file:
        json.dump(dict(device=device.device_kind, ell_max=args.ell_max, N=args.n, results=summary), file, indent=1)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
