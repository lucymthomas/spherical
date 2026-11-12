#!/usr/bin/env python
# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Benchmark `spherical.jaxwigner` on a GPU (or any device JAX can use).

See benchmarks/README.md for instructions.  In short:

    python benchmarks/gpu_benchmark.py --quick        # a few minutes; checks that everything works
    python benchmarks/gpu_benchmark.py                # the full benchmark
    python benchmarks/gpu_benchmark.py --plot-from benchmark_results/<file>.json

The script first checks accuracy on the device against the numba implementation
in `spherical`, then times each operation for a range of ℓ_max, batch sizes and
precisions.  Compile time is measured separately from run time.  Results are
printed as they arrive and saved as JSON and CSV.

"""

import argparse
import csv
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000, 16_000_000]
QUICK_SIZES = [1_000, 100_000, 1_000_000]
DEFAULT_ELL_MAX = [2, 4, 8]
ALL_OPS = [
    "wigner_D", "sYlm", "rotate", "rotate_via_D", "evaluate", "evaluate_via_sYlm",
    "waveform", "waveform_value_and_grad", "rotate_value_and_grad",
]
QUICK_OPS = ["wigner_D", "rotate", "evaluate", "waveform", "waveform_value_and_grad"]
SPIN_WEIGHT = -2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help=f"small run to check the setup: ℓ_max=8, sizes {QUICK_SIZES}, ops {QUICK_OPS}")
    p.add_argument("--ell-max", type=int, nargs="+", default=None, help=f"ℓ_max values (default {DEFAULT_ELL_MAX})")
    p.add_argument("--sizes", type=int, nargs="+", default=None, help=f"batch sizes N (default {DEFAULT_SIZES})")
    p.add_argument("--precision", choices=["float64", "float32", "both"], default="both")
    p.add_argument("--ops", nargs="+", choices=ALL_OPS, default=None, help="operations to time (default: all)")
    p.add_argument("--repeats", type=int, default=10, help="timed runs per measurement (median reported)")
    p.add_argument("--device", choices=["default", "cpu"], default="default",
                   help="'cpu' forces JAX onto the CPU, for comparison runs")
    p.add_argument("--numba-max-size", type=int, default=100_000,
                   help="largest N for the numba CPU reference timings (0 to skip them)")
    p.add_argument("--skip-accuracy", action="store_true", help="skip the accuracy checks")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--tag", default="", help="label added to output file names, e.g. 'a100'")
    p.add_argument("--plot", action="store_true", help="also save plots (needs matplotlib)")
    p.add_argument("--plot-from", metavar="JSON", help="only make plots from an existing results file")
    p.add_argument("--trace-dir", default=None,
                   help="write a JAX profiler trace of one run of each op at the largest size that fits "
                        "(view with TensorBoard or Perfetto)")
    args = p.parse_args(argv)
    if args.quick:
        args.ell_max = args.ell_max or [8]
        args.sizes = args.sizes or QUICK_SIZES
        args.ops = args.ops or QUICK_OPS
        args.repeats = min(args.repeats, 5)
    args.ell_max = args.ell_max or DEFAULT_ELL_MAX
    args.sizes = sorted(args.sizes or DEFAULT_SIZES)
    args.ops = args.ops or ALL_OPS
    return args


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def describe_environment(jax):
    devices = jax.devices()
    info = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        "backend": jax.default_backend(),
        "devices": [f"{d.platform}:{d.id} {getattr(d, 'device_kind', '')}" for d in devices],
        "device_used": f"{devices[0].platform}:{devices[0].id} {getattr(devices[0], 'device_kind', '')}",
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "default (true)"),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            info["nvidia_smi"] = [line.strip() for line in out.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import spherical
        info["spherical"] = spherical.__version__
        info["git_commit"] = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        pass
    return info


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def make_operation(name, ell_max, jax, jnp, jw, spherical):
    """Return (function to time, function making its inputs from (key, N, dtype))."""
    n_modes = spherical.Ysize(0, ell_max)

    def rotors(key, N, dtype):
        return jax.random.normal(key, (N, 4), dtype=dtype)

    def modes(key, N, dtype):
        k1, k2 = jax.random.split(key)
        real = jax.random.normal(k1, (N, n_modes), dtype=dtype)
        imag = jax.random.normal(k2, (N, n_modes), dtype=dtype)
        return real + 1j * imag

    def rotors_and_modes(key, N, dtype):
        k1, k2 = jax.random.split(key)
        return rotors(k1, N, dtype), modes(k2, N, dtype)

    def waveform_inputs(key, N, dtype):
        R, f = rotors_and_modes(key, N, dtype)
        return R, f, jnp.asarray(0.7, dtype=dtype), jnp.asarray(0.3, dtype=dtype)

    blocks = [
        (slice(spherical.Yindex(l, -l), spherical.Yindex(l, l) + 1),
         slice(spherical.WignerDindex(l, -l, -l), spherical.WignerDindex(l, l, l) + 1), l)
        for l in range(ell_max + 1)
    ]

    def rotate_via_D(R, f):
        D = jw.wigner_D(R, ell_max)
        return jnp.concatenate([
            jnp.einsum("...i,...ij->...j", f[..., a], D[..., b].reshape(D.shape[:-1] + (2 * l + 1, 2 * l + 1)))
            for a, b, l in blocks
        ], axis=-1)

    def waveform(R, f, θ, ϕ):
        # strain at one observer, for a different frame rotor per sample
        return jw.evaluate(f, jw.multiply(R, jw.from_spherical_coordinates(θ, ϕ)), SPIN_WEIGHT)

    def waveform_loss(R, f, θ, ϕ):
        return jnp.sum(jnp.abs(waveform(R, f, θ, ϕ)) ** 2)

    def rotate_loss(R, f):
        weights = jnp.arange(n_modes, dtype=jnp.real(f).dtype)
        return jnp.sum(jnp.abs(jw.rotate(f, R)) ** 2 * weights)

    table = {
        "wigner_D": (lambda R: jw.wigner_D(R, ell_max), lambda key, N, dt: (rotors(key, N, dt),)),
        "sYlm": (lambda R: jw.sYlm(SPIN_WEIGHT, R, ell_max), lambda key, N, dt: (rotors(key, N, dt),)),
        "rotate": (lambda R, f: jw.rotate(f, R), rotors_and_modes),
        "rotate_via_D": (rotate_via_D, rotors_and_modes),
        "evaluate": (lambda R, f: jw.evaluate(f, R, SPIN_WEIGHT), rotors_and_modes),
        "evaluate_via_sYlm": (lambda R, f: jnp.sum(f * jw.sYlm(SPIN_WEIGHT, R, ell_max), axis=-1), rotors_and_modes),
        "waveform": (waveform, waveform_inputs),
        "waveform_value_and_grad": (jax.value_and_grad(waveform_loss, argnums=(0, 2, 3)), waveform_inputs),
        "rotate_value_and_grad": (jax.value_and_grad(rotate_loss), rotors_and_modes),
    }
    return table[name]


def is_out_of_memory(error):
    text = f"{type(error).__name__}: {error}"
    return any(s in text for s in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OOM"))


def nbytes(tree, jax):
    return int(sum(getattr(x, "nbytes", 0) for x in jax.tree_util.tree_leaves(tree)))


def time_operation(fn, inputs, repeats, jax):
    """Compile, then time `repeats` runs.  Returns (compile s, median s, min s, output bytes)."""
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    output = compiled(*inputs)  # warm-up run
    jax.block_until_ready(output)
    output_bytes = nbytes(output, jax)
    del output
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*inputs))
        times.append(time.perf_counter() - start)
    return compile_time, statistics.median(times), min(times), output_bytes


# ---------------------------------------------------------------------------
# Accuracy checks against numba on the CPU
# ---------------------------------------------------------------------------

def special_rotors(np):
    δ = 1e-8
    q = np.array([
        [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0],  # identity and poles
        [1, δ, 0, 0], [δ, 1, 0, 0], [1, 1, 1, 1],
    ], dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def accuracy_checks(ell_max_values, precisions, jax, jnp, jw, np, spherical, quaternionic):
    """Max |JAX on device − numba on CPU| for each function, precision and ℓ_max."""
    rng = np.random.default_rng(12345)
    R = np.concatenate([special_rotors(np), rng.normal(size=(193, 4))])
    R /= np.linalg.norm(R, axis=-1, keepdims=True)
    q = quaternionic.array(R)
    results = []
    for ell_max in ell_max_values:
        wigner = spherical.Wigner(ell_max)
        n = spherical.Ysize(0, ell_max)
        f = rng.normal(size=(len(R), n)) + 1j * rng.normal(size=(len(R), n))
        f[:, : spherical.Yindex(min(2, ell_max), -min(2, ell_max))] = 0  # spin weight -2: no ℓ < 2
        D_ref = wigner.D(q)
        Y_ref = wigner.sYlm(SPIN_WEIGHT, q) if ell_max >= abs(SPIN_WEIGHT) else None
        rot_ref = np.array([wigner.rotate(spherical.Modes(f[i], spin_weight=0), q[i]).ndarray for i in range(len(R))])
        for precision in precisions:
            real = jnp.float64 if precision == "float64" else jnp.float32
            complex_ = jnp.complex128 if precision == "float64" else jnp.complex64
            R_dev = jnp.asarray(R, dtype=real)
            f_dev = jnp.asarray(f, dtype=complex_)
            row = {"ell_max": ell_max, "precision": precision}
            row["wigner_D"] = float(np.max(np.abs(np.asarray(jw.wigner_D(R_dev, ell_max)) - D_ref)))
            if Y_ref is not None:
                row["sYlm"] = float(np.max(np.abs(np.asarray(jw.sYlm(SPIN_WEIGHT, R_dev, ell_max)) - Y_ref)))
                ev_ref = np.sum(f * Y_ref, axis=-1)
                row["evaluate"] = float(np.max(np.abs(np.asarray(jw.evaluate(f_dev, R_dev, SPIN_WEIGHT)) - ev_ref)))
            row["rotate"] = float(np.max(np.abs(np.asarray(jw.rotate(f_dev, R_dev)) - rot_ref)))
            # Gradients: finite everywhere, including the poles
            grad = jax.grad(lambda R: jnp.sum(jnp.abs(jw.rotate(f_dev, R)) ** 2 * jnp.arange(n, dtype=real)))(R_dev)
            row["gradients_finite"] = bool(np.all(np.isfinite(np.asarray(grad))))
            results.append(row)
            print("  accuracy", ", ".join(
                f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            ), flush=True)
    return results


# ---------------------------------------------------------------------------
# numba reference timings (CPU)
# ---------------------------------------------------------------------------

def numba_timings(ell_max_values, sizes, max_size, np, spherical, quaternionic):
    results = []
    rng = np.random.default_rng(0)
    for ell_max in ell_max_values:
        wigner = spherical.Wigner(ell_max)
        for N in [N for N in sizes if N <= max_size]:
            R = quaternionic.array(rng.normal(size=(N, 4))).normalized
            for name, fn in (("wigner_D", lambda: wigner.D(R)),
                             ("sYlm", lambda: wigner.sYlm(SPIN_WEIGHT, R))):
                if name == "sYlm" and ell_max < abs(SPIN_WEIGHT):
                    continue
                fn()  # numba compiles on first use
                start = time.perf_counter()
                fn()
                elapsed = time.perf_counter() - start
                row = {"implementation": "numba (CPU)", "op": name, "ell_max": ell_max, "precision": "float64",
                       "N": N, "status": "ok", "median_s": elapsed, "min_s": elapsed,
                       "rotors_per_s": N / elapsed, "ns_per_rotor": elapsed / N * 1e9}
                results.append(row)
                print(f"  numba  {name:<24} ℓ≤{ell_max} N={N:>9}: {elapsed * 1e3:10.2f} ms", flush=True)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_FIELDS = ["implementation", "op", "ell_max", "precision", "N", "status", "compile_s",
              "median_s", "min_s", "rotors_per_s", "ns_per_rotor", "output_bytes", "error"]


def output_base(args):
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(args.output_dir, f"jaxwigner_benchmark{'_' + args.tag if args.tag else ''}_{stamp}")


def save(results, base):
    """Write results to base.json and base.csv, overwriting earlier partial saves."""
    with open(base + ".json", "w") as f:
        json.dump(results, f, indent=1)
    with open(base + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results["timings"])


def plot(results, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots")
        return
    rows = [r for r in results["timings"] if r["status"] == "ok"]
    device = results["environment"].get("device_used", "")
    # One fixed color per (implementation, op), the same in every panel and figure
    series = sorted({(r["implementation"], r["op"]) for r in rows}, key=lambda s: (s[0] != "jax", s[1]))
    cmap = plt.get_cmap("tab20")
    color = {s: cmap(i % 20) for i, s in enumerate(series)}
    for precision in sorted({r["precision"] for r in rows}):
        ells = sorted({r["ell_max"] for r in rows if r["precision"] == precision})
        fig, axes = plt.subplots(1, len(ells), figsize=(4.5 * len(ells) + 2.5, 4.5), squeeze=False, sharey=True)
        handles = {}
        for ax, ell in zip(axes[0], ells):
            for impl, op in series:
                pts = sorted((r["N"], r["ns_per_rotor"]) for r in rows if r["precision"] == precision
                             and r["ell_max"] == ell and r["op"] == op and r["implementation"] == impl)
                if not pts:
                    continue
                label = op if impl == "jax" else f"{op} [{impl}]"
                (line,) = ax.loglog(*zip(*pts), marker="o", ls="--" if impl != "jax" else "-",
                                    color=color[impl, op], label=label)
                handles[label] = line
            ax.set(title=f"ℓ_max = {ell}, {precision}", xlabel="batch size N")
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].set_ylabel("time per rotor (ns)")
        fig.legend(handles.values(), handles.keys(), loc="center right", fontsize=8, frameon=False)
        fig.suptitle(device)
        fig.tight_layout(rect=(0, 0, 1 - 2.3 / fig.get_figwidth(), 1))
        path = f"{base}_{precision}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("saved", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    if args.plot_from:
        with open(args.plot_from) as f:
            plot(json.load(f), os.path.splitext(args.plot_from)[0])
        return

    import jax
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)  # needed for float64; float32 inputs still run in float32
    import jax.numpy as jnp
    import numpy as np
    import quaternionic
    import spherical
    from spherical import jaxwigner as jw

    environment = describe_environment(jax)
    print("Environment:")
    for k, v in environment.items():
        print(f"  {k}: {v}")
    if environment["backend"] == "cpu" and args.device != "cpu":
        print("\nWARNING: JAX is using the CPU.  If you expected a GPU, see 'Troubleshooting' in "
              "benchmarks/README.md.\n")

    precisions = ["float64", "float32"] if args.precision == "both" else [args.precision]
    results = {"environment": environment, "arguments": vars(args), "complete": False,
               "accuracy": [], "timings": []}
    base = output_base(args)
    print(f"\nResults will be saved (and updated as the run goes) to {base}.json/.csv")

    if not args.skip_accuracy:
        print("\nAccuracy on the device, max |JAX − numba| over 200 rotors (7 at or near the poles):")
        results["accuracy"] = accuracy_checks(args.ell_max, precisions, jax, jnp, jw, np, spherical, quaternionic)
        save(results, base)

    print(f"\nTimings (median of {args.repeats} runs, after compiling):")
    key = jax.random.PRNGKey(0)
    largest_ok = {}
    for precision in precisions:
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        for ell_max in args.ell_max:
            for op in args.ops:
                fn, make_inputs = make_operation(op, ell_max, jax, jnp, jw, spherical)
                for N in args.sizes:
                    row = {"implementation": "jax", "op": op, "ell_max": ell_max, "precision": precision, "N": N}
                    inputs = None
                    try:
                        inputs = make_inputs(key, N, dtype)
                        jax.block_until_ready(inputs)
                        compile_s, median_s, min_s, out_bytes = time_operation(fn, inputs, args.repeats, jax)
                        row.update(status="ok", compile_s=compile_s, median_s=median_s, min_s=min_s,
                                   rotors_per_s=N / median_s, ns_per_rotor=median_s / N * 1e9,
                                   output_bytes=out_bytes)
                        largest_ok[(op, ell_max, precision)] = N
                        print(f"  {precision} {op:<24} ℓ≤{ell_max} N={N:>9}: {median_s * 1e3:10.3f} ms "
                              f"({median_s / N * 1e9:8.2f} ns/rotor; compile {compile_s:5.2f} s)", flush=True)
                    except Exception as error:  # noqa: BLE001 - record and move on
                        oom = is_out_of_memory(error)
                        row.update(status="out of memory" if oom else "error",
                                   error=f"{type(error).__name__}: {str(error).splitlines()[0][:300]}")
                        print(f"  {precision} {op:<24} ℓ≤{ell_max} N={N:>9}: {row['status']}"
                              + ("" if oom else f" — {row['error']}"), flush=True)
                    finally:
                        del inputs
                    results["timings"].append(row)
                    if row["status"] != "ok":
                        break  # larger N will not fit either
                save(results, base)  # so an interrupted run keeps what it has
                jax.clear_caches()

    if args.numba_max_size > 0:
        print("\nnumba reference timings (CPU, float64, one rotor at a time):")
        results["timings"] += numba_timings(args.ell_max, args.sizes, args.numba_max_size, np, spherical, quaternionic)

    if args.trace_dir:
        print(f"\nWriting profiler traces to {args.trace_dir}")
        for (op, ell_max, precision), N in largest_ok.items():
            dtype = jnp.float64 if precision == "float64" else jnp.float32
            fn, make_inputs = make_operation(op, ell_max, jax, jnp, jw, spherical)
            inputs = make_inputs(key, N, dtype)
            compiled = jax.jit(fn).lower(*inputs).compile()
            jax.block_until_ready(compiled(*inputs))
            with jax.profiler.trace(os.path.join(args.trace_dir, f"{op}_l{ell_max}_{precision}_N{N}")):
                jax.block_until_ready(compiled(*inputs))

    results["complete"] = True
    save(results, base)
    print(f"\nSaved {base}.json and {base}.csv")
    if args.plot:
        plot(results, base)


if __name__ == "__main__":
    main()
