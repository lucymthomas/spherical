# Benchmarking `spherical.jaxwigner` on a GPU

`gpu_benchmark.py` does two things on whatever device JAX finds:

1. **Checks accuracy** against the numba implementation in `spherical`, in float64 and
   float32, on 200 rotations including 7 at or near the poles.  It also checks that
   gradients are finite there.  Speed only matters if the results are right, and GPU math
   libraries can differ from CPU ones.
2. **Times each operation** for several values of ℓ_max, batch sizes N and precisions.
   Compile time is measured separately from run time.

Results are printed as they arrive and saved as JSON and CSV, with optional plots.

## 1. Setup on the GPU machine

You need an NVIDIA GPU with a recent driver, and Python 3.10 or newer.  JAX's pip wheels
bring their own CUDA libraries, so a separate CUDA toolkit is not needed.

```bash
git clone https://github.com/lucymthomas/spherical.git
cd spherical
git checkout jax-wigner

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . "jax[cuda12]" matplotlib
```

`jax[cuda12]` needs an NVIDIA driver that supports CUDA 12.  Newer JAX releases also offer
`jax[cuda13]`; check `nvidia-smi` (top-right, "CUDA Version") and the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) if unsure.

Check that JAX sees the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

This should print something like `[CudaDevice(id=0)]`.  If it prints `CpuDevice`, see
[Troubleshooting](#troubleshooting).

## 2. Run it

Always run from the repository root.

**First, a quick run** (a few minutes) to check that everything works:

```bash
python benchmarks/gpu_benchmark.py --quick --tag quick
```

**Then the full benchmark:**

```bash
python benchmarks/gpu_benchmark.py --tag <gpu-name> --plot
```

Use a short `--tag` naming the GPU, such as `a100` or `rtx4090`, so results from different
machines are easy to tell apart.  The full run times 9 operations × 3 ℓ_max × 2
precisions × up to 6 batch sizes.  Every distinct input shape compiles a new function, so
expect roughly 15–40 minutes, most of it compilation.  It stops increasing N for an
operation once it runs out of GPU memory, and moves on.

To make plots later from a saved results file:

```bash
python benchmarks/gpu_benchmark.py --plot-from benchmark_results/<file>.json
```

### Useful options

| option | default | meaning |
|---|---|---|
| `--quick` | off | ℓ_max = 8, N ∈ {10³, 10⁵, 10⁶}, five main operations |
| `--ell-max L [L ...]` | `2 4 8` | ℓ_max values |
| `--sizes N [N ...]` | `1000 … 16000000` | batch sizes |
| `--precision` | `both` | `float64`, `float32` or `both` |
| `--ops OP [OP ...]` | all | operations to time (listed below) |
| `--repeats` | `10` | timed runs per measurement; the median is reported |
| `--numba-max-size` | `100000` | largest N for the numba CPU reference; `0` skips it |
| `--device cpu` | — | force JAX onto the CPU, to compare on the same machine |
| `--trace-dir DIR` | — | also save a profiler trace of each operation at its largest N |
| `--output-dir` | `benchmark_results` | where results go |

### On a cluster with SLURM

Adapt the partition, GPU request and module names to your cluster:

```bash
#!/bin/bash
#SBATCH --job-name=jaxwigner-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jaxwigner-bench-%j.log

cd "$HOME/spherical"
source .venv/bin/activate
nvidia-smi
python benchmarks/gpu_benchmark.py --tag "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')" --plot
```

Submit with `sbatch bench.sh`.  Request a whole GPU: sharing one with other jobs makes the
timings unreliable.

### On Google Colab

[`colab_benchmark.ipynb`](colab_benchmark.ipynb) runs the same script on a Colab GPU:
[open it in Colab](https://colab.research.google.com/github/lucymthomas/spherical/blob/jax-wigner/benchmarks/colab_benchmark.ipynb),
choose a GPU runtime (*Runtime → Change runtime type*), and run the cells in order.  It
installs the package without replacing Colab's own JAX, saves results to Google Drive,
and splits the full benchmark into presets that fit in a free session.  Colab machines
have few CPU cores, so compile times there are longer than on a cluster node; the run
times are what measure the GPU.

## 3. What is measured

All operations use spin weight s = −2, as for gravitational-wave strain.  Rotors and modes
are random, generated directly on the device, with one rotor (and, where used, one set of
modes) per batch entry.

| operation | what it computes |
|---|---|
| `wigner_D` | the full 𝔇 matrices |
| `sYlm` | the s = −2 harmonics |
| `rotate` | rotated modes, without building 𝔇 |
| `rotate_via_D` | the same, by building 𝔇 and multiplying (for comparison) |
| `evaluate` | Σ fₗₘ ₛYₗₘ(R), without storing the harmonics separately |
| `evaluate_via_sYlm` | the same, by building the harmonics and summing (for comparison) |
| `waveform` | strain at one observer for a different frame rotor per sample: `evaluate(f, multiply(R, R_obs))` |
| `waveform_value_and_grad` | value and gradient of Σ\|h\|² with respect to the frame rotors and the observer angles |
| `rotate_value_and_grad` | value and gradient of a loss of `rotate` with respect to the rotors |

The numba reference times `spherical.Wigner.D` and `.sYlm` on the CPU in float64, one rotor at
a time, for N up to `--numba-max-size`.

Each JAX measurement compiles the function first (reported as `compile_s`), runs it once to
warm up, then reports the median and minimum of `--repeats` runs, waiting for the device to
finish each time.

## 4. Output

Files are written to `benchmark_results/` (ignored by git), named
`jaxwigner_benchmark_<tag>_<date>-<time>.*`.  They are rewritten after every operation,
so a run that is interrupted keeps what it had finished; the JSON's `"complete"` field
says whether the run reached the end.

- **`.json`**: everything.  This includes the environment (JAX version, GPU model, driver,
  git commit), the accuracy checks, and every timing row, including runs that ran out of
  memory.
- **`.csv`**: the timing rows only, one per line, for spreadsheets.
- **`_float64.png`, `_float32.png`** (with `--plot`): time per rotor against N, one panel per
  ℓ_max.

**To get these results analysed, send the `.json` file.**  It contains the platform string
and GPU details but not the hostname.  Profiler traces from `--trace-dir` *do* include the
hostname in their file names.

### Columns

| column | meaning |
|---|---|
| `implementation` | `jax`, or `numba (CPU)` for the reference |
| `op`, `ell_max`, `precision`, `N` | what was run |
| `status` | `ok`, `out of memory`, or `error` (with the message in `error`) |
| `compile_s` | compilation time, in seconds |
| `median_s`, `min_s` | run time, in seconds |
| `rotors_per_s`, `ns_per_rotor` | throughput, from the median |
| `output_bytes` | size of the result |

## 5. Reading the results

- **Time per rotor** should fall as N grows and then level off once the GPU is fully busy.
  The level it settles at is the number that matters for large batches.
- **float32 against float64.**  Consumer and workstation GPUs (GeForce, RTX, most of the
  L-series) run float64 far more slowly than float32; data-centre GPUs (A100, H100) much less
  so.  The accuracy section shows what float32 costs in accuracy (about 10⁻⁶ for ℓ ≤ 8).
- **`rotate` against `rotate_via_D`**, and **`evaluate` against `evaluate_via_sYlm`**, show
  whether avoiding 𝔇 and intermediate arrays pays off on this GPU.
- **`waveform` against `rotate` + `evaluate`.**  `waveform` is the pattern recommended for a
  waveform observed in one direction; it should be about as cheap as `evaluate`.
- **Compile time** grows with ℓ_max and is paid once per input shape, so in a real pipeline
  keep shapes fixed.

## Troubleshooting

**JAX uses the CPU instead of the GPU.**  The script prints a warning if so.  Usually the
CPU-only `jax` package was installed, or the CUDA wheels do not match the driver.  Reinstall
with `python -m pip install -U "jax[cuda12]"` and check `nvidia-smi`.  Running
`python -c "import jax; jax.devices('gpu')"` prints the reason a GPU backend failed to load.

**Out of memory at the largest sizes.**  This is expected for `wigner_D` and `rotate_via_D`
at ℓ_max = 8, which produce 969 complex numbers per rotor (about 15 GB at N = 10⁶ in
float64).  The script records it and moves on.  By default JAX reserves 75% of GPU memory on
start-up.  To let it use more, set `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`; to allocate only
what is needed, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

**Several GPUs.**  The script uses the first device.  Choose one with, for example,
`CUDA_VISIBLE_DEVICES=1 python benchmarks/gpu_benchmark.py …`.

**Installing `spherical` fails while building `spinsfast`.**  `spinsfast` is a dependency of
`spherical` with compiled code.  If pip cannot find a wheel for your platform and the build
fails, installing it from conda-forge first usually works:
`conda install -c conda-forge spinsfast`, then repeat the `pip install -e .` step.
