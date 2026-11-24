# Benchmark results

Output of [`../gpu_benchmark.py`](../gpu_benchmark.py), kept for reference.  Each run has
a `.json` file (everything: environment, accuracy checks, every timing), a `.csv` file
(timings only) and plots of time per rotor against batch size N.

## `colab-t4/`: Google Colab, NVIDIA Tesla T4

Run with [`../colab_benchmark.ipynb`](../colab_benchmark.ipynb): Tesla T4 (15 GB),
driver 580.82.07, JAX 0.11.1, Python 3.13, ℓ_max = 8, spin weight −2.

| run | code | what |
|---|---|---|
| `*-quick_20261112-094723` | before 7d45700 | `--quick`: five operations, N ∈ {10³, 10⁵, 10⁶} |
| `*-l8_20261112-100805` | before 7d45700 | all nine operations, N from 10³ to 4×10⁶, float64 and float32 |
| `*-quick_20261124-200012` | 6095104 | `--quick`, after the memory and gradient changes |
| `*-l8_20261124-200511` | 6095104 | all nine operations, after the memory and gradient changes |

An earlier ℓ_max = 8 run from the same session is not included.  It ran while the
notebook itself held most of the GPU's memory (fixed since), so nearly every operation
ran out of memory at N = 10⁶.  Its timings at smaller N agree with the run above.

### First run (2026-11-12)

**Accuracy on the GPU** (max |JAX − numba| over 200 rotors, 7 at or near the poles):
float64 ≤ 1×10⁻¹⁴ for every operation; float32 ≈ 3×10⁻⁶ for 𝔇 and ≈ 1×10⁻⁵ for
`rotate` and `evaluate`.  Gradients are finite at the poles in both precisions.

**Time per rotor at N = 10⁶** (median), where the GPU is close to fully busy:

| operation | float32 | float64 |
|---|---|---|
| `sYlm` | 33 ns | 129 ns |
| `evaluate` | 33 ns | 145 ns |
| `waveform` | 34 ns | 144 ns |
| `waveform_value_and_grad` | 68 ns | 266 ns |
| `rotate` | 96 ns | 527 ns |
| `rotate_via_D` | 117 ns | 405 ns |
| `rotate_value_and_grad` | 109 ns | 502 ns |
| `wigner_D` | 110 ns | 304 ns (N = 10⁵) |

For comparison, the numba implementation on the Colab CPU, one rotor at a time, took
about 12 µs per rotor for `sYlm` and 25 µs for `wigner_D`.  The CPU there is slow and
numba ran on one core, so these ratios overstate the speed-up over a good CPU node.

**Observations**

- `evaluate` and `waveform` cost the same as computing ₛYₗₘ alone, so evaluating
  observed strain costs about 30 ns per sample in float32.
- Gradients cost about 2× the value for `waveform`.
- float64 is 4–5× slower than float32 on the T4.  That is much less than the T4's
  1:32 ratio of float64 to float32 arithmetic, which suggests these kernels are
  limited by memory traffic rather than arithmetic.
- `rotate` is about 3× slower than `evaluate`, and in float64 it is *slower* than
  building 𝔇 and multiplying (`rotate_via_D`).  Worth investigating.
- At N = 4×10⁶ every operation except float32 `sYlm` ran out of the T4's 15 GB.  That
  is more memory per rotor than the outputs need, so XLA is keeping large
  intermediates.  (Largely fixed in 7d45700; see below.)
- `evaluate_via_sYlm` is no slower than `evaluate`: XLA fuses the two steps.

### Second run (2026-11-24): after the memory and gradient changes

Commit 6095104 includes 7d45700 (ₛYₗₘ, `evaluate` and `rotate` no longer build the whole
flattened H wedge) and the new derivative rule for `evaluate`, which needs no harmonics
for the rotor gradient.  Accuracy is unchanged (float64 ≤ 1×10⁻¹⁴, float32 ≈ 1×10⁻⁵,
gradients finite at the poles).

**Time per rotor at N = 10⁶**, old → new (speed-up):

| operation | float32 | float64 |
|---|---|---|
| `sYlm` | 33 → 29 ns (1.1×) | 129 → 105 ns (1.2×) |
| `evaluate` | 33 → 27 ns (1.2×) | 145 → 77 ns (1.9×) |
| `waveform` | 34 → 27 ns (1.3×) | 144 → 77 ns (1.9×) |
| `waveform_value_and_grad` | 68 → 60 ns (1.1×) | 266 → 177 ns (1.5×) |
| `rotate` | 96 → 134 ns (**0.7×**) | 527 → 468 ns (1.1×) |
| `rotate_value_and_grad` | 109 → 93 ns (1.2×) | 502 → 501 ns (1.0×) |
| `wigner_D`, `rotate_via_D` | unchanged | unchanged |

At N = 10⁴–10⁵ the gains are larger: `evaluate`, `waveform` and
`waveform_value_and_grad` are about 2× faster in both precisions.

**N = 4×10⁶** now runs in float32 for `sYlm`, `evaluate`, `evaluate_via_sYlm`,
`waveform` (27 ns) and `waveform_value_and_grad` (58 ns), all of which ran out of
memory before.  The remaining failures are:

- every float64 operation except `sYlm` (95 ns), while *creating the inputs*: the benchmark gives each rotor
  its own 81 complex128 modes, which is 4.83 GiB on its own.  This is a limit of the
  benchmark's inputs, not of the code being measured.
- float32 `wigner_D`: the output alone is 29 GiB.
- float32 `rotate`, `rotate_via_D` and `rotate_value_and_grad`, which still keep large
  intermediates.

**Observations**

- `evaluate` is now cheaper than computing ₛYₗₘ, because it never stores the harmonics.
- float32 `rotate` is about 30% slower at N ≥ 10⁵ than before, while float64 `rotate`
  got faster.  The per-ℓ gather in 7d45700 is the likely cause.  `rotate` is still
  the operation most worth investigating.
