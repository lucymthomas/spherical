# Benchmark results

Output of [`../gpu_benchmark.py`](../gpu_benchmark.py), kept for reference.  Each run has
a `.json` file (everything: environment, accuracy checks, every timing), a `.csv` file
(timings only) and plots of time per rotor against batch size N.

## `colab-t4/`: Google Colab, NVIDIA Tesla T4

Run on 2026-11-12 with [`../colab_benchmark.ipynb`](../colab_benchmark.ipynb): Tesla T4
(15 GB), driver 580.82.07, JAX 0.11.1, Python 3.13, ℓ_max = 8, spin weight −2.

| run | what |
|---|---|
| `*-quick_20261112-094723` | `--quick`: five operations, N ∈ {10³, 10⁵, 10⁶} |
| `*-l8_20261112-100805` | all nine operations, N from 10³ to 4×10⁶, float64 and float32 |

An earlier ℓ_max = 8 run from the same session is not included.  It ran while the
notebook itself held most of the GPU's memory (fixed since), so nearly every operation
ran out of memory at N = 10⁶.  Its timings at smaller N agree with the run above.

### Summary

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
  intermediates; until that is understood, split large batches into chunks of ~10⁶.
- `evaluate_via_sYlm` is no slower than `evaluate`: XLA fuses the two steps.
