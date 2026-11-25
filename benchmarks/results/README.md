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
| `*-quick_20261125-120336` | fc6ffa3 | `--quick`, with the new `rotate` |
| `*-l8_20261125-121038` | fc6ffa3 | all nine operations, with the new `rotate` |
| `rotate_group_sizes_*_20261125-122225` | fc6ffa3 | `rotate` with each grouping of its ℓ blocks, N = 10⁶ |
| `*-quick_20261125-144458` | 9fd0e07 | `--quick`, with group size 36 |
| `*-l8_20261125-144905` | 9fd0e07 | all nine operations, with group size 36 |
| `rotate_variants_*_20261125-150314` | 9fd0e07 | 16 implementations of `rotate`, N = 10⁶ |
| `profile_*_20261125-150433` | 9fd0e07 | time per GPU kernel for six operations, N = 10⁶ |
| `rotate_variants_*_20261125-171551` | 6348bbe | the same, with binary powers in `rotate` |
| `profile_*_20261125-171904` | 6348bbe | the same, with binary powers in `rotate` |
| `evaluate_variants_*_20261125-171748` | 6348bbe | 7 ways of writing the final sum in `evaluate`, N = 10⁶ |

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

### Third run (2026-11-25): `rotate` without the recursion

Commit fc6ffa3 computes `rotate` from dˡ(β) = i^{m-m′} Δᵀ diag(e^{ikβ}) Δ with the
constant Δ = dˡ(π/2): phase multiplications and two products with a fixed
block-diagonal matrix, all ℓ in one product.  Nothing else changed, and the other
operations' timings agree with the second run.

**`rotate`, time per rotor at N = 10⁶**, second run → third run:

| operation | float32 | float64 |
|---|---|---|
| `rotate` | 134 → 53 ns (2.5×) | 468 → 465 ns (1.0×) |
| `rotate_value_and_grad` | 93 → 62 ns (1.5×) | 501 → 653 ns (**0.8×**) |

float32 `rotate` now also runs at N = 4×10⁶ (54 ns), and is about as fast as
`waveform_value_and_grad`.  `rotate_value_and_grad` and `rotate_via_D` still run out
of memory there.

**Group sizes** (`rotate_group_sizes_*.json`, N = 10⁶; the size is the largest number
of modes in one product):

| group size | products | float32 | float64 |
|---|---|---|---|
| 1 (one per ℓ) | 9 | 65 ns | 338 ns |
| 16 | 6 | 62 ns | 327 ns |
| 36 | 3 | 52 ns | 325 ns |
| 81 (one product) | 1 | 56 ns | 484 ns |

The single 81×81 product, the default in fc6ffa3, spends 7× more arithmetic than
needed on zeros, which costs little in float32 but a lot in float64, where the T4 is
32× slower.  That is why float64 `rotate` did not improve and its gradient got
slower.  The default is now 36, the fastest here in both precisions; this should take
float64 `rotate` from about 465 to 325 ns.  Even so, float64 products with such small
matrices use the T4 poorly, so float64 `rotate` remains far slower than float32.

### Fourth run (2026-11-25): group size 36, `rotate` variants, kernel profile

**Group size 36 confirmed** in the full benchmark (N = 10⁶, third run → fourth run):
float64 `rotate` 465 → 305 ns (1.5×) and `rotate_value_and_grad` 653 → 486 ns
(1.3×); float32 unchanged (52 and 61 ns).  Every other operation is unchanged.

**`rotate` variants** (`rotate_variants_*.json`, N = 10⁶; an earlier run of the same
script agreed within 1%):

| variant | float32 | float64 |
|---|---|---|
| dense-36, chain powers (the package in 9fd0e07) | 53 ns | 318 ns |
| dense-36, binary powers | **41 ns** | **279 ns** |
| dense-81, either powers | 51–52 ns | 441–484 ns |
| parity blocks, best of 8 | 212 ns | 547 ns |

- The four-block parity split of Δ (see `docs/RotateFactorization.md`) is 2–5×
  *slower* than the dense products, despite a quarter of the arithmetic: moving
  modes into the f_m ± f_{−m} basis and out again costs more than the products
  save, and the small blocks use the GPU poorly.
- Building zᵏ from z, z², z⁴, z⁸ ("binary") instead of by repeated multiplication
  saves 22% in float32 and 12% in float64.  `rotate` now does this.

**Where the time goes** (`profile_*.json`, ns per rotor, N = 10⁶):

- `evaluate`/`waveform`, float32 (27 ns): about 18 ns (two thirds) is the final
  Σ fₗₘ ₛYₗₘ, which XLA compiles to one kernel in which each thread sums one
  rotor's 81 terms; the H recursion and phases take the rest.  Reading the modes
  needs only about 2 ns at the T4's memory bandwidth, so this kernel is far from
  the hardware limit.
- `evaluate`, float64 (75 ns): the same final sum is 27 ns, and the H recursion
  kernels about 35 ns, which is float64 arithmetic.
- `sYlm`: 70% (float32) and 61% (float64) is writing the 81 harmonics out.
- `rotate`, float64: 68% is the matrix products (cuBLAS dgemm), as expected on a
  GPU with 1:32 float64.  In float32 the products are only 28%; the rest is
  stacking real and imaginary parts and joining the groups back together.
- `rotate_value_and_grad`, float64: 137 of 500 ns is one reduction in the
  benchmark's own loss (Σ|f′|² weighted), not in `rotate`.

### Fifth run (2026-11-25): binary powers in `rotate`, `evaluate` variants

**`rotate` with binary powers** (4f9ff00), from the profile at N = 10⁶: float32
52 → 42 ns, float64 305 → 282 ns; `rotate_value_and_grad` 61 → 54 ns and 486 →
464 ns.

**`evaluate` variants** (`evaluate_variants_*.json`, N = 10⁶, spin weight −2; see
`../evaluate_variants.py`):

| variant | float32 | float64 |
|---|---|---|
| current | 27.5 ns | 75.0 ns |
| one reduction over all modes | **23.0** | 127.8 |
| sum over ℓ first, then Σₘ gₘ zₐᵐ | 46.7 | 102.5 |
| the same, by Horner's rule | 47.4 | 99.3 |

Moving the batch to the last axis before the sum (the `-T` variants) changed
nothing, within 1 ns.  Only the single reduction is faster, by 16% and in float32
only; in float64 it is 1.7× slower.  The current form is the best overall, so
`evaluate` is unchanged.
