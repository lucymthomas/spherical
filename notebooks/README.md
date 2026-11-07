# Notebooks: `spherical.jaxwigner`

Worked examples and documentation for the JAX implementation of Wigner 𝔇 matrices and
spin-weighted spherical harmonics, written alongside its development.

| Notebook | Contents |
|---|---|
| [01_wigner_D_in_jax.ipynb](01_wigner_D_in_jax.ipynb) | Conventions, the H wedge, d and 𝔇 matrices and their indexing, ₛYₗₘ, validation against the numba implementation, batching/`jit`/`vmap`, single precision |
| [02_rotating_and_evaluating_modes.ipynb](02_rotating_and_evaluating_modes.ipynb) | `rotate` and `evaluate` without building 𝔇, broadcasting of modes and rotors, composing rotors instead of rotating modes, a toy precessing waveform, cost comparison, quaternion helpers |
| [03_gradients.ipynb](03_gradients.ipynb) | Why plain autodiff fails at the poles, exact derivatives from the rotation group, gradients with respect to sky position and Euler angles, a Fisher matrix on the pole, cost, what not to differentiate |

## Running them

From the repository root:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/
```

The notebooks are committed with their outputs.  Timings in them were measured on a
laptop CPU and are illustrative only.
