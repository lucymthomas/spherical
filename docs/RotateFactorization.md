# Rotating modes through $d(\pi/2)$

This page explains how `spherical.jaxwigner.rotate` computes rotated mode weights:
by factorizing Wigner's $d$ matrix through the constant matrix $d(\pi/2)$.  It
also describes a symmetry of that constant matrix, which could cut the arithmetic
by about a factor of four.  Both are compared with what Gumerov & Duraiswami
(2014; arXiv:1403.7698, "G&D" below) derive and with what the `spherical`
code takes from that paper.

**In short.**

* The factorization is G&D's *axis flip transform*, their Eq. (38).  G&D use it
  only to check their recursion, because computing every coefficient of $d(\beta)$
  from it costs $O(\ell^3)$ per $\ell$.  Rotating a set of modes never needs
  $d(\beta)$ itself, and then the factorization costs only $O(\ell^2)$ and needs no
  recursion.
* The symmetry of $\Delta = d(\pi/2)$ is G&D's *third* symmetry (their
  Eq. (30), $\beta \to \pi - \beta$), evaluated at the one angle where
  $\pi - \beta = \beta$, together with their first symmetry.  At that angle it
  stops relating two different matrices and becomes a symmetry *within* $\Delta$.
* G&D use the third symmetry for their error analysis, to restrict $\beta$ to
  $[0, \pi/2]$.  In their Eq. (40) they use it at $\beta = \pi/2$ to fold the sum
  in Eq. (38) in half.
* The `spherical` code lists all four symmetries in the docstrings of
  `spherical/recursions/wignerH.py` and `Wigner`, but only *uses* the first two.
  They give the "wedge" $m \geq |m'|$ (step 6 of G&D's algorithm), and they hold
  at every $\beta$.

Equation numbers follow the arXiv PDF of G&D (dated 29 March 2014).  The arXiv
HTML rendering numbers the equations differently.  For example, the symmetries
appear there as Eq. (37).

## Conventions

As in `spherical`, with $\epsilon_k = 1$ for $k \leq 0$ and $(-1)^k$ for $k > 0$:

\begin{align}
  \mathfrak{D}^{\ell}_{m',m}(\mathbf{R}) &= d^{\ell}_{m',m}(\beta)\, z_\alpha^{m'}\, z_\gamma^{m},
  \qquad z_\alpha = e^{i\alpha},\ z_\beta = e^{i\beta},\ z_\gamma = e^{i\gamma}, \\\\
  d^{\ell}_{m',m}(\beta) &= \epsilon_{m'}\, \epsilon_{-m}\, H^{m',m}_{\ell}(\beta)
  \qquad \text{(G\&D Eq. (29))}.
\end{align}

`rotate` computes $f'_{\ell m} = \sum_{m'} f_{\ell m'}\, \mathfrak{D}^{\ell}_{m',m}(\mathbf{R})$.
Below, $\Delta^{\ell}_{k,m} = d^{\ell}_{k,m}(\pi/2)$ is a constant real
$(2\ell+1)\times(2\ell+1)$ matrix, computed once (in float64) by the numba code.

## The factorization

G&D (§2.5) write a rotation by $\beta$ about the $y$ axis as: a quarter turn
that brings the $y$ axis onto the $z$ axis, a rotation by $\beta$ about $z$
(which is diagonal), and the inverse quarter turn (their Eq. (34)).  In terms of
$H$ this is their Eq. (38):

$$
H^{m',m}_{\ell}(\beta) = \sum_{\nu=-\ell}^{\ell}
  e^{i m' \pi/2}\, H^{m',\nu}_{\ell}(\tfrac{\pi}{2})\, e^{i\nu\beta}\,
  H^{\nu,m}_{\ell}(\tfrac{\pi}{2})\, e^{i m\pi/2}.
$$

Converting to $d$ with Eq. (29), and using
$\Delta_{m,k} = (-1)^{m-k} \Delta_{k,m}$, gives the form used in the code:

$$
d^{\ell}_{m',m}(\beta) = i^{\,m-m'} \sum_{k=-\ell}^{\ell}
  \Delta^{\ell}_{k,m'}\, e^{ik\beta}\, \Delta^{\ell}_{k,m}.
$$

Substituting this into the rotation gives

$$
f'_{\ell m} = (i z_\gamma)^{m} \sum_{k} \Delta^{\ell}_{k,m}\, z_\beta^{k}
  \sum_{m'} \Delta^{\ell}_{k,m'}\, f_{\ell m'}\, (z_\alpha / i)^{m'} .
$$

So a rotation is: multiply by phases, multiply by the constant matrix $\Delta$,
multiply by the phases $z_\beta^{k}$, multiply by $\Delta^{\mathsf T}$, and
multiply by phases again.  There is no recursion in $\beta$.  The products with a
constant matrix are the operations that GPUs (and CPUs) are best at, and
they make up most of the work.

**Why G&D did not use it, and why `rotate` does.**  G&D (§7.2) note that computing
all the coefficients $H^{m',m}_{\ell}(\beta)$ from Eqs. (38)–(40) costs $O(\ell^3)$
per $\ell$, against $O(\ell^2)$ for their recursion.  So they use the flip
decomposition only to validate the recursion, and mention a hybrid method (their
ref. [24]).  `rotate` never forms $d(\beta)$.  It applies $d(\beta)$ to one
vector of modes, which costs $O(\ell^2)$ per $\ell$ either way.  The recursion is
then pure overhead.  It also compiled to hundreds of small GPU kernels, each of
which writes its result to memory.  G&D themselves say this flip composition "is
well-known and used in some algorithms".  It is the basis of several FFT-based
methods for Wigner functions and rotations (for example Risbo 1996; Trapani &
Navaza 2006; McEwen & Wiaux 2011, who call $d(\pi/2)$ "$\Delta$").  I have not
checked those papers in preparing this note.

The real-part form, G&D Eq. (39), does not help here: it relies on $H(\beta)$
being real, whereas `rotate` applies the complex factorization to complex modes.

## The symmetry of $\Delta$

G&D list four symmetries (Eq. (30)):

\begin{align}
  \text{(a)}\quad & H^{m',m}_{\ell}(\beta) = H^{m,m'}_{\ell}(\beta), \\\\
  \text{(b)}\quad & H^{m',m}_{\ell}(\beta) = H^{-m',-m}_{\ell}(\beta), \\\\
  \text{(c)}\quad & H^{m',m}_{\ell}(\pi-\beta) = (-1)^{\ell+m'+m}\, H^{-m',m}_{\ell}(\beta), \\\\
  \text{(d)}\quad & H^{m',m}_{\ell}(-\beta) = (-1)^{m'+m}\, H^{m',m}_{\ell}(\beta).
\end{align}

(a) and (b) hold at every $\beta$.  They are the reason only the wedge
$m \geq |m'|$ needs computing, and they are all that the recursion in `spherical`
(and its JAX version, `H_wedge`) uses.

(c) relates $H$ at two *different* angles.  At $\beta = \pi/2$ the two angles
coincide, so (c) becomes a relation between elements of the single matrix
$H(\pi/2)$:

$$
H^{m',m}_{\ell}(\tfrac{\pi}{2}) = (-1)^{\ell+m'+m}\, H^{-m',m}_{\ell}(\tfrac{\pi}{2}).
$$

Converting to $d$ with Eq. (29), using $\epsilon_{-k} = (-1)^k \epsilon_k$, and
combining with (a), gives the two symmetries of $\Delta$ used below:

\begin{align}
  \text{(i)}\quad  & \Delta^{\ell}_{-k,m} = (-1)^{\ell+m}\, \Delta^{\ell}_{k,m}, \\\\
  \text{(ii)}\quad & \Delta^{\ell}_{k,-m} = (-1)^{\ell+k}\, \Delta^{\ell}_{k,m}.
\end{align}

Together, (i) and (ii) imply
$\Delta_{-k,-m} = (-1)^{k-m}\, \Delta_{k,m}$, which is just (b) at $\pi/2$.  So
the new information beyond the wedge symmetries is (c) at $\beta = \pi/2$.
Geometrically, $d(\pi - \beta) = d(\pi)\, d(-\beta)$, and $d(\pi)$ is the
permutation $m \to -m$ with signs $(-1)^{\ell+m}$.  A quarter turn, followed by a
half turn about the same axis, is the inverse quarter turn.  That is why
$d(\pi/2)$ is mapped to itself, up to signs and transposition.

G&D use (c) in two places.  In §3.1 and later, it restricts $\beta$ to
$[0, \pi/2]$ for their bounds and stability analysis.  In Eq. (40), it is applied
at $\pi/2$ to fold the $\nu$ and $-\nu$ terms of Eq. (39) together, which halves
that sum.  Symmetry (i) is exactly that folding.  Symmetry (ii) folds the other
index as well.

## How the symmetry reduces the arithmetic

Take one product $y = \Delta x$ for a single $\ell$ (the other product, with
$\Delta^{\mathsf T}$, is analogous).  Write

$$
x^{\pm}_{m} = x_m \pm x_{-m}, \qquad y^{\pm}_{k} = \tfrac12 (y_k \pm y_{-k}),
\qquad k, m \geq 0,
$$

where for $m = 0$ and $k = 0$ the element itself is used.  By (ii), output $k$
sees only $x^{\sigma}$ with $\sigma = (-1)^{\ell+k}$.  By (i), the combination
$y^{\tau}_k$ sees only inputs with $(-1)^{\ell+m} = \tau$.  So

$$
y^{\tau}_{k} = \sum_{\substack{m \geq 0 \\\\ (-1)^{\ell+m} = \tau}}
  \Delta^{\ell}_{k,m}\, x^{\sigma}_{m},
  \qquad \sigma = (-1)^{\ell+k},\quad k \geq 0 .
$$

In the basis of sums and differences, $\Delta$ therefore splits into **four
dense blocks**, one for each pair $(\sigma, \tau)$, each of size about
$\tfrac{\ell}{2} \times \tfrac{\ell}{2}$.  Counting exactly, one product needs
$\ell^2 + \ell + 1$ multiply-adds instead of $(2\ell+1)^2$:

| $\ell$ | dense $(2\ell+1)^2$ | with (i) and (ii): $\ell^2+\ell+1$ |
|---|---|---|
| 2 | 25 | 7 |
| 4 | 81 | 21 |
| 8 | 289 | 73 |
| all $\ell \leq 8$ | 969 | 249 |
| large $\ell$ | $\approx 4\ell^2$ | $\approx \ell^2$ |

The price is cheap sums and differences: one set before each product, and one
after it to recover $y_k = y^{+}_k + y^{-}_k$ and $y_{-k} = y^{+}_k - y^{-}_k$.

For comparison, the previous `rotate` (recursion for $d(\beta)$, then
$\sum_{m'} f_{m'} d_{m',m}$) needed the recursion *plus* 969 multiply-adds for
$\ell \leq 8$.  The dense flip needs $2 \times 969$ multiply-adds and no
recursion.  The flip with (i) and (ii) needs $2 \times 249$.

## Does it pay off in practice?

Fewer operations does not always mean faster code.  On a laptop CPU (XLA, chain powers,
$\ell \leq 8$, $N = 5\times10^4$), the variants in
`benchmarks/rotate_variants.py` gave:

| implementation | float32 | float64 |
|---|---|---|
| dense, products of ≤36 modes | 682 ns/rotor | 1115 ns/rotor |
| dense, one 81×81 product | 633 | 900 |
| four blocks, grouped into one product | 1551 | 2731 |
| four blocks, written out as plain arithmetic | 2679 | 4875 |

The blocked versions are slower here.  XLA does not merge the sum-and-difference
steps and the many tiny blocks into the surrounding computation, so their
memory traffic outweighs the arithmetic saved.  The CPU is not where this
matters, though.

The expectation was that the symmetry would pay off for float64 on GPUs with
weak float64 arithmetic, such as the T4, where arithmetic matters: grouping
$\ell \leq 8$ into one 81×81 product (seven times the necessary arithmetic) made
float64 `rotate` 1.5× slower there than products of ≤36 modes.  It does not, at
least at $\ell \leq 8$.  On a Tesla T4 ($N = 10^6$,
`benchmarks/results/colab-t4/rotate_variants_*.json`):

| implementation | float32 | float64 |
|---|---|---|
| dense, products of ≤36 modes (current `rotate`) | 41 ns/rotor | 279 ns/rotor |
| dense, one 81×81 product | 52 | 441 |
| four blocks, best grouping | 236 | 591 |
| four blocks, written out as plain arithmetic | 212 | 547 |

The blocked versions are 2–5× slower here too.  Profiling shows the products
take about 70% of float64 `rotate` on the T4, so the arithmetic saved is real,
but rearranging the modes into the $f_m \pm f_{-m}$ basis and back, and running
many small products, costs more than the factor of four saves.

The symmetry could still pay off:

* **at larger $\ell$**, where the $(2\ell+1)^2$ products dominate everything else
  and the blocks are large enough to use the hardware well;
* **in a hand-written kernel** (Pallas or CUDA) that applies the four blocks while
  the modes are in registers, so the basis change costs no memory traffic.

It is unlikely to help on data-centre GPUs (A100, H100), whose fast float64
arithmetic leaves these kernels limited by memory, not arithmetic.

`benchmarks/rotate_variants.py` (also the optional cell in
`benchmarks/colab_benchmark.ipynb`) times all these variants, so the comparison
can be repeated on other hardware or at larger $\ell$ with `--ell-max`.

## Numerical checks

The following were checked numerically for $\ell \leq 6$ or $\ell \leq 8$, using
the numba `spherical.Wigner`.  All agree to within $10^{-15}$:

* G&D Eq. (38) and symmetry (c), in G&D's $H$ convention;
* the $d$ form of the factorization above;
* symmetries (i) and (ii), and $\Delta_{m,k} = (-1)^{m-k} \Delta_{k,m}$;
* the four-block formula, including the counts $\ell^2+\ell+1$.

All variants of `rotate` in `benchmarks/rotate_variants.py` agree with the current
implementation to about $10^{-14}$ in float64 and $10^{-6}$ in float32.

## References

* N. A. Gumerov and R. Duraiswami, *Recursive computation of spherical harmonic
  rotation coefficients of large degree*, arXiv:1403.7698 (2014), UMIACS-TR-2014-04.
  §2.3 (symmetries, Eq. (30)), §2.5 (axis flip transform, Eqs. (34)–(40)),
  §6.1 (the recursive algorithm used by `spherical`), §7.2 (validation with the
  flip decomposition).
* T. Risbo, *Fourier transform summation of Legendre series and D-functions*,
  J. Geodesy 70, 383 (1996).
* S. Trapani and J. Navaza, *Calculation of spherical harmonics and Wigner d
  functions by FFT*, Acta Cryst. A 62, 262 (2006).
* J. D. McEwen and Y. Wiaux, *A novel sampling theorem on the sphere*, IEEE
  Trans. Signal Process. 59, 5876 (2011).
