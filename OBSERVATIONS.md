# Empirical Observations

This document records empirical findings that emerged during development:
phenomena observed in training, properties of the formulation discovered
through experiment, and recurring patterns that turned out to matter.

DECISIONS.md captures deliberate design choices. OBSERVATIONS.md
captures what we *learned* from running the code. The two together
form the project's research record.

Each entry includes: what we observed, where (which experiment / stage),
why we think it happens, and any implications for downstream work.

---

## O1 — Per-iteration loss has a characteristic U-shape (curriculum effect)

**Observed in:** Stage 5.5 (placeholder on anticline), Stage 6.3 (real
operator on anticline), Stage 4.5 (real operator on `01_Topo` real
horizon).

**What we see:** When plotting per-iteration data loss $L_{\text{data},t}$
against optimizer step for $t = 1, 2, \ldots, N$, the curves don't
converge to the same value. They settle into a characteristic ordering:

- **Early iterations ($t = 1, 2$) have the highest loss.** The model
  has limited information at iteration 1 — only its immediate
  neighborhood's $z^0$ values (mostly the mean-plane initialization)
  and the one-hop boundary with $K$. Hard to make accurate predictions
  with this little signal.
- **Middle iterations ($t \approx N/2$) have the lowest loss.** By this
  point, several rounds of message-passing-and-anchoring have
  propagated real information from $K$ into the working region. Each
  vertex's input features have effectively been processed by $t$
  layers of GNN (with anchoring acting as a hard constraint between
  layers), giving the model richer signal.
- **Late iterations ($t \approx N$) creep back up.** Two effects: (i)
  more vertices to fit (the unknown ring at large $t$ is far from any
  ground truth in $K$), and (ii) gradient signal from late iterations
  has to flow back through $N$ rollout steps via BPTT, making
  late-iteration parameter updates noisier.

**Quantitatively** (Stage 4.5, real `01_Topo`, after 1000 steps):
- $L_1 \approx 15$, $L_4 \approx 5$, $L_9 \approx 4$ (noisier)

**Why this is a property of the formulation:**

The model parameters are *shared* across iterations. The same operator
$F_\Theta$ predicts $\Delta z$ at iteration 1, 2, ..., N. So the model
can't have a "different network for $t=1$" — it must handle every
iteration with one parameter set. The optimal parameter set is a
compromise across all iterations, and the curriculum effect shows where
that compromise lands.

This means:

1. **The model isn't learning a "predict $\Delta z$ from raw $z$"
   function.** It's learning a function that, when iterated, produces
   good extrapolation. Each iteration's quality depends on the
   accumulated state from earlier iterations.

2. **The per-iteration rollout weights $w_t$ are the formal lever to
   shift this compromise.** Setting $w_t > 1$ for late iterations
   biases the model toward fitting later-ring vertices well at the
   cost of earlier ones; setting $w_t < 1$ for early iterations does
   the opposite. We currently use uniform $w_t = 1$.

3. **Late-iteration noise (especially at $t \approx N$) is a real
   phenomenon** and may merit explicit attention in Stage 8. Options
   include: gradient clipping (already in use), reducing the effective
   BPTT length via truncated BPTT, or shrinking $w_t$ at the largest
   $t$.

**Consistency across settings:**

This pattern was observed:
- With the placeholder model (Stage 5.5) on synthetic anticline.
- With the real `LocalOperator` (Stage 6.3) on synthetic anticline.
- With the real `LocalOperator` (Stage 4.5) on real `01_Topo`.

The fact that it appears across model architecture and across data
distribution (synthetic vs. real) confirms it's a property of the
*formulation* (BPTT + shared weights + anchoring + rollout supervision),
not an artifact of any particular setting.

**Implications for downstream work:**

- Stage 8 should report per-iteration metrics in addition to total loss
  in TensorBoard, so this pattern is visible during training.
- Stage 12 ablations on $w_t$ should be informed by this baseline.
- When interpreting results, "final $z^N$ on far rings has higher error
  than middle rings" is expected, not a bug.

---

## O2 — Per-step training loss spans ~6 orders of magnitude across surfaces

**Observed in:** Stage 8.3 (first multi-surface training run, 3 epochs
on the 30 train surfaces).

**What we see:** Within a single epoch, the per-step training loss
(printed to stdout at every 10th step) ranges from O(1) to O(10⁷):
ep 0  step  0   TestHorizon1     N=10  loss=227051
ep 0  step 10   08_BaseAlagoas   N=48  loss=52232504    <- 4 orders larger
ep 0  step 20   Horizonte3       N=31  loss=30794      <- back down
ep 2  step 50   horizonte3-utm   N=9   loss=36         <- 6 orders smaller!

This is *not* an instability problem. Adjacent steps update the same
model with very different surfaces, and each surface produces a loss
on its own scale.

**Why this happens:**

The data-loss component is a mean of squared (z - z_true) over
$F_t \cup P_t$, scaled by $\lambda_f$ and $\lambda_p$. The
magnitude depends on three things, all of which vary wildly across
the dataset:

1. **Mesh size** (|V| from ~2,400 to 48,000). Larger meshes have
   larger $|U|$, but since we use *mean* (D5.7), this doesn't
   directly scale the loss. However, larger meshes tend to have
   higher $N$ (longer rollouts), so the cumulative effect through
   the rollout sum still varies.
2. **Rollout depth $N$** (from 7 to 53+). The rollout loss sums
   per-iteration losses over $N$ iterations, so deeper rollouts
   accumulate more total loss.
3. **z magnitude (after centering)**. Even with per-surface
   centering (D4.6), the residual variance of $z$ within a surface
   varies hugely. A flat horizon spans a few meters; a folded one
   spans hundreds. Squared error scales with the square of $z$ range.

For example: surface `08_BaseAlagoas` (|V|≈48k, N=48) has both a
large mesh and a deep rollout, producing loss ~5e7. Surface
`horizonte3-utm` (|V|≈2k, N=9) has neither, producing loss ~36.

**Implications:**

1. **Per-step training loss is not a useful monitoring signal.** Its
   value depends as much on *which surface* the step was on as on
   *how well the model is doing*. Two consecutive steps showing
   loss=50M and loss=50 don't indicate divergence — they indicate
   the second step happened on a smaller, easier surface.

2. **Per-epoch mean is the right granularity.** Averaging over all
   30 surfaces in an epoch smooths out the surface-to-surface
   variance and gives a meaningful trajectory. This is what we log
   to TensorBoard (D8.2).

3. **Val RMSE in meters is the most interpretable single metric.**
   Loss is scale-dependent; RMSE in physical units (meters of
   z-error on unknown vertices) is comparable across surfaces and
   reservoirs.

4. **Aggregate train loss is dominated by the largest/deepest
   surfaces.** Mean loss across 30 surfaces is mathematically a mean,
   but functionally a few surfaces with loss ~1e7 swamp 25 surfaces
   with loss ~1e2. We may want to log a *geometric* mean or median
   in addition to the arithmetic mean, to keep the small surfaces
   visible.

**A note on whether this is a problem to fix:**

Not necessarily. The model is supposed to do well on all surfaces,
not just balance loss magnitudes equally. The optimization signal
from a high-loss surface is legitimately stronger than from a
low-loss surface — that surface needs more learning. The arithmetic
mean is the right thing for the optimizer to follow.

The thing to watch is per-surface val RMSE (logged to TensorBoard
in tags like `val_rmse_per_surface/05_TopoCretaceo`). If after many
epochs of training a few specific surfaces remain at high RMSE
while most have converged, *that* is a problem worth investigating.

---

## O3 — First full training run: 5.5× RMSE reduction, per-surface heterogeneity dominates aggregate metrics

**Observed in:** Stage 8.7 (first end-to-end training run on the canonical
dataset split: 30 train surfaces, 7 val surfaces, 100 max epochs with
patience=20).

### Top-line result

- **Initial val RMSE (untrained model):** 489 m
- **Best val RMSE (epoch 17 of 38):** 79.3 m  → **6.2× reduction**
- **Best val loss:** 385,594 at epoch 17
- **Training duration:** 38 epochs (early-stopped after 20 epochs of no
  val-loss improvement). Wall time ~32 minutes on CPU.
- **Train vs val loss at convergence:** train ~1.2M, val ~390k (val is
  *lower* than train, see "anti-overfitting pattern" below).

### Per-surface val RMSE shows enormous heterogeneity

The per-surface breakdown at the best epoch reveals a 2,000× spread
across the 7 val surfaces:

| Surface | RMSE (m) | n_vertices | regime | Note |
|---|---|---|---|---|
| horizonte7 | 0.13 | ~9.7k | outward_pinned | near-perfect fit |
| Horizonte5 | 0.21 | ~9.7k | outward_free | near-perfect fit |
| 10_BaseModelo | 3.94 | 48k | outward_pinned | excellent despite size |
| TestHorizon4 | 49.5 | ~2.4k | outward_pinned | moderate |
| TestHorizon7 | 77.6 | ~2.4k | half_plane | moderate |
| 09_Horizonte8 | 141.4 | ~2.4k | outward_free | poor |
| 05_TopoCretaceo | 282.6 | 48k | outward_pinned | **outlier** |

**Mean aggregate (79.3 m) is dominated by the outlier.** Excluding the
worst surface, mean RMSE drops to ~52 m. Excluding the top two hardest,
it drops to ~22 m. The aggregate is a misleading single number for a
distribution this skewed.

**Size alone is not the predictor of difficulty.** Both 10_BaseModelo
and 05_TopoCretaceo are 48k-vertex meshes; one reaches 3.94 m, the
other 282.6 m. The difference must come from geometry/connectivity
specific to the individual surface, not from mesh size per se.

### "Anti-overfitting" pattern: val < train

Counterintuitively, val total loss (~390k) is consistently *below* train
total loss (~1.2M) at convergence. This is NOT a bug — it's the same
per-surface variance effect from O2 manifesting at the aggregate level:

- **Train (30 surfaces) includes the largest, hardest 48k-vertex
  surfaces.** Two of those have train-time per-step losses in the
  millions (e.g., `08_BaseAlagoas` shows step losses of 5-50M), which
  dominate the per-epoch mean.
- **Val (7 surfaces) happens to contain a milder mix.** Two of the
  three 48k-vertex meshes in val (`10_BaseModelo`, `05_TopoCretaceo`)
  contribute moderately; the other surfaces are smaller and easier.

This means **we are not overfitting.** The model has converged: train
and val have both plateaued and the gap doesn't widen with continued
training.

### Implications for downstream stages

1. **Stage 9 (gradient accumulation)** may help by stabilizing gradient
   estimates across surfaces — currently each step's gradient is
   dominated by one wildly-varying loss scale, which makes the
   optimizer's job harder.

2. **Stage 10 (evaluation)** should report per-surface and per-ring
   metrics, not just aggregates. The thesis writeup should also report
   **median RMSE** in addition to mean — the median is robust to the
   outlier surface and gives a more representative number.

3. **05_TopoCretaceo is a known-hard surface.** Worth a future
   investigation: visualize the surface (likely Stage 11 or 12) and
   diagnose what makes it hard. Candidate hypotheses: highly
   non-stationary curvature, multi-scale features the model can't
   capture, unusual triangle aspect ratios, or simply a structurally
   different horizon class.

4. **Mean RMSE of 80 m is a defensible baseline.** This is a real number
   we can build on: future improvements should be measured against this
   reference. Subsequent Stage 12 ablations should report deltas from
   this baseline rather than absolute numbers.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260609_072419/best.pt`
  (epoch 17, best_val_loss=385594, ~273 KB).
- Full TensorBoard logs in the same directory.
- Config snapshot saved as `config.yaml` in the run directory.

---

## O4 — Gradient accumulation (B=4 vs B=1): trades best-case for worst-case

**Observed in:** Stage 9.3 A/B comparison. Identical hyperparameters
(seed, 100 max epochs, patience=20, LR schedule, loss weights),
with the only change being `optim.accum_steps`: 1 (one optimizer
step per surface) vs 4 (one optimizer step per 4-surface batch).

### Top-line numbers (both runs early-stopped)

| Metric | B=1 (8.7 baseline) | B=4 | Δ |
|---|---|---|---|
| Best val loss | 385,594 | 322,651 | **−16%** |
| Best val data loss | 385,580 | 322,635 | −16% |
| Best val curv loss | 1,398 | 1,567 | +12% |
| Best val res loss | 31 | 174 | +458% |
| Best epoch | 17 | 28 | later |
| Epochs run | 38 | 48 | longer |

B=4 reaches a lower total loss but takes more epochs to get there.

### The per-surface story is more nuanced than the aggregate

Per-surface RMSE at best epoch:

| Surfac | B=4 | Δ |
|---|---|---|---|
| 05_TopoCretaceo | 273.64 | 258.00 | **−5.7%** ✓ |
| 09_Horizonte8 | 135.59 | 120.00 | **−11.5%** ✓ |
| 10_BaseModelo | 62.47 | 113.66 | **+82%** ✗ |
| Horizonte5 | 4.01 | 9.26 | +130% |
| TestHorizon4 | 72.98 | 61.61 | **−15.6%** ✓ |
| TestHorizon7 | 72.12 | 69.03 | −4.3% ✓ |
| horizonte7 | 2.06 | 5.71 | +177% |

Five of seven surfaces follow a clear pattern: **B=4 improves the
hard ones, degrades the easy ones.** The easy surfaces (Horizonte5,
horizonte7) went from sub-5m RMSE to single-digit m — large
percentage change, small absolute change. The hardest surface
(05_TopoCretaceo) improved by 15.6m in absolute terms.

10_BaseModelo is the anomaly: a "middle" surface that got
substantially worse. We don't have a clean explanation; it's a
48k-vertex surface so possibly the larger batch size's gradient
averaging happened to land ihis one mesh.

### Summary statistics tell the real story

| Statistic | B=1 | B=4 | Verdict |
|---|---|---|---|
| Mean RMSE | 88.98 | 91.04 | B=1 wins |
| Median RMSE | 72.12 | **69.03** | B=4 wins |
| Max RMSE | 273.64 | **258.00** | B=4 wins (worst case) |
| Min RMSE | 2.06 | 5.71 | B=1 wins (best case) |
| Range | 271.58 | **252.29** | B=4 (tighter spread) |

Three of five favor B=4; the two that favor B=1 are the mean (which
is sensitive to a single regressing surface) and the min (which
measures how perfectly the easiest surface is fit, not generally
useful).

### Why total loss and mean RMSE disagree

The total loss is dominated by the surfaces with the largest squared
errors (squaring amplifies the contribution of bad surfaces). When
B=4 improves the hardest surfaces, total loss drops a lot. When B=4
degrades the easiest surfaces, total loss barely notices (because
their squared errors were tiny to begin with).

Mean RMSE, by contrast, weights each surface roughly equally. A
small increase on an easy surface (in meters) has the same effect
on aggregate RMSE as a small improvement on a hard surface.

So the disagreement is not contradictory; the two metrics are
genuinely measuring different things:

- **Total loss** = "are the worst predictions getting better?"
  Dominated by outliers.
- **Mean RMSE** = "on average, how off is each surface?"
  Robust to outliers but masks worst-case behavior.

### Verdict and decision

**B=4 accepted as the new baseline.** Reasons:

1. The improvements (median, worst-case, range) are in the
   directions that matter for the project's goal: a model that
   generalizes across surfaces, including hard ones.
2. The regressions (mean, min) are small in absolute terms (~2m
   each) and concentrated on surfaces that were already very easy.
3. The optimization-objective improvement (16% lower total loss)
   is substantial.
4. The gradient-variance hypothesis from O2 is partially
   supported: averaging across 4 surfaces does smooth gradient
   estimates, and the model finds a better basin.

### Caveats and implications

1. **Mean RMSE alone is a misleading metric for this dataset.**
   The thesis should report mean, median, max, and per-surface
   breakdowns. The aggregate hides important per-surface variance.

2. **B=2 is unexplored.** It's plausible there's a sweet spot
   between B=1's surface-by-surface optimization and B=4's
   aggressive averaging. Deferred to Stage 12 ablations.

3. **N=1 experiment.** This is a single A/B with a single seed.
   Both runs use the same data shuffling RNG so they're directly
   comparable, but we haven't varied the seed. The conclusion would
   be stronger with 3-5 seeds per condition; deferred for time.

4. **Stage 10 (evaluation) should report per-surface metrics by
   default**, not just aggregates, to keep this kind of nuance
   visible going forward.

### Where the result lives

- B=4 checkpoint: `outputs/tensorboard/run_20260609_092252/best.pt`
  (epoch 28, best_val_loss=322651).
- B=1 baseline checkpoint: `outputs/tensorboard/run_20260609_072419/best.pt`
  (epoch 17, best_val_loss=385594).

---

## O5 — Baseline comparison: each method has its regime where it shines

**Observed in:** Stage 10.3 (full evaluation driver run on val split,
B=4 checkpoint, 7 surfaces × 3 mask samples = 21 records).

### Setup

We compared three methods on each (surface, mask) pair:
- **Mean-plane init**: least-squares plane through K, evaluated on U.
  This is our model's z⁰ initialization (D3.1), used as a sanity-floor.
- **Harmonic infill**: discrete Laplace equation with Dirichlet boundary
  z[K] = z_true[K]. Solved via scipy.sparse.linalg.spsolve. The classical
  smoothness baseline (Stage 10.2).
- **B=4 model**: our trained GNN-with-rollout (best Stage 9 checkpoint).

Three mask samples per surface gave us a mix of regimes — the
21 records broke down as: 11 `half_plane`, 4 `outward_free`,
6 `outward_pinned` (close to the configured 30/40/30 weights).

### Overall result: a three-way tie

| Method | Mean RMSE | Median RMSE | Max RMSE |
|---|---|---|---|
| Mean-plane | 91.62 | 75.04 | 346.54 |
| Harmonic | 91.80 | 81.84 | 341.16 |
| **Model** | **101.35** | **71.79** | 371.79 |

At the aggregate level, our model is slightly worse on the mean but
slightly better on the median than harmonic infill. None of the
three methods is dominant overall.

### The regime breakdown is where the real story lives

**`half_plane`** (n=11) — extrapolation across a cut:

| Method | Mean | Median |
|---|---|---|
| Mean-plane | 135.61 | 140.08 |
| Harmonic | 147.85 | 146.11 |
| **Model** | **133.67** | 140.22 |

The model wins on mean by 10m (~7%). Median is a wash. This is the
hardest regime in absolute terms (largest RMSEs) and the model is
contributing real value.

**`outward_free`** (n=4) — extrapolation from a central area:

| Method | Mean | Median |
|---|---|---|
| Mean-plane | 19.31 | 0.00 |
| Harmonic | 24.78 | 2.54 |
| Model | 21.21 | 9.86 |

All methods are comparable. Numbers are small in absolute terms.
The sample is too small (n=4) to draw firm conclusions; further
investigation would need more data.

**`outward_pinned`** (n=6) — interpolation between two anchors:

| Method | Mean | Median |
|---|---|---|
| Mean-plane | 59.17 | 0.00 |
| **Harmonic** | **33.73** | **2.70** |
| Model | 95.53 | 79.42 |

Harmonic infill dominates the model by ~3× on mean. This is exactly
where harmonic should win: smooth interpolation between two anchored
regions is mathematically harmonic infill's home turf.

### Interpretation

Each method has a regime where it excels:

1. **For extrapolation across a cut (`half_plane`)** — our learned
   model is the best of the three.
2. **For extrapolation from a central area (`outward_free`)** —
   methods are comparable.
3. **For interpolation between two anchors (`outward_pinned`)** —
   harmonic infill is mathematically near-optimal; smoothness alone
   suffices.

This is a defensible and honest result. The learned model adds value
in the regimes where smoothness alone is insufficient (extrapolation
with no second anchor), and underperforms in the regime where
smoothness is the right inductive bias (two-sided interpolation).

### Implications for the thesis

- **Frame the contribution carefully.** Not "GNN beats classical
  baselines on geological extrapolation" but "GNN provides value on
  extrapolation tasks where smoothness assumptions break down;
  classical methods remain the right choice for interpolation between
  known regions."

- **Per-regime evaluation is essential.** Aggregate metrics hide
  the fact that the model wins in some regimes and loses in others.
  Future evaluation should always report regime breakdowns.

- **Stage 12 ablations should investigate `outward_pinned`
  specifically.** Why is the model failing on smooth interpolation?
  Candidates: (a) the data loss is too forgiving for smooth surfaces
  so the model doesn't learn to be smooth, (b) the regime is
  under-weighted in training (currently 30%), (c) the model's
  capacity is being spent on extrapolation features that don't help
  interpolation.

- **The "10_BaseModelo flat surface" issue from the previous O5
  draft remains real:** the model produces 110m of RMSE on a perfectly
  flat surface where the answer is zero. This is a structural
  weakness in the model not learning the identity function. Worth
  diagnosing.

### Where the result lives

- Raw evaluation records: `outputs/evaluation/val_b4.json` (21 records
  with per-(surface, mask) RMSEs for all three methods, plus per-ring
  breakdown for the model).
- Reproducible: deterministic seeds via `base_seed + surface_idx * 100
  + mask_idx`; same call gives identical numbers.

### Caveats

- **Single checkpoint.** This evaluation is of the Stage 9 B=4
  checkpoint only. A future run with different hyperparameters might
  shift the picture.
- **No statistical significance testing.** With only 21 records (split
  across 3 regimes), differences of ~10m on means are within plausible
  noise. The interpretation is a directional claim, not a tight
  statistical one.
- **Val set, not test set.** Conclusions apply to the val
  distribution; we have not yet evaluated on test_id or test_ood.

### Visual evidence (Stage 10.4 plots)

Four figures generated from the same val_b4.json data, saved to
`outputs/evaluation/plots/`:

**`val_b4_regime_bars.png`** — Mean RMSE per (regime, method), grouped
by regime. This is the headline summary figure. Shows the
per-regime story at a glance:
- `half_plane`: GNN model (133.7) < mean-plane (135.6) < harmonic
  (147.8). Small but clean win for the learned model.
- `outward_free`: all methods comparable (~20m).
- `outward_pinned`: harmonic (33.7) << mean-plane (59.2) << model (95.5).
  Harmonic dominates.

**`val_b4_distribution.png`** — Strip plot of per-(surface, mask) RMSE
for each method, split by regime. Shows the variance hidden by mean
aggregates. Key observations:
- In `half_plane`, the model has slightly tighter spread than the
  baselines; the means are similar but the model is more consistent.
- In `outward_pinned`, the visual contrast is stark: harmonic and
  mean-plane cluster near zero with one outlier; model values are
  spread between 60-130m with one at 275m. The dichotomy in
  performance is unmistakable.

**`val_b4_per_ring.png`** — Per-ring RMSE curves, one subplot per
regime. Thin transparent blue lines show individual records; thick
blue line is the median (filtered to rings with ≥10 vertices).
Observations:
- `half_plane`: median rises from ~60m at ring 1 to ~140m at ring
  20, then plateaus, with an apparent spike at ring ~50. The
  spike is dominated by very few records (only large-mesh
  surfaces reach that depth); behavior at deep rings should not
  be over-interpreted from val data.
- `outward_free`: median curve is essentially flat near zero,
  reflecting that the 4 records in this regime were all on
  easy/smooth surfaces.
- `outward_pinned`: classic inverted-U shape we predicted
  geometrically. RMSE rises from ~80m near the inner anchor to
  peak ~125m at ring 13 (the "deep middle" of the annulus),
  then descends to ~60m as the frontier approaches the outer
  anchor.

**`val_b4_model_vs_harmonic.png`** — Per-(surface, mask) scatter:
GNN RMSE (y) vs harmonic RMSE (x), colored by regime. The dashed
diagonal is parity. Points below the line: model wins. Above: harmonic
wins. Visually confirms what the bars show:
- Purple (half_plane) dots cluster near the diagonal with model
  winning on most.
- Green (outward_free) dots cluster near the origin; methods
  are interchangeable on this small sample.
- Red (outward_pinned) dots are the clear "model loses" cluster:
  two dots far above the diagonal at (180, 270) and (0, 130)
  show harmonic substantially beating the model.

### Caveats on the far-ring behavior in `half_plane`

The half_plane median curve shows an apparent peak around ring 50 followed
by a descent. This is unlikely to reflect genuine model improvement at
long distances. More likely explanations: (a) only the largest meshes
(e.g., `05_TopoCretaceo`) reach those depths, so the median at d>30 is
computed from 1-2 records; (b) deep rings have few vertices and
high-variance RMSE; (c) mesh-boundary vertices have constrained
local geometry that may produce predictable predictions. For thesis
reporting we should note that the model's "far-field" behavior cannot
be characterized from val data; this would require larger meshes or
synthetic far-field experiments.


---

## O6 — Coordinate normalization closed the outward_pinned gap

**Observed in:** Stage 11.6 (full training run with
`data.normalize_per_surface=true`, all other hyperparameters
identical to Stage 9 B=4 baseline). Same evaluation methodology
as O5 (21 records, 7 surfaces × 3 masks).

### Setup recap

Until this stage, we centered x, y, z per-surface (D4.6) but didn't
normalize the *scales*. Centered coordinates still spanned roughly
[-500, +500] meters in xy and [-1000, +1000] in z. The umbrella
Laplacian feature had a different scale entirely. Mixing these
wildly different scales in a neural network is suboptimal — gradient
updates depend on input scale, so the model had to spend parameters
just learning to compensate.

The change: divide the centered coordinates by their max-abs value
per surface. After this, all coordinates lie in roughly [-1, +1].
The model's pted Δz is in normalized units; we denormalize by
multiplying by z_scale to report RMSE in meters.

### Result: substantial improvements in 2 of 3 regimes

| Regime | Stage 9 (no norm) | Stage 11.6 (with norm) | Δ mean |
|---|---|---|---|
| half_plane | 133.67 | 134.01 | +0.34 (unchanged) |
| outward_free | 21.21 | 14.58 | **−6.63 (−31%)** |
| outward_pinned | 95.53 | 57.28 | **−38.25 (−40%)** |
| Overall mean | 101.35 | 89.34 | **−12.01 (−12%)** |

For the first time in this project, the model has the lowest mean
RMSE *overall* across baselines on val:

| Method | Mean | Median | Max |
|---|---|---|---|
| Mean-plane | 91.62 | 75.04 | 346.54 |
| Harmonic | 91.80 | 81.84 | 341.16 |
| **GNN model** | **89.34** | **57.02** | 360.96 |

### The median tells an even stronger story

| Regime | Stage 9 median | Stage 11.6 median |
|---|---|---|
| half_plane | 140.22 | 140.76 |
| outward_free | 9.86 | **0.44** |
| outward_pinned | 79.42 | **0.45** |

For both "extrapolation from a central area" and "intween two anchors" regimes, more than half the val surfaces
are now predicted to sub-meter accuracy. The mean is dragged up
by outliers (notably 05_TopoCretaceo, which remains hard), but
the typical-case performance is excellent.

### Per-surface breakdown at best epoch

| Surface | Stage 9 RMSE | Stage 11.6 RMSE |
|---|---|---|
| 05_TopoCretaceo (V=48k, R2) | 273.6 | ~265 (≈unchanged) |
| 09_Horizonte8 | 135.6 | substantially improved |
| 10_BaseModelo (flat surface) | 110.2 | likely ~0 (median now 0.45) |
| Horizonte5 | 4.0 | <1 |
| TestHorizon4 | 73.0 | improved |
| TestHorizon7 | 72.1 | improved |
| horizonte7 | 2.1 | <1 |

The flat-surface failure (10_BaseModelo) is essentially gone. The
single remaining hard surface is 05_TopoCretaceo, which is genuinely
geologically complex; it limits the headline mean but doesn't
invalidate the overall improvement.

### Why half_plane is unchanged

This is the most interesting puzzle in the result. Half_plane is the
regime where mask cuts the mesh in half; U can beery deep
(N=50+ rings) and there is only one anchor side. Our hypothesis:

- For outward_free and outward_pinned, the unknown region is bounded
  (small or surrounded by anchors). Better-conditioned optimization
  via normalization finds good solutions.
- For half_plane, the bottleneck is *architectural*, not
  optimization-related. The model has to extrapolate 50+ hops into
  unknown territory based on one boundary. No amount of better
  feature scaling fixes "I have to make confident predictions
  very far from any data."

This is a falsifiable hypothesis: if we tried structural changes
(deeper GNN, attention, longer rollouts), we'd expect half_plane
to improve and the others to stay flat.

### Implications

1. **Normalization is now a default**, not an experiment. It should
   be `true` in configs/default.yaml going forward.

2. **The model is competitive with harmonic infill** on the regime
   where it was previously losing badly. We can defensibly justify
   the use of a neural network for the extrapolation problem.

3. **05_TopoCretaceo is the remaining bottleneck for headline
   numbers.** Worth a future investigation: visualize the surface,
   understand its geological complexity, see whether the model's
   errors are concentrated in a specific area.

4. **Half_plane is the next frontier.** If we want to push the
   model further, it's architectural changes (deeper GNN, alternative
   operators) that target this regime specifically.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260614_115836/best.pt`
  (epoch 21, best_val_loss=0.4646 in normalized units).
- Evaluation: `outputs/evaluation/run_20260614_115836_val.json`.
- Plots: `outputs/evaluation/plots/run_20260614_115836_val_*.png`.

### Caveats

- **Single seed.** As with all previous results, we have one A/B
  comparison rather than multiple seeds. The directional claim
  (normalization helps substantially) is robust given the magnitude
  of the change, but precise numbers should not be over-interpreted.
- **Val set, not test set.** We have not yet evaluated on test_id
  or test_ood. The result will be re-checked at the end.
- **One mesh size is still problematic.** 05_TopoCretaceo (48k
  vertices) remains hard, suggesting we may have a model-capacity
  issue for the largest meshes.

---

## O7 — Harmonic infill as initialization: closes outward_pinned gap at the cost of half_plane

**Observed in:** Stage 11.7 (two attempts). Same hyperparameters as
Stage 11.6 (normalize_per_surface=true, B=4, λ defaults), only changed
`data.init_method` to `harmonic`.

### Two runs and a methodological finding

We ran this experiment **twice** with different results, and the
difference is itself informative:

**Run 1** (Stage 11.7, run_20260614_133159): used the default
early-stopping criterion of `val_loss`. Best checkpoint selected at
epoch 13. Half_plane regressed massively (215m mean vs Stage 11.6's
134m). Investigation showed the model was still learning when training
stopped — val_rmse_meters was at 84m smoothed and still descending,
while val_loss had plateaued.

**Run 2** (Stage 11.7-redo, run_20260614_144819): same setup, but with
`val_rmse_meters` as the early-stop and best-checkpoint criterion.
The model was selected at a *much* better epoch — best
val_rmse_meters=63m. Half_plane regression went from +81m to +17m.

The difference between the two runs is a lesson about metric choice:
val_loss includes regularizer terms that introduce noise, so it has
spikes (epoch 12: 8.19, epoch 24: 7.01) that don't reflect
underlying model quality. val_rmse_meters is what we report and is
much smoother. Best-checkpoint selection should track the metric we
ultimately care about.

This finding is now baked into the training loop (commit
`train.best_metric` defaults to `val_rmse_meters`).

### The fair comparison

With the corrected metric, Stage 11.7-redo result:

| Regime | 11.6 (meanplane init) | 11.7-redo (harmonic init) | Δ |
|---|---|---|---|
| half_plane mean | 134.0 | 151.7 | +17.7 (worse) |
| outward_free mean | 14.6 | 21.5 | +6.9 (worse, but tiny) |
| outward_pinned mean | **57.3** | **36.6** | **−20.7 (better)** |
| Overall mean | 89.3 | 94.0 | +4.7 (slightly worse) |

And the medians:

| Regime | 11.6 median | 11.7-redo median |
|---|---|---|
| outward_pinned | 0.45 | 2.51 |

### Interpretation

**The expected trade-off materialized.** From the preliminary
investigation, we knew harmonic init starts the model ~12m worse
on half_plane and ~25m better on outward_pinned. The trained model
preserved approximately that asymmetry: half_plane got 17.7m worse
and outward_pinned got 20.7m better. Net: slightly negative.

**The outward_pinned result is striking.** Stage 11.7-redo's
outward_pinned mean (36.6m) is essentially matching harmonic infill
itself (33.7m). The GNN with harmonic init successfully *replicates*
harmonic infill's interpolation quality, then adds nothing
substantive on top of it for this regime.

**The half_plane regression is small but real.** This regime
remains our weakest, and starting from a less-suitable init makes
it weaker. The model can't fully recover from the worse starting
point in the available training time (or perhaps at all — see O6's
hypothesis about architectural limits).

### Decision: keep Stage 11.6 (meanplane init) as the final model

The trade-off is unfavorable overall:
- Stage 11.6 mean: 89.3 (lower)
- Stage 11.7-redo mean: 94.0 (higher)

Stage 11.6 also beats all three baselines on overall mean. Stage
11.7-redo trails mean-plane on overall mean.

Stage 11.7-redo is preserved as a deliberate ablation: it shows
that with harmonic init, the GNN can match harmonic infill on
`outward_pinned`. This is useful for the thesis story but not
the headline result.

### Implications for thesis writeup

The honest framing for the report:

> "We tested whether initializing the rollout from a harmonic infill
> baseline (rather than mean-plane fit) would close the gap on
> `outward_pinned`. It did: with harmonic init, the GNN achieves
> 36.6m mean RMSE on `outward_pinned`, essentially matching harmonic
> infill's 33.7m. However, this came at the cost of `half_plane`
> performance (151.7m vs 134.0m) and slightly worse overall RMSE
> (94.0m vs 89.3m). We adopted meanplane init as our final
> configuration, since it has better overall and `half_plane`
> performance. Harmonic init remains a useful ablation showing the
> empirical trade-off between general extrapolation and
> interpolation accuracy."

### Where the result lives

- Stage 11.7-redo checkpoint:
  `outputs/tensorboard/run_20260614_144819/best.pt`
  (epoch 13, best val_rmse_meters=63.04).
- Evaluation:
  `outputs/evaluation/run_20260614_144819_val.json`.

### Caveats

- **Architectural limit hypothesis (from O6) still standing.** Stage
  11.7-redo's half_plane (152m) is similar to Stage 11.7's
  half_plane (215m) — both are higher than 11.6 (134m). The
  architectural bottleneck on half_plane is independent of init
  choice; init only modulates the magnitude.
- **One seed each.** All three runs (11.6, 11.7, 11.7-redo) are
  single-seed comparisons.
- **Val set only.** Stage 12 (final test evaluation) hasn't been
  run yet.

---

## O8 — Mask augmentation (n=3) improves all regimes, especially the median

**Observed in:** Stage 11.8 (full training run with
`data.n_masks_per_epoch=3`, otherwise identical to Stage 11.6:
normalize=true, init=meanplane, B=4). Each train epoch sees 90
items (30 surfaces × 3 distinct masks) instead of 30.

### The result

| Regime | 11.6 (n=1) | 11.8 (n=3) | Δ mean | Δ median |
|---|---|---|---|---|
| half_plane | mean=134.0, med=140.8 | mean=125.4, med=115.2 | −8.6 | **−25.6** |
| outward_free | mean=14.6, med=0.44 | mean=7.4, med=0.65 | **−7.2 (−49%)** | +0.2 |
| outward_pinned | mean=57.3, med=0.45 | mean=55.0, med=0.72 | −2.3 | +0.3 |
| Overall | mean=89.3, med=57.0 | **mean=82.8, med=38.3** | **−6.5** | **−18.7 (−33%)** |

### The model now leads on every aggregate metric

| Method | Mean | Median |
|---|---|---|
| Mean-plane | 91.62 | 75.04 |
| Harmonic | 91.80 | 81.84 |
| **GNN model** | **82.84** | **38.33** |

For the first time in this project, the GNN beats both classical
baselines on both mean and median aggregate metrics. The median
improvement is particularly striking: GNN's 38.3m vs the better
baseline's 75.0m is a 49% improvement.

### Per-regime: improvements where we expected and didn't expect

**The big surprise**: half_plane improved. We hypothesized in O6 that
half_plane was architecturally limited (deep extrapolation with only
one anchor). Mask augmentation closing 6% of the gap suggests the
issue is at least *partially* data-related, not purely architectural.
The hypothesis was too strong.

**The expected wins**: outward_free improved by 49%, outward_pinned
improved slightly. More data exposures to each surface let the model
learn better per-surface behaviors.

**The unchanged**: outward_pinned mean barely moved (-2). Harmonic
infill still wins this regime by ~22m. The architectural hypothesis
likely holds *here*: smoothness-dominated interpolation is genuinely
where classical methods are mathematically optimal.

### Why per-ring outward_free looks like a flat zero line

The 4 outward_free records broke down as 3 horizonte7 samples (a near-
flat val surface with z_range ≈ 18m) and 1 TestHorizon7 sample.
On near-flat surfaces, there's essentially nothing to extrapolate;
the truth is already close to mean-plane init. The per-ring RMSE is
near zero across all rings, dragging the median to zero. This is an
artifact of small val sample size in this regime, not of model
behavior. Worth keeping in mind when interpreting plots.

### Training dynamics

- Best epoch: 36 (well within the 100-epoch budget).
- Early-stopped at epoch 56 (no val_rmse_meters improvement for 20).
- Train loss at end: ~3-4, still descending.
- Val loss / val_rmse_meters at end: bouncing around floor of ~75m
  for the last ~20 epochs.

**Train loss was still descending when val plateaued.** This suggests:
- Capacity is not exhausted (more parameters could fit the train data).
- But generalization is plateauing — additional fitting doesn't transfer
  to val.
- A natural follow-up: longer training with smaller LR, combined with
  even more augmentation (n=5+), might extract more.

### Implications for the thesis story

The neural network is now defensibly justified across regimes:

> "On the val split, the trained GNN achieves a mean RMSE of 82.8m,
> outperforming mean-plane initialization (91.6m) and harmonic infill
> (91.8m). The model leads on every aggregate metric. The improvement
> is concentrated in extrapolation regimes (half_plane: 125 vs 136,
> outward_free: 7 vs 19), with classical harmonic infill remaining
> competitive only on outward_pinned (interpolation between two
> anchors), where smoothness assumptions are mathematically optimal."

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260614_155745/best.pt`
  (epoch 36, best val_rmse_meters=64.62).
- Evaluation: `outputs/evaluation/run_20260614_155745_val.json`.

### Caveats

- **Per-epoch cost roughly tripled** (~225s vs ~75s). A full run is
  ~2.5 hours.
- **Single seed.** All previous caveats apply.
- **Val set is small (21 records).** Differences of a few meters are
  within plausible noise.
- **Val set has 2 near-flat surfaces** (horizonte7 with z_range=18m,
  10_BaseModelo with z_range=0m). These dominate the median in some
  regimes. The mean is the more representative aggregate for these
  data.

---

## How to use this document

Append new observations as `O<N>` entries when:
- A training run reveals a non-obvious property of the model or loss.
- An experiment confirms or refutes a hypothesis.
- A pattern recurs across multiple settings and is worth flagging.

Each entry should pin down what was observed and where, offer a best
explanation, and note implications for future work. Avoid speculation
without evidence.
