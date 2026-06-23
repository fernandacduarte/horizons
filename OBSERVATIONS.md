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

## Summary — the three-phase arc

The project set out to learn a GNN that extrapolates geological horizons through
an iterative rollout, and to ask whether it could beat the classical baseline,
harmonic infill. The answer turned out to hinge entirely on **rollout depth**,
and the work unfolded in three phases.

**Phase 1 — small data (V ≤ 48k): a tie, and an exhaustive null search.** On the
original 30-surface dataset the learned model matched harmonic infill but never
clearly beat it. Every lever we pulled — coordinate normalization (O6),
initialization (O7), mask augmentation (O8/O9), regime weighting (O10),
regularizer strength, network width (O12), depth (O13), and even a different
message-passing operator (EdgeConv, O18) — moved validation error by less than
the measurement noise (O16). The model was neither capacity- nor
operator-limited; it sat on a plateau, tied with the classical method. Plain
SAGE (Stage 11.8) was the best configuration.

**Phase 2 — the full dataset, and a clear failure with a clear cause.** Gradient
checkpointing (D12.2) lifted the memory wall that had excluded the large surfaces
(110k–455k vertices), so the study restarted on a magnitude-balanced split
(D12.3) that included them. Now the learned model clearly *lost* to harmonic
(O19) — but breaking the error down per surface showed exactly why. The GNN's
advantage holds at shallow rollout depth (small surfaces, true extrapolation)
and **inverts as the rollout deepens**, with a crossover around N ≈ 25–50. The
reason is structural: filling a large surface needs an N ≈ 150-step rollout, and
per-step error accumulates over that depth, while harmonic solves the whole field
in one global step with no depth penalty.

**Phase 2/3 — every fix to the rollout fails, which sharpens the diagnosis.** If
depth is the problem, what cures it? Nothing that tweaks the rollout did: more
capacity (O20), a harmonic starting point (O21), a heavier per-step penalty
(O22), and freezing already-filled rings (O23) all made the deep surfaces
*worse*. O23 was the most informative — freezing hurt, which means the rollout's
repeated updates are beneficial *refinement*, not harmful drift. The iterative
process does the right thing; it simply cannot converge well enough over ~150
steps. Every model-side intervention left the deepest surface worse than the
plain baseline.

**Phase 3 — replace the rollout, and the learned model finally wins (O24).** The
diagnosis pointed at a single fix: stop marching. The hybrid hands the long-range
reach to harmonic — a *global* solve that fills the entire field in one shot,
with no depth penalty — and restricts the GNN to a small, fixed number of local
refinement passes (no depth to accumulate). This is the first approach to **beat
harmonic infill**: it wins overall, fixes the deepest surface (the GNN improves
even harmonic's 443k field), and *strengthens* the shallow-surface extrapolation
it was always good at.

**In one line:** the learned operator's value is real but depth-limited — hand
the propagation to a classical global solver and keep the network to local
refinement, and it beats both the classical baseline and its own rollout.

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

## O9 — n=5 augmentation with smaller LR: roughly tied with n=3, much slower

**Observed in:** Stage 11.9 (full training run with `n_masks_per_epoch=5`,
`lr=5e-4`, `patience=40`, `n_epochs=200`). Hypothesis: more
augmentation + finer LR + more patience would push further than
n=3.

### Result: essentially flat

| Regime | 11.8 (n=3) | 11.9 (n=5, lr=5e-4) | Δ |
|---|---|---|---|
| half_plane mean | 125.4 | 128.3 | +2.9 |
| outward_free mean | 7.4 | 9.9 | +2.5 |
| outward_pinned mean | 55.0 | 54.9 | −0.1 |
| Overall mean | **82.8** | 84.8 | +2.0 |
| Overall median | **38.3** | 38.4 | +0.1 |

Stage 11.9 came in roughly tied with Stage 11.8, slightly worse on
mean (+2m) and identical on median. The early-stop fired at epoch 65
with best at epoch 25 (well within the patience budget). Train loss
was still slowly descending while val plateaued around 70-75m —
exactly t training pattern as 11.8, just stretched across
more epochs.

### What we learn

**The data-diversity ceiling is at or near n=3.** Doubling
augmentation (n=3 → n=5) didn't help. Tripling training time
(2.5h → 9.didn't help. Halving the LR didn't help.

This is informative: **the remaining gains are not in "more masks
per epoch" or "longer training."** They're in something else —
likely:
- Different data composition (e.g., re-weighting regimes for
  deployment focus, larger meshes brought back).
- Architectural changes (deeper GNN, alternative orators).
- Or we've genuinely reached the model's representational ceiling
  for this dataset.

### Implications

- **Stage 11.8's hyperparameters (n=3, lr=1e-3, patience=20) are
  near-optimal** for the current training data setup. We should
  stop sweeping these.
- **Future improvements need a different axis.** This is what
  motivates Stage 11.10's regime re-weighting.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260614_191007/best.pt`
  (epoch 25, best val_rmse_meters=62.72).
- Evaluation: `outputs/evaluation/run_20260614_191007_val.json`.

### Caveats

- **Compute cost was a real factor in deciding to stop.** Stage 11.9
  ran for ~9.7 hours. Each additional similar experiment doubles
  the wall-clock investment for similar marginal returns. Time is
  finite for thesis work.
- **n=4 was not tested.** The jump n=3 → n=5 might have skipped a
  point that would have helped. We're inferring monotonic
  diminishing returns, which isn't guaranteed.

---

## O10 — Regime re-weighting toward `outward_free` did not help (revised with n=10 eval)

**Observed in:** Stage 11.10 (full training run with `mask.regime_weights`
changed from 30/40/30 to 20/60/20). Re-evaluated with `--n-masks 10`
(70 records: 7 surfaces × 10 masks) after the initial n=3 evaluation
proved statistically unreliable.

### Motivation recap

The dominant inference scenario is `outward_free` (extrapolate from a
central region to a bounding box, no boundary anchors). We re-weighted
training masks 20/60/20 to give the model 3× more outward_free exposure,
expecting outward_free RMSE to drop substantially.

### Result with reliable evaluation (n=10 masks per surface)

| Regime | 11.8 (30/40/30 train, n=10 eval) | 11.10 (20/60/20 train, n=10 eval) | Δ |
|---|---|---|---|
| half_plane mean | 71.4 | 77.6 | +6.2 (worse) |
| outward_free mean | 76.9 | 80.8 | +3.9 (worse) |
| outward_pinned mean | 73.5 | 81.0 | +7.5 (worse) |
| Overall mean | **73.7** | 79.4 | +5.7 (worse) |

Stage 11.10 is worse than Stage 11.8 on every regime, including the
targeted outward_free regime. The directional conclusion of the
original O10 (with n=3 evaluation) was correct, but the magnitudes
were misleading — both runs looked worse on the small sample.

### What we learn

The hypothesis "the bottleneck on outward_free is insufficient
training exposure" is **false** in our setup. Re-weighting helps not
just outward_free but the other regimes too — and even outward_free
itself didn't gain.

Most plausible explanation: **the model benefits from regime diversity
during training**, even when only one regime matters at deployment.
Training mostly on outward_free (the 60% case in 11.10) may have
caused the model to overfit to specific outward_free mask geometries
rather than learning the generalizable extrapolation patterns that
all regimes share. Half_plane and outward_pinned training were
providing useful supervisory signal that we accidentally suppressed.

### Decision

**Stage 11.8 (30/40/30 regime weights) is the best model.** This is
the configuration to use going forward. Stage 11.10 is preserved as
a meaningful ablation showing that regime specialization doesn't
help, even on the specialized regime.

### Implications going forward

- **Regime weights stay at 30/40/30.**
- **Stop pursuing regime-weight tuning as a lever** — it doesn't help
  in any direction we've explored.
- **The remaining gains require structural changes** (architecture,
  larger meshes brought back, etc.) — see DECISIONS.md Tier 3
  candidates.
- **Validate on test set.** Stage 11.8 is the final hyperparameter
  configuration; the next step is to confirm its val performance
  generalizes to held-out test_id and test_ood splits.

### Where the result lives

- Stage 11.10 checkpoint: `outputs/tensorboard/run_20260615_092535/best.pt`
  (epoch 19, best val_rmse_meters=65.78).
- n=10 evaluation: `outputs/evaluation/run_20260615_092535_val.json`.
---

## O11 — Per-regime statistics need ≥10 masks per surface for reliability

**Observed in:** Stage 11.8 evaluation re-run with --n-masks 10 (versus
the original --n-masks 3 default).

### The finding

Stage 11.8's "best" per-regime numbers shifted substantially when we
moved from 3 masks per surface (21 records total) to 10 masks per
surface (70 records total):

| Regime | n=3 mean | n=10 mean | shift |
|---|---|---|---|
| half_plane | 125.4 | 71.4 | −54 (a lot) |
| outward_free | 7.4 | 76.9 | **+70 (huge)** |
| outward_pinned | 55.0 | 73.5 | +18 |
| Overall | 82.8 | 73.7 | −9 |

These are the same model evaluated on the same val surfaces. Only
the sampling density changed.

### What happened with outward_free

With n=3, the 21 evaluation records broke down as 11/4/6 per regime.
Three of the four outward_free records were on the same near-flat
val surface (horizonte7, z_range=18m), which has very little to
extrapolate. The mean was dragged down to 7.4m by these easy cases.

With n=10, the 70 records break down as 31/23/16 per regime. The
23 outward_free records cover all val surfaces multiple times,
giving a representative mix of easy and hard cases. The mean
rises to 76.9m — the honest number.

### What happened with half_plane

The reverse case. With n=3, half_plane happened to draw mostly
hard records (mean 125.4). With n=10, the mix is more
representative and the mean is more moderate (71.4).

### General lesson

When the val set has only 7 surfaces and you sample at random with
regime probabilities 30/40/30, **per-regime sample sizes are too
small for reliable statistics**. A handful of unlucky draws can
shift a mean by 10× in either direction.

**The fix is simple**: bump n_masks_per_surface to 10. The eval is
fast (no gradient computation), so the extra compute is minor (a
few minutes per evaluation).

This is now the default in `scripts/eval_run.py` and is documented
as D11.10.1 in DECISIONS.md.

### Implications for the project

- **All cross-experiment comparisons must use the same n_masks**.
  Old eval JSON files (with n=3) should be regenerated with n=10
  before being used for comparisons.
- **The n=3 numbers in OBSERVATIONS.md O6, O7, O8 were misleadingly
  precise** — qualitative directions were correct, but absolute
  values shifted with the bigger sample. The headline numbers in
  the thesis writeup should come from n=10 evaluations.
- **The val set is fundamentally small** (7 surfaces). Even with
  n=10 masks, we have only 7 *underlying surfaces* contributing to
  each metric. Per-surface variance limits how precise any
  aggregate can be. This is a real limitation of the dataset, not
  fixable by sampling more masks.

### Where the result lives

- Stage 11.8 n=10 evaluation:
  `outputs/evaluation/run_20260614_155745_val.json`.
- Stage 11.10 n=10 evaluation:
  `outputs/evaluation/run_20260615_092535_val.json`.

---

## O12 — Larger model (hidden_dim=128) does not help: we are not capacity-limited

**Observed in:** Stage 11.11 (full training run with
`model.hidden_dim=128`, otherwise identical to Stage 11.8: n=3,
lr=1e-3, normalize=true, init=meanplane, 30/40/30 regime weights).
Parameter count grew from ~21k (hidden_dim=64) to ~84k
(hidden_dim=128).

### Result: essentially identical to Stage 11.8

| Regime | 11.8 (hidden=64) | 11.11 (hidden=128) | Δ |
|---|---|---|---|
| half_plane mean | 71.4 | 69.3 | −2.1 (slightly better) |
| outward_free mean | 76.9 | 77.2 | +0.3 (essentially same) |
| outward_pinned mean | 73.5 | 73.8 | +0.3 (essentially same) |
| Overall mean | **73.7** | **73.0** | −0.7 (<1% improvement) |

All differences are **at or below the noise floor** of the val
evaluation (7 underlying surfaces, per-surface RMSE varies by
100m+). A 0.7m improvement in overall mean is not meaningful with
this sample size.

### Training-time vs evaluation-time numbers diverged

The training-time best val_rmse_meters was 60.3 (Stage 11.11) vs
64.6 (Stage 11.8) — a 4.3m apparent improvement. But the full
evaluation with n=10 masks per surface showed essentially no
difference (73.0 vs 73.7). This is consistent with the n_masks
methodological lesson from O11: small samples can give misleading
precision.

### The conclusion

**This is informative. It strongly suggests we are not
capacity-limited. The bottleneck is data or architectural
inductive biases, not model size.**

Quadrupling the parameter count from 21k to 84k did not move the
needle on val. If the model had been capacity-limited, this would
have improved things; it didn't. Whatever's holding us back is
not "the model is too small."

### What the remaining bottleneck might be

Plausible explanations, in rough order of likelihood:

1. **Data diversity.** We train on 30 surfaces (10 V>50k surfaces
   set aside in Stage 4). Bringing them back is a 33% data
   increase — more diverse geological structures, not just more
   masks of the same ones. This is the next experiment to try if
   we want to push further.

2. **Architectural inductive bias.** SAGEConv with umbrella
   Laplacian features may not be the optimal operator for this
   problem. Alternatives (EdgeConv, GAT, cotangent Laplacian) have
   different inductive biases that might fit the geological
   surface domain better.

3. **Information bottleneck in the rollout.** Each iteration
   propagates information one ring outward. After N=80 iterations,
   the signal from K must traverse 80 rings. The signal may decay
   or get noisy along the way, regardless of model capacity.

4. **Inherent ceiling of the dataset.** 30 surfaces with diverse
   geological provenance is a small dataset for learning a
   general-purpose extrapolation operator. We may have genuinely
   reached the model's representational ceiling for this dataset.

### Implications going forward

- **Stop sweeping hyperparameters.** Width, depth, LR, augmentation,
  regime weights, init method — all explored or considered. Further
  exploration in this space is unlikely to yield meaningful gains.
- **Architectural changes are the most promising remaining lever.**
  But each is a significant code change with non-trivial risk.
- **Final test evaluation is the natural next step**, with or
  without one more architectural experiment.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260615_145241/best.pt`
  (epoch 65, best val_rmse_meters=60.28).
- Evaluation: `outputs/evaluation/run_20260615_145241_val.json`.

### Caveats

- **Single seed.** As always, n=1 seed comparisons are limited.
- **Training was longer** (85 epochs vs 56 for Stage 11.8) due to
  the model having more parameters to fit. This added compute cost
  is real even though the model converged to a similar place.
- **The "not capacity-limited" conclusion is specific to this
  configuration.** A deeper model (n_layers=3) might still help —
  it's a different axis (depth vs width).

---

## O13 — Deeper GNN (n_layers=3) hurts slightly: not depth-limited either

**Observed in:** Stage 11.12 (full training run with `model.n_layers=3`,
otherwise identical to Stage 11.8). Parameter count grew from 21k
(2 layers) to 30k (3 layers).

### Result: small but consistent regression

| Regime | 11.8 (n_layers=2) | 11.12 (n_layers=3) | Δ |
|---|---|---|---|
| half_plane mean | 71.4 | 75.1 | +3.7 (worse) |
| outward_free mean | 76.9 | 81.4 | +4.4 (worse) |
| outward_pinned mean | 73.5 | 75.8 | +2.3 (worse) |
| Overall mean | **73.7** | 77.3 | +3.6 (worse) |

All regimes got slightly worse. The model trained normally
(converged at epoch 33, early-stop at 53), but the resulting model
generalizes slightly worse on val.

### Likely explanation: over-smoothing

Adding message-passing layers means each vertex aggregates
information from increasingly distant neighbors per rollout
iteration. With small graphs and 3 layers, this can cause
**over-smoothing** — vertex features become too similar to each
other, losing the local distinguishing information the model
needs to predict precise z values.

This is a known issue with deeper GNNs in the literature
(Chen et al. 2020, "Measuring and relieving the over-smoothing
problem"; Oono & Suzuki 2020, "Graph neural networks exponentially
lose expressive power"). Our result is consistent with these
findings.

### What this tells us, combined with O12

| Lever | Outcome |
|---|---|
| **Wider** (O12) | Tiny improvement (within noise) |
| **Deeper** (O13) | Small regression |
| **More augmentation** (O9) | No improvement past n=3 |
| **Regime re-weighting** (O10) | Slight regression |
| **Longer training, smaller LR** (O9) | No improvement |

A pattern emerges: **none of the model-side adjustments move the
needle.** Width, depth, regularization, training duration,
learning rate, augmentation, regime emphasis — all tried. None
produces a meaningful improvement.

This consistently points to **data being the bottleneck, not the
model.** We've fully explored the model side at this dataset
size. The remaining lever is data: bringing back the V>50k
surfaces (set aside in Stage 4) would add ~33% more training
data with genuinely different geological structure.

### Decision

**Stage 11.8 remains the best model.** The current architecture
(hidden_dim=64, n_layers=2) appears near-optimal for this
problem at this dataset size. Future improvements need data, not
model changes.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260616_072423/best.pt`
  (epoch 33, best val_rmse_meters=65.41).
- Evaluation: `outputs/evaluation/run_20260616_072423_val.json`.

### Caveats

- **Single seed**. As always.
- **n_layers=4 not tested.** The depth-vs-quality tradeoff has a
  curve; we tested {2, 3} and found 2 wins. 4 or more would likely
  hurt more, but we haven't verified.

---

## O14 — Adding one large surface (110k V) to training: essentially flat

**Observed in:** Stage 11.13 (full training run with one additional
training surface: `04BaseOligoMioceno`, V=110,240, brought back from
the V>50k exclusion list set during Stage 4). Otherwise identical to
Stage 11.8: n=3 augmentation, lr=1e-3, normalize=true, init=meanplane,
hidden_dim=64, n_layers=2.

### Motivation recap

After O12 (width didn't help) and O13 (depth hurt), we hypothesized the
remaining bottleneck was data diversity, not model capacity. The
original Stage 4 decision (D4.2) excluded V>50k surfaces as
"computationally unwieldy." With our trajectory-based rollout loss,
"computationally unwieldy" turned out to mean **OOM on 16GB RAM** when
trying to fit the largest surfaces (610k+ vertices).

We were able to fit one new surface (110k vertices) within our memory
budget and tested whether even one extra surface moves the needle.

### Result: essentially flat

| Regime | 11.8 (30 train surfaces) | 11.13 (31 train surfaces) | Δ |
|---|---|---|---|
| half_plane mean | 71.4 | 70.9 | −0.5 (slightly better) |
| outward_free mean | 76.9 | 77.6 | +0.7 (slightly worse) |
| outward_pinned mean | 73.5 | 72.1 | −1.4 (slightly better) |
| Overall mean | 73.7 | 73.4 | −0.3 (essentially same) |

All differences are within the noise floor of the val set (7 surfaces,
per-surface RMSE varies by 100m+). Adding one extra training surface
did not produce a measurable improvement.

### Note on val set composition

The val set remained at 7 surfaces (no V>50k additions). 70 records
total at --n-masks 10. Direct comparison with Stage 11.8's evaluation
is apples-to-apples because the *test* set is identical; only the
training data differs by one surface.

### Two interpretations

1. **One surface isn't enough.** The hypothesis "data diversity helps"
   may need substantially more new data, not just one new surface.
2. **More data won't help much regardless.** The training-vs-val gap
   may reflect a fundamental ceiling related to model architecture
   or task complexity, not data volume.

The conclusive experiment is to add all 10 V>50k surfaces (full size
range up to 673k vertices). This requires more RAM than our dev
machine (16GB) provides. The experiment was set up to run on a
separate Windows machine with 128GB RAM and i9-14900 CPU. **The
Windows experiment is the deciding test of the data-diversity
hypothesis** — to be documented as O15 when complete.

### Training dynamics

- Best epoch: 38 (well within budget).
- Early-stop at epoch 58 (no val_rmse_meters improvement for 20).
- Best val_rmse_meters: 64.98.
- Train loss reached ~4 at end, still slowly descending.
- The 110k surface appeared in step logs with N=69-80 rollout depth
  — substantially larger than typical training surfaces (N=5-50).

### Per-epoch cost

~270s per epoch on the Mac CPU (~225s for Stage 11.8). The added
surface increased per-epoch cost by only ~20% despite being 2.3× the
size of our previous largest training surface (48k vertices). This
suggests per-epoch cost is dominated by the **count** of (surface,
mask) items, not the **size** of individual surfaces — at least in
this range.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260616_164458/best.pt`
  (epoch 38, best val_rmse_meters=64.98).
- Evaluation: `outputs/evaluation/run_20260616_164458_val.json`.

### Caveats

- **Single seed.** As always.
- **One surface is a small experiment.** The interpretation gap (1 vs
  2 above) cannot be resolved with this data alone.
- **Memory limit was the constraint, not science.** We would have
  preferred to test the full V>50k set on this same hardware to
  isolate the data-diversity question more cleanly, but OOM made that
  infeasible.

---

## O15 — Bringing back V>50k surfaces: memory constraints and partial result

**Observed in:** Multiple experiments across two machines, attempting to
include the 10 large (V>50k) surfaces excluded during Stage 4
preprocessing. The motivation came from O12/O13/O14, which collectively
suggested data diversity (not model capacity) might be the remaining
bottleneck.

### Memory constraints we encountered

**On the macOS dev machine (16 GB RAM, CPU only):**
- All 10 V>50k surfaces caused immediate OOM during training (process
  killed by the OS).
- Only the smallest (04BaseOligoMioceno, V=110,240) fit in memory.
- Stage 11.13 documented the result with this one surface: essentially
  flat improvement (~0.3m on overall mean).

**On the GPU machine (NVIDIA RTX 6000 Ada, 48 GB VRAM, CUDA 13):**
- With `accum_steps=4` (Stage 11.8 default), training OOM'd at step 0
  even withhe 2 smallest surfaces dropped (610k, 443k still in train).
- With `accum_steps=1`, training got to step ~20 before OOMing, with
  the 412k surface still in train.
- After removing 412k from train, training OOM'd again around step 21
  — this time on a small surface, suggesting GPU memory fragmentation.
- After also removing 230k from train, and setting
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, training finally
  ran to completion.

**Root cause:** Our rollout loss keeps all intermediate states in the
backward computation graph. For an N-step rollout on a V-vertex
surface, peak memory scales roughly as O(V × N × hidden_dim). For the
largest surfaces, V=610k and N≈130 push peak memory beyond 48 GB.

This is **the same constraint that originally justified D4.2 (drop
V>50k surfaces)** — we have empirically confirmed that decision was
correct, and identified its true cause (trajectory-based backward graph
memory), not just compute speed.

### Final split for Stage 11.14 (GPU machine)

After driven removals, the dataset used for Stage 11.14 was:

| Split | Total | New V>50k included |
|---|---|---|
| train | 32 | 07TopoCenomaniano (165k), 16TopoAndarAlagoas (192k) |
| val | 8 | 04BaseOligoMioceno (110k) |
| test_id | 6 | 15TopoSal (195k) |
| test_ood | 5 | none |

Surfaces dropped from the original plan due to OOM:
- 03TopoOligoMioceno (230k) — was in train
- 06TopoCretaceoSuperior (412k) — was in train
- 02TopoMioceno (443k) — was in train
- 01FundoMar (455k) — was in val
- 18TopoEmbasamento (610k) — was in train
- 17TopoAndarJiquia (673k) — was in test_id

### Stage 11.14 result (with the reduced V>50k addition)

Trained with `accum_steps=1` (forced by GPU memory) on the new train
set. Evaluated on the new 8-surface val with `--n-masks 10` (80
records).

For an apples-to-apples comparison, we also re-evaluated Stage 11.8's
checkpoint on the same new 8-surface val:

| Method | Overall | half_plane | outward_free | outward_pinned |
|---|---|---|---|---|
| Mean-plane | 90.4 | 99.8 | 95.2Harmonic | **73.1** | 94.3 | 77.9 | **28.3** |
| GNN Stage 11.8 (orig val) | 76.6 | 92.2 | 73.2 | 53.6 |
| GNN Stage 11.14 (+165k +192k in train) | 81.2 | 96.3 | 78.3 | 58.0 |

### What we learn

**Three findings, in order of confidence:**

1. **Adding 2 mid-sized surfaces (165k, 192k) to training hurt the
   model.** Every regime got worse by 4-5m versus Stage 11.8 on the
   identical val set. Net: +4.6m overall mean RMSE. This is consistent
   with O10 (regime re-weighting hurt) and O13 (deeper hurt). The
   pattern: changes from Stage 11.8's "found" configuration consistently
   regress.

2. **The new val (with one 110k surface added) is intrinsically harder.**
   Stage 11.8 went from 73.7m mean RMSE on the original 7-surface val
   to 76.6m on the new 8-surface val. The 110k surface contributes
   high-magnitude errors that drag the mean up. This is informative for
   the writeup: smaller, simpler surfaces give optimistically-low
   numbers; larger ones reveal the model's actual difficulty.

3. **Harmonic infill now leads on overall mean.** On the new val,
   harmonic infill achieves 73.1m vs our model's 76.6m (Stage 11.8).
   This is partly because the new val surface (110k) is geologically
   smooth and well-suited to harmonic interpolation. The qualitative
   regime story is unchanged — GNN still wins on `outward_free` (73.2
   vs 77.9) and is close on `half_plane`; harmonic still dominates
   `outward_pinned` (28.3 vs 53.6).

### Caveats on the Stage 11.14 result

- **Forced `accum_steps=1` instead of Stage 11.8's `accum_steps=4`.**
  Single-sample gradients are noisier. The training pattern was
  consistent: best at epoch 15 of 35 (early stop at patience=20). We
  haven't yet tested whether more epochs with `accum_steps=1` would
  recover the lost performance.
- **Two interpretations remain plausible:**
    1. The noisier `accum_steps=1` updates need more epochs to converge.
       Longer training (200 epochs, patience=40) might close the gap.
    2. The added data genuinely makes the optimization ndscape harder
       with our model size, and additional training won't recover.
- **The Mac CPU run (with 128GB system RAM) is still in progress** and
  will provide an independent data point. That experiment includes the
  full set of 10 V>50k surfaces (or as many as fit during training).
  Results pending.

### Implications for the project

If the longer-training experiment with `accum_steps=1` doesn't close
the gap, **Stage 11.8 remains the best model** and the
data-diversity hypothesis is effectively rejected at the scales we
could test. Final test evaluation (Stage 12) would proceed with
Stage 11.8 as the chosen model.

### Where the results live

- Stage 11.14 checkpoint:
  `outputs/tensorboard/run_20260617_223754/best.pt` (epoch 15,
  best val_rmse_meters=69.46).
- Stage 11.14 evaluation:
  `outputs/evaluation/run_20260617_223754_val.json`.
- Stage 11.8 re-evaluation on new val:
  `outputs/evaluation/run_20260614_155745_val.json` (overwrote previous
  evaluation; the original 7-surface val results are gone unless
  recovered from git).

### Methodological note

When changing the val set during a research project, both old and new
models should be re-evaluated on the new set. Direct comparisons across
runs require the same evaluation set. We did this correctly here —
both Stage 11.8 and Stage 11.14 are scored on the same 80-record val.

---

## O16 — Noise-floor calibration: overall-mean differences below ~4 m are not real

**Observed in:** A dedicated calibration run before the EdgeConv experiment
(Stage 12). We re-scored the fixed Stage 11.8 checkpoint
(run_20260614_155745) on the 7-surface val under 5 different mask-draw
seeds (base_seed 1000–5000), n_masks=10, via `scripts/noise_band.py`. The
model is held constant; only the random masks change, isolating the
*eval-mask* component of run-to-run variance.

### Result

| Quantity | min–max | mean | range |
|---|---|---|---|
| model overall | 69.8 – 73.7 | 72.1 | **3.9 m** |
| harmonic overall | 70.6 – 77.7 | 74.0 | 7.1 m |
| model half_plane | 68.9 – 112.8 | 82.8 | **43.9 m** |
| model outward_free | 55.1 – 76.9 | 66.8 | 21.8 m |
| model outward_pinned | 50.3 – 87.5 | 69.5 | **37.2 m** |

### Two findings that reshape how we read every result

1. **On overall mean, the GNN and harmonic infill are tied within noise.**
   Model 72.1 vs harmonic 74.0 — a ~1.9 m gap, smaller than the 3.9 m
   eval-mask range. Earlier headline framings ("73.7 vs harmonic", the
   continuation brief's "76.6 vs 73.1") are both inside the noise; neither
   method is actually ahead on the aggregate. The honest claim is a tie on
   overall mean, with the real story in the regime breakdown.

2. **Per-regime single-seed numbers are unreliable.** half_plane swings
   44 m, outward_pinned 37 m, outward_free 22 m — purely from which masks
   were drawn, model fixed. A single-eval statement like "outward_pinned
   53.6 vs harmonic 28.3" can be off by ±20 m. Per-regime claims require
   averaging over several eval seeds with the spread reported.

### Why the per-regime spread is so large

The val set has only 7 surfaces; at n=10 masks each regime gets ~20–28
records, but they are dominated by a few hard surfaces (e.g.
05_TopoCretaceo). A handful of lucky/unlucky mask draws on those surfaces
moves the regime mean by tens of metres. This is a dataset limitation (O11
already flagged the small-val problem), not fixable by sampling more masks.

### Implications

- **Working decision threshold:** treat an overall-mean change below ~4 m as
  noise. Training-seed variance (re-running with a different seed) adds to
  this; we measured only the eval-mask half — the cheaper and dominant
  component — and deferred the ~2 retrains needed for the training half.
- For the EdgeConv comparison (O18), the test is whether its overall mean
  lands outside Stage 11.8's 69.8–73.7 m band, not whether it "beats 72.1."
- The thesis should report aggregate metrics as ranges over eval seeds, not
  single numbers, and frame model-vs-harmonic on overall mean as a tie.

### Where the result lives

- `scripts/noise_band.py outputs/tensorboard/run_20260614_155745`
  (5 seeds, n_masks=10).

### Caveats

- **Eval-mask variance only.** Training-seed variance (init + shuffle order)
  is not included and would widen the band further.
- **One checkpoint.** The spread is for the Stage 11.8 model; the
  dataset-driven per-regime instability would persist for any model.

---

## O17 — EdgeConv's per-edge memory makes the full-BPTT rollout OOM at 48k vertices

**Observed in:** First EdgeConv training attempt (Stage 12, D12.1), seed 42,
otherwise Stage 11.8 config, on the 16 GB macOS dev machine. The process was
OOM-killed (no Python traceback) in epoch 0, on reaching the first
48k-vertex surface.

### What happened

EdgeConv (PyG) materialises a feature vector *per edge*:
`hΘ([h_i, h_j − h_i])` produces tensors of shape `[E, 2H]` then `[E, H]`,
all retained for backward. A triangle mesh has E ≈ 6V directed edges, and
the rollout keeps the full N-step graph for BPTT (D5.6). So EdgeConv's peak
training memory is roughly `2 layers × N × E × 4H × 4 bytes`.

For a 48k-vertex surface (E ≈ 288k, H = 64):

| Operator | retained per layer per step | peak at N≈30 / N≈50 (2 layers) |
|---|---|---|
| SAGE (per-vertex scatter) | ~12 MB | ~0.7 / 1.2 GB |
| EdgeConv (per-edge MLP) | ~295 MB | ~18 / 30 GB |

EdgeConv's per-step footprint is ~24× SAGE's (≈6× the edges × ≈4× the
width), so it blows past 16 GB the moment training touches one of the four
48k-vertex train surfaces — and would strain even the 48 GB GPU on the
deepest masks.

### Interpretation

This is the same O(V × N × H) trajectory-retention wall from O15, but
EdgeConv's per-edge cost makes it bite at 48k vertices instead of 200k+.
More generally: **expressive per-edge operators (EdgeConv, and likely
GAT/GINE) are incompatible with full-trajectory BPTT at our surface sizes
without memory engineering.** The cheap scatter operators (SAGE, GCN) are
the exception, not the rule.

### Resolution

Gradient checkpointing the per-step operator call (D12.2) recomputes the
per-edge activations during backward instead of retaining them, dropping
peak from ~30 GB to under 1 GB for the 48k surface. With it enabled,
EdgeConv trains to completion on the 16 GB machine (~710 s/epoch, ~3×
SAGE's ~225 s, from the per-edge MLP plus the recompute). The checkpointing
is mathematically transparent (D12.2 test), so the result remains a fair
comparison to the non-checkpointed SAGE baseline.

### Implications

- Any future expressive-operator ablation must budget for checkpointing.
- The same machinery unblocks the V>50k surfaces for the data-diversity
  question (O14/O15), now trainable within 16 GB regardless of depth.

### Where it lives

- Memory scaling: the EdgeConv-vs-SAGE retained-tensor estimate above.
- Resolution: `horizons/training/rollout.py` (`use_checkpoint`), D12.2.

---

## O18 — EdgeConv (max) does not beat SAGE: the operator is not the lever

**Observed in:** Stage 12 (D12.1). EdgeConv with max aggregation, seed 42,
otherwise the exact Stage 11.8 config (30-surface train, normalize,
meanplane, n_masks=3, hidden=64, layers=2), trained on the GPU container
with gradient checkpointing (D12.2; verified the container used the 30-train
/ 7-val split). Evaluated with the same paired protocol as the O16 noise
probe: 5 mask-draw seeds (1000–5000), n_masks=10, identical masks to the
Stage 11.8 evaluation.

### Result: consistently ~2.3 m worse, within the noise envelope

Paired by mask-draw seed (same masks → controls for eval-mask variance):

| mask seed | SAGE 11.8 | EdgeConv | Δ (EC − SAGE) |
|---|---|---|---|
| 1000 | 73.69 | 76.41 | +2.72 |
| 2000 | 72.54 | 75.41 | +2.87 |
| 3000 | 71.07 | 73.78 | +2.71 |
| 4000 | 73.36 | 74.83 | +1.47 |
| 5000 | 69.78 | 71.75 | +1.97 |
| **mean** | **72.09** | **74.44** | **+2.35** |

EdgeConv is worse on all 5 paired draws (mean +2.35 m, std of the paired
difference 0.6 m). Controlling for masks, the regression is *consistent*,
but its magnitude (2.35 m) is within the ~4 m eval-mask noise floor (O16) —
and training-seed variance (unmeasured) would only widen that. Honest read:
**no improvement, plausibly a small regression, not confidently separable
from the training lottery.**

### The hypothesised win did not appear

Per-regime means (5 seeds):

| regime | SAGE 11.8 | EdgeConv | Δ |
|---|---|---|---|
| half_plane | 82.8 | 85.7 | +2.9 |
| outward_free | 66.8 | 70.5 | +3.7 |
| outward_pinned | 69.5 | 69.9 | +0.4 |

D12.1's hypothesis was that EdgeConv's neighbour-difference (local-gradient)
inductive bias would help the **extrapolation** regimes (half_plane,
outward_free). It did the opposite — nominally worse on exactly those (within
noise), tied on outward_pinned. There is no hint of the predicted gain.

### Interpretation

Given a fair fight — identical data and config, memory unblocked by
checkpointing — a genuinely different, more expressive operator lands on top
of SAGE (slightly under). Combined with the Stage 11 sweep (width O12, depth
O13, augmentation O9, regime weights O10, λ_c, LR, data O14/O15), this is
strong evidence that **the bottleneck is not the message-passing operator.**
The plateau at ~72 m (± noise) is set by something else — most plausibly the
rollout's long-range information propagation (signal from K must cross up to
~50–130 rings) or the small, hard dataset — not the choice of GNN layer.

### Decision

EdgeConv (max) is rejected as an improvement; **Stage 11.8 (SAGE) remains the
best model.** The operator axis is considered explored. aggr=mean is the one
untested EdgeConv variant, but being the more SAGE-like aggregation it is
expected to regress toward SAGE, so it is deprioritised.

### Where the result lives

- EdgeConv checkpoint: `outputs/tensorboard/run_20260619_155040/best.pt`
  (aggr=max, best epoch 41).
- Paired eval: `scripts/noise_band.py` on that run dir and on
  run_20260614_155745 (Stage 11.8), seeds 1000–5000.

### Caveats

- **One training seed each.** The +2.35 m is single-seed-vs-single-seed
  across eval masks; training-seed variance is unmeasured, so the small
  regression cannot be cleanly separated from the training lottery. Safe
  claim: "no improvement."
- **aggr=mean untested.**
- **Checkpointing is transparent (D12.2 test),** so it does not confound the
  comparison.

---

## Phase 1 concluded; Phase 2 (full dataset) begins

**Phase 1 (small-data regime, V ≤ 48k, 30 train surfaces).** Across O1–O18 we
ran an exhaustive search — coordinate normalization (O6), init (O7),
augmentation (O8/O9), regime weights (O10), width (O12), depth (O13), λ_c
(Stage 11.1), more data within memory limits (O14/O15), and a genuinely
different operator (EdgeConv, O18). **Stage 11.8 (SAGE, hidden=64, 2 layers,
normalize, meanplane, n_masks=3, 30/40/30) is the best model**, at ~72 m on the
7-surface val and tied with harmonic infill within the ~4 m noise floor (O16).
No model-side lever beats it; the bottleneck is the rollout dynamics or the
small dataset, not capacity or operator choice. **This concludes Phase 1.**

**Phase 2 (full dataset).** Gradient checkpointing (D12.2) removed the memory
wall (O17) that forced the V>50k surfaces out at Stage 4 (D4.2). Phase 2
restarts the study on a new canonical split (D12.3) that adds the eight V≤600k
large surfaces (110k–455k), magnitude-balanced across train / val / test_id so
the model is tested faithfully across mesh scales. Plan:
1. **Fresh baseline (O19):** Stage 11.8 hyperparameters on the new split, with
   n_epochs=200 / patience=40 (more and larger data may need longer to
   converge) and grad_checkpoint=true.
2. **Hypothesis-driven tuning.** Priority: re-test **capacity** (width/depth) —
   O12's "not capacity-limited" was specific to the small-data regime, and more
   data may flip it — then init and operator. Deprioritise levers with no
   reason to interact with data scale (regime weights, λ_c).

Phase-1 numbers (7-surface val, ~72 m floor) and Phase-2 numbers (9-surface val
incl. large surfaces) are NOT directly comparable; each phase is baselined on
its own val.

---

## O19 — Phase-2 baseline: with large surfaces in play, harmonic infill beats the GNN

**Observed in:** Phase-2 baseline run — Stage 11.8 hyperparameters (SAGE,
hidden=64, 2 layers, normalize, meanplane, n_masks=3) on the magnitude-balanced
split_v2 (D12.3), n_epochs=200, patience=40, grad_checkpoint=true, seed=42, on
the GPU container. The first attempt crashed at epoch 91 (NaN weights; fixed by
the gradient guard D12.4); the rerun completed, best epoch 54. Evaluated on the
9-surface val with `scripts/noise_band.py` (5 mask-draw seeds, n_masks=10,
device=cuda).

### Aggregate: harmonic wins by ~13 m, on every seed

| method | min–max | mean | std |
|---|---|---|---|
| GNN model | 90.0 – 104.8 | **96.6** | 5.7 |
| harmonic | 76.3 – 96.7 | **83.7** | 8.4 |

Paired by mask seed (harmonic − model): −12.0, −8.2, −17.1, −16.1, −11.3 —
harmonic better on all 5 (mean +12.9 m). This is the reverse of Phase 1, where
the two were tied at ~72 m (O16). Adding the large surfaces flips the aggregate
to harmonic's favour.

### Per-surface: the GNN's edge is intact on small surfaces, lost on large ones

Mean RMSE over masks × seeds (Δ = model − harmonic; positive = harmonic better):

| surface | V | model | harmonic | Δ |
|---|---|---|---|---|
| 02TopoMioceno | 443k | 202.2 | 174.3 | +28.0 |
| 04BaseOligoMioceno | 110k | 111.2 | 70.7 | +40.5 |
| 05_TopoCretaceo | 48k | 312.9 | 224.7 | +88.2 |
| 10_BaseModelo | 48k | 0.0 | 0.0 | 0.0 |
| 09_Horizonte8 | 2.4k | 133.5 | 122.3 | +11.3 |
| horizonte7 | 9.7k | 0.6 | 2.6 | −2.0 |
| Horizonte5 | 9.7k | 0.6 | 2.7 | −2.0 |
| TestHorizon7 | 2.4k | 58.9 | 78.1 | −19.2 |
| TestHorizon4 | 2.4k | 49.6 | 78.0 | −28.4 |

The GNN still **wins** on the small extrapolation surfaces (TestHorizon4 −28,
TestHorizon7 −19, horizonte7/Horizonte5 ~−2) — the Phase-1 picture holds. The
entire aggregate loss is three surfaces: the two large ones (443k +28, 110k +40)
and the pathologically hard 05_TopoCretaceo (+88, both methods awful).

### Interpretation: a rollout-depth scaling problem, not capacity

The deficit vs harmonic is **size/depth-graded**: the GNN wins on the 2.4k
surfaces, ties on the flat 48k one, and loses progressively on 110k → 443k.
Crucially the model *was* trained on four large surfaces (192k–455k), so this is
not "never seen large" — it is a failure to *generalise* to large val surfaces.
Most plausible mechanism: filling a large surface needs an N≈150–200 step
rollout and per-step error accumulates over that depth, whereas harmonic infill
solves the whole field in one global linear solve and pays no depth penalty.
This points at the **rollout formulation** (deep-rollout error accumulation) as
the scaling bottleneck — consistent with O18's finding that the operator and
capacity are not the lever.

### Rollout-depth crossover (quantified)

Re-evaluating per surface with the rollout depth N alongside the deficit
(`noise_band.py`, split_v2 val, seeds 1000–3000, n_masks=10) makes the mechanism
quantitative. Δ = model − harmonic (negative = GNN wins):

| surface | N | model | harmonic | Δ |
|---|---|---|---|---|
| TestHorizon4 | 11 | 48.9 | 75.5 | −26.6 |
| TestHorizon7 | 11 | 60.7 | 82.2 | −21.5 |
| 09_Horizonte8 | 12 | 132.7 | 119.8 | +12.9 |
| Horizonte5 | 19 | 0.6 | 2.5 | −2.0 |
| horizonte7 | 22 | 0.6 | 2.6 | −2.0 |
| 10_BaseModelo (flat) | 51 | 0.0 | 0.0 | 0.0 |
| 05_TopoCretaceo | 52 | 320.5 | 232.0 | +88.5 |
| 04BaseOligoMioceno | 69 | 116.2 | 75.0 | +41.2 |
| 02TopoMioceno | 132 | 213.1 | 191.7 | +21.4 |

The GNN's advantage is concentrated at **shallow depth (N ≲ 25)** — the genuine
small-surface extrapolation cases — and **inverts at deep depth (N ≳ 50)**, the
large surfaces. The crossover sits around **N ≈ 25–50**. Two qualifications: the
near-flat surfaces (horizonte7, Horizonte5, 10_BaseModelo) sit on Δ ≈ 0 and are
uninformative (both methods trivially correct), and 05_TopoCretaceo is a
pathologically hard *and* deep outlier. The deficit is also not perfectly
monotonic in N — 02TopoMioceno (N=132, Δ+21) is *less* bad than
04BaseOligoMioceno (N=69, Δ+41) because the former is hard for harmonic too — so
the deficit tracks **depth × harmonic-friendliness**, not depth alone. This is
the quantified form of the hypothesis above: per-step rollout error accumulates
with depth, so the learned operator's edge survives only while the rollout is
short.

**Figure:** `outputs/evaluation/plots/phase2_crossover.png` (deficit vs N),
regenerated by `python scripts/plot_crossover.py`.

### Implications

- Phase 1's conclusion strengthens rather than overturns: the learned operator's
  advantage is real but narrow (small-surface extrapolation); classical infill
  wins once large smooth surfaces dominate the evaluation.
- Capacity-first tuning (the Phase-2 plan) is now in doubt as a fix — a
  depth-of-rollout problem is unlikely to be solved by wider/deeper layers (O20
  tests this directly). The depth crossover above is the achievable,
  characterising result; closing it would need a change to the rollout
  formulation, not the operator.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260621_171110/best.pt` (best epoch 54).
- Eval: `scripts/noise_band.py` on that run, split_v2 val, seeds 1000–5000,
  n_masks=10, device=cuda.

### Caveats

- **Single training seed.** Per O16, per-regime aggregates are noisy (the 443k
  val surface widens the bands); the per-surface means (over masks × seeds) are
  the reliable cut, not the per-regime numbers.
- **best.pt is from a noisy plateau** (training-time val bounced ~98–130 m), so
  the checkpoint selection on this large-surface val is itself uncertain.

---

## O20 — Capacity (hidden_dim=128) does not move Phase 2: not capacity-limited (again)

**Observed in:** Phase-2 run, Stage 11.8 config but `model.hidden_dim=128`
(~84k params vs 21k), split_v2, n_epochs=200/patience=40, grad_checkpoint=true,
seed=42, GPU. Evaluated on split_v2 val (`noise_band.py`, 3 seeds, n_masks=10,
device=cuda).

### Result: tracks the baseline; slightly worse at the deepest surface

| metric | baseline (h=64) | O20 (h=128) |
|---|---|---|
| overall model | 99.3 | 103.4 |
| overall harmonic | 86.8 | 86.8 |

Per-surface, O20 lands on the baseline within noise everywhere **except the
deepest surface**: 02TopoMioceno (443k, N=132) gets *worse* (model 246.7 vs
213.1; deficit vs harmonic +55.0 vs +21.4). The shallow extrapolation wins are
essentially unchanged (TestHorizon4 −22.8 vs −26.6).

### Interpretation

Quadrupling parameters does not help — consistent with O12's small-data finding,
now confirmed at Phase-2 scale. If anything, more capacity is marginally *worse*
on the deepest rollout (more parameters → more per-step error to accumulate over
N=132 steps). Capacity is ruled out as the fix; the depth crossover (O19) is
unchanged.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260621_231004/best.pt` (best epoch 63;
  training-time val 98.5 ≈ the h=64 baseline's 98.3).
- Eval: `scripts/noise_band.py` on that run, split_v2 val, seeds 1000–3000.

---

## O21 — Harmonic-init: a depth-dependent trade, net negative — the rollout degrades a good start

**Observed in:** Phase-2 run, Stage 11.8 config but `data.init_method=harmonic`,
split_v2, n_epochs=80/patience=20, grad_checkpoint=true, seed=42, GPU
(early-stopped normally, best epoch 31). Evaluated on split_v2 val
(`noise_band.py`, 3 seeds, n_masks=10, device=cuda).

### Result: worse overall, but for an informative reason

Overall: model 107.3 vs baseline 99.3 (harmonic 86.8). Per-surface deficit
(Δ = model − harmonic), vs baseline:

| surface | N | baseline Δ | O21 Δ |
|---|---|---|---|
| TestHorizon4 | 11 | −26.6 | −7.9 |
| TestHorizon7 | 11 | −21.5 | −10.4 |
| 09_Horizonte8 | 12 | +12.9 | −2.8 |
| 05_TopoCretaceo | 52 | +88.5 | +73.7 |
| 04BaseOligoMioceno | 69 | +41.2 | +15.8 |
| 02TopoMioceno | 132 | +21.4 | +116.0 |

Three distinct effects:
1. **Helps moderate-depth / smooth-hard surfaces** where harmonic is a good
   start: 04BaseOligoMioceno (110k, N=69) Δ+16 vs +41; 09_Horizonte8 now wins
   (−2.8); 05_TopoCretaceo improves.
2. **Erodes the shallow-N extrapolation edge**: TestHorizon4/7 wins shrink from
   ~−24 to ~−9 — harmonic is a *poor* init for extrapolation (cf. O7), so the
   model's one strength weakens.
3. **Catastrophic at extreme depth**: 02TopoMioceno (443k, N=132) → model 307.7
   (Δ+116), *worse than meanplane-init (213) and far worse than harmonic alone
   (192)*.

### Interpretation: the strongest evidence for the rollout-depth bottleneck

A harmonic-init model that did nothing (Δz=0) would return harmonic's solution
exactly. Instead the trained model **degrades its own starting point** — most
severely on the deepest surface, where N≈130 rollout steps drift harmonic's
192 m solution to 307 m. So the limit is not the starting point (O21), nor
capacity (O20), nor the operator (O18): it is the **rollout accumulating error
with depth** (O19). Even handed the right answer, the deep rollout cannot
preserve it. Harmonic init also destabilised training (val spiked to ~1800 m
mid-run), consistent with the per-mask harmonic solve adding noise.

### Decision

Both Phase-2 interventions (O20 capacity, O21 init) are rejected; the depth
crossover (O19) stands, now bracketed by two failed fixes. The only untested
lever is the **rollout formulation itself** (e.g. truncated/anchored rollout, or
hybridising with the global solve) — out of scope here, flagged for future work.
Phase-2 conclusion: the GNN's value is depth-limited, and the limit is the
rollout — not the operator, capacity, or init.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260622_032320/best.pt` (best epoch 31;
  training-time val 120.1).
- Eval: `scripts/noise_band.py` on that run, split_v2 val, seeds 1000–3000.
- Figure: `outputs/evaluation/plots/phase2_crossover_compare.png` (baseline vs
  O20 vs O21, deficit vs N), via `scripts/plot_crossover.py`.

### Caveats

- **Single training seed; 3-seed eval.** Per-surface means are the reliable cut.
- The 110k improvement and 443k catastrophe are one surface each — directional,
  not precise.

---

## O22 — Residual penalty (λ_r=0.1) throttles the model and kills extrapolation

**Observed in:** Phase-2 run, Stage 11.8 config but `loss.lambda_r=0.1` (100× the
default), split_v2, seed=42, GPU. Eval: split_v2 val, 3 seeds, n_masks=10.

### Result: worse overall, extrapolation edge destroyed

Overall model 109.4 vs baseline 99.3 (harmonic 86.8). The best checkpoint came at
**epoch 2** and never improved — the heavy residual penalty drove per-step Δz
toward zero, so the model essentially stopped correcting and plateaued at once.
Per-surface deficit (Δ = model − harmonic), vs baseline:

| surface | N | baseline Δ | O22 Δ |
|---|---|---|---|
| TestHorizon4 | 11 | −26.6 | −0.1 |
| TestHorizon7 | 11 | −21.5 | +4.6 |
| 04BaseOligoMioceno | 69 | +41.2 | +27.6 |
| 02TopoMioceno | 132 | +21.4 | +76.8 |

The model's one strength — shallow-N extrapolation — is **gone** (TestHorizon4/7
collapse from ~−24 to ~0). Extrapolation requires meaningful per-step
corrections; penalising them removes exactly that, and the deepest surface is
much worse (do-nothing leaves it near the mean-plane init).

### Interpretation

λ_r is a negative across its range: at the default it is effectively off, and
cranked up it throttles the model into inaction. It suppresses the corrections
the rollout needs rather than addressing depth accumulation. Not the lever.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260622_145038/best.pt` (best epoch 2).
- Eval: `scripts/noise_band.py`, split_v2 val, seeds 1000–3000.

---

## O23 — Freeze-filled rollout (Phase 3c): the rollout's repeated updates are refinement, not drift

**Observed in:** Phase-2 run, Stage 11.8 config but `rollout.method=freeze_filled`
(a ring is locked once the frontier passes it, d < step), split_v2, seed=42, GPU.
Eval: split_v2 val, 3 seeds, n_masks=10, with the same freeze rollout.

### Result: worse on the deep surfaces, shallow wins preserved

Overall model 103.1 vs baseline 99.3 (harmonic 86.8). Per-surface deficit
(Δ = model − harmonic), vs baseline:

| surface | N | baseline Δ | O23 Δ |
|---|---|---|---|
| TestHorizon4 | 11 | −26.6 | −24.9 |
| TestHorizon7 | 11 | −21.5 | −18.9 |
| 05_TopoCretaceo | 52 | +88.5 | +73.2 |
| 04BaseOligoMioceno | 69 | +41.2 | +50.6 |
| 02TopoMioceno | 132 | +21.4 | +60.5 |

Freeze **preserves** the shallow-N extrapolation wins (those rings freeze early,
and their first-pass estimate is already good on small surfaces) but is **worse
on the deep surfaces** (443k +60.5 vs +21.4; 110k +50.6 vs +41.2) — the opposite
of the (c) hypothesis.

### Interpretation: refutes "re-drift is harmful"

(c) assumed that re-updating an already-filled ring is harmful drift, so freezing
would help, most of all deep. The data says the opposite: locking a ring at its
first-pass (under-informed) estimate and forbidding further updates is **worse**,
most on the deep surfaces. So the standard rollout's repeated updates are net
**refinement** — as information propagates further from K, later passes improve
the earlier rings, and that refinement is what large surfaces depend on. The
bottleneck (O19) is not that filled rings drift; it is that the iterative
refinement does not *converge well enough* over depth, and constraining it makes
it worse.

### Decision — Phase 3(c) rejected; what it implies

(c) is rejected. With O22, both "constrain/stabilise the existing rollout" tweaks
(freeze; throttle Δz) make things worse, because the rollout's iterative updates
are beneficial, not harmful. Together with O18 (operator), O20 (capacity), O21
(init), **every model-side and rollout-tweak intervention worsens the deepest
surface; the plain baseline is the best the GNN does there.** This points away
from tweaking the rollout and toward replacing its propagation mechanism — the
Phase-3(a) hybrid (global solve + single-shot residual) or the future-work (b)
multi-scale rollout.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260622_183336/best.pt` (best epoch 37).
- Eval: `scripts/noise_band.py`, split_v2 val, seeds 1000–3000 (freeze rollout).

---

## O24 — Hybrid (harmonic init + fixed-K GNN refine) beats harmonic: best of both

**Observed in:** Phase-3 capstone (a). SAGE operator, but a new prediction path
(`approach=hybrid`): init = harmonic infill (a global solve that fills all of U),
refined by a **fixed 3 GNN passes** (no surface-depth march), trained with all-U
MSE (`hybrid_rollout_loss`). hidden=64, split_v2, seed=42, GPU; early-stopped at
epoch 117 (best 77). Eval: split_v2 val, 3 seeds, n_masks=10, with the hybrid
path.

### Result: the learned approach finally beats harmonic

| method | mean (3 seeds) |
|---|---|
| Phase-2 baseline GNN (rollout) | 96.6 |
| harmonic infill | 86.8 |
| **hybrid** | **79.5** |

Paired by mask seed (hybrid − harmonic): −8.7, −9.3, −4.0 — the hybrid wins on
all three (mean −7.3 m). The baseline GNN *lost* to harmonic by ~13 m; the hybrid
*wins* by ~7 m — a ~20 m swing, and the first time in the study the learned model
leads.

### Per-surface: best-of-both, including the deep end

Deficit (Δ = model − harmonic; negative = beats harmonic):

| surface | N | hybrid | harmonic | Δ | (baseline Δ) |
|---|---|---|---|---|---|
| 02TopoMioceno | 132 | 178.8 | 191.7 | **−12.9** | (+21.4) |
| 04BaseOligoMioceno | 69 | 77.7 | 75.0 | +2.6 | (+41.2) |
| 05_TopoCretaceo | 52 | 267.6 | 232.0 | +35.5 | (+88.5) |
| 09_Horizonte8 | 12 | 113.2 | 119.8 | −6.6 | (+12.9) |
| TestHorizon4 | 11 | 35.4 | 75.5 | **−40.1** | (−26.6) |
| TestHorizon7 | 11 | 39.4 | 82.2 | **−42.8** | (−21.5) |

1. **The deepest surface is fixed.** 443k (N=132): hybrid 178.8 beats harmonic
   (191.7) *and* the baseline rollout (213.1). The crossover's worst point bent
   below zero — the deep-surface limit that survived O18–O23 is gone.
2. **The extrapolation edge got stronger, not traded away.** TestHorizon4/7 reach
   −40/−43 (hybrid 35/39 vs baseline rollout 49/61 vs harmonic 75/82) — beating
   *both* the rollout GNN and harmonic on the shallow surfaces.
3. **Best-of-both elsewhere:** matches harmonic on 110k (+2.6), now wins
   09_Horizonte8; only the pathological 05_TopoCretaceo still loses, and even
   there it improved from +88 to +35.

### Interpretation: confirms the whole diagnosis by construction

Harmonic's global solve does the long-range reach with no depth penalty; the
GNN's fixed 3 passes add the local, non-smooth detail harmonic misses, with no
surface-depth march and so no accumulation. Beating harmonic *even on the 443k*
shows the GNN learned genuinely useful local corrections; beating the rollout GNN
*even on shallow surfaces* shows refining a good global field in 3 clean passes
beats building the field over ~N error-accumulating steps from a mean plane. The
bottleneck was never operator, capacity, or init — it was the deep sequential
rollout (O19), and replacing it with global-solve + shallow-refinement is what
works.

### Decision

`approach=hybrid` is the **best model of the study** and the recommended
configuration for the large-surface regime: it beats harmonic infill and the
Phase-2 rollout baseline, robustly across eval seeds.

### Where the result lives

- Checkpoint: `outputs/tensorboard/run_20260623_115850/best.pt` (best epoch 77).
- Eval: `scripts/noise_band.py`, split_v2 val, seeds 1000–3000 (hybrid path).
- Figure: `outputs/evaluation/plots/phase2_hybrid.png` (baseline vs hybrid,
  deficit vs N), via `scripts/plot_crossover.py`.

### Caveats / next

- **Single training seed.** The win is paired across 3 eval seeds (robust to
  eval-mask noise) and the per-surface structure is mechanistically coherent, so
  it is a solid directional result — but a 2nd/3rd training seed would firm up
  the magnitude (the training-time val was noisy; best.pt was selected on a lucky
  low draw, yet the multi-seed eval confirms the level).
- **05_TopoCretaceo still loses** (pathologically hard for both methods).
- **Confirm on test_id / test_ood** before the final headline.
- **n_passes fixed at 3** — a 1/3/5 sweep is the obvious follow-up now that there
  is a signal to optimise.

---

## How to use this document

Append new observations as `O<N>` entries when:
- A training run reveals a non-obvious property of the model or loss.
- An experiment confirms or refutes a hypothesis.
- A pattern recurs across multiple settings and is worth flagging.

Each entry should pin down what was observed and where, offer a best
explanation, and note implications for future work. Avoid speculation
without evidence.
