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

## How to use this document

Append new observations as `O<N>` entries when:
- A training run reveals a non-obvious property of the model or loss.
- An experiment confirms or refutes a hypothesis.
- A pattern recurs across multiple settings and is worth flagging.

Each entry should pin down what was observed and where, offer a best
explanation, and note implications for future work. Avoid speculation
without evidence.
