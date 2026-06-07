# Design Decisions Log

This document records the deliberate design choices made while building
the GNN-based geological-horizon extrapolation system. Each entry states
what was decided, where it lives in code, why we chose it, and whether
it's open for revisiting later.

The decisions are grouped by the stage in which they were made; within a
stage they appear in the order they came up.

---

## Stage 0 — Project setup

### D0.1 — Python environment management
**Decision:** Conda environment named `horizons` with Python 3.11, plus
PyTorch installed via `python -m pip` to avoid pyenv-shim interference.
**Where:** No code; documented in the README (to be written) and the
conda env spec.
**Why:** Python 3.11 is the latest version with full PyG / torch-scatter
ecosystem support. The `python -m pip` pattern was needed because the
user's system has pyenv installed alongside conda, and a plain `pip`
command was redirected to the wrong Python. Using `python -m pip`
guarantees pip runs in the active interpreter.
**Status:** Fixed.

### D0.2 — Editable package install
**Decision:** The project is installed as an editable package via
`pip install -e .`, with a minimal `pyproject.toml`.
**Where:** `pyproject.toml`.
**Why:** Lets `from horizons.data.mesh import HorizonSurface` work from
any directory — scripts, tests, notebooks — without `sys.path` hacks or
running everything as `python -m`. Standard Python practice.
**Status:** Fixed.

### D0.3 — Hydra for config management
**Decision:** All hyperparameters live in `configs/default.yaml`, loaded
via Hydra at runtime. Overrides happen on the command line.
**Where:** `configs/default.yaml`, `scripts/*.py`.
**Why:** Centralizes the tuning surface in one file (no editing code to
change a learning rate); makes Stage 12 ablations one-line overrides.
The alternative (argparse + per-script defaults) scatters configuration
across many files and breaks down quickly when sweeps begin.
**Status:** Fixed.

### D0.4 — TensorBoard for experiment tracking
**Decision:** TensorBoard, not Weights & Biases.
**Where:** Logged via `configs/default.yaml` (`train.log_dir`); training
loop in Stage 8 will write events here.
**Why:** No accounts, no network dependencies, fully local. Adequate for
a single-developer research project. W&B would be better for team
collaboration but we don't need that.
**Status:** Fixed; could switch later without code changes (just config).

---

## Stage 1 — Mesh foundation

### D1.1 — Mesh data structure
**Decision:** `HorizonSurface` dataclass holding `V` (float32),
`F` (int64), `edge_index` (int64), and metadata (`surface_id`,
`reservoir_id`).
**Where:** `horizons/data/mesh.py`.
**Why:** Dataclass gives free `__repr__` and clear field declarations.
Storing everything as torch tensors (not numpy) means autograd works
through feature recomputation later (Stage 6) without any conversion
overhead.
**Status:** Fixed.

### D1.2 — Edge index format
**Decision:** PyG-style `(2, n_directed_edges)` with each undirected edge
appearing as two directed edges. Deduplicated via `torch.unique`.
**Where:** `build_edge_index` in `horizons/data/mesh.py`.
**Why:** PyG's message-passing layers expect this format. Deduplication
matters because shared mesh edges would otherwise contribute their
messages twice.
**Status:** Fixed.

### D1.3 — Vertex normals: area-weighted face-normal average
**Decision:** Normals computed as the area-weighted average of incident
face normals (raw cross products are summed before normalization,
which gives area-weighting automatically).
**Where:** `compute_vertex_normals` in `horizons/data/features.py`.
**Why:** Standard, robust, differentiable, ~10 lines, no edge cases.
Alternative (per-face normalize, then average) is less stable for
degenerate triangles.
**Status:** Fixed.

### D1.4 — Curvature: umbrella Laplacian (graph Laplacian)
**Decision:** Curvature for both the input feature and the regularizer
is computed as the umbrella Laplacian:
$\kappa_i = z_i - \frac{1}{|N_i|} \sum_{j \in N_i} z_j$
**Where:** `compute_umbrella_laplacian` in `horizons/data/features.py`.
**Why:** Simplest valid discrete Laplacian, ~3 lines, no cotangent edge
cases. Using the same expression as both input feature and regularizer
target means one function used twice — no risk of inconsistency.
**Status:** Open for ablation in Stage 12. The cotangent Laplacian is
the natural improvement: it handles irregular triangulation more
gracefully, especially near boundaries.

### D1.5 — Differentiability verified by gradcheck
**Decision:** Both `compute_vertex_normals` and
`compute_umbrella_laplacian` have `torch.autograd.gradcheck` tests.
**Where:** `tests/test_features.py`.
**Why:** These functions are called inside the rollout (Stage 6) where
gradients must flow back to model parameters. Silent autograd bugs
would surface only during training, where they're enormously harder to
diagnose. Verifying at the function level is cheap and decisive.
**Status:** Fixed.

### D1.6 — Synthetic fixtures for testing
**Decision:** Three synthetic fixtures (plane, sphere cap, anticline)
generated programmatically and saved as `.npz`.
**Where:** `tests/fixtures/generate_fixtures.py` and the three `.npz`
files.
**Why:** Tests need known geometry. The plane has zero curvature for
sanity checking; the sphere cap has known radial normals; the anticline
mimics a real horizon (tilted baseline + Gaussian bump) for end-to-end
smoke tests.
**Status:** Fixed for now. To be revisited when the GOCAD `.ts` loader
is written and real horizon data becomes available; the convex-hull-
boundary issue we hit on the synthetic fixtures won't apply to real
meshes.

---

## Stage 2 — Mask sampling

### D2.1 — Three mask regimes
**Decision:** Three mask geometries, each implemented as a separate
sampler function:
1. **Half-plane cut.** Rank-based partition along a random orientation
   `theta ~ Uniform[0, 2π)`.
2. **Outward from central rectangle (free boundary).** Central rectangle
   contains K; everything outside is U.
3. **Outward from central rectangle (pinned boundary).** Same as #2,
   plus the mesh-boundary vertices are added back to K. Unknown region
   is the annulus between the central rectangle and the outer ring.
**Where:** `horizons/data/masking.py`.
**Why:** Regime 2 is the deployment scenario (extrapolating outward from
a central observation area). Regime 3 is the bracketed variant (fill
between two known regions, per `final-remarks.pdf`). Regime 1 adds
directional diversity for the local operator to learn from.
**Status:** Fixed.

### D2.2 — Rank-based partition for fraction control
**Decision:** Both the half-plane and outward-rectangle samplers
partition vertices by *rank* in a chosen metric, not by a fixed
threshold or offset.
**Where:** `sample_half_plane_mask` and `sample_outward_rectangle_mask`
in `horizons/data/masking.py`.
**Why:** Guarantees the target unknown fraction φ exactly, regardless of
the mesh's vertex density or shape. Alternatives (fixed offset, fixed
rectangle size) would give variable φ depending on the mesh.
**Status:** Fixed.

### D2.3 — Regime mix: 30% / 40% / 30%
**Decision:** Default sampling mix is 30% half-plane, 40% outward-free,
30% outward-pinned.
**Where:** `configs/default.yaml`, `mask.regime_weights`.
**Why:** Regime 2 is the deployment case and gets the highest weight.
Regimes 1 and 3 are evenly split to provide variety. Configurable per
the Hydra knob, so we can change without editing code.
**Status:** Configurable; expected to be revisited in Stage 12 ablations
to test alternative mixes.

### D2.4 — No boundary perturbation
**Decision:** Mask boundaries are exact (straight lines for half-plane,
exact rectangle edges for outward regimes). No fuzzing/jittering of the
cut geometry.
**Where:** Throughout `horizons/data/masking.py`.
**Why:** Source-dictated by `final-remarks.pdf`, which explicitly
removed the boundary-perturbation step. Simpler implementation, faster
sampling, more reproducible.
**Status:** Source-dictated; not open to revisiting.

### D2.5 — Pinned ring thickness = 1
**Decision:** Regime 3 pins exactly the outermost ring of mesh-boundary
vertices (one vertex thick).
**Where:** `configs/default.yaml`, `mask.pinned_ring_thickness`. The
sampler explicitly raises `NotImplementedError` for thickness ≠ 1.
**Why:** Simplest implementation. Wider rings (2-3 vertices) are a
plausible Stage 12 ablation but not the default.
**Status:** Open for ablation in Stage 12.

### D2.6 — Topological distance via BFS
**Decision:** Multi-source BFS from K, vectorized using torch operations
on the edge index. Disconnected unknown vertices receive sentinel value
`UNREACHABLE = -1`.
**Where:** `horizons/data/topo_distance.py`.
**Why:** Straightforward and correct. The mask sampler uses
UNREACHABLE to detect and retry masks that produce disconnected U
components, which would break the rollout.
**Status:** Fixed.

### D2.7 — Connectivity retry in MaskSampler
**Decision:** `MaskSampler.sample()` retries up to `max_retries=32`
times if a sampled mask produces disconnected U. Raises `RuntimeError`
after that many failures.
**Where:** `horizons/data/masking.py`, class `MaskSampler`.
**Why:** Connectivity failures are rare for our regimes but possible
(e.g., a rectangle clipping off a corner). Resampling is cheap; failing
loudly on systematic pathology is better than silently producing broken
training data.
**Status:** Fixed.

---

## Stage 3 — Initialization

### D3.1 — Mean-plane initialization for z⁰
**Decision:** Unknown vertices initialize to a least-squares plane fit
through the known vertices: $z^0_i = a x_i + b y_i + c$ for $i \in U$.
**Where:** `horizons/data/init.py`, functions `fit_mean_plane` and
`init_z`.
**Why:** Geological horizons have strong regional trends (dip over
kilometers). A flat mean-z initialization would be far from any
reasonable horizon at the far edge of U. The mean plane captures the
regional trend with negligible extra cost (one 3×3 least-squares solve).
We empirically verified mean-plane RMSE beats mean-z RMSE on the
anticline fixture
(`tests/test_init.py::test_z0_is_better_than_global_mean`).
**Why same in training and inference:** Mismatch between train and
inference initialization would create a distribution shift the model
isn't prepared for.
**Status:** Fixed.

### D3.2 — Known vertices preserved exactly in z⁰
**Decision:** `init_z` sets `z⁰[i] = z_true[i]` exactly for known
vertices, with no plane evaluation passing through the known points.
**Where:** `init_z` in `horizons/data/init.py`.
**Why:** The rollout re-anchors known vertices at `z_true` every
iteration. `z⁰[K]` must equal `z_true[K]` exactly (not approximately
via the plane fit) so the anchoring is a no-op for `t=0`.
**Status:** Fixed.

---

## Stage 4 — Real data wiring

### D4.1 — GOCAD `.ts` loader supports PVRTX/VRTX and TRGL
**Decision:** `HorizonSurface.from_ts(path)` parses GOCAD TSurf-format
files. Recognized records: `PVRTX <id> <x> <y> <z>` (also `VRTX` as a
common variant) and `TRGL <i> <j> <k>`. All other records (headers,
coordinate-system metadata, `TFACE` markers, `END`, etc.) are ignored.
Vertex IDs from the file are 1-indexed and may be non-contiguous; we
sort them and map to 0-indexed positions.
**Where:** `HorizonSurface.from_ts` in `horizons/data/mesh.py`.
**Why:** Minimal viable parser for the format. We don't need to interpret
the coordinate-system metadata because we use the raw coordinates as-is.
Multiple `TFACE` blocks (if any) are treated as one mesh — the vertex
index space is shared across them.
**Status:** Fixed.

### D4.2 — Dataset filter: exclude pathological surfaces
**Decision:** When converting `.ts` files into the dataset, exclude
files matching any of these criteria:
  - Euler characteristic ≠ 1 (non-manifold or multi-component meshes —
    these are not horizons in the geometric sense the project expects).
  - n_vertices < 500 (degenerate stubs).
  - n_vertices > 50,000 (computationally unwieldy at present scale;
    the very large meshes (max ~670k) would dominate per-step cost
    and add huge variance across surfaces).
**Where:** `scripts/build_dataset.py` (to be written in 4B).
**Why:**
- The audit revealed three non-manifold files (`Base.ts`, `FundoMar.ts`,
  `Horizonte3_Base.ts`) and one degenerate stub (`01_fundo_mar.ts`,
  V=4). These cannot be trained on meaningfully.
- The big meshes (V > 50k) are out-of-distribution in size; about 7
  files would be excluded by this threshold.
**Future work (explicit user request):** the V > 50k cutoff is
configurable. After the main training pipeline works, the user wants to
re-run including these files to test whether the model handles them
well. Plan to expose `data.max_vertices` in the config so the threshold
can be raised or removed via Hydra override.
**Actual outcome on the dataset:** 17 of 64 files were excluded:
  - 3 non-manifold (`Base.ts`, `FundoMar.ts`, `Horizonte3_Base.ts`)
  - 4 degenerate stubs with V=4 (the R1 lowercase_underscore files —
    after this filter, R1 is empty and R3 has only 1 surviving file;
    these are documentation, not bugs)
  - 10 files with V > 50k, all from the R3 (concatenated naming)
    pattern: stratigraphic markers from a Brazilian basin
    (`01FundoMar`, `02TopoMioceno`, ..., `17TopoAndarJiquia`,
    `18TopoEmbasamento`) at 110k-670k vertices. These form a
    coherent batch of high-resolution interpretation-grade data
    likely from a single 3D seismic project. They're the strongest
    candidates for a follow-up experiment.
**Net dataset:** 47 surfaces across 6 effectively-populated reservoir
groups (R2 has 18, R5 has 10, R6 has 6, R4 has 5, R7 has 5, R8 has 2;
R1 and R3 are essentially empty). The natural gap in the size
distribution at ~50k (kept files top out at 48k; dropped files
start at 110k) means the threshold sits on a real distributional
break, not a judgment call.
**Status:** Fixed for the initial training run. Threshold to be
revisited (Stage 12 or later).

### D4.3 — Normalize z-sign convention: all-positive (depth)
**Decision:** Files whose z-values are entirely negative get their
z-coordinate sign flipped at load time, so the dataset is uniformly
depth-positive.
**Where:** `scripts/build_dataset.py`. Implemented as a preprocessing
step before the surface is saved to `.npz`.
**Why:** The audit revealed 53 files with depth-positive z (the
majority and the standard GOCAD `ZPOSITIVE Depth` convention) vs. 11
files with elevation-negative z. Mixing both conventions in training
would force the model to learn a bimodal z distribution for no
geological reason. Flipping the minority to match the majority unifies
the input distribution.
**Note:** We flip based on the *empirical* sign of z, not by parsing
`ZPOSITIVE Depth` vs `ZPOSITIVE Elevation` headers. This is simpler and
robust to files that lack the header. Files with z crossing zero
(genuinely mixed signs) would need special handling, but the audit
showed zero such files in the dataset.
**Status:** Fixed.

### D4.4 — Reservoir groups inferred from filename style
**Decision:** Surfaces are grouped into one of 8 "reservoir" categories
based on filename pattern (R1 through R8):
  - R1 = `NN_lowercase_name.ts` (e.g. `01_fundo_mar.ts`)
  - R2 = `NN_CamelName.ts` (e.g. `01_Topo.ts`, `02_Horizonte1.ts`)
  - R3 = `NNCamelName.ts` (e.g. `01FundoMar.ts`, `15TopoSal.ts`)
  - R4 = `Horizonte<N>[_Suffix].ts`
  - R5 = `TestHorizon<N>.ts`
  - R6 = `horizonte<N>[-utm].ts` (lowercase h)
  - R7 = `Horizon<N>-OutSpace.ts`
  - R8 = standalone files (`Base.ts`, `FundoMar.ts`, etc. — fallback)
**Where:** Classification function in `scripts/build_dataset.py`.
**Why:** The dataset has no proper reservoir metadata; filename style
is our best proxy. These groups likely reflect different processing
batches, different software, or different data sources rather than
distinct geological reservoirs. We are explicitly aware this is an
imperfect grouping.
**Status:** Fixed for split v1. May need revision if filename
classification turns out to misgroup specific files.

### D4.5 — Two test sets: in-distribution (Test-ID) and out-of-distribution (Test-OOD)
**Decision:** The split has four parts:
  - **train** (70% of non-R7 surfaces, stratified per group)
  - **val** (15% of non-R7 surfaces, stratified per group)
  - **test_id** (15% of non-R7 surfaces, stratified per group)
  - **test_ood** (all of R7 — `Horizon<N>-OutSpace.ts`, ~5 surfaces)
**Where:** `data/splits/split_v1.json` (to be written in 4C).
**Why:** Two test sets serve different purposes:
- **Test-ID** gives a stable headline RMSE with reasonable sample size
  (~10 surfaces). Used for the main results table.
- **Test-OOD** holds out an entire group the model has never seen,
  testing genuine cross-group generalization. Used for the
  generalization claim.
R7 was chosen because the name `OutSpace` suggests these surfaces were
intended for extrapolation/out-of-distribution testing, and the group
has 5 surfaces — small enough to lose for training, big enough to
yield a meaningful (if noisy) test estimate.
**Status:** Fixed.

### D4.6 — Per-surface (x, y, z) centering deferred to Stage 8
**Decision:** Per-surface centering of x, y, and z is NOT applied at
data-load time. It will be applied inside the dataset / training loop
in Stage 8. The overfit_real.py script applies it ad-hoc to demonstrate
end-of-Stage-4 proof of life.
**Where:** Future work in `horizons/data/dataset.py`. Currently lives in
`scripts/overfit_real.py` as a temporary measure.
**Why:** Raw coordinates across the dataset are wildly large:
  - UTM x coordinates can be ~3.5e5
  - UTM y coordinates can be ~7.5e6
  - z (depth) can be 0 to 6000 m, different per surface
Feeding these directly to the GNN's input projection causes float32
precision loss and unstable training. The overfit experiment on
01_Topo demonstrated this concretely: without centering, the loss
curve was chaotic with order-of-magnitude jumps; with per-surface
(x, y, z) centering applied, the loss curves became smooth and the
final RMSE dropped by 2x (~17m → ~9m).
The fix for `fit_mean_plane` (D4.1) handles the precision issue
during initialization but does not propagate to the model's input
features — the model still sees raw coordinates in its 9-feature
vector. The proper fix is to center per-surface in the dataset class
in Stage 8.
**For x and y**: centering can use the full (x, y) since they're
fully observed (no information leakage between K and U).
**For z**: centering must use only z[K] since z[U] is what we're
predicting; using z[U] would leak ground truth.
**Status:** Open. Ad-hoc fix in overfit_real.py; formal implementation
deferred to Stage 8.

---

## Stage 5 — End-to-end skeleton

### D5.1 — HorizonDataset returns a plain dict
**Decision:** `HorizonDataset.__getitem__` returns a Python dict, not a
PyG `Data` object.
**Where:** `horizons/data/dataset.py`.
**Why:** Simpler to inspect and debug. PyG `Data` adds machinery we
don't need for batch-size-1 training. We can migrate to `Data` later if
batching demands it (Stage 9 might).
**Status:** Open. May revisit if Stage 9 (gradient accumulation) is
easier with PyG batching.

### D5.2 — Deterministic mask seeding via SHA-256 of (surface_id, epoch, split)
**Decision:** Each mask is sampled with an RNG whose seed is derived
from SHA-256 of `f"{surface_id}|{epoch}|{split}"`. Train masks use the
current epoch; val/test masks use epoch=0 always.
**Where:** `_make_rng` in `horizons/data/dataset.py`.
**Why:**
- **Train masks vary per epoch** → mask augmentation works (different
  mask each epoch for each surface).
- **Val/test masks are stable** → val loss is comparable across epochs.
- **Hash-based seeding** → portable, doesn't depend on access order.
  Python's built-in `hash` is salted per process for security, so we
  use SHA-256 instead for cross-run determinism.
- **Different (split, surface) combinations produce different masks**
  → train and val see different masks even for the same surface.
**Status:** Fixed.

### D5.3 — TinySAGE placeholder with small output-layer init
**Decision:** Stage 5's placeholder model is a single SAGEConv layer
followed by a linear head, with the head initialized to small weights
(`output_init_scale=0.01`) so initial Δz is small.
**Where:** `horizons/models/placeholder.py`.
**Why:** Decouples "is the rollout correct" from "is the model
architecture correct" for Stage 5's keystone overfit test. The small
init prevents the first rollout iteration from making large
destabilizing corrections. The same init pattern will carry over to
the real `LocalOperator` in Stage 6.
**Status:** Placeholder, to be replaced by `LocalOperator` in Stage 6.

### D5.4 — Operator signature: `forward(z, V_xy, edge_index, F, mask, d) -> dz`
**Decision:** All operators (placeholder, real) implement this
six-argument forward signature, even when they don't use all the
inputs.
**Where:** `horizons/models/placeholder.py`, `horizons/training/rollout.py`
(LocalOperator protocol).
**Why:** A uniform signature lets the rollout call any operator without
branching. The placeholder ignores most arguments; the Stage 6 operator
will use all of them.
**Status:** Fixed.

### D5.5 — Re-anchoring via `torch.where(mask, z_true, z_t + dz)`
**Decision:** Known vertices are re-anchored at every rollout iteration
using `torch.where`. The known branch returns `z_true` exactly; the
unknown branch returns `z_t + dz`.
**Where:** `rollout` in `horizons/training/rollout.py`.
**Why:**
- Anchoring is exact (no floating-point drift accumulating over
  iterations).
- `torch.where` is differentiable, and gradients only flow through the
  selected branch — so gradients on K vertices are structurally zero
  (verified by `test_no_gradient_through_known_vertices`).
- No in-place ops, no `.detach()` hacks, no manual gradient surgery.
**Status:** Fixed.

### D5.6 — Full-trajectory BPTT, optimizer step at end of rollout
**Decision:** The rollout keeps the full trajectory `[z^0, z^1, ..., z^N]`
in the autograd graph; the loss sums per-iteration contributions; one
backward and one optimizer step happen at the end of the rollout.
**Where:** `rollout` and `RolloutResult` in
`horizons/training/rollout.py`; loss in `horizons/training/loss.py`;
training loop to be written in Stage 8.
**Why:** Source-dictated by `final-remarks.pdf`: "weights only update at
the end of a rollout." The full-trajectory autograd graph means
gradient signal reaches the model from every iteration, not just the
final one — verified by `test_gradient_through_all_iterations`.
**Status:** Source-dictated; not open to revisiting.

### D5.7 — Per-iteration data loss uses MEAN, not SUM (deviation from source)
**Decision:** The data loss uses `mean` over $F_t$ and $P_t$, whereas
the source files write `sum`.
**Where:** `per_iteration_data_loss` in `horizons/training/loss.py`,
with an explanatory note in the docstring.
**Why:** With `sum`, late iterations naturally dominate because $P_t$
grows with $t$. With `mean`, each iteration is roughly comparable in
magnitude, and the explicit rollout weights $w_t$ control the per-
iteration balance cleanly — without an implicit interaction with
ring sizes. This makes Stage 12 ablations on $w_t$ much easier to
interpret.
**Status:** Open. If results are poor, switching to `sum` is the first
thing to try. Could be promoted to a config knob
(`loss.reduction: mean | sum`) later.

---

## Stage 6 — Real GNN operator

### D6.1 — Per-iteration feature recomputation
**Decision:** Inside `LocalOperator.forward`, vertex normals and the
umbrella Laplacian are recomputed from the *current* $z^t$ at every
rollout iteration, not held fixed from $z^0$.
**Where:** `horizons/models/operator.py`, inside `forward`.
**Why:** At $t=0$, all unknown vertices share the same mean-plane
initialization, so their normals are uniform and their curvature is
zero — the geometric features carry no signal on $U$. As the rollout
evolves $z^t$, the surface in $U$ acquires real geometry, and these
features become informative. Holding them fixed at the $t=0$ values
would waste the geometric inductive bias the GNN is meant to exploit.
**Cost:** Deepens the autograd graph (normals computation involves a
cross product and an `index_add_`; curvature uses neighbor averaging).
Both functions are differentiable (D1.5) and the path-isolated tests in
`tests/test_operator.py` confirm gradients flow through them.
**Status:** Fixed. The frozen-feature variant is a Stage 12 ablation
candidate.

### D6.2 — Architecture: 2-layer SAGEConv with mean aggregation
**Decision:** The local operator is a 2-layer SAGEConv stack with mean
aggregation, sandwiched between an input projection (9 → H) and a
2-layer MLP output head (H → H → 1).
**Where:** `horizons/models/operator.py`, `LocalOperator.__init__`;
configured via `model.n_layers`, `model.hidden_dim`, `model.aggr` in
`configs/default.yaml`.
**Why:**
- **SAGEConv with mean aggregation** is the simplest geometry-aware GNN
  operator that's compatible with our formulation (chosen in pre-
  implementation Q1). EdgeConv and GMMConv are Stage 12 alternatives.
- **2 layers** gives a 2-hop receptive field per iteration. Combined
  with feature recomputation (which uses 1-hop neighborhoods), the
  effective receptive field is 3 hops per rollout iteration. Across
  N iterations, the model sees 3N hops of structure.
- **Hidden dim 64** is the default starting point; configurable.
- **2-layer MLP output head** (rather than a single linear) gives the
  readout some non-linearity to combine SAGE features before
  projecting to the scalar Δz. Negligible parameter cost.
**Status:** All choices configurable. Architecture, layer count, and
hidden dim are Stage 12 ablation candidates.

### D6.3 — ReLU between every SAGE layer, including before the head
**Decision:** Every SAGEConv layer is followed by a ReLU activation,
including the last one (i.e. the activation before the readout MLP).
**Where:** `LocalOperator.forward`, the message-passing loop.
**Why:** The readout MLP head benefits from receiving non-linearly-
transformed features rather than the raw output of a linear-ish SAGE
aggregation. The alternative — skipping the final activation so the
head's first linear layer reads raw SAGE outputs — is a known pattern
in some architectures (e.g. transformer pre-norm), but offers no
advantage at our depth (2 SAGE layers). The simpler "ReLU after every
layer" pattern is both more readable and standard practice.
**Status:** Fixed. If we later want to test the "skip the last
activation" variant, it's a Stage 12 ablation; it only changes one
line of code.

### D6.4 — Nine-feature input vector
**Decision:** Each vertex's input to the GNN is the 9-tuple
$(x, y, z^t, n_x, n_y, n_z, \kappa, m, d)$:
- $(x, y)$: static spatial coordinates.
- $z^t$: current prediction.
- $(n_x, n_y, n_z)$: vertex normal, recomputed from current $z^t$.
- $\kappa$: umbrella Laplacian, recomputed from current $z^t$.
- $m$: known/unknown mask (1 = known, 0 = unknown).
- $d$: topological distance from $K$ (in graph hops).
**Where:** `LocalOperator.forward`, the feature assembly block.
**Why:** Each feature contributes a different signal:
- Coordinates anchor predictions in space.
- Current $z$ is the obvious self-input.
- Normals expose surface orientation.
- Curvature exposes local non-smoothness.
- Mask tells the model which vertices have ground-truth signal.
- Topological distance is the curriculum coordinate — it tells the
  model how far it is from anchored data.
**Note on information leakage from d (Q8 in design discussion):**
Including $d$ as a feature means the model *knows* which ring it's on,
and the loss schedule also indexes by $d$. This could in principle
let the model learn a degenerate "per-ring lookup" strategy. We
keep $d$ because it's in the source formulation; a Stage 12 ablation
will drop it and check whether performance collapses (signal that
$d$ is doing real work) or stays the same (signal that the model
isn't using it).
**Status:** Fixed; $d$-dropping ablation planned for Stage 12.

### D6.5 — No explicit feature standardization
**Decision:** Input features are concatenated raw (different scales:
coordinates $\sim 10$, normals $\sim 1$, mask in $\{0,1\}$, $d$
from 0 to $\sim 30$) and rely on the learnable input projection layer
to handle per-feature scaling.
**Where:** `LocalOperator.forward`, before `self.input_proj(features)`.
**Why:** The linear projection from 9 → H can learn arbitrary per-
feature scaling, so explicit batch norm / layer norm / per-feature
standardization is redundant. Explicit normalization adds another
moving part and another hyperparameter; the Stage 6 overfit experiment
showed the unnormalized version works fine (1000× loss reduction).
**Status:** Fixed for now. If real-data training (Stage 8+) shows
slow or unstable convergence, adding feature normalization is one of
the first things to try.

### D6.6 — `model_kind` selector (operator vs. placeholder)
**Decision:** A top-level config key `model_kind: operator | placeholder`
selects which model the overfit script and (later) training loop
instantiate.
**Where:** `configs/default.yaml`; `scripts/overfit_one.py`.
**Why:** Lets us re-run the keystone overfit test with the placeholder
at any time, as a baseline / regression check. Hydra's struct mode
requires the key to be declared in the config (not added with `+` at
the CLI), so it lives at the top level.
**Status:** Fixed.

---

## Stage 7 — Full loss (data + regularizers)

### D7.1 — Two regularizers added: curvature and residual
**Decision:** The full per-iteration loss is now
$L_t = L_{data,t} + \lambda_c L_{curv,t} + \lambda_r L_{res,t}$
with defaults $\lambda_c = 0.01$, $\lambda_r = 0.001$:
- $L_{curv,t}$ penalizes squared umbrella Laplacian of $z^t$ on $U$,
  encouraging locally smooth predictions.
- $L_{res,t}$ penalizes squared $\Delta z^{t-1}$ on $U$, encouraging
  gradual refinement rather than wild jumps. Defends against the
  late-training instability observed in the Stage 6 overfit experiment.
**Where:** `per_iteration_curvature_loss` and `per_iteration_residual_loss`
in `horizons/training/loss.py`; composed in `rollout_loss`.
**Why:** Both come directly from the source files. $L_{curv}$ provides
a geometric prior (the predicted surface should be locally smooth, like
real geological horizons). $L_{res}$ stabilizes the rollout dynamics by
discouraging large per-iteration corrections.
**Status:** Fixed; weights configurable for Stage 12 ablations.

### D7.2 — Regularizers also use MEAN (consistent with D5.7)
**Decision:** $L_{curv,t}$ and $L_{res,t}$ use mean over $U$, not sum
(deviation from source files, consistent with the same choice made for
$L_{data,t}$ in D5.7).
**Where:** `per_iteration_curvature_loss`, `per_iteration_residual_loss`.
**Why:** Same rationale as D5.7. Using mean keeps the magnitudes of all
three loss components broadly comparable in a per-vertex-error sense,
so $\lambda_f, \lambda_p, \lambda_c, \lambda_r$ control the relative
importance cleanly rather than getting entangled with the size of $U$.
**Status:** Open. If results are bad, the first thing to try is
switching all three to sum (and potentially making it a config knob,
`loss.reduction: mean | sum`).

### D7.3 — `rollout_loss` returns a dict, not a scalar
**Decision:** `rollout_loss` returns
`{"total": tensor, "data": tensor, "curv": tensor, "res": tensor}`.
The total is what gets `.backward()`'d; the components are for logging.
**Where:** `rollout_loss` in `horizons/training/loss.py`.
**Why:** Stage 8's training loop will log all three component values
separately, plus the total. Returning a dict avoids recomputing the
components for logging or threading multiple return values through
the caller.
**Status:** Fixed.

### D7.4 — Keep `rollout_data_loss` as a backwards-compat alias
**Decision:** The original Stage 5 `rollout_data_loss` is preserved as
a callable (computing the data-only loss), even though it's no longer
the primary training objective.
**Where:** `rollout_data_loss` at the bottom of
`horizons/training/loss.py`.
**Why:** Stage 5's tests (`tests/test_loss.py`, `tests/test_overfit.py`,
`scripts/overfit_one.py`) all use the data-only loss. Keeping the
alias means we don't have to modify those files concurrently with
adding the regularizers, which would tangle two changes together.
The alias is also a useful baseline: training with the full loss
vs. training with data-only is a natural ablation.
**Status:** Fixed for now. May be deprecated later if no longer used.

---

## Future / open decisions

These are decisions we know we need to make but haven't yet, or
ablations queued for Stage 12.

**Architecture and features (Stage 12 ablations):**
- **Curvature variant (D1.4):** umbrella → cotangent Laplacian.
- **GNN backbone (D6.2):** SAGEConv → EdgeConv or GMMConv.
- **Frozen vs. recomputed features (D6.1):** the per-iteration-recompute
  decision can be compared against holding features at $t=0$.
- **Drop topological distance feature (D6.4):** verifies whether $d$
  is doing real curriculum work or whether the model could learn
  without it.
- **Pinned ring thickness (D2.5):** 1 → 2 or 3 vertices wide.
- **Feature normalization (D6.5):** add per-feature standardization or
  layer norm if convergence is slow on real data.

**Loss design:**
- **Mean vs. sum reduction (D5.7, D7.2):** currently mean for all three
  loss components. May become a config knob if needed.
- **Per-iteration rollout weights $w_t$:** currently uniform. Stage 12
  may explore weighting later rings more.

**Data and infrastructure:**
- **GOCAD .ts loader and stratified split (D4.1):** scheduled between
  Stages 7 and 8.
- **Geometric augmentation (Tier 2):** rotation/flip/translation of
  surfaces, deferred until mask augmentation proves insufficient.

**Other:**
- **Whether to migrate dataset items to PyG `Data` objects (D5.1):** may
  revisit if Stage 9 batching is easier that way.

---

## How to use this document

When making a new design decision, add an entry to the appropriate
stage section using the format above. When revisiting an existing
decision (e.g., during Stage 12 ablations), edit the **Status** line
and add a note explaining what changed.

This document is the single source of truth for "why is the code this
way." If a future reader (yourself in three months, an advisor, a
reviewer) asks "why mean and not sum?", point them at D5.7.

For empirical findings discovered while running the code (what we
*learned* from experiments, rather than what we *chose*), see
`OBSERVATIONS.md`.
