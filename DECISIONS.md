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
**Status:** Fixed. (Bonus fix in Stage 8.1: `fit_mean_plane` now
promotes to float64 internally for the least-squares solve. Float32
mean/centering on UTM-scale coords loses enough precision that the
centering trick alone wasn't sufficient — the dataset's per-surface
centering path triggered the rank-deficient error. Float64 gives ~15
digits of precision which is enough for any realistic coordinate
system; the returned a, b, c are Python floats so callers see no
dtype change.)

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
**Status:** Fixed in Stage 8.1. `HorizonDataset` now applies
per-surface centering by default when `center_per_surface=True` (the
default). `__getitem__` returns the centering offsets (`xy_mean`,
`z_mean`) so downstream code can invert back to original units when
reporting metrics. The ad-hoc fix in `overfit_real.py` remains as a
working example but is no longer the only path.

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

## Stage 8 — Training loop

### D8.1 — Training loop structure: shuffle, NaN skip, per-epoch metrics, dataclass state
**Decision:** Several structural choices in `horizons/training/loop.py::train()`:

1. **Train order is shuffled each epoch with a fixed-seed RNG.** A
   `random.Random(seed)` instance is created once at the start of
   training and used to shuffle the surface order at the top of each
   epoch. Same seed → same shuffle pattern across runs.
2. **NaN losses are skipped, not fatal.** If a training step produces
   a non-finite total loss, the loop logs a warning, increments the
   step counter, and continues to the next surface. The optimizer is
   not stepped for that surface.
3. **Per-epoch metrics in history, not per-step.** `state.train_history`
   accumulates one dict per epoch with mean loss across all successful
   steps in that epoch. Per-step losses are printed to stdout but not
   stored or logged to TensorBoard (see O2 for why).
4. **TrainState as a dataclass** holds: current epoch/step, best-val
   tracking, full train_history and val_history lists. Returned to the
   caller so they can serialize, plot, or analyze the run after the
   loop finishes.

**Where:** `horizons/training/loop.py`.

**Why:**
1. **Shuffling** prevents the model from learning surface-order
   artifacts; **fixed seed** keeps runs reproducible. Train masks
   already vary per epoch (D5.2), so shuffling adds independent
   variation that helps generalization.
2. **NaN-skip** beats crashing: a single pathological surface
   shouldn't abort a 100-epoch run. The warning makes it visible in
   logs so we can investigate later. If NaN persists across surfaces,
   that's a different problem (LR too high, etc.) that will surface
   anyway.
3. **Per-epoch granularity** matches what's meaningful for human
   interpretation; the per-step view is too noisy (see O2).
4. **Dataclass state** is cleaner than threading 5+ tuples through
   the loop or stashing things on `self`.

**Status:** Fixed.

### D8.2 — TensorBoard logging scope
**Decision:** TensorBoard logs:
- `train/loss_total`, `train/loss_data`, `train/loss_curv`,
  `train/loss_res` (per-epoch means).
- `train/lr` (current learning rate, per-epoch).
- `val/loss_total`, `val/loss_data`, `val/loss_curv`, `val/loss_res`
  (means over val set, on val epochs).
- `val/rmse_meters` (mean RMSE on U across val set, on val epochs).
- `val_rmse_per_surface/<surface_id>` — RMSE for each val surface.
- `val_rmse_per_reservoir/<reservoir_id>` — mean RMSE within each
  reservoir group represented in val.

Things explicitly NOT logged:
- Per-step train loss (too noisy, see O2).
- Gradient norms or weight statistics.
- Activations or intermediate feature distributions.

**Where:** `horizons/training/loop.py`, the conditional `if writer
is not None` blocks at end of train epoch and end of val epoch.

**Why:**
- **Per-surface RMSE** lets us see *which* surfaces are hard. Without
  it, an aggregate val RMSE could hide that 1 of 7 surfaces is
  catastrophically bad. The 8.3 smoke run already showed RMSE
  varying 30× across surfaces (43m to 1556m on the same epoch).
- **Per-reservoir means** group the surfaces into the categories from
  D4.4, letting us see if some reservoirs systematically underperform.
- **Per-epoch granularity** for train metrics — see O2 for why per-step
  is unusable.
- **Gradient/weight/activation logging** is for debugging specific
  issues, not for normal monitoring. We can add later if needed.

**Run directory naming:** `outputs/tensorboard/run_<YYYYMMDD_HHMMSS>/`.
Each run also writes `config.yaml` (snapshot of the Hydra config used)
and `summary.json` (best-val and top-line numbers) alongside the
events file. This lets us answer "what hyperparameters produced this
curve?" weeks later without git archaeology.

**Status:** Fixed for Stage 8. Will add gradient-norm logging in
Stage 8.5 if LR scheduling needs debugging.

---

## Stage 9 — Gradient accumulation

### D9.1 — Gradient accumulation: divide each loss by actual batch size, clip after
**Decision:** Training accumulates gradients across `accum_steps`
surfaces before each optimizer step. The accumulation is implemented
as a nested loop in `horizons/training/loop.py::train()`:

```python
for batch_start in range(0, n_train, accum_steps):
    batch_indices = order[batch_start : batch_start + accum_steps]
    batch_size = len(batch_indices)  # may be smaller than accum_steps on
                                     # the last batch of the epoch

    optimizer.zero_grad()
    for idx in batch_indices:
        loss = ... (forward + rollout_loss)
        if not torch.isfinite(loss):
            continue                        # skip surface; don't skip batch
       ss / batch_size).backward()      # accumulate the MEAN gradient

    if n_successful_in_batch > 0:
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
```

The choices encoded here:

1. **Divide each loss by the actual `batch_size`, not by `accum_steps`.**
   The last batch of an epoch may be smaller than `accum_steps`
   (e.g., 30 surfaces / 4 = 7 full batches + 1 partial batch of 2).
   Dividing by the actual size keeps the accumulated gradient equal
   to the true *mean*, regardless of batch size. Dividing by
   `accum_steps` instead would under-scale partial batches and bias
   the optimizer toward them.

2. **Gradient clipping is applied to the accumulated (mean) gradient,
   once, before the step.** Not per-surface. The clip threshold is
   independent of batch size this way — same `grad_clip_norm=1.0`
   works for `accum_steps=1` and `accum_steps=4` alike.

3. **NaN handling is at two levels.** A non-finite loss on one
   surface causes us to skip that surface but continue the batch.
   If *all* surfaces in a batch produced NaN losses, the entire
   batch is skipped (no optimizer step). This keeps the loop robust
   to pathological surfaces without losing legitimate updates from
   other surfaces in the same batch.

4. **End-of-epoch metric is mean per successful *surface*, not per
   *optimizer step*.** The train_record reports `loss_total` as the
   sum of per-surface losses divided by `n_successful_surfaces`.
   This keeps numbers directly comparable across different
 `accum_steps` values: a per-surface average is invariant under
   batch size, but a per-step average would scale up with batch size.

**Where:** `horizons/training/loop.py`, the batch loop in `train()`.

**Why:**
- Per-surface losses span ~6 orders of magnitude (O2). Without
  accumulation, every optimizer step is dominated by whichever
  surface happened to be drawn, making the optimizer's signal
  high-variance. Accumulating across a mini-batch averages this out.
- Dividing by `actual batch_size` (rather than `accum_steps`) is the
  same trick used by standard PyTorch trainers; it's the only way
  to handle partial-batch edge cases without bias.
- Per-surface metrics for logging are essential because we want to
  compare runs at different `accum_steps` values; per-step metrics
  would conflate batch size with training quality.

**Trade-off:** With `accum_steps=4` and 30 train surfaces, we get
only ~8 optimizer steps per epoch (vs. ~30 without accumulation).
The model receives 4× fewer updates per epoch ut each is more
stable. Whether this is a net win depends on the data; the A/B
result will be recorded in OBSERVATIONS.md once Stage 9.3 completes.

**Status:** Fixed. The choice of `accum_steps` itself remains a
hyperparameter (Hydra `optim.accum_steps`, default 4).

---

## Stage 11 — Hyperparameter improvements (planning)

This section is different from the D*.* "decisions" above: it's a
**planning document for candidate improvements** we intend to try
before the final test evaluation. Each candidate has a hypothesis
(why we think it will help), a plan (what we'll change), and a
results slot (what actually happened, filled in after the run).

The motivation comes from O5: our model wins on `half_plane` but
loses to harmonic infill on `outward_pinned`, and fails badly on
`10_BaseModelo` (a perfectly flat surface where the answer is
zero). The goal is to close those gaps while preserving the
`half_plane` advantage, so that we can defensibly justify the use
of a neural network across regimes.

Candidates are ordered by expected impact / effort ratio.

### Candidate 1 (Tier 1): Increase regularizer weights (λ_c, λ_r)

**Hypothesis:** At the current best epoch, loss_curv ≈ 1500 and
loss_res ≈ 30. With λ_c=0.01 and λ_r=0.001, their contributions to
total loss are 15 and 0.03 — completely dominated by the data loss
of ~322,000. The regularizers are essentially turned off. By
increasing λ_c we force the model to be smoother, which should
help `outward_pinned` (interpolation between anchors, where
smoothness is the right inductive bias). By increasing λ_r we
penalize large per-iteration Δz, which should help the flat-surface
case where the answer is "do nothing."

**Plan:**
- Run 3 configurations to sweep:
  - `λ_c=0.1, λ_r=0.01` (10× current).
  - `λ_c=0.5, λ_r=0.05` (50× current).
  - `λ_c=1.0, λ_r=0.1` (100× current).
- All other hyperparameters as in Stage 9 B=4 baseline.
- Full 100-epoch training with patience=20.
- Use the Stage 10 evaluation suite on val for comparison.

**Diagnostic:** Watch the per-regime breakdown carefully. We
*expect* `outward_pinned` to improve. We expect `half_plane` to
get slightly worse (smoothness penalty hurts extrapolation
fidelity). The trade-off needs to favor `outward_pinned`
significantly more than it hurts `half_plane`.

**Risk:** Too much smoothness regularization could collapse the
model to "always output harmonic infill," which beats us on
`outward_pinned` but loses our `half_plane` advantage.

**Results (Stage 11.1):** Negative.
Ran with `λ_c=0.1, λ_r=0.01`. Compared to Stage 9 B=4 baseline
(λ_c=0.01, λ_r=0.001):

| Regime | baseline mean RMSE | λ_c=0.1 mean RMSE | Δ |
|---|---|---|---|
| half_plane | 133.7 | 155.6 | +21.9 (worse) |
| outward_free | 21.2 | 20.1 | −1.1 (similar) |
| outward_pinned | 95.5 | 106.3 | +10.8 (worse) |
| Overall | 101.4 | 115.7 | +14.3 (worse) |

The intended effect (improve `outward_pinned` via smoothness) did
not materialize. All regimes got worse or stayed similar.

**Interpretation:** Penalizing the per-iteration umbrella
Laplacian of z is not the same as forcing the final z to be
smooth. The harmonic infill baseline minimizes the *final* z's
Laplacian via a global linear solve; our regularizer constrains
the *per-iteration* output of an iterative learned operator,
which is a different mathematical object. The two notions of
"smoothness" aren't equivalent, and the latter doesn't push
toward the former.

In addition, stronger regularization slowed convergence: best
epoch moved from 28 (baseline) to 47 (this run), and training
early-stopped later (67 vs 38). Some of the regression may be
training inefficiency, but the directional finding is robust:
this lever is not the right one for the `outward_pinned` gap.

**Decision:** Skip Candidates 1b and 1c. The trade-off curve
goes the wrong direction; sweeping λ_c higher will likely make
things worse, not better. Move directly to Candidate 2.

**Lesson:** Closing the `outward_pinned` gap requires a stronger
inductive bias toward smoothness *as input*, not as a loss term.
Candidate 2 (harmonic init) is exactly that — give the model
harmonic infill as the starting point and let it learn corrections.

**Where the run lives:** `outputs/tensorboard/run_20260613_175830/`.
Evaluation: `outputs/evaluation/run_20260613_175830_val.json`.

---

### Candidate 2 (Tier 1): Harmonic infill as initialization

**Hypothesis:** Currently the model has to learn extrapolation
from a mean-plane init, which is a poor starting point on
non-trivial geometries. If we instead initialize z⁰ to be the
harmonic infill (Stage 10.2's baseline), the model only needs to
learn the *residual* from a strong baseline. Mathematically, this
is iterative refinement from numerical analysis. Practically, it
gives us harmonic infill's performance as a floor; the model can
only do better, not worse (if trained correctly).

**Plan:**
- Modify `horizons.data.init` to add a `harmonic_init()` function.
- Add a config flag `init.method: meanplane | harmonic` (default
  `meanplane` for now; change to `harmonic` for this experiment).
- Run with the best Candidate 1 config (i.e., whatever regularizer
  weights worked best).
- Compare against the Candidate 1 result.

**Diagnostic:** The model's predictions should be very close to
harmonic infill at the beginning of training (Δz ≈ 0 produces the
input z⁰). Over training, it should learn corrections. If the
model learns to *only* output zeros (Δz = 0 everywhere), we get
exactly harmonic infill — that's the worst case, and it's still
much better than our current `outward_pinned` performance.

**Risk:** Implementation complexity. Harmonic infill requires
solving a sparse linear system per surface, which adds overhead.
We'll need to pre-compute it during dataset construction or
cache it. Also: harmonic infill needs the FULL z_true on K to
compute, but during training the K varies per epoch. We need to
recompute harmonic infill per mask, not per surface.

**Preliminary investigation (Stage 11.5.1):** Before training, we
compared harmonic vs mean-plane init RMSE on val surfaces across
all three regimes (21 records: 7 surfaces × 3 masks). Finding:
harmonic init is NOT universally better.

| Regime | meanplane init RMSE | harmonic init RMSE | Gap |
|---|---|---|---|
| half_plane (8 non-flat records) | ~150 | ~190 | harmonic 40 worse |
| outward_free (1 non-flat) | 77 | 92 | harmonic 15 worse |
| outward_pinned (2 non-flat) | 178 | 97 | harmonic 81 better |

Harmonic init starts the model substantially BETTER on
`outward_pinned` (interpolation between anchors, where harmonic's
smoothness assumption is right) but WORSE on `half_plane` and
`outward_free` (extrapolation regimes, where the truth has
non-harmonic global trends that harmonic infill suppresses).

**Implication:** A universal switch to harmonic init might hurt
`half_plane` performance while helping `outward_pinned`. The
honest expectation is mixed: the model would need to learn to
"undo" the smoothness bias on extrapolation regimes.

**Decision:** Defer running Candidate 2 until after Candidate 9
(coordinate normalization). The reasoning: normalization is a
more fundamental fix that should improve all regimes; once that's
in place, we can revisit Candidate 2 with a better baseline.

**Results (Stage 11.7-redo):** Mixed.
With harmonic init on top of normalization, the GNN essentially
matches harmonic infill on outward_pinned (36.6 vs 33.7), but
loses some ground on half_plane (151.7 vs 134.0 baseline) and
outward_free (21.5 vs 14.6). Overall mean is slightly worse
(94.0 vs 89.3).

We adopt meanplane init as the final default since it has lower
overall mean and better half_plane. Harmonic init is documented
as a useful ablation in O7 (OBSERVATIONS.md).

Methodological side-finding from this experiment: best-checkpoint
selection should use val_rmse_meters rather than val_loss. The
val_loss criterion was masking continued model improvement
because it has regularizer noise that val_rmse doesn't. This was
fixed by adding the `train.best_metric` config field, default
`val_rmse_meters`.

### Candidate 9 (NEW — Tier 1): Coordinate normalization

**Hypothesis:** Currently we center x, y, z per surface (D4.6,
Stage 8.1) but don't normalize the scales. After centering,
coordinates still span roughly [-500, +500] meters in x, y and
[-1000, +1000] in z. The umbrella Laplacian feature has a
different scale entirely (z / length²). Mixing these wildly
different scales in a neural network is suboptimal — gradient
updates depend on input scale, so the model has to spend
parameters learning to compensate for scale differences.

Normalizing all spatial coordinates to roughly [-1, +1] per
surface should:
1. Make features uniform in scale → better-conditioned
   optimization.
2. Make the model scale-invariant → same architecture handles
   surfaces of any geographic extent.
3. Improve numerical stability → no precision issues.

**Plan:**
- Add `xy_scale` and `z_scale` to `HorizonDataset` output (the
  L∞ extent of each coordinate, computed per surface).
- Normalize V[:, :2] by dividing by xy_scale, V[:, 2] by z_scale.
- The model's output Δz is in normalized units; we denormalize
  by multiplying by z_scale before computing RMSE in meters.
- Update `validate()` and `evaluate_surface()` to handle the
  denormalization consistently.

**Diagnostic:** Watch the training curves. Normalization should
make all loss components more similar in scale (currently
loss_data is millions, loss_curv is thousands). The model should
also train more uniformly across regimes (less per-surface RMSE
variance once everything is in O(1)).

**Risk:**
- The xy_scale and z_scale are per-surface, so the model's
  predictions are now also per-surface scale. We have to
  denormalize correctly when reporting in meters.
- The curvature feature scaling changes too. We may need to
  recompute the umbrella Laplacian after normalization (it's
  a function of z, not just connectivity).
- All metric values (RMSE in m) become invariant under
  normalization, but intermediate "loss numbers" in TensorBoard
  will shift. We should retain the meters scale for RMSE.

**Results (Stage 11.6):** **Substantial win.** Full results in
OBSERVATIONS.md O6. Summary:
- overall mean RMSE: 101.4 → 89.3 (−12%)
- outward_pinned mean: 95.5 → 57.3 (−40%)
- outward_free mean: 21.2 → 14.6 (−31%)
- half_plane: essentially unchanged
- median RMSE on outward_pinned: 79 → 0.45 (most surfaces
  now solved to sub-meter accuracy)

The GNN model now has the lowest overall mean across the three
methods evaluated (vs mean-plane and harmonic). Normalization
is adopted as default. The remaining limitation is
half_plane, which we hypothesize is architectural rather than
optimization-related (the deep-extrapolation problem cannot be
fixed by better feature scaling alone).

### Candidate 3 (Tier 2): Mask augmentation

**Hypothesis:** Sample 2-3 masks per surface per epoch instead of 1.
This triples the effective training data and exposes the model to
more mask variations, improving generalization.

**Plan:** Modify `HorizonDataset` to optionally sample multiple
masks per surface per epoch. Default 1 (current behavior), config
flag to set 2-3.

**Risk:** Per-epoch cost increases proportionally.

**Results (Stage 11.8):** Win.
With n=3, all regimes improved or stayed stable:
- Overall mean: 89.3 → 82.8 (−7%)
- Overall median: 57.0 → 38.3 (−33%)
- half_plane mean: 134.0 → 125.4 (−6%)
- outward_free mean: 14.6 → 7.4 (−49%)
- outward_pinned mean: 57.3 → 55.0 (−4%)

The GNN now beats both baselines on every aggregate metric for
the first time. n=3 is now the default for training.

Particularly notable: half_plane improved. This partially
refutes O6's hypothesis that half_plane is purely architectural
— more diverse mask exposure helps. Suggests room for further
improvement via even higher augmentation (Candidate 3.5: n=5).

### Candidate 4 (Tier 2): Longer training with smaller LR

**Hypothesis:** Our B=4 run early-stopped at epoch 48 of 100. With
a smaller LR (e.g., 5e-4 instead of 1e-3) and more patience, the
model might find a better basin.

**Plan:** Run with lr=5e-4, patience=40, n_epochs=200.

**Risk:** Pure compute cost; could also just confirm we've already
converged.

**Results:** _(to be filled in)_

### Candidate 5 (Tier 2): Rebalance regime weights toward `outward_free` (revised)

**Original hypothesis** (pre-O8): Rebalance regime weights toward
`outward_pinned`, since that was the regime where we trailed
classical baselines.

**Revised hypothesis** (post-O8, post Stage 11.9): After conversation
with the project owner about deployment scenarios, the priority
changed. At inference time, the dominantse case is "extrapolate
from a central observation region outward to a bounding box, with
no anchors at the boundary" — which is precisely `outward_free`.
Secondary cases involve boundary anchors (`outward_pinned`) or
fault-like discontinuities (`half_plane`).

This reframes the optimization target: **we should specialize the
model toward `outward_free` while preserving reasonable performance
on the others.**

**Plan:**
- Change regime weights from 30/40/30 to 20/60/20 (half_plane /
  outward_free / outward_pinned).
- Retrain with otherwise the best config so far (Stage 11.8: n=3,
  lr=1e-3, normalize=true, init=meanplane).
- Re-evaluate. The headline metric is now `outward_free mean RMSE`
  rather than the overall mean.

**Risk:**
- half_plane and outward_pinned will likely get worse.
- Acceptable if outward_free improves enough to make the trade
  worthwhile for deployment.
- Risk of overspecializing: if half_plane regresses dramatically
  (e.g., above 200m mean), we may have gone too far.

**Methodologal notes:**
- The val sampler during training also uses the new weights, so the
  val_rmse_meters that drives early-stopping is now biased toward
  the deployment regime. This is *correct* — we want the model
  selection criterion to track the regime we care about.
- The post-training `evaluate_split` (in `eval_run.py`) keeps the
  default 30/40/30 regime weights. This makes evaluation results
  directly comparable across runs, even when training-time weights
  differ.
- See also the eval methodology change in Stage 11 (below) about
  bumping `--n-masks` from 3 to 10 for statistical power on the
  underrepresented outward_free regime.

**Results (Stage 11.10):** Negative.
With weights 20/60/20, every regime got slightly worse (-2 to -9m),
including the targeted outward_free regime. Documented in O10.

Hypothesis on why: regime diversity during training may matter for
generalization, even when only one regime matters at deployment.
Restricting mask geometry diversity may have narrowed the model's
learned skills.

**Decision:** Revert to 30/40/30 default regime weights. Stage 11.8
remains the best model.

### Candidate 6 (Tier 3): Larger model

**Hypothesis:** We have 21k parameters. Modern GNNs often use
100k+. Maybe we're capacity-limited.

**Plan:** Try hidden_dim=128 (vs current 64), and/or
n_message_passing=3 (vs current 2).

**Risk:** Slower training; could also just overfit our small
dataset.

**Results:** _(to be filled in)_

### Candidate 7 (Tier 3): EdgeConv or GAT operator

**Hypothesis:** SAGEConv is a basic choice. Edge-conditioned
operators or attention-based GNNs might capture geological
features better.

**Plan:** Implement an EdgeConv-based LocalOperator variant.

**Risk:** Significant refactoring; impact unclear.

**Results:** Implemented in Stage 12 (D12.1). EdgeConv (aggr=max) added as a
`conv_type` option on `LocalOperator`; it required gradient checkpointing
(D12.2) to fit in memory (O17 — EdgeConv OOM'd at 48k vertices). Accuracy
vs the Stage 11.8 baseline and the O16 noise floor: O18 (pending the running
experiment). GAT not yet tried.

### Candidate 8 (Tier 3): Cotangent Laplacian

**Hypothesis:** The umbrella Laplacian (D1.6) ignores mesh angles.
The cotangent Laplacian respects local geometry and is more
"correct" for surfaces. Could give better curvature features.

**Plan:** Implement cotangent Laplacian as an alternative
`compute_umbrella_laplacian` function; use as a feature.

**Risk:** Numerical issues at sliver triangles; more complex code.

**Results:** _(to be filled in)_

### Execution order

Plan: do Candidates 1 and 2 first (cheap, directly target failure
modes). Then revisit the candidate list based on what we learn:
- If Candidate 1 already closes most of the `outward_pinned` gap,
  Candidate 2 may not be needed.
- If neither closes the gap, we'll know the issue is architectural
  (Candidates 6, 7, 8) rather than hyperparameter-related.

After improvements, run the final evaluation on **test_id and
test_ood** with the chosen final config (Stage 10 final).

If after all attempts, the model still loses to harmonic infill on
`outward_pinned`, that's a legitimate empirical finding that
deserves a clean writeup in the thesis: "neural networks add value
in extrapolation regimes but not in interpolation regimes; for
the latter, classical methods remain the right tool."

---

## Stage 11.10 — Evaluation methodology improvements

### D11.10.1 — Bump default `n_masks_per_surface` for eval from 3 to 10
**Decision:** When evaluating on val (or test), use 10 masks per
surface by default, not 3.

**Where:** `scripts/eval_run.py` CLI flag `--n-masks`. The default
value of `n_masks_per_surface` in `evaluate_split()` remains 3 for
backward compatibility with scripts that pass it explicitly, but
the eval_run CLI now defaults to 10.

**Why:**
- With 7 val surfaces × 3 masks = 21 records and default regime
  weights 30/40/30, the expected per-regime counts are ~6 / ~8 / ~6.
  Actual counts vary due to sampling variance. In several
  experiments we ended up with only **4 records** in `outward_free`
  — barely enough to compute a mean, certainly not enough for
  median or per-ring breakdowns.
- With 10 masks per surface, toords grow to 70 and expected
  per-regime counts are ~21 / ~28 / ~21. Outward_free becomes
  statistically reliable.
- Eval is fast (no backward pass, no gradient computation), so the
  extra cost is small (~5-10 minutes instead of ~2 for val
  evaluation).

**Why not change `evaluate_split`'s default itself:** Existing test
suites and historical experiments use n=3. Changing the function
default would invalidate those. The CLI default change for `eval_run`
covers the common case (interactive eval after training) while
leaving the function contract intact.

**Status:** Fixed for `eval_run.py` from Stage 11.10 onward.

### D11.10.2 — Eval sampler keeps default regime weights regardless of training
**Decision:** `evaluate_split()` uses `MaskSamplerConfig()` (default
30/40/30 regime weights), independent of the training config.

**Where:** `horizons/eval/driver.py::evaluate_split` does not read
the training config's mask weights; it instantiates a default
`MaskSamplerConfig`.

**Why:**
- Different training eeriments may use different regime weights
  (e.g., Stage 11.10's 20/60/20 vs default 30/40/30). If the
  evaluation also used the experimental weights, **per-regime mean
  RMSE could not be directly compared across experiments** because
  the mean would be over different subsets of (surface, mask) pairs.
- By using the same eval distribution everywhere, the per-regime
  mean RMSE is a comparable cross-experiment number.
- The "fairness" concern (training on 20/60/20 but evaluating on
  30/40/30) is illusory: the model's per-regime mean RMSE is a
  property of the regime, not of how often the sampler picks it.
  Re-weighting the sampler only changes the overall mean (which is
  why the overall mean is no longer the headline metric in Stage
  11.10).

**Status:** Always fixed (this is how it was built); now explicitly
documented because Stage 11.10's training/eval weight divergence
makes it a deliberate design point rather than an accident.

---

## Stage 11.7 — Checkpoint selection metric

### D11.7.1 — Use `val_rmse_meters` as the early-stop and best-checkpoint criterion

**Decision:** During training, the best checkpoint and early-stop
criterion are driven by **val_rmse_meters** (RMSE in meters on
the held-out validation surfaces' unknown vertices), not val_loss.

**Where:** `train.best_metric: val_rmse_meters` in
`configs/default.yaml`. Implemented in `horizons/training/loop.py`.
Selectable via CLI override if a comparison run is needed.

**Why we changed from val_loss to val_rmse_meters:**

1. **val_loss includes regularizer terms** (data fit + λ_s × smoothness +
   λ_c × curvature + λ_r × residual norm). The regularizers add
   per-step variance from their own gradient noise, not from
   prediction quality. We saw this empirically in Stage 11.7 — the
   model's val_loss pattern (epoch 3: 6.71, epoch 12:
   8.19, epoch 24: 7.01) that masked underlying improvement in
   val_rmse_meters.
2. **val_rmse_meters is what we report.** Aligning the
   checkpoint-selection criterion with the reported metric is more
   honest than selecting by a composite signal we don't ultimately
   evaluate against.
3. **At inference time, only RMSE matters.** Users want accurate
   predictions. Smoothness regularization shapes training but does
   not feature in deployment-time evaluation.

**Trade-offs we accept:**

- val_rmse_meters optimizes data fit only; the selected checkpoint
  may have slightly worse smoothness/curvature properties than one
  selected by val_loss.
- The smoothness/curvature penalties are still active **during
  training** (the model is pushed toward smooth outputs at every
  step), so the selected model is not maximally rough — it just
  isn't *additionally* selected for smoothness at the final
  checkpoint.
- This trade is acceptable because (a) the reported metric is
  RMSE(b) we have direct evidence that val_loss is noisier and
  can mislead checkpoint selection.

**For the thesis writeup:**

A reasonable framing is: "We selected the best checkpoint by
val_rmse_meters since this is our reported evaluation metric. The
alternative would be a composite val_loss including the
regularization penalties used during training; we chose val_rmse
to align checkpoint selection with how the model is evaluated
and how it will be used at inference. The regularizers remain
active during training, ensuring the model converges toward smooth
predictions, but they are not used as a selection criterion
afterward."

**Status:** In effect from Stage 11.7 onward. All subsequent runs
use val_rmse_meters as the criterion.

---

## Stage 12 — Alternative operators and memory engineering

This stage begins the architecture ablations deferred from Stage 11 (see
Candidates 7/8 and the "Future / open decisions" list). The first two
entries add an alternative message-passing operator (EdgeConv) and the
gradient-checkpointing machinery needed to train it.

### D12.1 — Operator selector: `conv_type` ("sage" | "edgeconv")

**Decision:** `LocalOperator` now takes `conv_type` (default "sage") and
`aggr` (default "mean") and builds its message-passing stack through a
`_make_conv(conv_type, hidden_dim, aggr)` helper. "sage" returns the
existing `SAGEConv(H, H, aggr)`; "edgeconv" returns
`EdgeConv(MLP(2H→H→H), aggr)`. `forward()` is unchanged — both layers share
the `conv(h, edge_index)` signature. Wired through `cfg.model.type` /
`cfg.model.aggr` (previously a dead config stub) in `scripts/train.py`, and
through `load_checkpoint(..., conv_type, aggr)` so eval rebuilds the right
architecture from the saved `config.yaml`.

**Where:** `horizons/models/operator.py` (`_make_conv`), `scripts/train.py`,
`horizons/eval/checkpoint.py`, `scripts/eval_run.py`, `scripts/noise_band.py`.
Test: `tests/test_operator.py::TestConvType`.

**Why EdgeConv:** every Stage 11 lever (width O12, depth O13, augmentation
O9, regime weights O10, λ_c Stage 11.1, LR, more data O14/O15) moved val
RMSE by less than the noise floor — but all were "more SAGE." SAGE's update
`W1·h_i + W2·mean(h_j)` is an *averaging* operator; a weighted mean of
neighbours cannot extend a trend, only smooth it. EdgeConv's edge message
`hΘ([h_i, h_j − h_i])` carries the explicit neighbour *difference*
`h_j − h_i` — a local-gradient inductive bias, which is what extending a
geological dip outward needs. It is the one genuinely different axis on the
model side. `hΘ` is an MLP (not a single linear) because a linear edge
function collapses EdgeConv back to an affine aggregator; the
ReLU-separated MLP is the DGCNN (Wang et al. 2019) form and keeps params
comparable (29.7k vs SAGE's 21.4k).

**aggr="max" for the first run:** canonical EdgeConv uses max
(salient-difference selection), the genuinely different operator.
`aggr="mean"` (smoother local-gradient flavour, one variable from SAGE) is
the queued follow-up if max is borderline.

**Status:** Implemented. First experiment (EdgeConv, aggr=max, seed=42,
otherwise Stage 11.8 config) running; accuracy result to be recorded in
O18. "gmm" is accepted in the config vocabulary but not implemented (raises
ValueError).

### D12.2 — Gradient checkpointing in the rollout

**Decision:** `rollout()` takes `use_checkpoint: bool = False`. When true,
each per-step `model(...)` call is wrapped in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, so the
operator's activations are recomputed during backward instead of retained.
Threaded through `train()` (`loop.py`) and `cfg.train.grad_checkpoint`
(`train.py`, default false). Eval is unaffected — it runs under
`@torch.no_grad`, so no graph is retained.

**Where:** `horizons/training/rollout.py`, `horizons/training/loop.py`,
`scripts/train.py`, `configs/default.yaml` (`train.grad_checkpoint`).
Test: `tests/test_rollout.py::TestCheckpoint::test_checkpoint_is_transparent`.

**Why:** the trajectory-based rollout retains the full N-step graph for BPTT
(D5.6), so peak training memory is O(V × N × H) — the wall documented in
O15. EdgeConv makes it far worse: it materialises a feature vector *per
edge* (~6× the vertices, ~4× the width), so its peak is ~24× SAGE's and it
OOM'd at just 48k vertices on 16 GB (O17). Checkpointing drops retention
from O(N) to O(1) in the rollout depth — EdgeConv's peak on a 48k surface
falls from ~30 GB to under 1 GB — at the cost of one extra forward in
backward (~1.5–2× compute). It is mathematically transparent (the test
asserts identical output, loss, and parameter gradients vs the plain
rollout), so a checkpointed run is a fair comparison to a non-checkpointed
one.

**`use_reentrant=False` specifically:** the reentrant variant drops
parameter gradients when no *input* tensor requires grad — exactly our
step-0 case (z⁰ is detached init; only the weights need grad). The
non-reentrant variant tracks the params correctly.

**Bonus — reopens the large-surface question:** the O15 data-diversity
experiments were blocked by the same O(V × N × H) wall. With checkpointing,
the V>50k surfaces become trainable within 16 GB (CPU) regardless of depth,
so the data-diversity hypothesis can be revisited cleanly (deferred).

**Status:** Implemented and tested. Default off (SAGE runs pay no recompute
cost); enabled for EdgeConv and any future memory-heavy operator or
large-surface run.

### D12.3 — Phase-2 split: large surfaces, magnitude-balanced

**Decision:** A new canonical split (`data/splits/split_v2.json`) for Phase 2
that adds the eight V>50k, V≤600k surfaces (110k–455k) to the Stage-4 small
split, placed by hand to balance mesh magnitude across sets:
- **train (+4):** 15TopoSal (195k), 16TopoAndarAlagoas (192k),
  03TopoOligoMioceno (230k), 01FundoMar (455k)
- **val (+2):** 04BaseOligoMioceno (110k), 02TopoMioceno (443k)
- **test_id (+2):** 07TopoCenomaniano (165k), 06TopoCretaceoSuperior (412k)

Each set gets one ~400k+ surface plus moderate-large ones, so train / val /
test_id all span the size range. The two >600k giants (610k, 673k) are
excluded for tractable epoch time. Result: train 34, val 9, test_id 7,
test_ood 5 (R7, unchanged).

**Where:** generated reproducibly by `scripts/build_phase2_split.py` (explicit,
non-seeded placement); the resulting `split_v2.json` is committed — unlike
O15's split, which lived only on the container and was lost.

**Why hand-balanced rather than stratified:** with only 8 large surfaces a
random/stratified draw can clump the big ones into one set; explicit balance
guarantees each set tests a very-large surface, needed for a faithful
generalization claim across scales.

**Trade-off accepted:** the 443k surface in val dominates the val aggregate and
widens its run-to-run band, lowering tuning sensitivity (cf. O16). Mitigated by
always reporting the per-surface decomposition.

**Status:** Phase-2 canonical split. Phase-1 results were on the prior
7-surface val and are not directly comparable; each phase is baselined on its
own val.

### D12.4 — Non-finite gradient guard in the training loop

**Decision:** Before clipping and the optimizer step, the loop checks that every
accumulated parameter gradient is finite; if any is NaN/Inf it zeroes the grads,
logs a warning, and skips the step — the same philosophy as the existing
non-finite-*loss* skip (D8.1).

**Where:** `horizons/training/loop.py` (batch loop, before `clip_grad_norm_`).
Test: `tests/test_loop.py::test_nonfinite_gradient_guard_preserves_weights`.

**Why:** Phase-2's first run died at epoch 91 after 90 clean epochs. A *finite*
loss back-propagated a *non-finite* gradient — most likely the `1/‖n‖` term in
vertex-normal normalization (`compute_vertex_normals`, eps=1e-12) blowing up on
a near-degenerate normal during a deep rollout, far more likely now with the
large surfaces' N≈150–200 rollouts. The existing guards could not catch it: the
loss-finiteness check skips NaN *losses*, not finite-loss/NaN-gradient cases,
and `clip_grad_norm_` cannot sanitize NaN/Inf (clipping by a NaN norm yields NaN
grads). So the poisoned gradient flowed into `optimizer.step()` and turned every
weight to NaN; thereafter every surface NaN'd, the loop skipped them all, and
the epoch aborted with "No successful surfaces." The guard makes a non-finite
*gradient* a skip, exactly like a non-finite loss, so one pathological batch can
no longer corrupt the weights.

**Trade-off:** a genuinely informative-but-explosive batch is dropped rather
than clipped. Acceptable — a corrupted run loses everything, a skipped batch
loses one update. If skips become frequent, the root-cause fix is to raise the
normals `eps` (1e-12 → ~1e-6), bounding the normalization gradient at source.

**Status:** In effect from Phase 2. Default-on (no flag); applies to every run.

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
