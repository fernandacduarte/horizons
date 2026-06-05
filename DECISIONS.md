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

## Stage 4 — Data split (deferred)

### D4.1 — Defer the train/val/test split file (until after Stage 7)
**Decision:** Do not create `data/splits/split_v1.json` until the GOCAD
`.ts` loader exists and real surface data is available. Develop Stages
5, 6, 7 against the three synthetic fixtures (`plane`, `sphere_cap`,
`anticline`). Return to Stage 4 between Stages 7 and 8.
**Where:** No code; this is a sequencing decision.
**Why:** The split file requires real surface IDs and reservoir IDs.
Stages 5–7 only need a single mesh to validate their machinery (the
keystone overfit test is more diagnostic on a known fixture than on a
real horizon). Stage 8 is the first stage where multi-surface
train/val/test semantics matter, so that's the natural moment to wire
in real data.
**Plan when we get there:**
  - 4A: write `HorizonSurface.from_ts(path)` parsing GOCAD TSurf format.
  - 4B: one-time script converting all `.ts` files to `.npz` under
    `data/surfaces/` with (surface_id, reservoir_id) metadata.
  - 4C: stratified-by-reservoir 70/15/15 split written to
    `data/splits/split_v1.json`.
  - 4D: regenerate synthetic-fixture-like test fixtures from real-data
    subsamples to replace the current synthetic anticline; the
    convex-hull-boundary issue from D1.6 goes away naturally.
**Status:** Open. Scheduled for completion between Stage 7 and Stage 8.

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

## Future / open decisions

These are decisions we know we need to make but haven't yet:

- **Number and thickness of pinned rings (D2.5).** Currently 1. Stage 12
  ablation candidate.
- **Curvature variant: umbrella vs. cotangent (D1.4).** Currently
  umbrella. Stage 12 ablation candidate.
- **GNN backbone (Stage 6).** Currently planned as 2× SAGEConv with
  mean aggregation. EdgeConv and GMMConv are Stage 12 ablation
  candidates.
- **Feature recomputation per iteration (Stage 6).** Decided yes (D from
  question 3 in our pre-implementation discussion), but the per-vs.-
  frozen comparison is a Stage 12 ablation.
- **Real-data loader for GOCAD .ts files.** Format understood; loader
  to be written when we move past synthetic fixtures.
- **Geometric augmentation (Tier 2).** Decided to defer until proven
  necessary; currently doing mask augmentation only (Tier 1).
- **Loss reduction: mean vs. sum (D5.7).** Currently mean. May become
  a config knob.

---

## How to use this document

When making a new design decision, add an entry to the appropriate
stage section using the format above. When revisiting an existing
decision (e.g., during Stage 12 ablations), edit the **Status** line
and add a note explaining what changed.

This document is the single source of truth for "why is the code this
way." If a future reader (yourself in three months, an advisor, a
reviewer) asks "why mean and not sum?", point them at D5.7.
EOFcat > DECISIONS.md << 'EOF'
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

## Stage 4 — Data split (deferred)

### D4.1 — Defer the train/val/test split file (until after Stage 7)
**Decision:** Do not create `data/splits/split_v1.json` until the GOCAD
`.ts` loader exists and real surface data is available. Develop Stages
5, 6, 7 against the three synthetic fixtures (`plane`, `sphere_cap`,
`anticline`). Return to Stage 4 between Stages 7 and 8.
**Where:** No code; this is a sequencing decision.
**Why:** The split file requires real surface IDs and reservoir IDs.
Stages 5–7 only need a single mesh to validate their machinery (the
keystone overfit test is more diagnostic on a known fixture than on a
real horizon). Stage 8 is the first stage where multi-surface
train/val/test semantics matter, so that's the natural moment to wire
in real data.
**Plan when we get there:**
  - 4A: write `HorizonSurface.from_ts(path)` parsing GOCAD TSurf format.
  - 4B: one-time script converting all `.ts` files to `.npz` under
    `data/surfaces/` with (surface_id, reservoir_id) metadata.
  - 4C: stratified-by-reservoir 70/15/15 split written to
    `data/splits/split_v1.json`.
  - 4D: regenerate synthetic-fixture-like test fixtures from real-data
    subsamples to replace the current synthetic anticline; the
    convex-hull-boundary issue from D1.6 goes away naturally.
**Status:** Open. Scheduled for completion between Stage 7 and Stage 8.

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

## Future / open decisions

These are decisions we know we need to make but haven't yet:

- **Number and thickness of pinned rings (D2.5).** Currently 1. Stage 12
  ablation candidate.
- **Curvature variant: umbrella vs. cotangent (D1.4).** Currently
  umbrella. Stage 12 ablation candidate.
- **GNN backbone (Stage 6).** Currently planned as 2× SAGEConv with
  mean aggregation. EdgeConv and GMMConv are Stage 12 ablation
  candidates.
- **Feature recomputation per iteration (Stage 6).** Decided yes (D from
  question 3 in our pre-implementation discussion), but the per-vs.-
  frozen comparison is a Stage 12 ablation.
- **Real-data loader for GOCAD .ts files.** Format understood; loader
  to be written when we move past synthetic fixtures.
- **Geometric augmentation (Tier 2).** Decided to defer until proven
  necessary; currently doing mask augmentation only (Tier 1).
- **Loss reduction: mean vs. sum (D5.7).** Currently mean. May become
  a config knob.

---

## How to use this document

When making a new design decision, add an entry to the appropriate
stage section using the format above. When revisiting an existing
decision (e.g., during Stage 12 ablations), edit the **Status** line
and add a note explaining what changed.

This document is the single source of truth for "why is the code this
way." If a future reader (yourself in three months, an advisor, a
reviewer) asks "why mean and not sum?", point them at D5.7.
