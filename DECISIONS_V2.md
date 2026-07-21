# DECISIONS — V2 campaign (extended-rollout experiments)

This document tracks a **new experimental campaign**, separate from the phases
recorded in `DECISIONS.md` and `OBSERVATIONS.md` (Phase 1 small-data, Phase 2
full-data rollout, Phase 3 hybrid). It reuses the **Phase-2 approach**
(`approach=rollout` — the standard full-depth rollout, no hybrid) on the
full-data split (`split_v2`) as its baseline, and adds two configurable
modifications to the rollout and its loss.

Everything documented in `DECISIONS.md` (data pipeline, mask sampler, features,
model, harmonic baseline, per-surface centering, etc.) still holds. Only the
items below are new. Both new knobs **default to the old behavior**, so
`rollout.n_multiplier=1` + `loss.equal_ring_weight=false` reproduces the Phase-2
baseline exactly.

Format matches `DECISIONS.md`: **Decision / Where / Why / Status.**

---

## Baseline inherited for this campaign

- **Approach:** `approach=rollout` (standard full-surface-depth rollout; no hybrid).
- **Split:** `split_v2` — 34 train / 9 val / 7 test_id / 5 test_ood.
- **Model, features, masks, optimizer, loss regularizers (λ_c, λ_r):** unchanged.
- **Metric:** RMSE on the unknown set U (meters).

---

## DV2.1 — Extended rollout depth: `k · N` iterations

**Decision:** The standard rollout runs a configurable **k × surface_depth**
iterations instead of exactly `N = surface_depth`. `k` is the hyperparameter
`rollout.n_multiplier` (default 1).

**Where:**
- Config: `rollout.n_multiplier` in `configs/default.yaml`.
- Threaded `scripts/train.py` → `train()` (`loop.py`) → `validate()`
  (`validate.py`); and for post-training evaluation `evaluate_surface`
  (`per_surface.py`) via `evaluate_split` (`driver.py`). At each rollout
  N-decision: `N = max(1, round(rollout_n_multiplier * surface_depth))`.

**Why:**
- Phase 2 showed the rollout collapses on large surfaces because error
  accumulates over the depth-`N` march (the depth crossover — `OBSERVATIONS.md`,
  O19). Once the front reaches the last ring (step `N`), the surface is fully
  filled; extra steps (`t > N`) let the operator keep **refining** the whole
  filled surface instead of stopping. This campaign tests whether additional
  refinement passes help or hurt the standard rollout.
- The loss needs no change: for `t > N` the frontier ring `F_t` is empty
  (contributes zero) and the filled region `P_t` = all of U, so the extra passes
  are supervised as all-U refinement (weight depends on the loss mode — see DV2.2).

**Notes / constraints:**
- Requires `rollout.method=standard`. `freeze_filled` freezes every ring once
  `d < step`, so past step `N` it would freeze the whole surface and the extra
  passes would do nothing.
- **Cost:** k× the BPTT depth → ~k× epoch time and (without checkpointing) k×
  activation memory. Use `train.grad_checkpoint=true` for `k ≥ 2`.
- **Evaluation must use the same k.** The model is trained to run `k·N` steps, so
  inference must too, or there is train/inference skew. `noise_band.py` should
  read `rollout.n_multiplier` from the run's saved `config.yaml` and pass it as
  `rollout_n_multiplier` into `evaluate_split`. (Verify this is wired before
  reporting eval numbers for `k > 1` runs.)
- Backward-compatible: `k = 1` reproduces the Phase-2 rollout exactly.

**Status:** Implemented; sweeping `k` in this campaign (start at `k = 2`).

---

## DV2.2 — Optional equal-weight (uniform) per-iteration data loss

**Decision:** A toggle `loss.equal_ring_weight` (default false) selects how the
per-iteration data loss weights vertices:
- **false (default):** the original weighted split — frontier ring
  `F_t = {d = t}` at `λ_f`, filled region `P_t = {0 < d < t}` at `λ_p`, each
  averaged separately.
- **true:** a single **uniform mean** over every reached unknown vertex
  (`0 < d ≤ t`); every vertex gets equal weight and `λ_f` / `λ_p` are ignored.

**Where:**
- Config: `loss.equal_ring_weight` in `configs/default.yaml`.
- `per_iteration_data_loss` + `rollout_loss` (`loss.py`), threaded through
  `train()` (`loop.py`) and `validate()` (`validate.py`) exactly like `λ_f`/`λ_p`.
- **Training-only.** This flag changes the loss, not the forward rollout, so it
  has **no effect on inference/evaluation** — the eval RMSE and predictions are
  identical regardless. No eval threading needed.

**Why:**
- The default loss deliberately emphasizes the propagation front (higher weight
  on the current ring). Because `F_t` (one ring) and `P_t` (all filled rings) are
  averaged separately, each frontier vertex also carries a much larger
  *per-vertex* weight than each filled vertex. This campaign tests supervising
  **every reached vertex equally**, removing the frontier emphasis — especially
  relevant now that the extended rollout (DV2.1) spends many steps refining the
  already-filled surface.

**Interaction with DV2.1:**
- With `equal_ring_weight=true`, the extra passes (`t > N`) supervise all of U at
  **uniform weight 1** (not the low `λ_p = 0.1` of the default mode). So this
  toggle also decides how the `t > N` refinement passes are weighted.

**Status:** Implemented; `equal_ring_weight=true` is the intended setting for this
campaign. Default false preserves the Phase-2 baseline.

---

## Open questions

- **Loss weighting for `t > N` (post-fill iterations).** With the default loss the
  extra passes are all-U at `λ_p = 0.1`; with `equal_ring_weight=true` they are
  all-U at weight 1. A third option (not implemented) is to *cycle* the strong
  frontier supervision back through the rings with a modulo — e.g. supervise ring
  `r_t = ((t − 1) mod N) + 1` at each extra step. Raised with the professor; the
  current choice is uniform (`equal_ring_weight`).

---

## How to run this campaign

**Train** (extended rollout + equal-weight loss):
```bash
python scripts/train.py \
  approach=rollout \
  rollout.n_multiplier=2 \
  loss.equal_ring_weight=true \
  train.grad_checkpoint=true
```

**Evaluate** a resulting run (reads `k` back from the run's config):
```bash
python scripts/noise_band.py outputs/tensorboard/run_XXXX --split test_id
python scripts/noise_band.py outputs/tensorboard/run_XXXX --split test_ood
```

**Baseline for comparison** (Phase-2 rollout, both knobs off):
```bash
python scripts/train.py approach=rollout rollout.n_multiplier=1 loss.equal_ring_weight=false
```

---

## Results log (fill in as runs complete)

| run_id | k | equal_ring_weight | best val_rmse (m) | test_id (m) | test_ood (m) | notes |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
