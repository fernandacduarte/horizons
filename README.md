# horizons

GNN-based extrapolation of geological horizon surfaces.

## Setup

```bash
conda create -n horizons python=3.11 -y
conda activate horizons
conda install pip -y
python -m pip install -e .
python -m pip install torch torchvision torch-geometric \
    trimesh meshio scipy matplotlib pyvista hydra-core tensorboard pytest
```

## Data directory

Scripts that touch raw `.ts` files (`scripts/audit_ts_files.py`,
`scripts/build_dataset.py`) read the data location from the
`HORIZONS_TS_DIR` environment variable. Set it before running:

```bash
export HORIZONS_TS_DIR="/path/to/your/.ts/files"
```

Or pass `--ts-dir /path/to/files` to either script.

## Common commands

### Training
```bash
# Use defaults from configs/default.yaml
python scripts/train.py

# Override any config field at the command line (Hydra)
python scripts/train.py train.n_epochs=100 train.patience=20 optim.accum_steps=4
python scripts/train.py loss.lambda_c=0.1 loss.lambda_r=0.01
```

Each run produces:
- `outputs/tensorboard/run_<TIMESTAMP>/` with `best.pt` (checkpoint),
  `config.yaml` (snapshot), `summary.json`, and the TensorBoard events file.
- `outputs/<date>/<time>/` (Hydra working directory) with the
  resolved config and full stdout log.

### Viewing TensorBoard
```bash
tensorboard --logdir=outputs/tensorboard
```
Then open http://localhost:6006

### Evaluation suite on a trained checkpoint
```bash
python scripts/eval_run.py outputs/tensorboard/run_<TIMESTAMP>
```
This evaluates on val (10 masks per surface), saves a JSON record to
`outputs/evaluation/`, and generates the four diagnostic plots to
`outputs/evaluation/plots/`.

### Running the test suite
```bash
python -m pytest tests/ -v
```

### Inspecting a checkpoint
```bash
python -c "
from horizons.eval.checkpoint import load_checkpoint, latest_checkpoint
ckpt = load_checkpoint(latest_checkpoint())
print(f'epoch: {ckpt.epoch}, best_val_loss: {ckpt.best_val_loss:.2f}')
"
```

## Reproducible results

Two trained checkpoints are committed to this repository so the results can be
re-checked **without retraining**. Both were trained with the split
(`data/splits/split_v2.json`), `seed: 42`, and per-surface normalization. Each
run directory ships its `best.pt`, the exact `config.yaml` snapshot,
`summary.json`, and the TensorBoard log.

| Run directory | Approach | Init | What it is |
|---|---|---|---|
| `outputs/tensorboard/run_20260621_171110` | rollout (standard) | meanplane | the masked-rollout model |
| `outputs/tensorboard/run_20260623_115850` | hybrid (harmonic init + 3 GNN refine passes) | harmonic | the hybrid model |

### Checking the results (no training needed)

`eval_run.py` reads each run's `config.yaml`, so it auto-detects the
architecture, init method, split, and approach — just point it at the run
directory:

```bash
# rollout model
python scripts/eval_run.py outputs/tensorboard/run_20260621_171110

# hybrid model
python scripts/eval_run.py outputs/tensorboard/run_20260623_115850
```

Each command evaluates on the `val` split (10 masks per surface, with fixed
mask-draw seeds so the numbers are deterministic and reproducible), prints the
overall and per-regime RMSE of the model against the mean-plane and harmonic
baselines, writes a JSON record to `outputs/evaluation/`, and regenerates the
diagnostic plots under `outputs/evaluation/plots/`.

To check generalization on the held-out test surfaces, switch the split:

```bash
python scripts/eval_run.py outputs/tensorboard/run_20260623_115850 --split test_id   # in-distribution
python scripts/eval_run.py outputs/tensorboard/run_20260623_115850 --split test_ood  # out-of-distribution
```

The training summary for each run (best epoch, early-stop reason, final losses)
is recorded in its `summary.json`; the RMSE tables are produced by the
`eval_run.py` commands above.

### Inspecting the predictions visually

```bash
python scripts/viz_prediction.py outputs/tensorboard/run_20260623_115850 --split test_ood --index 0
python scripts/viz_prediction_with_error.py outputs/tensorboard/run_20260623_115850 --split test_ood --index 0
```

See [Visualization](#visualization) below for all options; add `--out fig.png`
to save a PNG headlessly instead of opening a window.

### Retraining from scratch (optional)

The full hyperparameters live in each run's `config.yaml`. The rollout run uses
the shipped defaults; the hybrid run only overrides the approach and init:

```bash
python scripts/train.py                                              # ~ run_20260621_171110 (rollout)
python scripts/train.py approach=hybrid data.init_method=harmonic    # ~ run_20260623_115850 (hybrid)
```

Both were trained on GPU (`train.device=cuda`); on different hardware the numbers
should land very close but may not be bit-for-bit identical, so the committed
checkpoints above are the reference for exact reproduction.

## Visualization

Three PyVista scripts render 3-D views of a run's predictions and of the mask
regimes. All accept `--surface <id>` (or `--index <i>` for the i-th surface) and
`--list` to print the surfaces in a split. The two prediction scripts take a run
directory and an optional `--out fig.png` to save a PNG off-screen instead of
opening an interactive window; `--show-edges` overlays the triangle mesh edges.

### Prediction vs. ground truth
Three linked panels — the masked input, the network's extrapolation, and the
ground-truth surface (blue = K known, orange = U extrapolated):
```bash
python scripts/viz_prediction.py outputs/tensorboard/run_<TIMESTAMP> --surface TestHorizon4
python scripts/viz_prediction.py outputs/tensorboard/run_<TIMESTAMP> --split test_ood --index 0
python scripts/viz_prediction.py outputs/tensorboard/run_<TIMESTAMP> --surface TestHorizon4 --show-edges
python scripts/viz_prediction.py outputs/tensorboard/run_<TIMESTAMP> --surface TestHorizon4 --out fig.png
```

### Prediction with an error map
Same first two panels, but the third replaces the ground truth with a per-node
error map on the predicted surface (signed `prediction − truth`, in metres). The
known K region is ~0 by construction, so the colour concentrates where the model
actually extrapolated:
```bash
python scripts/viz_prediction_with_error.py outputs/tensorboard/run_<TIMESTAMP> --surface TestHorizon4
python scripts/viz_prediction_with_error.py outputs/tensorboard/run_<TIMESTAMP> --surface TestHorizon4 --abs-error
```
`--abs-error` colours by `|error|` (sequential map) instead of signed error
(diverging map). `--show-edges` and `--out fig.png` work here too.

### Mask regimes
One PNG per masking regime (`half_plane`, `outward_free`, `outward_pinned`) on a
single surface, coloured blue = K / orange = U. No trained run is needed:
```bash
python scripts/viz_mask_regimes.py                            # val split, first surface
python scripts/viz_mask_regimes.py --surface TestHorizon4 --out-dir figures/masks
python scripts/viz_mask_regimes.py --top-down                 # straight-down view (clearest mask shape)
```
Writes `mask_half_plane.png`, `mask_outward_free.png`, and
`mask_outward_pinned.png` to `--out-dir` (default `figures/masks`).

## Building the dataset (one-time)

```bash
python scripts/build_dataset.py   # produces data/surfaces/*.npz
python scripts/build_split.py     # produces data/splits/split_v1.json
```

## Decision log

See `DECISIONS.md` for the structured record of design decisions made
during development.
