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

## Running the test suite

```bash
python -m pytest tests/ -v
```

## Building the dataset (one-time)

```bash
python scripts/build_dataset.py   # produces data/surfaces/*.npz
python scripts/build_split.py     # produces data/splits/split_v1.json
```

## Decision log

See `DECISIONS.md` for the structured record of design decisions made
during development.
