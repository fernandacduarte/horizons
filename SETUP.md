# Environment setup

## On any platform (macOS/Linux/Windows)

```bash
conda env create -f environment.yml
conda activate horizons
```

## Then install PyTorch (platform-specific)

### macOS (Apple Silicon, CPU/MPS)
```bash
pip install torch torchvision
```

### Windows with NVIDIA GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Adjust `cu121` to match your CUDA version:
- CUDA 11.8 → `cu118`
- CUDA 12.1 → `cu121` (most common as of 2026)
- CUDA 12.4 → `cu124`

Check your CUDA version with `nvidia-smi`.

### Linux with NVIDIA GPU
Same as Windows.

### CPU-only (any platform)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Install remaining requirements

```bash
pip install -r requirements.txt
```

## Install this project (editable)

```bash
pip install -e .
```

## Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_ave()}, MPS: {torch.backends.mps.is_available()}')"
python -m pytest tests/ -q
```

You should see "195 passed" or similar.

## Get the raw data

The `.ts` source files aren't in the repo. On the machine where you're running:
1. Copy or download the `Dados de Horizontes/triangulated/` folder.
2. Set the environment variable:
```bash
   export HORIZONS_TS_DIR="/path/to/Dados de Horizontes/triangulated"
```
   (Windows: `set HORIZONS_TS_DIR=C:\path\to\Dados de Horizontes\triangulated`)

## Build the dataset

For the standard dataset (V≤50k surfaces only):
```bash
python scripts/build_dataset.py
```

For the full dataset (including the 10 large V>50k surfaces — requires ≥32GB RAM):
```bash
python scripts/setup_full_dataset.py
```

## Train

```bash
python scripts/train.py \
    train.n_epochs=100 \
    train.patience=20 \
    optim.accum_steps=4 \
    data.normalize_per_surface=true \
    data.init_method=meanplane \
    data.n_masks_per_epoch=3
```

For CUDA training, append `train.device=cuda`.
