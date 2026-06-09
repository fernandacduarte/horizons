"""Helpers for loading saved training checkpoints.

A checkpoint is a single .pt file produced by horizons.training.loop.train()
when val loss improves. It contains:
    - model_state       : torch state dict
    - optimizer_state   : torch state dict (useful only for resume)
    - epoch, step       : training position at the time of save
    - best_val_loss     : the val loss that triggered the save
    - train_history     : list of per-epoch train metrics dicts
    - val_history       : list of per-val-check metrics dicts


Example of usage:
# Simplest case: load the latest checkpoint
from horizons.eval.checkpoint import load_checkpoint, latest_checkpoint
ckpt = load_checkpoint(latest_checkpoint())

# Use the model
from horizons.data.dataset import load_split_dataset
from horizons.eval.validate import validate
val_ds = load_split_dataset("val")
results = validate(ckpt.model, val_ds)
print(f"RMSE: {results['rmse_meters']:.2f} m")

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from horizons.models.operator import LocalOperator


@dataclass
class LoadedCheckpoint:
    """A loaded checkpoint with the model ready to use.

    Attributes
    ----------
    model : torch.nn.Module
        The model with weights loaded and set to eval mode.
    epoch : int
        Epoch index at which the checkpoint was saved.
    step : int
        Optimizer step index at the time of save.
    best_val_loss : float
        The val loss that caused this checkpoint to be saved.
    train_history : list[dict]
        Per-epoch train metrics up to and including this epoch.
    val_history : list[dict]
        Per-val-check metrics up to and including this epoch.
    raw : dict
        The raw checkpoint dict, in case you need optimizer_state or other fields.
    """
    model: torch.nn.Module
    epoch: int
    step: int
    best_val_loss: float
    train_history: list[dict]
    val_history: list[dict]
    raw: dict[str, Any]


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    *,
    hidden_dim: int = 64,
    n_message_passing: int = 2,
    output_init_scale: float = 0.01,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a checkpoint from disk.

    Parameters
    ----------
    path : str or Path
        Path to the .pt file.
    model : torch.nn.Module, optional
        If provided, the weights are loaded into this model. Must match the
        architecture of the saved model. If None, a fresh LocalOperator with
        the hidden_dim / n_message_passing arguments is constructed.
    hidden_dim, n_message_passing, output_init_scale : float
        Used only when constructing a fresh LocalOperator (i.e., when
        `model` is None). Must match the architecture the checkpoint was
        saved from. The defaults match the project's standard config.
    device : str | torch.device
        Device to place the model on. The model is also set to eval mode.

    Returns
    -------
    LoadedCheckpoint with the model ready to use and metadata.

    Examples
    --------
    >>> ckpt = load_checkpoint("outputs/tensorboard/run_.../best.pt")
    >>> # ckpt.model is in eval mode on cpu, ready to use
    >>> from horizons.data.dataset import load_split_dataset
    >>> from horizons.eval.validate import validate
    >>> val_ds = load_split_dataset("val")
    >>> results = validate(ckpt.model, val_ds)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(path, weights_only=False, map_location=device)

    # Validate the checkpoint shape
    required_keys = {"model_state", "epoch", "step", "best_val_loss"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise ValueError(
            f"Checkpoint at {path} is missing required keys: {missing}"
        )

    # Build the model if not provided
    if model is None:
        model = LocalOperator(
            hidden_dim=hidden_dim,
            n_message_passing=n_message_passing,
            output_init_scale=output_init_scale,
        )

    # Load weights
    try:
        model.load_state_dict(raw["model_state"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load model state from {path}. The checkpoint's "
            f"architecture may not match the model you passed. Original "
            f"error: {e}"
        )

    model.to(device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        epoch=raw["epoch"],
        step=raw["step"],
        best_val_loss=raw["best_val_loss"],
        train_history=raw.get("train_history", []),
        val_history=raw.get("val_history", []),
        raw=raw,
    )


def latest_checkpoint(
    tensorboard_dir: str | Path = "outputs/tensorboard",
) -> Path:
    """Find the most recent run's best.pt file.

    Useful for quickly grabbing the latest result without typing the
    timestamped directory name.
    """
    tensorboard_dir = Path(tensorboard_dir)
    runs = sorted([
        d for d in tensorboard_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    if not runs:
        raise FileNotFoundError(
            f"No run_* directories found in {tensorboard_dir}"
        )
    for run in reversed(runs):
        ckpt = run / "best.pt"
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(
        f"No best.pt found in any run under {tensorboard_dir}"
    )
