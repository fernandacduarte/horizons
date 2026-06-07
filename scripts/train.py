"""Train horizon-extrapolation model on the canonical dataset split.

Stage 8.3 scope: bare skeleton training. No LR scheduling, no
checkpointing yet.

Usage:
    python scripts/train.py                           # use config defaults
    python scripts/train.py train.n_epochs=20         # override epochs
    python scripts/train.py optim.lr=5e-4             # override LR
"""
from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from horizons.data.dataset import load_split_dataset
from horizons.data.masking import MaskSamplerConfig
from horizons.models.operator import LocalOperator
from horizons.models.placeholder import TinySAGE
from horizons.training.loop import train


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    # Build datasets
    mask_cfg = MaskSamplerConfig.from_dictconfig(cfg.mask)
    train_ds = load_split_dataset("train", mask_config=mask_cfg)
    val_ds = load_split_dataset("val", mask_config=mask_cfg)

    # Build model
    model_kind = cfg.model_kind
    if model_kind == "operator":
        model = LocalOperator(
            hidden_dim=cfg.model.hidden_dim,
            n_message_passing=cfg.model.n_layers,
            output_init_scale=cfg.model.output_init_scale,
        )
    elif model_kind == "placeholder":
        model = TinySAGE(hidden_dim=32, output_init_scale=0.01)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_kind} | params: {n_params:,}\n")

    # Train
    state = train(
        model, train_ds, val_ds,
        n_epochs=cfg.train.n_epochs,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        grad_clip_norm=cfg.optim.grad_clip_norm,
        lambda_f=cfg.loss.lambda_f,
        lambda_p=cfg.loss.lambda_p,
        lambda_c=cfg.loss.lambda_c,
        lambda_r=cfg.loss.lambda_r,
        val_every=cfg.train.val_every,
        log_every_steps=cfg.train.log_every_steps,
        device=cfg.train.device,
        seed=cfg.seed,
    )

    print("Training complete.")
    print(f"Best val loss: {state.best_val_loss:.4f} at epoch {state.best_val_epoch}.")


if __name__ == "__main__":
    main()
