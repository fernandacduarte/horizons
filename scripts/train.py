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

import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

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
    train_ds = load_split_dataset(
        "train", mask_config=mask_cfg,
        normalize_per_surface=cfg.data.normalize_per_surface,
        init_method=cfg.data.init_method,
        n_masks_per_epoch=cfg.data.n_masks_per_epoch,
    )
    val_ds = load_split_dataset(
        "val", mask_config=mask_cfg,
        normalize_per_surface=cfg.data.normalize_per_surface,
        init_method=cfg.data.init_method,
    )

    # Build model
    model_kind = cfg.model_kind
    if model_kind == "operator":
        model = LocalOperator(
            hidden_dim=cfg.model.hidden_dim,
            n_message_passing=cfg.model.n_layers,
            output_init_scale=cfg.model.output_init_scale,
            conv_type=cfg.model.type,
            aggr=cfg.model.aggr,
        )
    elif model_kind == "placeholder":
        model = TinySAGE(hidden_dim=32, output_init_scale=0.01)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_kind} ({cfg.model.type}, aggr={cfg.model.aggr}) | params: {n_params:,}\n")

    # Set up TensorBoard run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.train.log_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # Save config snapshot alongside the events so we can reproduce later
    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"TensorBoard logs: {run_dir}")
    print(f"  tensorboard --logdir={cfg.train.log_dir}\n")

    try:
        checkpoint_path = run_dir / "best.pt"
        state = train(
            model, train_ds, val_ds,
            n_epochs=cfg.train.n_epochs,
            lr=cfg.optim.lr,
            lr_min=cfg.optim.lr_min,
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
            writer=writer,
            warmup_epochs=cfg.train.warmup_epochs,
            lr_schedule=cfg.train.lr_schedule,
            patience=cfg.train.patience,
            checkpoint_path=str(checkpoint_path),
            accum_steps=cfg.optim.accum_steps,
            best_metric=cfg.train.best_metric,
            use_checkpoint=cfg.train.grad_checkpoint,
        )
    finally:
        writer.close()

    # Save final training state summary
    summary = {
        "best_val_loss": state.best_val_loss,
        "best_val_epoch": state.best_val_epoch,
        "n_epochs_completed": len(state.train_history),
        "early_stopped": state.early_stopped,
        "early_stop_reason": state.early_stop_reason,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "final_train_loss": state.train_history[-1]["loss_total"]
            if state.train_history else None,
        "final_val_loss": state.val_history[-1]["loss_total"]
            if state.val_history else None,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(f"Best {state.best_metric_name}: {state.best_metric_value:.4f} at epoch {state.best_val_epoch}.")


if __name__ == "__main__":
    main()
