"""Training loop for masked-rollout horizon extrapolation."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from torch.utils.tensorboard import SummaryWriter

from horizons.data.dataset import HorizonDataset
from horizons.eval.validate import validate
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_loss, hybrid_rollout_loss


def compute_lr(
    epoch: int,
    *,
    lr_max: float,
    lr_min: float,
    warmup_epochs: int,
    n_epochs: int,
    schedule: str = "cosine",
) -> float:
    """Compute the learning rate for a given epoch.

    For schedule="cosine":
        - epochs [0, warmup_epochs): linear ramp from lr_min to lr_max
        - epochs [warmup_epochs, n_epochs): half-cosine decay from lr_max to lr_min

    For schedule="constant":
        - LR is always lr_max (warmup is ignored)

    Parameters
    ----------
    epoch : int
        Current epoch index (0-based).
    lr_max : float
        Peak LR (the LR at the end of warmup, before decay starts).
    lr_min : float
        Floor LR (the LR at the start of warmup and at the end of decay).
    warmup_epochs : int
        Number of warmup epochs.
    n_epochs : int
        Total epochs in the run.
    schedule : str
        "cosine" or "constant".
    """
    if schedule == "constant":
        return lr_max

    if schedule != "cosine":
        raise ValueError(
            f"Unknown schedule {schedule!r}; expected 'cosine' or 'constant'"
        )

    if epoch < warmup_epochs:
        # Linear ramp from lr_min to lr_max over warmup_epochs steps.
        # At epoch=0 we get lr_min; at epoch=warmup_epochs-1 we get
        # close to lr_max (one step shy); the lr_max is hit at warmup_epochs.
        if warmup_epochs <= 0:
            return lr_max
        frac = (epoch + 1) / warmup_epochs
        return lr_min + (lr_max - lr_min) * frac

    # Cosine decay from lr_max at epoch=warmup_epochs to lr_min at epoch=n_epochs-1.
    decay_epochs = max(1, n_epochs - warmup_epochs - 1)
    progress = (epoch - warmup_epochs) / decay_epochs
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cos


@dataclass
class TrainState:
    """Holds the state that evolves across the training run."""
    epoch: int = 0
    step: int = 0
    best_val_loss: float = float("inf")
    best_val_epoch: int = -1
    best_metric_value: float = float("inf")
    best_metric_name: str = "val_loss"
    train_history: list[dict] = field(default_factory=list)
    val_history: list[dict] = field(default_factory=list)
    early_stopped: bool = False
    early_stop_reason: str | None = None


def train(
    model: torch.nn.Module,
    train_dataset: HorizonDataset,
    val_dataset: HorizonDataset,
    *,
    n_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
    lambda_c: float = 0.01,
    lambda_r: float = 0.001,
    val_every: int = 5,
    log_every_steps: int = 10,
    device: str | torch.device = "cpu",
    seed: int = 42,
    verbose: bool = True,
    writer: SummaryWriter | None = None,
    lr_min: float = 1e-5,
    warmup_epochs: int = 5,
    lr_schedule: str = "cosine",
    patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    accum_steps: int = 1,
    best_metric: str = "val_loss",
    use_checkpoint: bool = False,
    rollout_method: str = "standard",
    approach: str = "rollout",
    hybrid_n_passes: int = 3,
) -> TrainState:
    """Train a horizon-extrapolation model.

    Parameters
    ----------
    model : torch.nn.Module
        A LocalOperator (or compatible).
    train_dataset, val_dataset : HorizonDataset
        Datasets for training and validation. The train dataset's masks
        should vary per epoch (split='train'); val masks should be stable.
    n_epochs : int
        Number of full passes over the training set.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    grad_clip_norm : float
        Max gradient L2 norm; clipped each step.
    lambda_f, lambda_p, lambda_c, lambda_r : float
        Loss weights (passed to rollout_loss).
    val_every : int
        Run validation every N epochs (and at the end).
    log_every_steps : int
        Print a one-line train status every N optimizer steps.
    device : str | torch.device
    seed : int
        For reproducible shuffling of train order each epoch.
    verbose : bool
        Whether to print progress to stdout.

    Returns
    -------
    TrainState : the final state, including train and val history.
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    state = TrainState()

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    if n_train == 0:
        raise ValueError("train_dataset is empty")
    if n_val == 0:
        raise ValueError("val_dataset is empty")

    if verbose:
        print(f"Training: {n_train} train surfaces, {n_val} val.")
        print(f"Device: {device}. Epochs: {n_epochs}. LR: {lr}.")
        print()

    t_start = time.time()

    for epoch in range(n_epochs):
        state.epoch = epoch
        train_dataset.set_epoch(epoch)

        # Update LR according to schedule. We set the param_group lr
        # before the first optimizer.step() of this epoch, so all steps
        # in the epoch use the same LR.
        current_lr = compute_lr(
            epoch, lr_max=lr, lr_min=lr_min,
            warmup_epochs=warmup_epochs,
            n_epochs=n_epochs, schedule=lr_schedule,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Shuffle the order of train surfaces this epoch
        order = list(range(n_train))
        rng.shuffle(order)

        epoch_loss_sum = 0.0
        epoch_data_sum = 0.0
        epoch_curv_sum = 0.0
        epoch_res_sum = 0.0
        n_successful_surfaces = 0
        n_optimizer_steps = 0

        model.train()
        # Iterate in batches of accum_steps. The last batch may be smaller
        # if n_train is not divisible by accum_steps.
        for batch_start in range(0, n_train, accum_steps):
            batch_indices = order[batch_start : batch_start + accum_steps]
            batch_size = len(batch_indices)

            optimizer.zero_grad()
            batch_loss_sum = 0.0
            batch_data_sum = 0.0
            batch_curv_sum = 0.0
            batch_res_sum = 0.0
            n_successful_in_batch = 0
            last_surface_id = None  # for logging
            last_N = 0

            for idx in batch_indices:
                item = train_dataset[idx]
                last_surface_id = item["surface_id"]
                last_N = item["N"]

                # Move tensors to device
                z0 = item["z0"].to(device)
                z_true = item["z_true"].to(device)
                V_xy = item["V"][:, :2].to(device)
                F = item["F"].to(device)
                edge_index = item["edge_index"].to(device)
                mask = item["mask"].to(device)
                d = item["d"].to(device)
                                # hybrid: harmonic-filled init, refined by a fixed shallow number
                # of passes (no surface-depth march), supervised by all-U MSE.
                # rollout: the standard surface-depth rollout + per-ring loss.
                N = hybrid_n_passes if approach == "hybrid" else item["N"]

                result = rollout(
                    model,
                    z0=z0, z_true=z_true,
                    V_xy=V_xy, F=F, edge_index=edge_index,
                    mask=mask, d=d, N=N,
                    use_checkpoint=use_checkpoint,
                    rollout_method=rollout_method,
                )
                if approach == "hybrid":
                    loss_dict = hybrid_rollout_loss(result.z_trajectory, z_true, mask)
                else:
                    loss_dict = rollout_loss(
                        z_trajectory=result.z_trajectory,
                        dz_trajectory=result.dz_trajectory,
                        z_true=z_true, d=d, edge_index=edge_index, mask=mask,
                        lambda_f=lambda_f, lambda_p=lambda_p,
                        lambda_c=lambda_c, lambda_r=lambda_r,
                    )
                loss = loss_dict["total"]

                if not torch.isfinite(loss):
                    if verbose:
                        print(
                            f"  step {state.step}: NON-FINITE loss "
                            f"({loss.item()}) on surface "
                            f"{item['surface_id']}. Skipping this surface."
                        )
                    continue

                # Divide by batch_size so accumulated gradient = mean gradient.
                # We use the actual batch_size (not accum_steps) so partial
                # batches at the end of an epoch still produce a mean, not
                # an under-scaled gradient.
                (loss / batch_size).backward()

                batch_loss_sum += loss.item()
                batch_data_sum += loss_dict["data"].item()
                batch_curv_sum += loss_dict["curv"].item()
                batch_res_sum += loss_dict["res"].item()
                n_successful_in_batch += 1

            if n_successful_in_batch == 0:
                # Entire batch failed; skip the optimizer step
                if verbose:
                    print(
                        f"  step {state.step}: entire batch failed; "
                        f"skipping optimizer step"
                    )
                state.step += 1
                continue

            # Guard: a finite loss can still backprop a NON-finite gradient
            # (e.g. the 1/||n|| term in vertex-normal normalization on a
            # near-degenerate normal during a deep rollout). clip_grad_norm_
            # cannot sanitize NaN/Inf — clipping by a NaN norm yields NaN
            # grads — so an unguarded step corrupts the weights to NaN, after
            # which every surface NaNs and the run cannot recover. Skip instead.
            grads_finite = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters() if p.grad is not None
            )
            if not grads_finite:
                if verbose:
                    print(
                        f"  step {state.step}: non-finite GRADIENT; skipping "
                        f"optimizer step (weights preserved)"
                    )
                optimizer.zero_grad()
                state.step += 1
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm,
            )
            optimizer.step()

            # Aggregate stats
            mean_batch_loss = batch_loss_sum / n_successful_in_batch
            epoch_loss_sum += batch_loss_sum
            epoch_data_sum += batch_data_sum
            epoch_curv_sum += batch_curv_sum
            epoch_res_sum += batch_res_sum
            n_successful_surfaces += n_successful_in_batch
            n_optimizer_steps += 1

            if verbose and state.step % log_every_steps == 0:
                print(
                    f"  ep {epoch:3d}  step {state.step:5d}  "
                    f"batch_size={n_successful_in_batch}  "
                    f"last={last_surface_id:<22} N={last_N:<3} "
                    f"batch_mean_loss={mean_batch_loss:.4f}"
                )
            state.step += 1

        # End-of-epoch summary
        if n_successful_surfaces == 0:
            raise RuntimeError(f"No successful surfaces in epoch {epoch}")

        train_record = {
            "epoch": epoch,
            "loss_total": epoch_loss_sum / n_successful_surfaces,
            "loss_data": epoch_data_sum / n_successful_surfaces,
            "loss_curv": epoch_curv_sum / n_successful_surfaces,
            "loss_res": epoch_res_sum / n_successful_surfaces,
            "n_surfaces": n_successful_surfaces,
            "n_optimizer_steps": n_optimizer_steps,
            "n_skipped": n_train - n_successful_surfaces,
        }
        state.train_history.append(train_record)

        # TensorBoard: per-epoch train metrics
        if writer is not None:
            writer.add_scalar("train/loss_total", train_record["loss_total"], epoch)
            writer.add_scalar("train/loss_data", train_record["loss_data"], epoch)
            writer.add_scalar("train/loss_curv", train_record["loss_curv"], epoch)
            writer.add_scalar("train/loss_res", train_record["loss_res"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        is_last_epoch = epoch == n_epochs - 1
        if (epoch + 1) % val_every == 0 or is_last_epoch:
            val_results = validate(
                model, val_dataset,
                lambda_f=lambda_f, lambda_p=lambda_p,
                lambda_c=lambda_c, lambda_r=lambda_r,
                approach=approach, hybrid_n_passes=hybrid_n_passes,
            )
            val_record = {
                "epoch": epoch,
                "loss_total": val_results["loss_total"],
                "loss_data": val_results["loss_data"],
                "loss_curv": val_results["loss_curv"],
                "loss_res": val_results["loss_res"],
                "rmse_meters": val_results["rmse_meters"],
                "per_surface": val_results["per_surface"],
            }
            state.val_history.append(val_record)

            # Map best_metric name to the key in val_record.
            # "val_loss" tracks loss_total; "val_rmse_meters" tracks RMSE in meters.
            metric_keys = {
                "val_loss": "loss_total",
                "val_rmse_meters": "rmse_meters",
            }
            if best_metric not in metric_keys:
                raise ValueError(
                    f"best_metric must be one of {list(metric_keys)}; "
                    f"got {best_metric!r}"
                )
            current_metric = val_record[metric_keys[best_metric]]
            state.best_metric_name = best_metric

            is_new_best = current_metric < state.best_metric_value
            if is_new_best:
                state.best_metric_value = current_metric
                state.best_val_loss = val_record["loss_total"]
                state.best_val_epoch = epoch
                # Save best checkpoint
                if checkpoint_path is not None:
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": state.step,
                        "best_val_loss": state.best_val_loss,
                        "best_metric_value": state.best_metric_value,
                        "best_metric_name": state.best_metric_name,
                        "train_history": state.train_history,
                        "val_history": state.val_history,
                    }
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    if verbose:
                        print(
                            f"  saved checkpoint: {checkpoint_path} "
                            f"({best_metric}={current_metric:.4f})"
                        )

            # TensorBoard: validation metrics
            if writer is not None:
                writer.add_scalar("val/loss_total", val_record["loss_total"], epoch)
                writer.add_scalar("val/loss_data", val_record["loss_data"], epoch)
                writer.add_scalar("val/loss_curv", val_record["loss_curv"], epoch)
                writer.add_scalar("val/loss_res", val_record["loss_res"], epoch)
                writer.add_scalar("val/rmse_meters", val_record["rmse_meters"], epoch)
                # Per-surface RMSE (lets us see which surfaces are hard)
                for s in val_record["per_surface"]:
                    writer.add_scalar(
                        f"val_rmse_per_surface/{s['surface_id']}",
                        s["rmse_meters"], epoch,
                    )
                # Per-reservoir mean RMSE
                by_reservoir: dict[str, list[float]] = {}
                for s in val_record["per_surface"]:
                    rid = s["reservoir_id"] or "unknown"
                    by_reservoir.setdefault(rid, []).append(s["rmse_meters"])
                for rid, rmses in by_reservoir.items():
                    writer.add_scalar(
                        f"val_rmse_per_reservoir/{rid}",
                        sum(rmses) / len(rmses), epoch,
                    )

            # Early stopping check: if patience epochs have passed since
            # the last best val loss, stop. Patience is measured in EPOCHS
            # (not val checks), so val_every interacts with it: if
            # val_every=5 and patience=30, we get ~6 val checks before
            # giving up.
            if patience is not None and patience > 0:
                epochs_since_best = epoch - state.best_val_epoch
                if epochs_since_best >= patience:
                    state.early_stopped = True
                    state.early_stop_reason = (
                        f"no {best_metric} improvement for {epochs_since_best} "
                        f"epochs (patience={patience})"
                    )
                    if verbose:
                        print(f"\n  EARLY STOP: {state.early_stop_reason}")
                    break

            if verbose:
                t_elapsed = time.time() - t_start
                print(
                    f"\n--- epoch {epoch} ---"
                    f"\n  train loss: {train_record['loss_total']:.4f}"
                    f"\n  val loss:   {val_record['loss_total']:.4f}"
                    f"  val RMSE:   {val_record['rmse_meters']:.2f} m"
                    f"\n  elapsed:    {t_elapsed:.1f}s"
                )
                print()

    return state
