"""Training loop for masked-rollout horizon extrapolation."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import torch

from horizons.data.dataset import HorizonDataset
from horizons.eval.validate import validate
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_loss


@dataclass
class TrainState:
    """Holds the state that evolves across the training run."""
    epoch: int = 0
    step: int = 0
    best_val_loss: float = float("inf")
    best_val_epoch: int = -1
    train_history: list[dict] = field(default_factory=list)
    val_history: list[dict] = field(default_factory=list)


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

        # Shuffle the order of train surfaces this epoch
        order = list(range(n_train))
        rng.shuffle(order)

        epoch_loss_sum = 0.0
        epoch_data_sum = 0.0
        epoch_curv_sum = 0.0
        epoch_res_sum = 0.0
        n_successful_steps = 0

        model.train()
        for idx in order:
            item = train_dataset[idx]

            # Move tensors to device
            z0 = item["z0"].to(device)
            z_true = item["z_true"].to(device)
            V_xy = item["V"][:, :2].to(device)
            F = item["F"].to(device)
            edge_index = item["edge_index"].to(device)
            mask = item["mask"].to(device)
            d = item["d"].to(device)
            N = item["N"]

            optimizer.zero_grad()
            result = rollout(
                model,
                z0=z0, z_true=z_true,
                V_xy=V_xy, F=F, edge_index=edge_index,
                mask=mask, d=d, N=N,
            )
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
                        f"{item['surface_id']}. Skipping step."
                    )
                state.step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm,
            )
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_data_sum += loss_dict["data"].item()
            epoch_curv_sum += loss_dict["curv"].item()
            epoch_res_sum += loss_dict["res"].item()
            n_successful_steps += 1

            if verbose and state.step % log_every_steps == 0:
                print(
                    f"  ep {epoch:3d}  step {state.step:5d}  "
                    f"{item['surface_id']:<22} N={N:<3} "
                    f"loss={loss.item():.4f}"
                )
            state.step += 1

        # End-of-epoch summary
        if n_successful_steps == 0:
            raise RuntimeError(f"No successful steps in epoch {epoch}")

        train_record = {
            "epoch": epoch,
            "loss_total": epoch_loss_sum / n_successful_steps,
            "loss_data": epoch_data_sum / n_successful_steps,
            "loss_curv": epoch_curv_sum / n_successful_steps,
            "loss_res": epoch_res_sum / n_successful_steps,
            "n_steps": n_successful_steps,
            "n_skipped": n_train - n_successful_steps,
        }
        state.train_history.append(train_record)

        # Validation
        is_last_epoch = epoch == n_epochs - 1
        if (epoch + 1) % val_every == 0 or is_last_epoch:
            val_results = validate(
                model, val_dataset,
                lambda_f=lambda_f, lambda_p=lambda_p,
                lambda_c=lambda_c, lambda_r=lambda_r,
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

            if val_record["loss_total"] < state.best_val_loss:
                state.best_val_loss = val_record["loss_total"]
                state.best_val_epoch = epoch

            if verbose:
                t_elapsed = time.time() - t_start
                print(
                    f"\n--- epoch {epoch} ---"
                    f"\n  train loss: {train_record['loss_total']:.4f}"
                    f"\n  val loss:   {val_record['loss_total']:.4f}"
                    f"  val RMSE:   {val_record['rmse_meters']:.2f} m"
                    f"\n  elapsed:    {t_elapsed:.1f}s"
                )
                if val_record["loss_total"] == state.best_val_loss:
                    print(f"  (new best val loss)")
                print()

    return state
