"""Validation: compute loss and metrics over a held-out dataset.

The validation function is deliberately separate from training so it
can be reused for the test sets at the end of training, or for ad-hoc
evaluation of a saved checkpoint.
"""
from __future__ import annotations

import torch

from horizons.data.dataset import HorizonDataset
from horizons.training.rollout import rollout
from horizons.training.loss import (
    rollout_loss,
    per_iteration_data_loss,
    hybrid_rollout_loss
)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataset: HorizonDataset,
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
    lambda_c: float = 0.01,
    lambda_r: float = 0.001,
    approach: str = "rollout",
    hybrid_n_passes: int = 3,
) -> dict:
    """Compute mean validation loss and per-surface diagnostics.

    Parameters
    ----------
    model : torch.nn.Module
        A LocalOperator (or compatible). Will be set to eval() mode.
    dataset : HorizonDataset
        The val (or test) dataset. Masks should be stable (split != 'train').

    Returns
    -------
    dict with keys:
      - "loss_total" : mean total loss across surfaces (float)
      - "loss_data", "loss_curv", "loss_res" : mean component losses (floats)
      - "rmse_centered" : mean RMSE on U in *centered* z units (float)
      - "rmse_meters" : mean RMSE on U in original (uncentered) meters (float)
      - "per_surface" : list of dicts, one per surface, with keys:
          "surface_id", "reservoir_id", "regime", "N",
          "loss_total", "rmse_centered", "rmse_meters"
    """
    was_training = model.training
    model.eval()

    n = len(dataset)
    sum_total = 0.0
    sum_data = 0.0
    sum_curv = 0.0
    sum_res = 0.0
    sum_rmse_centered = 0.0
    sum_rmse_meters = 0.0
    per_surface: list[dict] = []

    # Detect the model's device so we can move val tensors to match.
    # This is important when training on CUDA: the dataset returns CPU
    # tensors but the model expects them on the same device as its params.
    device = next(model.parameters()).device

    for idx in range(n):
        item = dataset[idx]
        z_true = item["z_true"].to(device)
        z0 = item["z0"].to(device)
        V_xy = item["V"][:, :2].to(device)
        F = item["F"].to(device)
        edge_index = item["edge_index"].to(device)
        mask = item["mask"].to(device)
        d = item["d"].to(device)
        surface_N = item["N"]
        N = hybrid_n_passes if approach == "hybrid" else surface_N

        result = rollout(
            model,
            z0=z0, z_true=z_true,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
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

        # Compute RMSE on U at final iteration. The model operates in
        # (possibly normalized) centered units, so residuals are in those
        # units. To report in meters, multiply by z_scale (= 1.0 if not
        # normalized). Centering offsets are additive and cancel in the
        # difference, so we don't need z_mean.
        z_final = result.z_trajectory[-1]
        unknown = ~mask
        err_centered = (z_final[unknown] - z_true[unknown])
        rmse_centered = err_centered.pow(2).mean().sqrt().item()
        z_scale = item.get("z_scale", torch.tensor(1.0))
        rmse_meters = rmse_centered * float(z_scale)

        sum_total += loss_dict["total"].item()
        sum_data += loss_dict["data"].item()
        sum_curv += loss_dict["curv"].item()
        sum_res += loss_dict["res"].item()
        sum_rmse_centered += rmse_centered
        sum_rmse_meters += rmse_meters

        per_surface.append({
            "surface_id": item["surface_id"],
            "reservoir_id": item["reservoir_id"],
            "regime": item["regime"],
            "N": surface_N,
            "loss_total": loss_dict["total"].item(),
            "rmse_centered": rmse_centered,
            "rmse_meters": rmse_meters,
        })

    if was_training:
        model.train()

    return {
        "loss_total": sum_total / n,
        "loss_data": sum_data / n,
        "loss_curv": sum_curv / n,
        "loss_res": sum_res / n,
        "rmse_centered": sum_rmse_centered / n,
        "rmse_meters": sum_rmse_meters / n,
        "per_surface": per_surface,
    }
