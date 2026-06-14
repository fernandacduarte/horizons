"""Per-surface evaluation: rollout + RMSE breakdown by topological ring.

This is the workhorse of Stage 10. Given a trained model and a single
surface (with its mask + topological distance), it runs the rollout
and returns a structured result with:
  - overall RMSE on U
  - per-ring RMSE breakdown (r_1, ..., r_N)
  - raw residuals (for plotting or further analysis)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler
from horizons.data.init import init_z, init_z_dispatch
from horizons.training.rollout import rollout


@dataclass
class SurfaceEvalResult:
    """Per-surface evaluation result.

    Attributes
    ----------
    surface_id : str
    reservoir_id : str | None
    regime : str
    N : int
        Rollout depth (max d on U).
    n_K : int
        Number of known vertices.
    n_U : int
        Number of unknown vertices.
    rmse_overall : float
        Overall RMSE on U, in z's original units (after un-centering).
    rmse_per_ring : list[float]
        rmse_per_ring[k-1] is the RMSE on ring r_k = {i : d_i = k} for
        k = 1, ..., N. Length N. Same units as rmse_overall.
    ring_sizes : list[int]
        ring_sizes[k-1] is |r_k|. Length N.
    residuals_U : torch.Tensor
        Shape (n_U,). Raw residuals z^N[i] - z_true[i] for i in U,
        in centered units (NOT meters). Useful for plotting histograms
        or computing other summary statistics.
    d_U : torch.Tensor
        Shape (n_U,). The d value of each unknown vertex (same order
        as residuals_U).
    """
    surface_id: str
    reservoir_id: str | None
    regime: str
    N: int
    n_K: int
    n_U: int
    rmse_overall: float
    rmse_per_ring: list[float]
    ring_sizes: list[int]
    residuals_U: torch.Tensor
    d_U: torch.Tensor


@torch.no_grad()
def evaluate_surface(
    model: torch.nn.Module,
    surface: HorizonSurface,
    mask_sampler: MaskSampler,
    rng_seed: int,
    *,
    center_per_surface: bool = True,
    normalize_per_surface: bool = False,
    init_method: str = "meanplane",
    device: str | torch.device = "cpu",
) -> SurfaceEvalResult:
    """Run the model on one surface and compute per-ring metrics.

    Parameters
    ----------
    model : torch.nn.Module
        A trained LocalOperator. Set to eval mode by this function.
    surface : HorizonSurface
        The mesh to evaluate.
    mask_sampler : MaskSampler
        For sampling the mask. Use the same config that was used during
        validation (regime mix, fractions, etc.).
    rng_seed : int
        Seed for the mask sampler. For deterministic evaluation across
        runs, use the same seed.
    center_per_surface : bool
        Whether to apply (x, y, z) per-surface centering before feeding
        to the model (matches D4.6 / Stage 8.1).
    device : str | torch.device

    Returns
    -------
    SurfaceEvalResult
    """
    device = torch.device(device)
    was_training = model.training
    model.eval()
    model = model.to(device)

    # Sample mask deterministically for this surface
    rng = torch.Generator().manual_seed(rng_seed)
    mask, d, regime = mask_sampler.sample(surface, rng)

    # Apply per-surface centering (same logic as HorizonDataset)
    if center_per_surface:
        xy_mean = surface.V[:, :2].mean(dim=0)
        z_mean = surface.V[mask, 2].mean()
        V_centered = surface.V.clone()
        V_centered[:, :2] = surface.V[:, :2] - xy_mean
        V_centered[:, 2] = surface.V[:, 2] - z_mean
    else:
        V_centered = surface.V

    # Apply per-surface normalization (same logic as HorizonDataset)
    if normalize_per_surface:
        if not center_per_surface:
            raise ValueError(
                "normalize_per_surface requires center_per_surface=True"
            )
        xy_scale = V_centered[:, :2].to(torch.float64).abs().max().item()
        z_scale = V_centered[mask, 2].to(torch.float64).abs().max().item()
        xy_scale = max(xy_scale, 1.0)
        z_scale = max(z_scale, 1.0)
        V_centered = V_centered.clone()
        V_centered[:, :2] = V_centered[:, :2] / xy_scale
        V_centered[:, 2] = V_centered[:, 2] / z_scale
    else:
        z_scale = 1.0

    z0 = init_z_dispatch(
        V_centered, mask, surface.edge_index, method=init_method,
    )
    N = int(d.max().item())

    # Move to device
    V_xy = V_centered[:, :2].to(device)
    F = surface.F.to(device)
    edge_index = surface.edge_index.to(device)
    z_true = V_centered[:, 2].to(device)
    z0 = z0.to(device)
    mask = mask.to(device)
    d = d.to(device)

    # Run rollout (no autograd needed because of the @torch.no_grad above)
    result = rollout(
        model,
        z0=z0, z_true=z_true,
        V_xy=V_xy, F=F, edge_index=edge_index,
        mask=mask, d=d, N=N,
    )

    z_final = result.z_trajectory[-1]
    unknown = ~mask
    residuals_U = (z_final[unknown] - z_true[unknown]).cpu()
    d_U = d[unknown].cpu()

    # Overall RMSE in METERS. Centering is an additive shift (cancels in
    # the difference), but normalization is a multiplicative scale, so we
    # multiply by z_scale to get back to meters. When not normalized,
    # z_scale = 1.0 and this is identical to the previous behavior.
    rmse_overall = residuals_U.pow(2).mean().sqrt().item() * z_scale

    # Per-ring RMSE (also in meters)
    rmse_per_ring: list[float] = []
    ring_sizes: list[int] = []
    for k in range(1, N + 1):
        ring_mask = (d_U == k)
        n_in_ring = ring_mask.sum().item()
        ring_sizes.append(int(n_in_ring))
        if n_in_ring > 0:
            ring_rmse = residuals_U[ring_mask].pow(2).mean().sqrt().item() * z_scale
        else:
            ring_rmse = float("nan")
        rmse_per_ring.append(ring_rmse)

    if was_training:
        model.train()

    return SurfaceEvalResult(
        surface_id=surface.surface_id,
        reservoir_id=surface.reservoir_id,
        regime=regime,
        N=N,
        n_K=int(mask.sum().item()),
        n_U=int((~mask).sum().item()),
        rmse_overall=rmse_overall,
        rmse_per_ring=rmse_per_ring,
        ring_sizes=ring_sizes,
        residuals_U=residuals_U,
        d_U=d_U,
    )
