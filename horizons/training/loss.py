"""Loss functions for masked-rollout training.

Three terms (all defined per-iteration t and summed over the rollout):
  - data loss: supervises z^t on the frontier ring F_t and the filled
    region P_t, with weights lambda_f and lambda_p.
  - curvature regularizer: penalizes squared umbrella Laplacian of z^t
    on the unknown set U, weighted by lambda_c.
  - residual regularizer: penalizes squared Δz^{t-1} on U, weighted by
    lambda_r. Defends against unstable large per-iteration corrections.

All three terms use MEAN rather than SUM (see D5.7 in DECISIONS.md).
"""
from __future__ import annotations

import torch

from horizons.data.features import compute_umbrella_laplacian


# ======================================================================
# Per-iteration loss components
# ======================================================================
def per_iteration_data_loss(
    z_t: torch.Tensor,
    z_true: torch.Tensor,
    d: torch.Tensor,
    t: int,
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
) -> torch.Tensor:
    """Per-iteration data loss L_{data,t}.

    L_{data,t} = lambda_f * mean_{i in F_t} (z^t_i - z_true_i)^2
               + lambda_p * mean_{i in P_t} (z^t_i - z_true_i)^2

    where F_t = {i : d_i = t} (frontier) and P_t = {i : 0 < d_i < t}
    (already-filled region; we exclude i in K because d_i = 0 there and
    z^t equals z_true exactly on K by construction).

     Note
    ----
    We use mean instead of sum. This stabilizes the relative magnitude of L_t across iterations.
    For example, in outward rectangle regime, early t has tiny |F_t|,
    later t has growing |P_t|, making the rollout weights w_t in the total loss easier to reason about.

    Parameters
    ----------
    z_t : (n,) float — predicted z at iteration t
    z_true : (n,) float — ground truth
    d : (n,) int64 — topological distance from K
    t : int — iteration index, 1 <= t <= N
    lambda_f, lambda_p : float — weights

    Returns
    -------
    L_t : scalar torch.Tensor
        The per-iteration data loss. Differentiable w.r.t. z_t.
    """
    if t < 1:
        raise ValueError(f"t must be >= 1; got {t}")

    sq_err = (z_t - z_true).pow(2)

    # Frontier: d_i = t
    frontier = d == t
    L_f = sq_err[frontier].mean() if frontier.any() else sq_err.new_zeros(())

    # Already-filled: 0 < d_i < t. Strictly > 0 excludes K (where d_i = 0
    # and the squared error is structurally 0 anyway).
    filled = (d > 0) & (d < t)
    L_p = sq_err[filled].mean() if filled.any() else sq_err.new_zeros(())

    return lambda_f * L_f + lambda_p * L_p


def per_iteration_curvature_loss(
    z_t: torch.Tensor,
    edge_index: torch.Tensor,
    unknown_mask: torch.Tensor,
) -> torch.Tensor:
    """L_{curv,t} = mean_{i in U} kappa_i(z^t)^2.

    The umbrella Laplacian of z^t squared and averaged over unknown
    vertices. Zero where z^t is locally planar; large where it has
    high local variation. Encourages locally smooth predictions.

    Parameters
    ----------
    z_t : (n,) float
    edge_index : (2, n_directed_edges) int64
    unknown_mask : (n,) bool — True for vertices in U
    """
    if unknown_mask.dtype != torch.bool:
        raise TypeError(f"unknown_mask must be bool; got {unknown_mask.dtype}")

    kappa = compute_umbrella_laplacian(z_t, edge_index)         # (n,)
    sq = kappa.pow(2)
    if unknown_mask.any():
        return sq[unknown_mask].mean()
    return sq.new_zeros(())


def per_iteration_residual_loss(
    dz_prev: torch.Tensor,
    unknown_mask: torch.Tensor,
) -> torch.Tensor:
    """L_{res,t} = mean_{i in U} (Δz^{t-1}_i)^2.

    The squared per-iteration correction averaged over unknown vertices.
    Penalizes large jumps; encourages gradual refinement.

    Parameters
    ----------
    dz_prev : (n,) float — the Δz applied at the previous step
    unknown_mask : (n,) bool — True for vertices in U
    """
    if unknown_mask.dtype != torch.bool:
        raise TypeError(f"unknown_mask must be bool; got {unknown_mask.dtype}")

    sq = dz_prev.pow(2)
    if unknown_mask.any():
        return sq[unknown_mask].mean()
    return sq.new_zeros(())


# ======================================================================
# Total rollout loss (data + curvature + residual)
# ======================================================================
def rollout_loss(
    z_trajectory: list[torch.Tensor],
    dz_trajectory: list[torch.Tensor],
    z_true: torch.Tensor,
    d: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
    lambda_c: float = 0.01,
    lambda_r: float = 0.001,
    rollout_weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Total rollout loss: L = sum_t w_t (L_{data,t} + lc * L_{curv,t} + lr * L_{res,t}).

    Parameters
    ----------
    z_trajectory : list of length N+1
        z^0, z^1, ..., z^N from rollout().
    dz_trajectory : list of length N
        Δz^0, ..., Δz^{N-1} from rollout().
    z_true : (n,)
    d : (n,) int64
    edge_index : (2, n_directed_edges)
    mask : (n,) bool — True for KNOWN; ~mask gives U
    lambda_f, lambda_p, lambda_c, lambda_r : float
    rollout_weights : optional list of length N. Default: uniform (all 1).

    Returns
    -------
    dict with keys:
      - "total" : scalar tensor — the loss to backprop on
      - "data", "curv", "res" : per-component totals (each summed over t,
        with rollout weights applied). Useful for logging.
    """
    N = len(z_trajectory) - 1
    if N < 1:
        raise ValueError("z_trajectory must contain at least z^0 and z^1")
    if len(dz_trajectory) != N:
        raise ValueError(
            f"dz_trajectory has length {len(dz_trajectory)}, expected {N}"
        )
    if rollout_weights is None:
        rollout_weights = [1.0] * N
    if len(rollout_weights) != N:
        raise ValueError(
            f"rollout_weights has length {len(rollout_weights)}, expected {N}"
        )

    unknown_mask = ~mask

    zero = z_trajectory[0].new_zeros(())
    total_data = zero.clone()
    total_curv = zero.clone()
    total_res = zero.clone()

    for t in range(1, N + 1):
        w = rollout_weights[t - 1]

        L_data = per_iteration_data_loss(
            z_trajectory[t], z_true, d, t,
            lambda_f=lambda_f, lambda_p=lambda_p,
        )
        L_curv = per_iteration_curvature_loss(
            z_trajectory[t], edge_index, unknown_mask
        )
        L_res = per_iteration_residual_loss(
            dz_trajectory[t - 1], unknown_mask
        )

        total_data = total_data + w * L_data
        total_curv = total_curv + w * L_curv
        total_res = total_res + w * L_res

    total = total_data + lambda_c * total_curv + lambda_r * total_res

    return {
        "total": total,
        "data": total_data,
        "curv": total_curv,
        "res": total_res,
    }


# ======================================================================
# Backwards-compat alias: keep rollout_data_loss for Stage 5/6 tests
# ======================================================================
def rollout_data_loss(
    z_trajectory: list[torch.Tensor],
    z_true: torch.Tensor,
    d: torch.Tensor,
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
    rollout_weights: list[float] | None = None,
) -> torch.Tensor:
    """Data-only rollout loss (no regularizers). Used by Stage 5's overfit
    test and the keystone regression guard; the full rollout_loss is the
    real training objective from Stage 7 onward."""
    N = len(z_trajectory) - 1
    if N < 1:
        raise ValueError("z_trajectory must contain at least z^0 and z^1")
    if rollout_weights is None:
        rollout_weights = [1.0] * N
    if len(rollout_weights) != N:
        raise ValueError(
            f"rollout_weights has length {len(rollout_weights)}, expected {N}"
        )

    total = z_trajectory[0].new_zeros(())
    for t in range(1, N + 1):
        total = total + rollout_weights[t - 1] * per_iteration_data_loss(
            z_trajectory[t], z_true, d, t,
            lambda_f=lambda_f, lambda_p=lambda_p,
        )
    return total
