"""Loss functions for masked-rollout training.

This module starts with just the data term. The full loss with
curvature and residual regularizers will be added later.
"""
from __future__ import annotations

import torch


def per_iteration_data_loss(
    z_t: torch.Tensor,           # (n,) — z^t, the prediction at iteration t
    z_true: torch.Tensor,        # (n,) — ground truth
    d: torch.Tensor,             # (n,) — topological distance from K
    t: int,                      # iteration index, 1 <= t <= N
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
    t : int — iteration index (>= 1)
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


def rollout_data_loss(
    z_trajectory: list[torch.Tensor],  # length N+1: z^0, z^1, ..., z^N
    z_true: torch.Tensor,              # (n,)
    d: torch.Tensor,                   # (n,) int64
    lambda_f: float = 1.0,
    lambda_p: float = 0.1,
    rollout_weights: list[float] | None = None,
) -> torch.Tensor:
    """Total data loss over the full rollout: L = sum_t w_t L_{data,t}.

    Parameters
    ----------
    z_trajectory : list[torch.Tensor], length N+1
        Output of rollout(). Index 0 is z^0 (not supervised); indices 1..N
        are z^1, ..., z^N (each supervised against its frontier ring).
    z_true : (n,)
    d : (n,) int64
    lambda_f, lambda_p : float
        Per-iteration weights (see per_iteration_data_loss).
    rollout_weights : optional list of length N
        w_t for each iteration. Default: uniform (all 1).

    Returns
    -------
    L : scalar torch.Tensor
    """
    N = len(z_trajectory) - 1
    if N < 1:
        raise ValueError(f"z_trajectory must contain at least z^0 and z^1")
    if rollout_weights is None:
        rollout_weights = [1.0] * N
    if len(rollout_weights) != N:
        raise ValueError(
            f"rollout_weights has length {len(rollout_weights)}, expected {N}"
        )

    total = z_trajectory[0].new_zeros(())
    for t in range(1, N + 1):
        L_t = per_iteration_data_loss(
            z_t=z_trajectory[t],
            z_true=z_true,
            d=d,
            t=t,
            lambda_f=lambda_f,
            lambda_p=lambda_p,
        )
        total = total + rollout_weights[t - 1] * L_t
    return total
