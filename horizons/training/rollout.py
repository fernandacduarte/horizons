"""Iterative rollout: apply the local operator F_Theta for N steps with
re-anchoring of known vertices at each step.

This is the core training-time computation. At each iteration:
  Δz^{t-1} = F_Theta(z^{t-1}, V_xy, edge_index, F, mask, d)
  z^t = z^{t-1} + Δz^{t-1}
  z^t[K] = z_true[K]   # re-anchor

The function returns the full trajectory of z^t and Δz^{t-1}, which the
loss function consumes (training supervises every t, not just the final z^N).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class LocalOperator(Protocol):
    """Type protocol for the operator F_Theta. Both TinySAGE 
    and the real LocalOperator satisfy this."""
    def __call__(
        self,
        z: torch.Tensor,             # (n,)
        V_xy: torch.Tensor,          # (n, 2)
        edge_index: torch.Tensor,    # (2, E)
        F: torch.Tensor,             # (n_faces, 3)
        mask: torch.Tensor,          # (n,) bool
        d: torch.Tensor,             # (n,) int64
    ) -> torch.Tensor:               # returns (n,) — Δz
        ...


@dataclass
class RolloutResult:
    """Outputs of a rollout.

    Attributes
    ----------
    z_trajectory : list[torch.Tensor]
        z^0, z^1, ..., z^N — each of shape (n_vertices,). Length N+1.
        z_trajectory[0] is the input z^0; z_trajectory[t] is the state
        after iteration t.
    dz_trajectory : list[torch.Tensor]
        Δz^0, Δz^1, ..., Δz^{N-1} — each of shape (n_vertices,). Length N.
        dz_trajectory[t] is the residual predicted at iteration t+1 (i.e.,
        the one applied to get from z^t to z^{t+1}).
    N : int
        The number of rollout iterations performed.
    """
    z_trajectory: list[torch.Tensor]
    dz_trajectory: list[torch.Tensor]
    N: int

    def z_final(self) -> torch.Tensor:
        return self.z_trajectory[-1]


def rollout(
    model: LocalOperator,
    *,
    z0: torch.Tensor,             # (n,)
    z_true: torch.Tensor,         # (n,) — for re-anchoring K
    V_xy: torch.Tensor,           # (n, 2)
    F: torch.Tensor,              # (n_faces, 3)
    edge_index: torch.Tensor,     # (2, E)
    mask: torch.Tensor,           # (n,) bool, True=known
    d: torch.Tensor,              # (n,) int64
    N: int,
    use_checkpoint: bool = False,
) -> RolloutResult:
    """Run the iterative rollout for N steps.

    Parameters
    ----------
    model : LocalOperator
        Any nn.Module conforming to the LocalOperator protocol.
    z0, z_true, V_xy, F, edge_index, mask, d :
        See the protocol and the dataset item structure.
    N : int
        Number of iterations to run. Must be >= 1.

    Returns
    -------
    RolloutResult
        Trajectory of z^t (length N+1) and Δz (length N). The autograd
        graph spans all N iterations; calling .backward() on a loss
        constructed from this trajectory will backprop through all of them.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1; got {N}")

    z_traj: list[torch.Tensor] = [z0]
    dz_traj: list[torch.Tensor] = []

    z_t = z0
    for _ in range(N):
        # Predict the residual. With use_checkpoint, gradient-checkpoint the
        # operator forward: its (per-edge) activations are recomputed during
        # backward instead of retained, turning peak memory from O(N) to O(1)
        # in the rollout depth. Needed for memory-heavy operators (EdgeConv)
        # and large surfaces. use_reentrant=False so gradients still reach the
        # model parameters even at step 0, where z_t (the only tensor input)
        # does not yet require grad.
        if use_checkpoint:
            dz_t = grad_checkpoint(
                model, z_t, V_xy, edge_index, F, mask, d,
                use_reentrant=False,
            )
        else:
            dz_t = model(z_t, V_xy, edge_index, F, mask, d)

        # Apply correction
        z_t_unanchored = z_t + dz_t

        # Re-anchor: known vertices return to z_true; unknown vertices
        # keep the corrected value. torch.where is differentiable;
        # gradients only flow through the unknown branch.
        z_t = torch.where(mask, z_true, z_t_unanchored)

        z_traj.append(z_t)
        dz_traj.append(dz_t)

    return RolloutResult(z_trajectory=z_traj, dz_trajectory=dz_traj, N=N)
