"""LocalOperator: the real GNN that predicts the per-iteration residual.

Architecture
------------
Input features per vertex (9 dims):
    (x, y, z^t, n_x, n_y, n_z, kappa, mask, d)

where (n_x, n_y, n_z) are the recomputed vertex normals from V_xy and z^t,
and kappa is the umbrella Laplacian of z^t. Both are recomputed at every
rollout iteration to capture the evolving geometry.

Pipeline:
    input MLP (9 -> H)
    SAGEConv (H -> H, mean aggr)
    ReLU
    SAGEConv (H -> H, mean aggr)
    output MLP (H -> H -> 1)

The final layer is initialized with small weights so that Δz ≈ 0 at the
start of training (prevents the first rollout iteration from making
destabilizing corrections).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from horizons.data.features import (
    compute_vertex_normals,
    compute_umbrella_laplacian,
)


class LocalOperator(nn.Module):
    """Real per-iteration operator F_Theta.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension H for the input projection and SAGE layers.
    n_message_passing : int
        Number of SAGEConv layers. Default 2 (the receptive field per
        rollout iteration is then 2 hops; iterating N times gives
        2*N effective hops).
    output_init_scale : float
        Std of the normal initialization for the final linear layer's
        weight. Smaller means smaller initial Δz, which is more stable
        for the rollout.
    """

    N_INPUT_FEATURES = 9  # (x, y, z, n_x, n_y, n_z, kappa, mask, d)

    def __init__(
        self,
        hidden_dim: int = 64,
        n_message_passing: int = 2,
        output_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if n_message_passing < 1:
            raise ValueError(
                f"n_message_passing must be >= 1; got {n_message_passing}"
            )

        # Input projection: 9 features -> hidden_dim
        self.input_proj = nn.Linear(self.N_INPUT_FEATURES, hidden_dim)

        # Message-passing stack
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(n_message_passing)
        ])

        # Output head: hidden -> hidden -> scalar
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Small init on the final linear so initial Δz ≈ 0
        final_linear = self.head[-1]
        nn.init.normal_(final_linear.weight, std=output_init_scale)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        z: torch.Tensor,             # (n,) — current scalar z^t
        V_xy: torch.Tensor,          # (n, 2) — fixed (x, y) per vertex
        edge_index: torch.Tensor,    # (2, n_directed_edges)
        F: torch.Tensor,             # (n_faces, 3) for normal computation
        mask: torch.Tensor,          # (n,) bool — True = known
        d: torch.Tensor,             # (n,) int64 — topological distance
    ) -> torch.Tensor:
        """Predict Δz from the current state z^t and static geometry.

        Returns
        -------
        dz : torch.Tensor, shape (n,), float
        """
        if z.dim() != 1:
            raise ValueError(f"z must be 1-D; got shape {tuple(z.shape)}")

        # Recompute dynamic geometric features from the current z^t.
        # V_t = (x, y, z^t) is the current 3D position of each vertex.
        V_t = torch.cat([V_xy, z.unsqueeze(1)], dim=1)            # (n, 3)
        normals = compute_vertex_normals(V_t, F)                   # (n, 3)
        kappa = compute_umbrella_laplacian(z, edge_index)          # (n,)

        # Assemble the 9-dim feature vector
        # Cast mask and d to float for tensor concat
        mask_f = mask.to(z.dtype).unsqueeze(1)                     # (n, 1)
        d_f = d.to(z.dtype).unsqueeze(1)                           # (n, 1)
        features = torch.cat([
            V_xy,                       # (n, 2): x, y
            z.unsqueeze(1),             # (n, 1): z^t
            normals,                    # (n, 3): n_x, n_y, n_z
            kappa.unsqueeze(1),         # (n, 1): kappa
            mask_f,                     # (n, 1): mask (0 or 1 as float)
            d_f,                        # (n, 1): d (int as float)
        ], dim=1)                       # (n, 9)

        # Input projection
        h = self.input_proj(features)                              # (n, H)

        # Message-passing stack (ReLU between layers, not after last)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = torch.relu(h)
        h = torch.relu(h)  # final activation before head

        # Output head -> scalar per vertex
        dz = self.head(h).squeeze(1)                               # (n,)
        return dz
