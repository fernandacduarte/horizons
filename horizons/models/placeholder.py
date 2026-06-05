"""Placeholder GNN: a single SAGEConv layer mapping z^t to Δz.

Used only to exercise the rollout machinery. The real operator
is in horizons/models/operator.py with normals, curvature,
topological distance, mask, and a 2-layer SAGE backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class TinySAGE(nn.Module):
    """Single SAGEConv layer: z_t -> Δz.

    Input features per vertex: just the scalar z^t.
    Output: scalar Δz per vertex.

    The model is deliberately tiny. Its only purpose is to validate that
    the rollout/loss/backward pipeline can drive a single-example loss to
    zero. The real model handles all 9 features and uses 2 layers.
    """

    def __init__(self, hidden_dim: int = 16, output_init_scale: float = 0.01):
        super().__init__()
        self.conv = SAGEConv(in_channels=1, out_channels=hidden_dim,
                              aggr="mean")
        self.head = nn.Linear(hidden_dim, 1)

        # Initialize the head with small weights so Δz ≈ 0 at the start
        # of training. This is the same trick we'll use for the real model,
        # it prevents the first rollout iteration from
        # making large, destabilizing corrections.
        nn.init.normal_(self.head.weight, std=output_init_scale)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        z: torch.Tensor,             # (n_vertices,) — current scalar prediction
        V_xy: torch.Tensor,          # (n_vertices, 2) — UNUSED by placeholder
        edge_index: torch.Tensor,    # (2, n_directed_edges)
        F: torch.Tensor,             # (n_faces, 3) — UNUSED by placeholder
        mask: torch.Tensor,          # (n_vertices,) bool — UNUSED by placeholder
        d: torch.Tensor,             # (n_vertices,) int64 — UNUSED by placeholder
    ) -> torch.Tensor:
        """Predict Δz per vertex from z^t.

        The placeholder ignores all features except z. The real operator
        uses all of them.

        Returns
        -------
        dz : torch.Tensor, shape (n_vertices,)
        """
        if z.dim() != 1:
            raise ValueError(f"z must be 1-D; got shape {tuple(z.shape)}")

        x = z.unsqueeze(1)                  # (n, 1) — SAGEConv wants (n, F)
        h = self.conv(x, edge_index)        # (n, hidden_dim)
        h = torch.relu(h)
        dz = self.head(h).squeeze(1)        # (n,)
        return dz

"""
Diagram of the forward pass:

z
shape: (n,)
one scalar height per vertex
        │
        ▼
unsqueeze(1)
shape: (n, 1)
        │
        ▼
SAGEConv
shape: (n, hidden_dim)
each vertex now includes neighborhood information
        │
        ▼
ReLU
shape: (n, hidden_dim)
        │
        ▼
Linear head
shape: (n, 1)
        │
        ▼
squeeze(1)
shape: (n,)
one predicted Δz per vertex
"""