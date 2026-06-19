"""LocalOperator: the real GNN that predicts the per-iteration residual.

Architecture
------------
Input features per vertex (9 dims):
    (x, y, z^t, n_x, n_y, n_z, kappa, mask, d)

where (n_x, n_y, n_z) are the recomputed vertex normals from V_xy and z^t,
and kappa is the umbrella Laplacian of z^t. Both are recomputed at every
rollout iteration to capture the evolving geometry.

Default pipeline:
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
from torch_geometric.nn import SAGEConv, EdgeConv

from horizons.data.features import (
    compute_vertex_normals,
    compute_umbrella_laplacian,
)


class LocalOperator(nn.Module):
    """Real per-iteration operator F_Theta.

        Parameters
    ----------
    hidden_dim : int
        Hidden dimension H for the input projection and message-passing layers.
    n_message_passing : int
        Number of message-passing layers. Default 2 (the receptive field per
        rollout iteration is then 2 hops; iterating N times gives
        2*N effective hops).
    output_init_scale : float
        Std of the normal initialization for the final linear layer's
        weight. Smaller means smaller initial Δz, which is more stable
        for the rollout.
    conv_type : str
        Which message-passing operator to use: "sage" (SAGEConv, default) 
        or "edgeconv" (EdgeConv / DGCNN,
        whose edge messages use the neighbour difference h_j - h_i).
    aggr : str
        Neighbour aggregation for each layer ("mean", "max", ...), passed
        straight to the underlying conv. SAGE uses "mean"; EdgeConv is
        canonically "max".
    """

    N_INPUT_FEATURES = 9  # (x, y, z, n_x, n_y, n_z, kappa, mask, d)

    def __init__(
        self,
        hidden_dim: int = 64,
        n_message_passing: int = 2,
        output_init_scale: float = 0.01,
        conv_type: str = "sage",
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if n_message_passing < 1:
            raise ValueError(
                f"n_message_passing must be >= 1; got {n_message_passing}"
            )

        self.conv_type = conv_type
        self.aggr = aggr

        # Input projection: 9 features -> hidden_dim
        self.input_proj = nn.Linear(self.N_INPUT_FEATURES, hidden_dim)

        # Message-passing stack (operator chosen by conv_type)
        self.convs = nn.ModuleList([
            self._make_conv(conv_type, hidden_dim, aggr)
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

    @staticmethod
    def _make_conv(conv_type: str, hidden_dim: int, aggr: str) -> nn.Module:
        """Build one message-passing layer of the requested type.

        - "sage": SAGEConv — keeps a self-term W1·h_i plus aggregated
          neighbour features. Default;
        - "edgeconv": EdgeConv (DGCNN) — each edge message is
          h_Θ([h_i, h_j - h_i]); the explicit neighbour *difference*
          gives a local-gradient inductive bias (the reason we're trying it).
        """
        if conv_type == "sage":
            return SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        if conv_type == "edgeconv":
            mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return EdgeConv(mlp, aggr=aggr)
        raise ValueError(
            f"Unknown conv_type {conv_type!r}; expected 'sage' or 'edgeconv'"
        )

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

        # Message-passing stack with ReLU between every layer.
        # The final ReLU before the head gives the readout MLP a
        # non-linear input to combine.
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)

        # Output head -> scalar per vertex
        dz = self.head(h).squeeze(1)                               # (n,)
        return dz
