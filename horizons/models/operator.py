"""LocalOperator: the real GNN that predicts the per-iteration residual.

Architecture
------------
Input features per vertex (9 dims):
    (x, y, z^t, n_x, n_y, n_z, kappa, mask, d)

where (n_x, n_y, n_z) are the recomputed vertex normals from V_xy and z^t,
and kappa is the umbrella Laplacian of z^t. Both are recomputed at every
rollout iteration to capture the evolving geometry.

With use_mask_feature=False the mask column is dropped and the input is
8-dim — the ablation that measures whether the mask feature helps.

Default pipeline:
    input MLP (9 -> H -> H)
    SAGEConv (H -> H, mean aggr)
    ReLU
    SAGEConv (H -> H, mean aggr)
    output MLP (H -> H -> 1)

input_proj_layers=1 replaces the input MLP with a single Linear (9 -> H).
That was the form used for the runs up to July 2026, so their checkpoints
need it to load.

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
from horizons.models.gated_sage import GatedSAGEConv


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
        Which message-passing operator to use: "sage" (SAGEConv, default),
        "edgeconv" (EdgeConv / DGCNN, whose edge messages use the neighbour
        difference h_j - h_i), or "gated_sage" (GatedSAGEConv, which weights
        each neighbour message by a learned sigmoid gate so the operator can
        discount unreliable neighbours instead of averaging them in).
    aggr : str
        Neighbour aggregation for each layer ("mean", "max", ...), passed
        straight to the underlying conv. SAGE uses "mean"; EdgeConv is
        canonically "max"; GatedSAGE accepts only "mean" or "add".
    mask_mode : str
        How the mask input feature is encoded:
        - "binary" (default): 1.0 for known vertices, 0.0 for unknown.
        - "soft_distance": known vertices keep 1.0; unknown vertices get
          a reliability ramp m_U = 1 - d/(N+1), where N = max d over the
          surface, so reliability decays linearly with distance from the
          known set. Unreachable vertices (d = -1) get 0.0.
        Ignored when use_mask_feature is False.
    use_mask_feature : bool
        Whether the mask is fed to the network as an input feature.
        True (default) gives the 9-dim input above. False drops the mask
        column, leaving 8 dims — the ablation arm for measuring whether
        the mask feature helps or hurts. Note this only removes the
        *explicit* mask channel: the topological-distance feature d is
        still derived from the mask (d = 0 exactly on the known set), so
        the network is not blind to which vertices are known.
        This changes the input layer's shape, so checkpoints trained with
        one setting cannot be loaded into a model built with the other.
    input_proj_layers : int
        Depth of the input projection: 2 (default) for the MLP
        (Linear -> ReLU -> Linear), 1 for a single Linear. Only affects
        which checkpoints can be loaded; see `load_checkpoint`, which infers
        it from the saved state dict.
    """

    N_INPUT_FEATURES = 9  # (x, y, z, n_x, n_y, n_z, kappa, mask, d)

    def __init__(
        self,
        hidden_dim: int = 64,
        n_message_passing: int = 2,
        output_init_scale: float = 0.01,
        conv_type: str = "sage",
        aggr: str = "mean",
        mask_mode: str = "binary",
        use_mask_feature: bool = True,
        input_proj_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_message_passing < 1:
            raise ValueError(
                f"n_message_passing must be >= 1; got {n_message_passing}"
            )
        if mask_mode not in ("binary", "soft_distance"):
            raise ValueError(
                f"unknown mask_mode {mask_mode!r}; "
                f"expected 'binary' or 'soft_distance'"
            )
        if input_proj_layers not in (1, 2):
            raise ValueError(
                f"input_proj_layers must be 1 or 2; got {input_proj_layers}"
            )

        self.conv_type = conv_type
        self.aggr = aggr
        self.mask_mode = mask_mode
        self.use_mask_feature = use_mask_feature

        # Input projection: 9 (or 8 without the mask column) -> hidden_dim
        self.n_input_features = (
            self.N_INPUT_FEATURES if use_mask_feature
            else self.N_INPUT_FEATURES - 1
        )
        self.input_proj_layers = input_proj_layers
        if input_proj_layers == 1:
            self.input_proj = nn.Linear(self.n_input_features, hidden_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(self.n_input_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

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
        - "gated_sage": GatedSAGEConv — SAGE's self-term plus a *gated* mean,
          each neighbour message scaled by e_ij = sigmoid(A·h_i + B·h_j).
          The anisotropy is the point: neighbours here are unequally
          reliable (they sit at different distances from the known set) and
          a plain mean cannot discount them.
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
        if conv_type == "gated_sage":
            return GatedSAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        raise ValueError(
            f"Unknown conv_type {conv_type!r}; "
            f"expected 'sage', 'edgeconv' or 'gated_sage'"
        )

    def _mask_feature(
        self,
        mask: torch.Tensor,          # (n,) bool — True = known
        d: torch.Tensor,             # (n,) int64 — topological distance
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode the mask input feature according to mask_mode.

        "binary": 1.0 for known, 0.0 for unknown.
        "soft_distance": known vertices keep 1.0; unknown vertices get the
        reliability ramp m_U = 1 - d/(N+1) with N = max d over the surface
        (so the farthest unknown ring still gets 1/(N+1) > 0, and the ramp
        is scale-free across surfaces of different depths). Unreachable
        vertices (d = -1 sentinel) get 0.0 — least reliable of all.

        Returns
        -------
        mask_f : torch.Tensor, shape (n, 1), dtype `dtype`
        """
        if self.mask_mode == "binary":
            return mask.to(dtype).unsqueeze(1)

        # soft_distance. N = surface depth (max reachable d); clamp guards
        # the degenerate all-known / all-unreachable cases (denominator >= 2).
        depth = d.max().clamp(min=1).to(dtype)
        m = 1.0 - d.to(dtype) / (depth + 1.0)
        m = torch.where(mask, torch.ones_like(m), m)   # known: exactly 1
        m = torch.where(d < 0, torch.zeros_like(m), m)  # unreachable: 0
        return m.unsqueeze(1)

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

        # Assemble the feature vector (9-dim, or 8-dim without the mask
        # column when use_mask_feature is False).
        # Cast mask and d to float for tensor concat
        d_f = d.to(z.dtype).unsqueeze(1)                           # (n, 1)
        columns = [
            V_xy,                       # (n, 2): x, y
            z.unsqueeze(1),             # (n, 1): z^t
            normals,                    # (n, 3): n_x, n_y, n_z
            kappa.unsqueeze(1),         # (n, 1): kappa
        ]
        if self.use_mask_feature:
            columns.append(self._mask_feature(mask, d, z.dtype))   # (n, 1)
        columns.append(d_f)             # (n, 1): d (int as float)
        features = torch.cat(columns, dim=1)        # (n, 9) or (n, 8)

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
