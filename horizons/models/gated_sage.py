"""GatedSAGEConv: SAGE with learned per-neighbour trust gates.

Motivation
----------
`SAGEConv` is *isotropic*: its update `W1·h_i + W2·mean_j(h_j)` weights every
neighbour equally. For this problem that is the wrong prior. The rollout
pushes information from the *known* region outward into the *unknown* one, so
a vertex's neighbours are systematically unequal in reliability — a neighbour
one ring from the known data carries far better information than one seven
rings out, which during the early iterations still mostly carries the initial
guess. The network sees `mask` and `d` as input *features*, but a mean
aggregator can only let them shift a message, never discount it.

GatedSAGEConv adds the GatedGCN / anisotropic mechanism (Bresson & Laurent
2017) to SAGE's self-term + neighbour-term structure:

    gate:    e_ij = sigmoid(A·h_i + B·h_j)              (n_edges, H)
    message: m_ij = e_ij * (W_nbr·h_j)                  (n_edges, H)
    aggr:    m_i  = sum_j m_ij / (sum_j e_ij + eps)     gated mean
    update:  h_i' = W_self·h_i + m_i

Design notes
------------
* Normalising by `sum_j e_ij` rather than by degree is what makes this a
  gated *mean* rather than a gated sum: the output scale stays comparable to
  `SAGEConv(aggr="mean")`, so `output_init_scale`, the learning rate and the
  loss weights transfer from the SAGE baseline without retuning. `eps` guards
  the fully-closed-gate and no-neighbour cases.
* The self term `W_self·h_i` is kept (SAGE's `root_weight=True`); without it
  a closed gate would zero the vertex out entirely.
* The gate is per-channel (H dims, not a scalar) so different feature
  channels can have different trust profiles.
* Parameters are ~2x SAGEConv at equal width (4·H² vs 2·H²) — worth keeping
  in mind when reading a gated-vs-sage comparison at matched `hidden_dim`.
* Numerator and denominator are aggregated in a single `propagate` call by
  concatenating `[e_ij * W_nbr·h_j, e_ij]` and splitting afterwards.

The gate depends only on the pair `(h_i, h_j)`, so the layer stays
permutation-equivariant and strictly 1-hop local — the two invariants
`tests/test_operator.py` checks for the operator as a whole.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GatedSAGEConv(MessagePassing):
    """SAGE-style convolution with a learned sigmoid gate per edge.

    Parameters
    ----------
    in_channels : int
        Input feature width.
    out_channels : int
        Output feature width.
    aggr : str
        How the gated messages are combined:
        - "mean" (default): gated mean, normalised by the gate sum.
        - "add" / "sum": gated sum, no normalisation. Degree-sensitive, so
          expect a different output scale than the SAGE baseline.
        Anything else raises: a gated "max" is not well defined here (the
        gate would rescale candidates before a non-linear selection), and
        silently ignoring the argument would be worse than failing.
    eps : float
        Added to the gate-sum denominator. Also what makes a vertex with no
        incoming edges return exactly `W_self·h_i` instead of 0/0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        if aggr not in ("mean", "add", "sum"):
            raise ValueError(
                f"GatedSAGEConv does not support aggr={aggr!r}; "
                f"expected 'mean', 'add' or 'sum'"
            )
        # The layer always sums internally; `aggr` selects the normalisation.
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gated_aggr = aggr
        self.normalize = aggr == "mean"
        self.eps = eps

        self.lin_self = nn.Linear(in_channels, out_channels)               # W_self
        self.lin_nbr = nn.Linear(in_channels, out_channels, bias=False)    # W_nbr
        self.gate_dst = nn.Linear(in_channels, out_channels)               # A
        self.gate_src = nn.Linear(in_channels, out_channels, bias=False)   # B

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_self.reset_parameters()
        self.lin_nbr.reset_parameters()
        self.gate_dst.reset_parameters()
        self.gate_src.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,             # (n, in_channels)
        edge_index: torch.Tensor,    # (2, n_directed_edges)
    ) -> torch.Tensor:
        """Returns (n, out_channels)."""
        # Node-level projections done once; `message` only does the
        # elementwise work per edge.
        v = self.lin_nbr(x)
        g_dst = self.gate_dst(x)
        g_src = self.gate_src(x)

        out = self.propagate(edge_index, v=v, g_dst=g_dst, g_src=g_src)
        num, den = out.split(self.out_channels, dim=-1)
        if self.normalize:
            num = num / (den + self.eps)

        return self.lin_self(x) + num

    def message(
        self,
        v_j: torch.Tensor,        # (n_edges, out_channels) — W_nbr·h_j
        g_dst_i: torch.Tensor,    # (n_edges, out_channels) — A·h_i
        g_src_j: torch.Tensor,    # (n_edges, out_channels) — B·h_j
    ) -> torch.Tensor:
        """Gated message plus the gate itself, so one `add` aggregation
        yields both the numerator and the denominator of the gated mean."""
        e = torch.sigmoid(g_dst_i + g_src_j)
        return torch.cat([e * v_j, e], dim=-1)

    def __repr__(self) -> str:
        # MessagePassing defines its own __repr__ (which ignores extra_repr
        # and drops aggr), so override it to keep `print(model)` informative.
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.gated_aggr!r})"
        )
