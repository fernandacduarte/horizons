"""Unit tests for GatedSAGEConv (horizons/models/gated_sage.py).

Layer-level only — small hand-built graphs, no mesh fixture. The operator-level
tests (shape, gradients, locality through LocalOperator) live in
tests/test_operator.py::TestConvType.
"""
import pytest
import torch

from horizons.models.gated_sage import GatedSAGEConv


def _open_all_gates(conv: GatedSAGEConv) -> None:
    """Zero the gate parameters so every e_ij is exactly sigmoid(0) = 0.5.

    With a constant gate the normalisation cancels and the layer must reduce
    to plain SAGE: W_self·h_i + mean_j(W_nbr·h_j).
    """
    with torch.no_grad():
        conv.gate_dst.weight.zero_()
        conv.gate_dst.bias.zero_()
        conv.gate_src.weight.zero_()


# A 4-node path 0-1-2 plus an isolated node 3, as directed edges (j -> i).
PATH_EDGES = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)


class TestShape:
    def test_output_shape_and_dtype(self) -> None:
        conv = GatedSAGEConv(5, 7)
        x = torch.randn(4, 5)
        out = conv(x, PATH_EDGES)
        assert out.shape == (4, 7)
        assert out.dtype == x.dtype

    def test_gradients_flow_to_every_parameter(self) -> None:
        conv = GatedSAGEConv(5, 7)
        x = torch.randn(4, 5)
        conv(x, PATH_EDGES).sum().backward()
        for name, p in conv.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


class TestGatedMean:
    def test_constant_gate_reduces_to_sage_mean(self) -> None:
        """The key correctness check: a constant gate must give the plain
        SAGE update W_self·h_i + mean_j(W_nbr·h_j)."""
        conv = GatedSAGEConv(3, 3)
        _open_all_gates(conv)
        x = torch.randn(4, 3)

        out = conv(x, PATH_EDGES)

        v = conv.lin_nbr(x)
        expected = conv.lin_self(x).clone()
        expected[0] = expected[0] + v[1]                 # nbrs of 0: {1}
        expected[1] = expected[1] + (v[0] + v[2]) / 2    # nbrs of 1: {0, 2}
        expected[2] = expected[2] + v[1]                 # nbrs of 2: {1}
        # node 3 is isolated -> self term only
        assert torch.allclose(out, expected, atol=1e-6)

    def test_isolated_vertex_is_self_term_only(self) -> None:
        """No incoming edges means num = den = 0; eps must keep it finite
        and leave exactly W_self·h_i."""
        conv = GatedSAGEConv(3, 3)
        x = torch.randn(4, 3)
        out = conv(x, PATH_EDGES)
        assert torch.isfinite(out).all()
        assert torch.allclose(out[3], conv.lin_self(x)[3], atol=1e-6)

    def test_closed_gate_removes_the_neighbour(self) -> None:
        """Driving the gate to ~0 for one of two neighbours must leave the
        other one's message alone (this is what the gating buys)."""
        conv = GatedSAGEConv(1, 1)
        with torch.no_grad():
            conv.lin_self.weight.fill_(1.0)
            conv.lin_self.bias.zero_()
            conv.lin_nbr.weight.fill_(1.0)
            # gate = sigmoid(0·h_i + 20·h_j): saturates open for h_j = +1,
            # shut for h_j = -1.
            conv.gate_dst.weight.zero_()
            conv.gate_dst.bias.zero_()
            conv.gate_src.weight.fill_(20.0)

        # Node 0 has neighbours 1 (h = +1, gate open) and 2 (h = -1, shut).
        x = torch.tensor([[0.0], [1.0], [-1.0]])
        edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)

        out = conv(x, edge_index)
        # Gated mean over an open (1.0) and a shut (~0) gate -> the open
        # neighbour's value, +1, plus the self term (0).
        assert out[0].item() == pytest.approx(1.0, abs=1e-3)

    def test_add_aggr_skips_normalisation(self) -> None:
        """aggr='add' is the gated *sum*: degree-sensitive, no division."""
        conv = GatedSAGEConv(3, 3, aggr="add")
        _open_all_gates(conv)
        x = torch.randn(4, 3)

        out = conv(x, PATH_EDGES)

        v = conv.lin_nbr(x)
        expected = conv.lin_self(x).clone()
        expected[0] = expected[0] + 0.5 * v[1]
        expected[1] = expected[1] + 0.5 * (v[0] + v[2])
        expected[2] = expected[2] + 0.5 * v[1]
        assert torch.allclose(out, expected, atol=1e-6)


class TestEquivariance:
    def test_permutation_equivariance(self) -> None:
        """Relabelling the vertices must permute the output identically —
        the same invariant tests/test_operator.py checks for the operator."""
        conv = GatedSAGEConv(4, 4)
        conv.eval()
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4],
                                   [1, 0, 2, 1, 4, 3]], dtype=torch.long)

        out = conv(x, edge_index)

        perm = torch.tensor([3, 0, 4, 2, 1])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        out_perm = conv(x[perm], inv[edge_index])

        assert torch.allclose(out_perm, out[perm], atol=1e-6)


class TestValidation:
    def test_unsupported_aggr_raises(self) -> None:
        with pytest.raises(ValueError, match="aggr"):
            GatedSAGEConv(4, 4, aggr="max")
