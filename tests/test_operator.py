"""Unit tests for horizons.models.operator.LocalOperator."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.topo_distance import compute_topological_distance
from horizons.models.operator import LocalOperator


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


def _build_inputs(surface: HorizonSurface) -> dict:
    """Build a complete input bundle for the operator (no masking; just
    everything-known so we can focus on the operator itself)."""
    n = surface.n_vertices
    return {
        "z": surface.V[:, 2],
        "V_xy": surface.V[:, :2],
        "edge_index": surface.edge_index,
        "F": surface.F,
        "mask": torch.ones(n, dtype=torch.bool),
        "d": torch.zeros(n, dtype=torch.int64),
    }


# ----------------------------------------------------------------------
# Shape & dtype
# ----------------------------------------------------------------------
class TestShape:
    def test_output_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        model = LocalOperator()
        inputs = _build_inputs(anticline)
        dz = model(**inputs)
        assert dz.shape == (anticline.n_vertices,)
        assert dz.dtype == anticline.V.dtype

    def test_initial_dz_is_small(self, anticline: HorizonSurface) -> None:
        """With output_init_scale=0.01, initial Δz should be much smaller
        than typical z magnitudes — keeps the rollout stable at iteration 1."""
        model = LocalOperator(output_init_scale=0.01)
        inputs = _build_inputs(anticline)
        dz = model(**inputs)
        # Anticline z spans roughly -1 to 5. Δz should be much smaller.
        assert dz.abs().mean() < 0.5


# ----------------------------------------------------------------------
# Permutation equivariance
# ----------------------------------------------------------------------
class TestEquivariance:
    def test_permutation_equivariance(self, anticline: HorizonSurface) -> None:
        """Permuting vertex order should permute outputs identically.

        Equivariance is the defining property of a correctly-implemented
        GNN — it's what justifies the model treating "the i-th vertex"
        as having no special meaning beyond its connectivity and features.
        """
        torch.manual_seed(0)
        model = LocalOperator()
        model.eval()  # for deterministic outputs

        inputs = _build_inputs(anticline)
        dz_orig = model(**inputs)

        # Random permutation of the n_vertices vertices
        n = anticline.n_vertices
        perm = torch.randperm(n)
        inv_perm = torch.argsort(perm)  # so inv_perm[perm] = identity

        # Permute V, mask, d directly
        V_perm = anticline.V[perm]
        mask_perm = torch.ones(n, dtype=torch.bool)  # all True regardless
        d_perm = torch.zeros(n, dtype=torch.int64)

        # Faces must be re-indexed: old index i becomes inv_perm[i]
        F_perm = inv_perm[anticline.F]

        # Edge index must be re-indexed too
        ei_perm = inv_perm[anticline.edge_index]

        inputs_perm = {
            "z": V_perm[:, 2],
            "V_xy": V_perm[:, :2],
            "edge_index": ei_perm,
            "F": F_perm,
            "mask": mask_perm,
            "d": d_perm,
        }
        dz_perm = model(**inputs_perm)

        # The permuted output, undone, should equal the original output
        dz_unpermuted = dz_perm[inv_perm]
        assert torch.allclose(dz_orig, dz_unpermuted, atol=1e-5), (
            f"Permutation equivariance violated. "
            f"Max diff: {(dz_orig - dz_unpermuted).abs().max().item()}"
        )


# ----------------------------------------------------------------------
# Locality: 2-layer SAGE has 2-hop receptive field
# ----------------------------------------------------------------------
class TestLocality:
    def test_far_perturbation_does_not_affect_local_output(
        self, anticline: HorizonSurface
    ) -> None:
        """The model has 2 SAGEConv layers, so each vertex's output depends
        only on its 2-hop neighborhood. Perturbing z on a vertex 3+ hops
        away from i should not change Δz_i at all.

        Note about features: x, y, and d are static — perturbing z doesn't
        change them. Normals and curvature DO depend on z, but they're
        computed per-vertex from the LOCAL neighborhood of each vertex
        (curvature uses only direct neighbors; normals use only the
        incident faces of that vertex). So a vertex's features depend on
        a 1-hop neighborhood. With 2 SAGEConv layers on top, the total
        receptive field is 3 hops.

        So: perturbing z at vertex j only changes Δz_i if i is within
        3 hops of j.
        """
        torch.manual_seed(0)
        model = LocalOperator()
        model.eval()

        inputs = _build_inputs(anticline)
        dz_orig = model(**inputs)

        # Find some pair (i, j) where j is far from i in graph distance
        # Use BFS distance from a single source as a stand-in.
        source = torch.zeros(anticline.n_vertices, dtype=torch.bool)
        source[0] = True  # vertex 0 is the source
        d_from_0 = compute_topological_distance(anticline.edge_index, source)

        # Find a vertex j that is far (>= 5 hops) from vertex 0
        far_vertices = (d_from_0 >= 5).nonzero(as_tuple=True)[0]
        assert far_vertices.numel() > 0, (
            "No vertex 5+ hops from vertex 0 in this fixture; pick a different fixture."
        )
        j = far_vertices[0].item()

        # Perturb z at vertex j by a large amount
        z_perturbed = inputs["z"].clone()
        z_perturbed[j] += 100.0
        inputs_perturbed = {**inputs, "z": z_perturbed}
        dz_perturbed = model(**inputs_perturbed)

        # Vertex 0 (4+ hops from j) should be unaffected
        assert dz_orig[0].item() == pytest.approx(dz_perturbed[0].item(), abs=1e-5), (
            f"Locality violated: perturbing vertex {j} (d=5+ from vertex 0) "
            f"changed Δz[0] from {dz_orig[0].item():.6f} to "
            f"{dz_perturbed[0].item():.6f}"
        )

    def test_local_perturbation_does_affect_local_output(
        self, anticline: HorizonSurface
    ) -> None:
        """The complement of the above test: perturbing z at a vertex's
        immediate neighbor SHOULD change Δz at that vertex. This confirms
        the model isn't trivially constant."""
        torch.manual_seed(0)
        model = LocalOperator()
        model.eval()

        inputs = _build_inputs(anticline)
        dz_orig = model(**inputs)

        # Find a neighbor of vertex 0
        src, dst = anticline.edge_index
        nbrs_of_0 = dst[src == 0]
        assert nbrs_of_0.numel() > 0, "Vertex 0 has no neighbors?"
        j = nbrs_of_0[0].item()

        z_perturbed = inputs["z"].clone()
        z_perturbed[j] += 1.0  # smaller perturbation, still detectable
        dz_perturbed = model(**{**inputs, "z": z_perturbed})

        assert dz_orig[0].item() != pytest.approx(dz_perturbed[0].item(), abs=1e-5), (
            "Output insensitive to local input changes — model is "
            "effectively constant w.r.t. z"
        )


# ----------------------------------------------------------------------
# Feature recomputation differentiability
# ----------------------------------------------------------------------
class TestDifferentiability:
    def test_gradient_flows_to_model_parameters(
        self, anticline: HorizonSurface
    ) -> None:
        """A loss on Δz should produce non-zero gradients on all model
        parameters: input projection, both SAGE layers, and the head."""
        torch.manual_seed(0)
        model = LocalOperator()
        inputs = _build_inputs(anticline)
        dz = model(**inputs)
        loss = dz.pow(2).sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient on {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"

    # ------------------------------------------------------------------
    # The output Δz depends on z through three independent paths:
    #   (a) z as a direct input feature in the feature tensor
    #   (b) z via the recomputed vertex normals (V_t includes z)
    #   (c) z via the recomputed umbrella Laplacian (kappa)
    #
    # A test using only the full model can pass even if paths (b) and (c)
    # are accidentally broken — path (a) alone would produce non-zero
    # gradients. So we test each path *independently* by calling the
    # feature functions directly. These tests guard the actual claim:
    # that feature recomputation is wired into the autograd graph.
    # ------------------------------------------------------------------
    def test_gradient_flows_via_direct_z_input(
        self, anticline: HorizonSurface
    ) -> None:
        """Path (a): z enters the feature tensor directly. Verified via
        the full model. This is the weakest claim but the most direct test."""
        torch.manual_seed(0)
        model = LocalOperator()
        inputs = _build_inputs(anticline)
        z = inputs["z"].clone().requires_grad_(True)
        inputs["z"] = z
        dz = model(**inputs)
        dz.sum().backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0
        # Most vertices should have non-zero gradient
        assert (z.grad.abs() > 1e-6).float().mean() > 0.5

    def test_gradient_flows_through_recomputed_normals(
        self, anticline: HorizonSurface
    ) -> None:
        """Path (b): test compute_vertex_normals in isolation. If anyone
        adds .detach() or a numpy conversion inside that function, the
        gradient w.r.t. z will be zero and this test will catch it.

        Note: gradcheck in tests/test_features.py verifies the gradient
        VALUE; this test verifies the gradient EXISTS through z (since
        z is part of V_t = (x, y, z) feeding into compute_vertex_normals).
        """
        from horizons.data.features import compute_vertex_normals
        inputs = _build_inputs(anticline)
        z = inputs["z"].clone().requires_grad_(True)
        V_t = torch.cat([inputs["V_xy"], z.unsqueeze(1)], dim=1)
        normals = compute_vertex_normals(V_t, inputs["F"])
        normals.pow(2).sum().backward()
        assert z.grad is not None, (
            "compute_vertex_normals does not pass gradient through z. "
            "Check for .detach() or numpy conversion."
        )
        assert z.grad.abs().sum() > 0

    def test_gradient_flows_through_recomputed_kappa(
        self, anticline: HorizonSurface
    ) -> None:
        """Path (c): test compute_umbrella_laplacian in isolation."""
        from horizons.data.features import compute_umbrella_laplacian
        inputs = _build_inputs(anticline)
        z = inputs["z"].clone().requires_grad_(True)
        kappa = compute_umbrella_laplacian(z, inputs["edge_index"])
        kappa.pow(2).sum().backward()
        assert z.grad is not None, (
            "compute_umbrella_laplacian does not pass gradient through z. "
            "Check for .detach() or numpy conversion."
        )
        assert z.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestValidation:
    def test_invalid_n_message_passing_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_message_passing"):
            LocalOperator(n_message_passing=0)

# ----------------------------------------------------------------------
# Operator variants (conv_type)
# ----------------------------------------------------------------------
class TestConvType:
    def test_edgeconv_forward_shape(self, anticline: HorizonSurface) -> None:
        """EdgeConv variant builds and produces the same (n,) output."""
        model = LocalOperator(conv_type="edgeconv", aggr="max")
        inputs = _build_inputs(anticline)
        dz = model(**inputs)
        assert dz.shape == (anticline.n_vertices,)
        assert dz.dtype == anticline.V.dtype

    def test_edgeconv_gradients_flow(self, anticline: HorizonSurface) -> None:
        """Gradients reach every EdgeConv parameter (needed for training)."""
        model = LocalOperator(conv_type="edgeconv", aggr="max")
        inputs = _build_inputs(anticline)
        model(**inputs).sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_unknown_conv_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown conv_type"):
            LocalOperator(conv_type="gmm")