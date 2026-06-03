"""Unit tests for horizons.data.topo_distance."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface, build_edge_index
from horizons.data.masking import (
    sample_half_plane_mask,
    sample_outward_rectangle_mask,
)
from horizons.data.topo_distance import (
    compute_topological_distance,
    UNREACHABLE,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


# ----------------------------------------------------------------------
# Hand-built tiny graphs
# ----------------------------------------------------------------------
class TestTinyGraphs:
    def test_path_graph(self) -> None:
        """5-vertex path: 0 - 1 - 2 - 3 - 4. Only vertex 0 is known.
        Expected distances: 0, 1, 2, 3, 4."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4],
             [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.int64,
        )
        known = torch.tensor([True, False, False, False, False])
        d = compute_topological_distance(edge_index, known)
        assert torch.equal(d, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64))

    def test_multiple_sources(self) -> None:
        """5-vertex path: 0 - 1 - 2 - 3 - 4. Vertices 0 and 4 are known.
        Distances should be 0, 1, 2, 1, 0 — the BFS meets in the middle."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4],
             [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.int64,
        )
        known = torch.tensor([True, False, False, False, True])
        d = compute_topological_distance(edge_index, known)
        assert torch.equal(d, torch.tensor([0, 1, 2, 1, 0], dtype=torch.int64))

    def test_disconnected_component(self) -> None:
        """Two components: {0, 1, 2} as a path, {3, 4} as a separate edge.
        Known = {0}. Vertex 3 and 4 should be UNREACHABLE."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4],
             [1, 0, 2, 1, 4, 3]],
            dtype=torch.int64,
        )
        known = torch.tensor([True, False, False, False, False])
        d = compute_topological_distance(edge_index, known)
        assert d[0].item() == 0
        assert d[1].item() == 1
        assert d[2].item() == 2
        assert d[3].item() == UNREACHABLE
        assert d[4].item() == UNREACHABLE

    def test_all_known(self) -> None:
        """If everyone is known, all distances are 0."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64,
        )
        known = torch.tensor([True, True, True])
        d = compute_topological_distance(edge_index, known)
        assert torch.equal(d, torch.zeros(3, dtype=torch.int64))


# ----------------------------------------------------------------------
# Fixture meshes with realistic masks
# ----------------------------------------------------------------------
class TestOnFixtures:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        assert d.shape == (anticline.n_vertices,)
        assert d.dtype == torch.int64

    def test_known_vertices_have_distance_zero(
        self, anticline: HorizonSurface
    ) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        assert (d[mask] == 0).all()

    def test_unknown_vertices_have_positive_distance(
        self, anticline: HorizonSurface
    ) -> None:
        """For a half-plane cut, U is connected to K (no disconnected
        components), so all unknown vertices should have d > 0."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        unknown_d = d[~mask]
        assert (unknown_d > 0).all()
        # No vertex should be unreachable on a Delaunay-connected mesh
        assert (unknown_d != UNREACHABLE).all()

    def test_rollout_depth_reasonable(
        self, anticline: HorizonSurface
    ) -> None:
        """For our fixture sizes, the BFS depth should be in single or
        low double digits. This is the value that will become the rollout N."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        max_d = d.max().item()
        # Anticline is a 30x30 grid; max BFS depth ~ 30 in worst case
        assert 1 < max_d < 50, f"Suspicious BFS depth: {max_d}"

    def test_outward_rectangle_depth(
        self, anticline: HorizonSurface
    ) -> None:
        """Outward-rectangle regime: BFS expands from a central rectangle
        outward. Depth should be roughly the radial distance from rectangle
        edge to mesh boundary, in graph hops."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_outward_rectangle_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        assert (d[mask] == 0).all()
        assert (d[~mask] > 0).all()
        assert (d != UNREACHABLE).all()

    def test_bfs_correctness_via_triangle_inequality(
        self, anticline: HorizonSurface
    ) -> None:
        """A correct BFS must satisfy: for every edge (u, v),
        |d[u] - d[v]| <= 1. This is a necessary and easy-to-check property
        of shortest-path distances."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        d = compute_topological_distance(anticline.edge_index, mask)
        src, dst = anticline.edge_index[0], anticline.edge_index[1]
        diff = (d[src] - d[dst]).abs()
        assert (diff <= 1).all(), (
            f"BFS violates triangle inequality on some edges: "
            f"max |d[u] - d[v]| = {diff.max().item()}"
        )


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
class TestInputValidation:
    def test_edge_index_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="edge_index must"):
            compute_topological_distance(
                torch.zeros(5, dtype=torch.int64),
                torch.tensor([True, False, False]),
            )

    def test_known_mask_wrong_dtype(self) -> None:
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        with pytest.raises(TypeError, match="known_mask must be bool"):
            compute_topological_distance(ei, torch.tensor([1, 0]))
