"""Unit tests for horizons.data.mesh."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface, build_edge_index


FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ["plane", "sphere_cap", "anticline"]


# ----------------------------------------------------------------------
# Pytest fixtures: load each mesh once, share across tests
# ----------------------------------------------------------------------
@pytest.fixture(scope="module", params=FIXTURE_NAMES)
def surface(request) -> HorizonSurface:
    """Parameterized fixture: every test using `surface` runs once per fixture."""
    name = request.param
    return HorizonSurface.from_npz(FIXTURES_DIR / f"{name}.npz")


# ----------------------------------------------------------------------
# Loading invariants
# ----------------------------------------------------------------------
class TestLoading:
    def test_V_shape_and_dtype(self, surface: HorizonSurface) -> None:
        assert surface.V.ndim == 2
        assert surface.V.shape[1] == 3
        assert surface.V.dtype == torch.float32

    def test_F_shape_and_dtype(self, surface: HorizonSurface) -> None:
        assert surface.F.ndim == 2
        assert surface.F.shape[1] == 3
        assert surface.F.dtype == torch.int64

    def test_F_indices_in_range(self, surface: HorizonSurface) -> None:
        """Every face index must point to a valid vertex."""
        assert surface.F.min() >= 0
        assert surface.F.max() < surface.n_vertices

    def test_no_degenerate_triangles(self, surface: HorizonSurface) -> None:
        """No triangle may have a repeated vertex index."""
        F = surface.F
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        assert ((a != b) & (b != c) & (a != c)).all()

    def test_surface_id_set(self, surface: HorizonSurface) -> None:
        """surface_id should be populated (we default it from the filename)."""
        assert surface.surface_id in FIXTURE_NAMES


# ----------------------------------------------------------------------
# Edge index correctness
# ----------------------------------------------------------------------
class TestEdgeIndex:
    def test_shape_and_dtype(self, surface: HorizonSurface) -> None:
        ei = surface.edge_index
        assert ei.ndim == 2
        assert ei.shape[0] == 2
        assert ei.dtype == torch.int64

    def test_no_self_loops(self, surface: HorizonSurface) -> None:
        src, dst = surface.edge_index
        assert (src != dst).all()

    def test_bidirectional(self, surface: HorizonSurface) -> None:
        """Every directed edge (i, j) should have a partner (j, i)."""
        src, dst = surface.edge_index
        forward = torch.stack([src, dst], dim=1)   # (E, 2)
        reverse = torch.stack([dst, src], dim=1)   # (E, 2)
        # Both sets should contain the same rows. Compare via lexsort.
        forward_sorted = forward[
            torch.argsort(forward[:, 0] * surface.n_vertices + forward[:, 1])
        ]
        reverse_sorted = reverse[
            torch.argsort(reverse[:, 0] * surface.n_vertices + reverse[:, 1])
        ]
        assert torch.equal(forward_sorted, reverse_sorted)

    def test_no_duplicate_edges(self, surface: HorizonSurface) -> None:
        """edge_index should be deduplicated."""
        edges = surface.edge_index.t()  # (E, 2)
        unique_edges = torch.unique(edges, dim=0)
        assert edges.shape[0] == unique_edges.shape[0]

    def test_euler_formula(self, surface: HorizonSurface) -> None:
        """For a disk-like triangulation: V - E + F = 1."""
        V = surface.n_vertices
        E = surface.n_edges  # undirected
        F = surface.n_faces
        assert V - E + F == 1, (
            f"Expected V - E + F = 1 (disk topology); "
            f"got V={V}, E={E}, F={F}, V-E+F={V - E + F}"
        )


# ----------------------------------------------------------------------
# Edge index <-> Face consistency
# ----------------------------------------------------------------------
class TestEdgeFaceConsistency:
    def test_every_face_edge_is_in_edge_index(
        self, surface: HorizonSurface
    ) -> None:
        """Every (i,j) appearing as a triangle edge must appear in edge_index
        (in some direction)."""
        F = surface.F
        face_edges = torch.cat(
            [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], dim=0
        )  # (3 * n_faces, 2)
        # Canonical form: sort each row so (i, j) and (j, i) match
        face_edges_canon, _ = torch.sort(face_edges, dim=1)

        ei_pairs = surface.edge_index.t()  # (E, 2)
        ei_canon, _ = torch.sort(ei_pairs, dim=1)

        # Every canonicalized face-edge should appear in canonicalized ei
        face_set = {tuple(row.tolist()) for row in face_edges_canon}
        ei_set = {tuple(row.tolist()) for row in ei_canon}
        assert face_set.issubset(ei_set)


# ----------------------------------------------------------------------
# build_edge_index, in isolation
# ----------------------------------------------------------------------
class TestBuildEdgeIndex:
    def test_single_triangle(self) -> None:
        """A single triangle (0, 1, 2) has 3 undirected edges (6 directed)."""
        F = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        ei = build_edge_index(F)
        assert ei.shape == (2, 6)
        # All edges must be among the three triangle edges
        edges_canon, _ = torch.sort(ei.t(), dim=1)
        edge_set = {tuple(row.tolist()) for row in edges_canon}
        assert edge_set == {(0, 1), (0, 2), (1, 2)}

    def test_shared_edge_deduplicated(self) -> None:
        """Two triangles sharing an edge produce 5 undirected edges,
        not 6: (0,1), (1,2), (0,2) [shared], (2,3), (1,3)."""
        F = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        ei = build_edge_index(F)
        # 5 undirected edges * 2 directions = 10 directed
        assert ei.shape == (2, 10)
