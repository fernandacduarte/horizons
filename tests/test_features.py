"""Unit tests for horizons.data.features."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.features import (
    compute_vertex_normals,
    compute_umbrella_laplacian,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def plane() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "plane.npz")


@pytest.fixture(scope="module")
def sphere_cap() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "sphere_cap.npz")


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


# ----------------------------------------------------------------------
# compute_vertex_normals
# ----------------------------------------------------------------------
class TestVertexNormals:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        N = compute_vertex_normals(anticline.V, anticline.F)
        assert N.shape == anticline.V.shape
        assert N.dtype == anticline.V.dtype

    def test_unit_length(self, anticline: HorizonSurface) -> None:
        """Every vertex normal should have unit length (up to eps)."""
        N = compute_vertex_normals(anticline.V, anticline.F)
        norms = torch.linalg.norm(N, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_plane_normals_uniform(self, plane: HorizonSurface) -> None:
        """On a flat tilted plane, all vertex normals should be identical
        (modulo sign — the orientation depends on triangle winding)."""
        N = compute_vertex_normals(plane.V, plane.F)
        # All normals should be parallel: their pairwise dot products
        # should all be +/- 1.
        ref = N[0]
        dots = (N * ref).sum(dim=1).abs()
        assert torch.allclose(dots, torch.ones_like(dots), atol=1e-4)

    def test_plane_normal_perpendicular(self, plane: HorizonSurface) -> None:
        """The plane is z = 0.3 x + 0.1 y + 2.0, so its normal direction
        is proportional to (-0.3, -0.1, 1) (or its negative)."""
        N = compute_vertex_normals(plane.V, plane.F)
        expected = torch.tensor([-0.3, -0.1, 1.0])
        expected = expected / torch.linalg.norm(expected)
        dot = (N[0] * expected).sum().abs()  # abs handles sign ambiguity
        assert torch.allclose(dot, torch.tensor(1.0), atol=1e-4), (
            f"Expected normal parallel to {expected.tolist()}, got {N[0].tolist()}"
        )

    def test_sphere_cap_normals_radial(self, sphere_cap: HorizonSurface) -> None:
        """On a sphere centered at (0, 0, 20), each vertex normal should
        be parallel to (vertex - center)."""
        center = torch.tensor([0.0, 0.0, 20.0])
        N = compute_vertex_normals(sphere_cap.V, sphere_cap.F)
        radial = sphere_cap.V - center
        radial = radial / torch.linalg.norm(radial, dim=1, keepdim=True)
        # Allow sign ambiguity: take absolute value of dot product
        dots = (N * radial).sum(dim=1).abs()
        # Tolerate triangulation-induced discretization error.
        # The fixture uses random points inside a disk, which produces some
        # long thin triangles near the boundary; those vertices can have
        # area-weighted normals that deviate up to ~10 degrees from radial.
        # This is a property of the mesh, not the algorithm.
        # 99%% of vertices should still be very close to radial.
        assert (dots > 0.98).all(), (
            f"Normals deviate from radial: min dot = {dots.min().item()}"
        )
        # And the bulk should be tightly aligned
        assert (dots > 0.99).float().mean() > 0.95, (
            f"Only {(dots > 0.99).float().mean().item():.1%} of normals are "
            f"tightly radial; expected >95%%"
        )

    def test_differentiable(self) -> None:
        """gradcheck: the analytic gradient w.r.t. V matches finite differences.
        Uses a tiny mesh (cheap; gradcheck is O(n_inputs^2))."""
        # Build a tiny mesh: 5 vertices, 3 triangles forming a small fan
        V = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1],
                [1.0, 1.0, 0.2],
                [0.0, 1.0, 0.05],
                [-1.0, 0.5, 0.15],
            ],
            dtype=torch.float64,  # gradcheck needs float64
            requires_grad=True,
        )
        F = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4]],
            dtype=torch.int64,
        )

        def f(V_):
            return compute_vertex_normals(V_, F)

        # gradcheck returns True if analytic == finite-diff (within tol)
        assert torch.autograd.gradcheck(f, (V,), atol=1e-5, rtol=1e-3)


# ----------------------------------------------------------------------
# compute_umbrella_laplacian
# ----------------------------------------------------------------------
class TestUmbrellaLaplacian:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        z = anticline.V[:, 2]
        kappa = compute_umbrella_laplacian(z, anticline.edge_index)
        assert kappa.shape == z.shape
        assert kappa.dtype == z.dtype

    def test_constant_field_zero(self, anticline: HorizonSurface) -> None:
        """A constant scalar field has zero Laplacian everywhere."""
        z = torch.full((anticline.n_vertices,), 5.0, dtype=torch.float32)
        kappa = compute_umbrella_laplacian(z, anticline.edge_index)
        assert torch.allclose(kappa, torch.zeros_like(kappa), atol=1e-6)

    def test_linear_field_near_zero_on_plane(
        self, plane: HorizonSurface
    ) -> None:
        """For a linear field on a fairly regular mesh, the umbrella Laplacian
        is small (it would be exactly zero on a perfectly regular mesh; on an
        irregular one it picks up small residuals from valence variation)."""
        z = plane.V[:, 2]  # the linear field z = 0.3x + 0.1y + 2
        kappa = compute_umbrella_laplacian(z, plane.edge_index)
        # Most vertices should have small |kappa|. We check the median, not
        # the max, because boundary vertices have asymmetric neighborhoods.
        assert kappa.abs().median() < 0.05

    def test_anticline_peak_negative_or_positive(
        self, anticline: HorizonSurface
    ) -> None:
        """At the apex of an upward bump, neighbors lie below the apex,
        so z_apex > mean(z_neighbors), so kappa > 0.
        (Our convention: kappa = z - mean(z_neighbors))
        Find the highest vertex and check the sign."""
        z = anticline.V[:, 2]
        kappa = compute_umbrella_laplacian(z, anticline.edge_index)
        apex_idx = torch.argmax(z)
        assert kappa[apex_idx] > 0, (
            f"Expected positive kappa at apex; got {kappa[apex_idx].item()}"
        )

    def test_differentiable(self, anticline: HorizonSurface) -> None:
        """gradcheck on z. The function is linear in z, so the gradient
        should match finite differences exactly."""
        # Use a subset for speed: build a tiny path graph.
        n = 6
        z = torch.linspace(0, 1, n, dtype=torch.float64, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
             [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
            dtype=torch.int64,
        )

        def f(z_):
            return compute_umbrella_laplacian(z_, edge_index)

        assert torch.autograd.gradcheck(f, (z,), atol=1e-6, rtol=1e-4)
