"""Tests for the triangulated-surface rendering helpers."""
from pathlib import Path

import numpy as np
import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.viz.mesh import (
    SurfacePanel,
    auto_z_exaggeration,
    triangulated_polydata,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


class TestTriangulatedPolydata:
    def test_preserves_vertices_and_faces(
        self, anticline: HorizonSurface
    ) -> None:
        mesh = triangulated_polydata(anticline.V, anticline.F)
        assert mesh.n_points == anticline.V.shape[0]
        assert mesh.n_cells == anticline.F.shape[0]
        assert np.allclose(mesh.points, anticline.V.numpy(), atol=1e-5)

    def test_all_cells_are_triangles(self, anticline: HorizonSurface) -> None:
        """A wrong cell-size prefix silently produces garbage topology, and
        the triangle edges are exactly what these figures are meant to show."""
        mesh = triangulated_polydata(anticline.V, anticline.F)
        faces = mesh.faces.reshape(-1, 4)
        assert np.all(faces[:, 0] == 3)
        assert np.array_equal(faces[:, 1:], anticline.F.numpy())

    def test_z_overrides_third_column(
        self, anticline: HorizonSurface
    ) -> None:
        """Swapping in a predicted depth field must keep xy untouched."""
        z = torch.zeros(anticline.V.shape[0])
        mesh = triangulated_polydata(anticline.V, anticline.F, z)
        assert np.allclose(mesh.points[:, :2], anticline.V[:, :2].numpy(),
                           atol=1e-5)
        assert np.allclose(mesh.points[:, 2], 0.0)

    def test_accepts_xy_only(self, anticline: HorizonSurface) -> None:
        mesh = triangulated_polydata(
            anticline.V[:, :2], anticline.F, anticline.V[:, 2]
        )
        assert np.allclose(mesh.points, anticline.V.numpy(), atol=1e-5)

    def test_xy_only_requires_z(self, anticline: HorizonSurface) -> None:
        with pytest.raises(ValueError, match="z is required"):
            triangulated_polydata(anticline.V[:, :2], anticline.F)

    def test_z_exaggeration_scales_only_z(
        self, anticline: HorizonSurface
    ) -> None:
        plain = triangulated_polydata(anticline.V, anticline.F)
        stretched = triangulated_polydata(
            anticline.V, anticline.F, z_exaggeration=4.0
        )
        assert np.allclose(stretched.points[:, :2], plain.points[:, :2])
        assert np.allclose(stretched.points[:, 2], plain.points[:, 2] * 4.0)

    def test_mismatched_z_length_raises(
        self, anticline: HorizonSurface
    ) -> None:
        with pytest.raises(ValueError, match="vertices"):
            triangulated_polydata(
                anticline.V, anticline.F, torch.zeros(3)
            )


class TestAutoZExaggeration:
    def test_stretches_a_near_planar_surface(self) -> None:
        """xy spanning 10 km with 100 m of relief needs stretching to read."""
        V = torch.tensor([
            [0.0, 0.0, 0.0],
            [10_000.0, 0.0, 100.0],
            [0.0, 10_000.0, 50.0],
        ])
        factor = auto_z_exaggeration(V, target_ratio=0.2)
        assert factor == pytest.approx(0.2 * 10_000 / 100)

    def test_never_shrinks_a_steep_surface(self) -> None:
        V = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 10_000.0],
            [0.0, 100.0, 5_000.0],
        ])
        assert auto_z_exaggeration(V) == 1.0

    def test_flat_surface_is_left_alone(self) -> None:
        """A constant-z surface has no relief to exaggerate (no div-by-zero)."""
        V = torch.tensor([
            [0.0, 0.0, 7.0],
            [100.0, 0.0, 7.0],
            [0.0, 100.0, 7.0],
        ])
        assert auto_z_exaggeration(V) == 1.0


class TestSurfacePanel:
    def test_from_mesh_builds_polydata(self, anticline: HorizonSurface) -> None:
        panel = SurfacePanel.from_mesh(
            anticline.V[:, :2], anticline.F, anticline.V[:, 2],
            title="ground truth",
        )
        assert panel.title == "ground truth"
        assert panel.mesh.n_cells == anticline.F.shape[0]
