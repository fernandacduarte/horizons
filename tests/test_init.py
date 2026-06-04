"""Unit tests for horizons.data.init."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import sample_half_plane_mask
from horizons.data.init import fit_mean_plane, init_z


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def plane() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "plane.npz")


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


# ----------------------------------------------------------------------
# fit_mean_plane
# ----------------------------------------------------------------------
class TestFitMeanPlane:
    def test_recovers_planar_coefficients(self) -> None:
        """If the input is exactly planar, we recover (a, b, c) exactly."""
        torch.manual_seed(0)
        xy = torch.randn(50, 2)
        a_true, b_true, c_true = 0.7, -0.4, 1.2
        z = a_true * xy[:, 0] + b_true * xy[:, 1] + c_true
        a, b, c = fit_mean_plane(xy, z)
        assert abs(a - a_true) < 1e-5
        assert abs(b - b_true) < 1e-5
        assert abs(c - c_true) < 1e-5

    def test_planar_fixture(self, plane: HorizonSurface) -> None:
        """The planar fixture is z = 0.3 x + 0.1 y + 2.0. Fitting through
        all its vertices should recover those coefficients."""
        xy = plane.V[:, :2]
        z = plane.V[:, 2]
        a, b, c = fit_mean_plane(xy, z)
        assert abs(a - 0.3) < 1e-4
        assert abs(b - 0.1) < 1e-4
        assert abs(c - 2.0) < 1e-4

    def test_minimum_three_points_required(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            fit_mean_plane(
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                torch.tensor([0.0, 1.0]),
            )

    def test_collinear_points_rejected(self) -> None:
        """Three collinear points -> singular fit; should raise."""
        xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        z = torch.tensor([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="collinear"):
            fit_mean_plane(xy, z)

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="xy_K must have shape"):
            fit_mean_plane(torch.zeros(5), torch.zeros(5))
        with pytest.raises(ValueError, match="z_K must have shape"):
            fit_mean_plane(torch.zeros(5, 2), torch.zeros(3))


# ----------------------------------------------------------------------
# init_z
# ----------------------------------------------------------------------
class TestInitZ:
    def test_known_vertices_exact(self, anticline: HorizonSurface) -> None:
        """z^0 on K must equal z_true on K (exact equality, not just close).
        This is critical: the rollout re-anchors at z_true every iteration,
        so z^0 on K had better be z_true."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        z0 = init_z(anticline.V, mask)
        z_true = anticline.V[:, 2]
        assert torch.equal(z0[mask], z_true[mask])

    def test_plane_input_means_plane_output(
        self, plane: HorizonSurface,
    ) -> None:
        """If the surface IS a plane, z^0 should equal z_true everywhere —
        the unknowns are initialized to the same plane the knowns lie on."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(plane.V, phi=0.5, rng=rng)
        z0 = init_z(plane.V, mask)
        z_true = plane.V[:, 2]
        # Allow tiny numerical error from the lstsq solve
        assert torch.allclose(z0, z_true, atol=1e-4)

    def test_unknown_vertices_match_plane_eval(
        self, anticline: HorizonSurface,
    ) -> None:
        """On U, z^0 should equal the plane fitted to K evaluated at U's (x,y)."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        z0 = init_z(anticline.V, mask)

        # Independently recompute the plane and evaluate at U
        xy_K = anticline.V[mask, :2]
        z_K = anticline.V[mask, 2]
        a, b, c = fit_mean_plane(xy_K, z_K)
        xy_U = anticline.V[~mask, :2]
        expected_z_U = a * xy_U[:, 0] + b * xy_U[:, 1] + c

        assert torch.allclose(z0[~mask], expected_z_U, atol=1e-5)

    def test_z0_is_better_than_global_mean(
        self, anticline: HorizonSurface,
    ) -> None:
        """The mean-plane initialization should beat a flat mean-z
        initialization in RMSE on U. This justifies the choice of mean-plane
        over mean-z (the decision we made in question 11)."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)

        z_true = anticline.V[:, 2]

        # Mean-plane init
        z0_plane = init_z(anticline.V, mask)
        err_plane = (z0_plane[~mask] - z_true[~mask]).pow(2).mean().sqrt()

        # Mean-z init (baseline)
        mean_z = z_true[mask].mean()
        z0_meanz = z_true.clone()
        z0_meanz[~mask] = mean_z
        err_meanz = (z0_meanz[~mask] - z_true[~mask]).pow(2).mean().sqrt()

        # Mean plane should be better. On the anticline (which has a tilted
        # baseline + a bump), the difference is substantial.
        assert err_plane < err_meanz, (
            f"Mean-plane RMSE ({err_plane:.4f}) should be less than "
            f"mean-z RMSE ({err_meanz:.4f})"
        )

    def test_shape_validation(self, anticline: HorizonSurface) -> None:
        mask = torch.zeros(anticline.n_vertices, dtype=torch.bool)
        mask[0] = True
        with pytest.raises(ValueError, match="V must have shape"):
            init_z(torch.zeros(10), mask)
        wrong_mask = torch.zeros(5, dtype=torch.bool)
        with pytest.raises(ValueError, match="mask shape"):
            init_z(anticline.V, wrong_mask)
        bad_dtype_mask = torch.zeros(anticline.n_vertices, dtype=torch.int64)
        with pytest.raises(TypeError, match="mask must be bool"):
            init_z(anticline.V, bad_dtype_mask)
