"""Tests for harmonic_infill."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.features import compute_umbrella_laplacian
from horizons.eval.harmonic_infill import harmonic_infill


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


class TestHarmonicInfill:
    def test_known_vertices_preserved_exactly(
        self, anticline: HorizonSurface
    ) -> None:
        """z_pred[K] must equal z_true[K] exactly — anchoring is hard."""
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        mask, _, _ = sampler.sample(anticline, rng)

        z_pred = harmonic_infill(
            anticline.V[:, 2], anticline.edge_index, mask,
        )
        assert torch.allclose(z_pred[mask], anticline.V[mask, 2])

    def test_harmonic_on_U(self, anticline: HorizonSurface) -> None:
        """The umbrella Laplacian of the solution should be ~0 on U.
        That's literally the property we're solving for."""
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        mask, _, _ = sampler.sample(anticline, rng)

        z_pred = harmonic_infill(
            anticline.V[:, 2], anticline.edge_index, mask,
        )
        kappa = compute_umbrella_laplacian(z_pred, anticline.edge_index)

        # On U, kappa should be near zero (within solver tolerance)
        max_kappa_on_U = kappa[~mask].abs().max().item()
        assert max_kappa_on_U < 1e-4, (
            f"Harmonic property violated: max |kappa| on U = {max_kappa_on_U}"
        )

    def test_constant_field_recovered_exactly(
        self, anticline: HorizonSurface
    ) -> None:
        """If z is constant on K, the harmonic solution must be constant
        on U too (constant fields are harmonic)."""
        n = anticline.n_vertices
        z_true = torch.full((n,), 5.0)
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        mask, _, _ = sampler.sample(anticline, rng)

        z_pred = harmonic_infill(z_true, anticline.edge_index, mask)
        assert torch.allclose(z_pred, torch.full((n,), 5.0), atol=1e-6)

    def test_linear_field_recovered_approximately(
        self, anticline: HorizonSurface
    ) -> None:
        """A linear field z = a*x + b*y + c is harmonic (umbrella Laplacian
        is approximately zero everywhere on the interior of a regular mesh).
        On irregular meshes there's boundary effects, so we test that the
        harmonic infill of a linear field is at least close to the original.
        """
        a, b, c = 0.3, -0.2, 5.0
        z_true = (
            a * anticline.V[:, 0] + b * anticline.V[:, 1] + c
        )
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        mask, _, _ = sampler.sample(anticline, rng)

        z_pred = harmonic_infill(z_true, anticline.edge_index, mask)
        # On a linear field, harmonic infill is approximately exact
        # (modulo discretization). The unknown region should be close.
        rmse_U = (z_pred[~mask] - z_true[~mask]).pow(2).mean().sqrt().item()
        assert rmse_U < 0.5, (
            f"Harmonic infill of linear field has large error: RMSE={rmse_U}"
        )
