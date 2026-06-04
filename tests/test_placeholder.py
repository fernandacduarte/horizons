"""Unit tests for the placeholder GNN."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.models.placeholder import TinySAGE


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


class TestTinySAGE:
    @staticmethod
    def _call(model, surface):
        """Helper: call model with the standard operator signature."""
        z = surface.V[:, 2]
        V_xy = surface.V[:, :2]
        mask = torch.ones(surface.n_vertices, dtype=torch.bool)
        d = torch.zeros(surface.n_vertices, dtype=torch.int64)
        return model(z, V_xy, surface.edge_index, surface.F, mask, d)

    def test_output_shape(self, anticline: HorizonSurface) -> None:
        model = TinySAGE()
        dz = self._call(model, anticline)
        assert dz.shape == (anticline.n_vertices,)
        assert dz.dtype == torch.float32

    def test_initial_dz_is_small(self, anticline: HorizonSurface) -> None:
        """With output_init_scale=0.01, Δz at initialization should be much
        smaller than typical z values, so the first rollout iteration is gentle."""
        model = TinySAGE(output_init_scale=0.01)
        dz = self._call(model, anticline)
        # On the anticline, z ranges roughly -1 to 5, so std(z) ~ 1.
        # Δz should be much smaller — say, <0.5 in magnitude on average.
        assert dz.abs().mean() < 0.5

    def test_gradient_flows_to_parameters(
        self, anticline: HorizonSurface
    ) -> None:
        """Verify that backward() through the model produces non-trivial
        gradients on its parameters. If this fails, the rollout will not learn."""
        model = TinySAGE()
        dz = self._call(model, anticline)
        loss = dz.pow(2).sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient on {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"

    def test_overfitting_single_example_capability(
        self, anticline: HorizonSurface
    ) -> None:
        """The placeholder must be capable of overfitting a single z-prediction
        task even though it's tiny. This is a precondition for the rollout
        overfit test in 5.6 to be meaningful.

        Task: given z, predict z itself (trivially possible since input == target).
        After enough optimizer steps, loss should drop near zero."""
        torch.manual_seed(0)
        model = TinySAGE(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        z_target = anticline.V[:, 2]

        initial_loss = None
        for step in range(500):
            optimizer.zero_grad()
            dz = self._call(model, anticline)
            # Try to match z itself
            loss = (dz - z_target).pow(2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # We expect a substantial reduction. The placeholder is weak so it
        # won't hit zero, but it should improve by orders of magnitude.
        assert final_loss < initial_loss * 0.1, (
            f"Placeholder failed to overfit: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )
        