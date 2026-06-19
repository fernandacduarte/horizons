"""Unit tests for horizons.training.rollout.

These tests verify the three properties that matter most:
  1. Known vertices are anchored exactly at z_true after every iteration.
  2. Gradients flow to model parameters.
  3. Setting model output to 0 produces z^N = z^0 on U (anchoring sanity).
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import sample_half_plane_mask
from horizons.data.init import init_z
from horizons.data.topo_distance import compute_topological_distance
from horizons.models.placeholder import TinySAGE
from horizons.training.rollout import rollout, RolloutResult


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


@pytest.fixture(scope="module")
def setup(anticline: HorizonSurface) -> dict:
    """A complete (surface, mask, z0, d, N) bundle for the rollout."""
    rng = torch.Generator().manual_seed(0)
    mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
    d = compute_topological_distance(anticline.edge_index, mask)
    z0 = init_z(anticline.V, mask)
    return {
        "z0": z0,
        "z_true": anticline.V[:, 2],
        "V_xy": anticline.V[:, :2],
        "F": anticline.F,
        "edge_index": anticline.edge_index,
        "mask": mask,
        "d": d,
        "N": int(d.max().item()),
    }


# ----------------------------------------------------------------------
# Shape and structure
# ----------------------------------------------------------------------
class TestStructure:
    def test_trajectory_lengths(self, setup) -> None:
        model = TinySAGE()
        result = rollout(model, **setup)
        # z trajectory: z^0, z^1, ..., z^N → N+1 elements
        # dz trajectory: Δz^0, Δz^1, ..., Δz^{N-1} → N elements
        assert len(result.z_trajectory) == setup["N"] + 1
        assert len(result.dz_trajectory) == setup["N"]
        assert result.N == setup["N"]

    def test_per_iteration_shapes(self, setup) -> None:
        model = TinySAGE()
        result = rollout(model, **setup)
        n = setup["z0"].shape[0]
        for z_t in result.z_trajectory:
            assert z_t.shape == (n,)
        for dz_t in result.dz_trajectory:
            assert dz_t.shape == (n,)

    def test_first_state_is_z0(self, setup) -> None:
        model = TinySAGE()
        result = rollout(model, **setup)
        assert torch.equal(result.z_trajectory[0], setup["z0"])


# ----------------------------------------------------------------------
# Property 1: Known vertices are anchored at every iteration
# ----------------------------------------------------------------------
class TestAnchoring:
    def test_known_vertices_unchanged_throughout_rollout(self, setup) -> None:
        """At every iteration t, z^t[K] must equal z_true[K] exactly.
        Not approximately — exactly."""
        model = TinySAGE()
        result = rollout(model, **setup)
        z_true_K = setup["z_true"][setup["mask"]]
        for t, z_t in enumerate(result.z_trajectory):
            assert torch.equal(z_t[setup["mask"]], z_true_K), (
                f"Anchoring failed at iteration {t}: "
                f"max diff = {(z_t[setup['mask']] - z_true_K).abs().max()}"
            )


# ----------------------------------------------------------------------
# Property 2: Gradients flow correctly
# ----------------------------------------------------------------------
class TestGradientFlow:
    def test_gradient_reaches_model_parameters(self, setup) -> None:
        """A loss on z^N must produce non-zero gradients on model parameters."""
        torch.manual_seed(0)
        model = TinySAGE()
        result = rollout(model, **setup)
        # Loss on unknown vertices only — known ones are anchored, no signal there
        z_N = result.z_trajectory[-1]
        loss = (z_N[~setup["mask"]] - setup["z_true"][~setup["mask"]]).pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient on {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"

    def test_no_gradient_through_known_vertices(self, setup) -> None:
        """A loss computed only on known vertices should produce zero
        gradient on model parameters (since K is anchored at z_true, the
        model's prediction can't affect z^t[K])."""
        torch.manual_seed(0)
        model = TinySAGE()
        result = rollout(model, **setup)
        z_N = result.z_trajectory[-1]
        # Loss only on K — these values are torch.where(mask, z_true, ...),
        # which is exactly z_true for K, so any "loss" is structurally 0
        # and any "gradient" must be 0 too.
        loss = (z_N[setup["mask"]] - setup["z_true"][setup["mask"]]).pow(2).mean()
        # Loss should be exactly 0
        assert loss.item() == 0.0
        # backward() on a 0-tensor still works; it just produces 0 grads.
        loss.backward()
        # Gradients should all be zero
        for name, param in model.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Nonzero gradient on {name} despite K-only loss"
            )

    def test_gradient_through_all_iterations(self, setup) -> None:
        """Use a custom 'identity-output' tracer to verify that gradients
        flow through all N iterations, not just the last one.

        The test: replace the model with one whose Δz is a vector parameter
        (so the gradient on Δz directly reaches that parameter). After N
        iterations, the parameter should have received gradient contributions
        from N different time steps."""
        n = setup["z0"].shape[0]

        class TracerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.zeros(n))

            def forward(self, z, V_xy, edge_index, F, mask, d):
                return self.scale  # Δz = self.scale, regardless of z

        torch.manual_seed(0)
        model = TracerModel()
        result = rollout(model, **setup)
        # Use the final z to compute loss
        z_N = result.z_trajectory[-1]
        loss = z_N[~setup["mask"]].sum()  # arbitrary differentiable scalar
        loss.backward()

        # On each iteration t, Δz_t = self.scale gets added to z^{t-1}, and
        # z^t propagates onward. The gradient of z^N w.r.t. self.scale[i]
        # (for i in U) should be approximately N (each iteration contributes 1).
        grad_U = model.scale.grad[~setup["mask"]]
        N = setup["N"]
        assert grad_U.mean().item() == pytest.approx(float(N), abs=0.5), (
            f"Expected gradient ~{N} (one per iteration); got {grad_U.mean()}"
        )


# ----------------------------------------------------------------------
# Property 3: Identity-output model gives z^N = z^0 (re-anchoring sanity)
# ----------------------------------------------------------------------
class TestIdentityModel:
    def test_zero_dz_means_zN_equals_z0_on_U(self, setup) -> None:
        """If the model always outputs 0, then z^{t+1} = z^t (on U), so
        z^N = z^0 (on U). On K, z^N = z_true at every iteration anyway."""
        class ZeroModel(nn.Module):
            def forward(self, z, V_xy, edge_index, F, mask, d):
                return torch.zeros_like(z)

        model = ZeroModel()
        result = rollout(model, **setup)
        z_N = result.z_trajectory[-1]
        # On U, z^N should equal z^0 because nothing changed
        assert torch.equal(z_N[~setup["mask"]], setup["z0"][~setup["mask"]])
        # On K, z^N should equal z_true
        assert torch.equal(z_N[setup["mask"]], setup["z_true"][setup["mask"]])


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestValidation:
    def test_N_zero_rejected(self, setup) -> None:
        model = TinySAGE()
        setup_bad = {**setup, "N": 0}
        with pytest.raises(ValueError, match="N must be"):
            rollout(model, **setup_bad)

# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------
class TestCheckpoint:
    def test_checkpoint_is_transparent(self):
        """Gradient checkpointing must be mathematically transparent: identical
        forward output AND identical parameter gradients vs the plain rollout.
        Tested on EdgeConv (the memory-heavy operator that motivated it)."""
        import torch
        from pathlib import Path
        from horizons.data.mesh import HorizonSurface
        from horizons.data.masking import MaskSampler, MaskSamplerConfig
        from horizons.data.init import init_z
        from horizons.models.operator import LocalOperator
        from horizons.training.rollout import rollout
        from horizons.training.loss import rollout_loss

        surface = HorizonSurface.from_npz(
            Path(__file__).parent / "fixtures" / "anticline.npz"
        )
        sampler = MaskSampler(MaskSamplerConfig())
        mask, d, _ = sampler.sample(surface, torch.Generator().manual_seed(0))
        N = int(d.max().item())
        z0 = init_z(surface.V, mask)
        V_xy, F, edge_index = surface.V[:, :2], surface.F, surface.edge_index
        z_true = surface.V[:, 2]

        def run(use_ckpt: bool):
            torch.manual_seed(0)
            model = LocalOperator(conv_type="edgeconv", aggr="max")
            res = rollout(
                model, z0=z0, z_true=z_true, V_xy=V_xy, F=F,
                edge_index=edge_index, mask=mask, d=d, N=N,
                use_checkpoint=use_ckpt,
            )
            loss = rollout_loss(
                z_trajectory=res.z_trajectory, dz_trajectory=res.dz_trajectory,
                z_true=z_true, d=d, edge_index=edge_index, mask=mask,
            )["total"]
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in model.parameters()])
            return res.z_trajectory[-1].detach(), loss.detach(), grads

        z_plain, loss_plain, g_plain = run(False)
        z_ckpt, loss_ckpt, g_ckpt = run(True)

        assert torch.allclose(z_plain, z_ckpt, atol=1e-6)
        assert torch.allclose(loss_plain, loss_ckpt, atol=1e-6)
        assert torch.allclose(g_plain, g_ckpt, atol=1e-5)