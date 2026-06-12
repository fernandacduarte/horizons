"""Tests for gradient accumulation in the training loop.

The mathematical property we're testing: accumulating gradients from
N surfaces (each loss divided by N) should equal the gradient from a
single combined loss equal to the mean of the N losses.

This is straightforward in principle — torch's backward() adds to the
.grad buffer. But the loop code does it via a non-trivial sequence
(zero_grad, loop calling backward with loss/B, then step), so we test
the realized behavior, not the theory.
"""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z
from horizons.models.operator import LocalOperator
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_loss


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_sample(surface_id: str, seed: int):
    """Build a complete forward+loss inputs bundle from one fixture."""
    surface = HorizonSurface.from_npz(FIXTURES_DIR / f"{surface_id}.npz")
    sampler = MaskSampler(MaskSamplerConfig())
    mask, d, _regime = sampler.sample(
        surface, torch.Generator().manual_seed(seed),
    )
    z0 = init_z(surface.V, mask)
    N = int(d.max().item())
    return {
        "z0": z0, "z_true": surface.V[:, 2],
        "V_xy": surface.V[:, :2], "F": surface.F,
        "edge_index": surface.edge_index,
        "mask": mask, "d": d, "N": N,
    }


def _forward_loss(model, sample):
    """Run rollout + loss for one sample, return total loss (scalar tensor)."""
    result = rollout(
        model,
        z0=sample["z0"], z_true=sample["z_true"],
        V_xy=sample["V_xy"], F=sample["F"],
        edge_index=sample["edge_index"],
        mask=sample["mask"], d=sample["d"], N=sample["N"],
    )
    loss_dict = rollout_loss(
        z_trajectory=result.z_trajectory,
        dz_trajectory=result.dz_trajectory,
        z_true=sample["z_true"], d=sample["d"],
        edge_index=sample["edge_index"], mask=sample["mask"],
    )
    return loss_dict["total"]


class TestGradientAccumulation:
    """Numerical equivalence: accumulating grads should equal one combined backward."""

    def test_accumulated_grads_equal_combined_backward(self) -> None:
        """For B=3 samples, sum of (loss_i / B).backward() should equal
        ((loss_1 + loss_2 + loss_3) / B).backward()."""
        torch.manual_seed(0)
        samples = [
            _make_sample("plane", seed=1),
            _make_sample("sphere_cap", seed=2),
            _make_sample("anticline", seed=3),
        ]
        B = len(samples)

        # --- Path A: accumulate gradients across B backward passes ---
        model_a = LocalOperator(hidden_dim=16, n_message_passing=1)
        for p in model_a.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for s in samples:
            loss = _forward_loss(model_a, s)
            (loss / B).backward()
        grads_a = {
            name: p.grad.clone() for name, p in model_a.named_parameters()
            if p.grad is not None
        }

        # --- Path B: single backward on mean of losses ---
        model_b = LocalOperator(hidden_dim=16, n_message_passing=1)
        # Copy weights from model_a so they're identical
        model_b.load_state_dict(model_a.state_dict())
        for p in model_b.parameters():
            if p.grad is not None:
                p.grad.zero_()
        losses = [_forward_loss(model_b, s) for s in samples]
        mean_loss = sum(losses) / B
        mean_loss.backward()
        grads_b = {
            name: p.grad.clone() for name, p in model_b.named_parameters()
            if p.grad is not None
        }

        # --- Compare ---
        for name in grads_a:
            assert torch.allclose(
                grads_a[name], grads_b[name], rtol=1e-5, atol=1e-7,
            ), (
                f"Gradient mismatch on parameter {name!r}.\n"
                f"  accumulated path: norm={grads_a[name].norm().item():.6e}\n"
                f"  combined path:    norm={grads_b[name].norm().item():.6e}"
            )

    def test_batch_size_1_matches_no_accumulation(self) -> None:
        """When batch_size=1, the accumulation loop should produce the
        exact same gradient as a single direct backward (no /B scaling
        applied because 1/1 = 1)."""
        torch.manual_seed(0)
        sample = _make_sample("anticline", seed=42)

        # Path A: with the /B = /1 scaling
        model_a = LocalOperator(hidden_dim=16, n_message_passing=1)
        for p in model_a.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss_a = _forward_loss(model_a, sample)
        (loss_a / 1).backward()
        grads_a = {n: p.grad.clone() for n, p in model_a.named_parameters()}

        # Path B: direct backward, no scaling
        model_b = LocalOperator(hidden_dim=16, n_message_passing=1)
        model_b.load_state_dict(model_a.state_dict())
        for p in model_b.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss_b = _forward_loss(model_b, sample)
        loss_b.backward()
        grads_b = {n: p.grad.clone() for n, p in model_b.named_parameters()}

        for name in grads_a:
            assert torch.allclose(grads_a[name], grads_b[name]), (
                f"B=1 produces different grads from direct backward on {name}"
            )
