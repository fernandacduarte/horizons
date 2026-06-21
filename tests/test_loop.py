"""Tests for the training loop, focused on the non-finite-gradient guard.

Regression for the Phase-2 epoch-91 failure: a *finite* loss back-propagated a
*non-finite* gradient (the 1/||n|| term in vertex-normal normalization on a
near-degenerate normal during a deep rollout). clip_grad_norm_ cannot sanitize
NaN/Inf (clipping by a NaN norm yields NaN grads), so the poisoned gradient
flowed into optimizer.step() and turned every weight to NaN — after which every
surface NaN'd and the run could not recover.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from horizons.data.dataset import load_fixture_dataset
from horizons.models.operator import LocalOperator
from horizons.training.loop import train


def _fixture_datasets(names=("anticline", "sphere_cap")):
    train_ds = load_fixture_dataset(list(names), split="train")
    val_ds = load_fixture_dataset(list(names), split="val")
    return train_ds, val_ds


class _NaNGradOnFirstCall(nn.Module):
    """Operator-protocol module whose forward is always finite (dz = 0), but the
    FIRST forward call back-propagates a NaN gradient to ``w``
    (d/dw sqrt(w*0) = inf * 0 = NaN); later calls give a finite (zero) gradient.

    With >=2 train surfaces and accum_steps=1, the first surface's batch carries
    the NaN gradient (and must be skipped, weights preserved); a later surface
    steps normally, so the epoch still has a successful surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, z, V_xy, edge_index, F, mask, d):
        self.calls += 1
        corrupt = torch.sqrt(self.w * 0.0) if self.calls == 1 else self.w * 0.0
        return torch.zeros_like(z) + corrupt


def test_train_runs_one_epoch_finite():
    """Sanity: a real operator trains one epoch on fixtures without NaNs."""
    train_ds, val_ds = _fixture_datasets()
    model = LocalOperator(hidden_dim=16, n_message_passing=1)
    state = train(
        model, train_ds, val_ds,
        n_epochs=1, val_every=1, verbose=False,
        warmup_epochs=0, lr_schedule="constant",
    )
    assert len(state.train_history) == 1
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_nonfinite_gradient_guard_preserves_weights():
    """A finite loss that back-props a NaN gradient must NOT corrupt the weights:
    the optimizer step is skipped and parameters stay finite. Without the guard
    the NaN gradient would flow through clip_grad_norm_ into optimizer.step()
    and poison the weights — the Phase-2 epoch-91 failure mode."""
    train_ds, val_ds = _fixture_datasets()
    model = _NaNGradOnFirstCall()

    state = train(
        model, train_ds, val_ds,
        n_epochs=1, val_every=1, verbose=False,
        warmup_epochs=0, lr_schedule="constant",
    )

    # The guard skipped the poisoned step; weights are intact.
    assert torch.isfinite(model.w).all(), "NaN gradient leaked into the weights"
    # At least one clean surface still produced a successful optimizer step.
    assert state.train_history[0]["n_surfaces"] >= 1
