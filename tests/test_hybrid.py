"""Tests for the hybrid approach (Phase 3a): harmonic init + fixed-K GNN refine."""
from __future__ import annotations

from pathlib import Path

import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z_dispatch
from horizons.models.operator import LocalOperator
from horizons.training.rollout import rollout
from horizons.training.loss import hybrid_rollout_loss
from horizons.eval.per_surface import evaluate_surface

FIX = Path(__file__).parent / "fixtures"


def test_hybrid_rollout_loss_supervises_all_U_and_backprops():
    """The hybrid loss is an all-U MSE (dict-shaped like rollout_loss), finite,
    and differentiable to the model parameters."""
    surface = HorizonSurface.from_npz(FIX / "anticline.npz")
    mask, d, _ = MaskSampler(MaskSamplerConfig()).sample(surface, torch.Generator().manual_seed(0))
    z0 = init_z_dispatch(surface.V, mask, surface.edge_index, method="harmonic")
    model = LocalOperator(hidden_dim=16, n_message_passing=1)
    res = rollout(model, z0=z0, z_true=surface.V[:, 2], V_xy=surface.V[:, :2],
                  F=surface.F, edge_index=surface.edge_index, mask=mask, d=d, N=2)
    out = hybrid_rollout_loss(res.z_trajectory, surface.V[:, 2], mask)
    assert set(out) == {"total", "data", "curv", "res"}
    assert torch.isfinite(out["total"]) and out["total"].item() > 0
    out["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_hybrid_eval_reports_surface_depth_not_pass_count():
    """evaluate_surface(approach='hybrid') runs only hybrid_n_passes, but the
    reported N (and per-ring breakdown) is the surface's true depth — so hybrid
    points land at their real depth on the crossover figure."""
    surface = HorizonSurface.from_npz(FIX / "anticline.npz")
    sampler = MaskSampler(MaskSamplerConfig())
    _, d, _ = sampler.sample(surface, torch.Generator().manual_seed(7))
    surface_depth = int(d.max().item())
    assert surface_depth > 2

    model = LocalOperator(hidden_dim=16, n_message_passing=1)
    res = evaluate_surface(model, surface, sampler, rng_seed=7,
                           init_method="harmonic", approach="hybrid", hybrid_n_passes=2)
    assert res.N == surface_depth
    assert len(res.rmse_per_ring) == surface_depth
