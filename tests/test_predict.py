"""Tests for predict_surface, which reconstructs the depth fields a figure
or export needs.

It repeats the centering / normalization / rollout-depth logic of
evaluate_surface, so the tests here mostly pin the two together: if they ever
disagree, a plotted prediction would no longer be the one being scored.
"""
from pathlib import Path

import pytest
import torch

from horizons.data.harmonic_infill import harmonic_infill
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.mesh import HorizonSurface
from horizons.eval.per_surface import evaluate_surface
from horizons.eval.predict import predict_surface
from horizons.models.operator import LocalOperator


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SEED = 3


@pytest.fixture
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


@pytest.fixture
def model() -> LocalOperator:
    torch.manual_seed(0)
    return LocalOperator(hidden_dim=16, n_message_passing=2)


@pytest.fixture
def sampler() -> MaskSampler:
    return MaskSampler(MaskSamplerConfig())


class TestPredictSurface:
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize(
        "approach,n_multiplier", [("rollout", 1.0), ("rollout", 2.0),
                                  ("hybrid", 1.0)]
    )
    def test_rmse_matches_evaluate_surface(
        self, anticline, model, sampler, normalize, approach, n_multiplier,
    ) -> None:
        init_method = "harmonic" if approach == "hybrid" else "meanplane"
        kwargs = dict(
            normalize_per_surface=normalize,
            init_method=init_method,
            approach=approach,
            rollout_n_multiplier=n_multiplier,
        )
        pred = predict_surface(model, anticline, sampler, SEED, **kwargs)
        ref = evaluate_surface(model, anticline, sampler, SEED, **kwargs)

        assert pred.regime == ref.regime
        assert pred.N == ref.N
        assert pred.n_K == ref.n_K
        assert pred.rmse_model == pytest.approx(ref.rmse_overall, rel=1e-5)

    def test_known_vertices_are_anchored(
        self, anticline, model, sampler
    ) -> None:
        """Every method must reproduce the given depths on K exactly; a drift
        there would mean the figure is not showing the same input."""
        pred = predict_surface(model, anticline, sampler, SEED)
        for z in (pred.z_model, pred.z_harmonic, pred.z_init):
            assert torch.allclose(
                z[pred.mask], pred.z_true[pred.mask], atol=1e-4
            )

    def test_harmonic_field_is_the_baseline(
        self, anticline, model, sampler
    ) -> None:
        pred = predict_surface(model, anticline, sampler, SEED)
        expected = harmonic_infill(
            pred.z_true, anticline.edge_index, pred.mask
        )
        assert torch.allclose(pred.z_harmonic, expected)

    def test_fields_are_in_metres_not_normalized_units(
        self, anticline, model, sampler
    ) -> None:
        """Normalization must be undone, or panels would be unit-inconsistent
        and the RMSE captions meaningless."""
        plain = predict_surface(
            model, anticline, sampler, SEED, normalize_per_surface=False
        )
        scaled = predict_surface(
            model, anticline, sampler, SEED, normalize_per_surface=True
        )
        assert torch.allclose(plain.z_true, scaled.z_true)
        assert torch.allclose(plain.z_harmonic, scaled.z_harmonic)

    def test_z_is_centered_on_known_mean(
        self, anticline, model, sampler
    ) -> None:
        pred = predict_surface(model, anticline, sampler, SEED)
        assert pred.z_true[pred.mask].mean().abs() < 1e-3
        assert pred.xy.mean(dim=0).abs().max() < 1e-2

    def test_without_model_only_the_baseline_is_computed(
        self, anticline, sampler
    ) -> None:
        pred = predict_surface(None, anticline, sampler, SEED)
        expected = harmonic_infill(
            pred.z_true, anticline.edge_index, pred.mask
        )
        assert torch.allclose(pred.z_harmonic, expected)
        assert torch.isnan(torch.tensor(pred.rmse_model))
        assert torch.allclose(pred.z_model, pred.z_init)

    def test_same_seed_gives_the_same_mask(
        self, anticline, model, sampler
    ) -> None:
        a = predict_surface(model, anticline, sampler, SEED)
        b = predict_surface(model, anticline, sampler, SEED)
        assert torch.equal(a.mask, b.mask)
        assert a.rmse_model == b.rmse_model
