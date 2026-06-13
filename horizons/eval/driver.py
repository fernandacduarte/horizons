"""End-to-end evaluation driver.

Evaluates a trained model + two baselines (mean-plane, harmonic infill)
across all surfaces in a split, with multiple mask samples per surface
to cover the three mask regimes. Aggregates by regime, by reservoir,
and overall.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from horizons.data.loaders import load_split
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z
from horizons.data.mesh import HorizonSurface
from horizons.eval.per_surface import evaluate_surface
from horizons.eval.harmonic_infill import harmonic_infill


@dataclass
class MaskEvalRecord:
    """Per-(surface, mask) result with all three methods."""
    surface_id: str
    reservoir_id: str | None
    regime: str
    seed: int
    N: int
    n_K: int
    n_U: int
    # RMSE on U from each method (in original units, i.e., meters)
    rmse_meanplane: float
    rmse_harmonic: float
    rmse_model: float
    # Per-ring RMSE for the model only (the baselines don't depend on
    # rollout, so per-ring breakdown isn't as meaningful for them).
    rmse_per_ring_model: list[float]
    ring_sizes: list[int]


@dataclass
class EvalResult:
    """Aggregate result of one full evaluation run."""
    split_name: str
    n_surfaces: int
    n_masks_per_surface: int
    records: list[MaskEvalRecord] = field(default_factory=list)


def evaluate_split(
    model: torch.nn.Module,
    split_name: str,
    *,
    n_masks_per_surface: int = 3,
    base_seed: int = 1000,
    mask_config: MaskSamplerConfig | None = None,
    device: str | torch.device = "cpu",
) -> EvalResult:
    """Evaluate model + baselines on every surface in a split, with
    multiple mask samples per surface.

    Parameters
    ----------
    model : trained LocalOperator
    split_name : "val", "test_id", or "test_ood"
    n_masks_per_surface : how many distinct masks to sample per surface
    base_seed : base for the per-surface, per-mask seed
    mask_config : sampler config (default: MaskSamplerConfig() with defaults)
    device : torch device for the model
    """
    if mask_config is None:
        mask_config = MaskSamplerConfig()
    sampler = MaskSampler(mask_config)

    surfaces = load_split(split_name)
    result = EvalResult(
        split_name=split_name,
        n_surfaces=len(surfaces),
        n_masks_per_surface=n_masks_per_surface,
    )

    for surface_idx, surface in enumerate(surfaces):
        for mask_idx in range(n_masks_per_surface):
            seed = base_seed + surface_idx * 100 + mask_idx

            # Use the model's evaluate_surface helper (which handles
            # centering and per-ring breakdown).
            model_result = evaluate_surface(
                model, surface, sampler, rng_seed=seed, device=device,
            )

            # Now compute baselines on the SAME mask. We have to re-sample
            # with the same seed to ensure we get the same mask.
            rng = torch.Generator().manual_seed(seed)
            mask, d, regime = sampler.sample(surface, rng)

            # Per-surface centering (mirror evaluate_surface's logic)
            xy_mean = surface.V[:, :2].mean(dim=0)
            z_mean = surface.V[mask, 2].mean()
            V_centered = surface.V.clone()
            V_centered[:, :2] = surface.V[:, :2] - xy_mean
            V_centered[:, 2] = surface.V[:, 2] - z_mean
            z_true = V_centered[:, 2]

            z_meanplane = init_z(V_centered, mask)
            z_harmonic = harmonic_infill(
                z_true, surface.edge_index, mask,
            )

            unknown = ~mask
            rmse_meanplane = (
                z_meanplane[unknown] - z_true[unknown]
            ).pow(2).mean().sqrt().item()
            rmse_harmonic = (
                z_harmonic[unknown] - z_true[unknown]
            ).pow(2).mean().sqrt().item()

            # Verify consistency: regime/N from sampler vs from evaluate_surface
            # should match because they used the same seed
            assert regime == model_result.regime, (
                f"Regime mismatch on {surface.surface_id}: "
                f"{regime} vs {model_result.regime}"
            )

            record = MaskEvalRecord(
                surface_id=surface.surface_id,
                reservoir_id=surface.reservoir_id,
                regime=regime,
                seed=seed,
                N=model_result.N,
                n_K=model_result.n_K,
                n_U=model_result.n_U,
                rmse_meanplane=rmse_meanplane,
                rmse_harmonic=rmse_harmonic,
                rmse_model=model_result.rmse_overall,
                rmse_per_ring_model=model_result.rmse_per_ring,
                ring_sizes=model_result.ring_sizes,
            )
            result.records.append(record)

    return result


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------
def aggregate_by_regime(
    result: EvalResult,
) -> dict[str, dict[str, dict[str, float]]]:
    """Group records by regime, compute mean / median / max RMSE for
    each method within each regime.

    Returns
    -------
    {
        "half_plane": {
            "meanplane": {"mean": ..., "median": ..., "max": ..., "n": ...},
            "harmonic":  {...},
            "model":     {...},
        },
        ...
    }
    """
    by_regime: dict[str, list[MaskEvalRecord]] = {}
    for r in result.records:
        by_regime.setdefault(r.regime, []).append(r)

    output: dict[str, dict[str, dict[str, float]]] = {}
    for regime, records in by_regime.items():
        output[regime] = {}
        for method in ["meanplane", "harmonic", "model"]:
            values = [getattr(r, f"rmse_{method}") for r in records]
            output[regime][method] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "n": len(values),
            }
    return output


def aggregate_overall(
    result: EvalResult,
) -> dict[str, dict[str, float]]:
    """Compute mean / median / max RMSE for each method across all records."""
    output: dict[str, dict[str, float]] = {}
    for method in ["meanplane", "harmonic", "model"]:
        values = [getattr(r, f"rmse_{method}") for r in result.records]
        output[method] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "max": max(values),
            "min": min(values),
            "n": len(values),
        }
    return output


def save_result(result: EvalResult, path: str | Path) -> None:
    """Persist an EvalResult to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to plain dict; records become list of dicts
    data = asdict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_result(path: str | Path) -> EvalResult:
    """Load an EvalResult from JSON."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    records = [MaskEvalRecord(**r) for r in data["records"]]
    return EvalResult(
        split_name=data["split_name"],
        n_surfaces=data["n_surfaces"],
        n_masks_per_surface=data["n_masks_per_surface"],
        records=records,
    )
