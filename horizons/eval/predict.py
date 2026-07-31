"""Reconstruct the depth fields of a single surface: input, baseline, model.

`evaluate_surface` returns metrics; this module returns the *surfaces* — the
full z fields needed to draw or export a prediction. One call gives the
ground truth, the harmonic infill baseline and the model output on the same
mask, all in centered metres, so they can be compared directly.

`RunSpec` carries the inference settings of a training run (architecture,
init, approach, normalization, mask mixture) read from its saved
`config.yaml`, which keeps callers from re-deriving them from the YAML by
hand.

    spec = RunSpec.from_dir("outputs/tensorboard/run_20260720_184019")
    model = spec.load_model()
    surfaces = load_split("test_id", split_file=spec.split_file)
    pred = spec.predict(model, surfaces[0], seed=1000)
    pred.z_model, pred.z_harmonic, pred.z_true   # (n,) each, centered metres
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from horizons.data.harmonic_infill import harmonic_infill
from horizons.data.init import init_z_dispatch
from horizons.data.loaders import load_split
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.mesh import HorizonSurface
from horizons.eval.checkpoint import load_checkpoint
from horizons.training.rollout import rollout


def split_seed(base_seed: int, index: int, mask_index: int = 0) -> int:
    """Mask seed of one case, using the evaluation driver's scheme.

    Keeping this identical to `evaluate_split` is what makes a figure or an
    export show the very masks the reported metrics were computed on.
    """
    return base_seed + 100 * index + mask_index


def load_split_cases(
    split_name: str,
    *,
    split_file: str | Path,
    surface_ids: Sequence[str] | None = None,
) -> list[tuple[int, HorizonSurface]]:
    """Surfaces of a split, each paired with its index in the *full* split.

    The index is what `split_seed` consumes, so narrowing to a few surfaces
    still reproduces the masks they would have had in a whole-split run.

    Raises
    ------
    KeyError
        If a requested surface_id is not in the split.
    """
    cases = list(enumerate(load_split(split_name, split_file=split_file)))
    if surface_ids is None:
        return cases

    wanted = set(surface_ids)
    selected = [(i, s) for i, s in cases if s.surface_id in wanted]
    missing = wanted - {s.surface_id for _, s in selected}
    if missing:
        raise KeyError(
            f"not in {split_name}: {sorted(missing)}"
        )
    return selected


@dataclass
class SurfacePrediction:
    """Depth fields and metrics for one (surface, mask) pair.

    All z fields are in centered metres: the per-surface xy mean and the mean
    z over K are subtracted (as during training), but any per-surface
    normalization has been undone, so differences are metres. `xy_offset` and
    `z_offset` are the shifts that were subtracted, so `to_world` puts a field
    back in the survey's own coordinates without altering it otherwise.

    Attributes
    ----------
    surface_id, reservoir_id, regime : identification of the case.
    N : rollout depth of the surface (max topological distance on U).
    n_K, n_U : number of known / unknown vertices.
    xy : (n, 2) centered horizontal coordinates.
    faces : (n_faces, 3) triangle indices.
    mask : (n,) bool, True on known vertices.
    d : (n,) topological distance from K.
    z_true : (n,) ground-truth depth.
    z_init : (n,) initialization the model started from (z0).
    z_harmonic : (n,) harmonic infill baseline.
    z_model : (n,) model prediction after the rollout.
    xy_offset : (2,) horizontal mean subtracted during centering.
    z_offset : mean z over K subtracted during centering.
    rmse_harmonic, rmse_model : RMSE over U in metres.
    """

    surface_id: str
    reservoir_id: str | None
    regime: str
    N: int
    n_K: int
    n_U: int
    xy: torch.Tensor
    faces: torch.Tensor
    mask: torch.Tensor
    d: torch.Tensor
    z_true: torch.Tensor
    z_init: torch.Tensor
    z_harmonic: torch.Tensor
    z_model: torch.Tensor
    xy_offset: torch.Tensor
    z_offset: float
    rmse_harmonic: float
    rmse_model: float

    def world_xy(self) -> torch.Tensor:
        """The horizontal coordinates as they are in the source data."""
        return self.xy + self.xy_offset

    def to_world(self, z: torch.Tensor) -> torch.Tensor:
        """Undo the z centering of a depth field.

        Centering is a pure additive shift, so this recovers the original
        depths exactly rather than rescaling anything.
        """
        return z + self.z_offset


@torch.no_grad()
def predict_surface(
    model: torch.nn.Module | None,
    surface: HorizonSurface,
    mask_sampler: MaskSampler,
    seed: int,
    *,
    normalize_per_surface: bool = False,
    init_method: str = "meanplane",
    approach: str = "rollout",
    hybrid_n_passes: int = 3,
    rollout_method: str = "standard",
    rollout_n_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
) -> SurfacePrediction:
    """Run the harmonic baseline and (optionally) the model on one surface.

    Centering, normalization and rollout depth follow `evaluate_surface`, so
    the returned RMSEs match what the evaluation driver reports for the same
    seed and settings.

    Parameters
    ----------
    model : trained LocalOperator, or None to compute the baseline only
        (in which case `z_model` is the initialization and `rmse_model` NaN).
    mask_sampler, seed : the mask to hold out; the same seed reproduces it.
    normalize_per_surface, init_method, approach, hybrid_n_passes,
    rollout_method, rollout_n_multiplier :
        Must match the settings the checkpoint was trained with.
    """
    device = torch.device(device)

    rng = torch.Generator().manual_seed(seed)
    mask, d, regime = mask_sampler.sample(surface, rng)

    # Per-surface centering (mirrors HorizonDataset): xy about its mean, z
    # about the mean over the known vertices.
    xy_offset = surface.V[:, :2].mean(dim=0)
    z_offset = surface.V[mask, 2].mean()
    V_centered = surface.V.clone()
    V_centered[:, :2] = surface.V[:, :2] - xy_offset
    V_centered[:, 2] = surface.V[:, 2] - z_offset

    z_true_m = V_centered[:, 2].clone()
    xy_m = V_centered[:, :2].clone()

    if normalize_per_surface:
        xy_scale = max(
            V_centered[:, :2].to(torch.float64).abs().max().item(), 1.0
        )
        z_scale = max(
            V_centered[mask, 2].to(torch.float64).abs().max().item(), 1.0
        )
        V_model = V_centered.clone()
        V_model[:, :2] /= xy_scale
        V_model[:, 2] /= z_scale
    else:
        V_model, z_scale = V_centered, 1.0

    z0 = init_z_dispatch(
        V_model, mask, surface.edge_index, method=init_method
    )
    z_harmonic = harmonic_infill(z_true_m, surface.edge_index, mask)

    N = int(d.max().item())
    unknown = ~mask

    if model is None:
        z_model_m = z0 * z_scale
        rmse_model = float("nan")
    else:
        model = model.to(device).eval()
        rollout_N = (
            hybrid_n_passes
            if approach == "hybrid"
            else max(1, round(rollout_n_multiplier * N))
        )
        result = rollout(
            model,
            z0=z0.to(device),
            z_true=V_model[:, 2].to(device),
            V_xy=V_model[:, :2].to(device),
            F=surface.F.to(device),
            edge_index=surface.edge_index.to(device),
            mask=mask.to(device),
            d=d.to(device),
            N=rollout_N,
            rollout_method=rollout_method,
        )
        z_model_m = result.z_trajectory[-1].cpu() * z_scale
        rmse_model = (
            (z_model_m[unknown] - z_true_m[unknown]).pow(2).mean().sqrt().item()
        )

    rmse_harmonic = (
        (z_harmonic[unknown] - z_true_m[unknown]).pow(2).mean().sqrt().item()
    )

    return SurfacePrediction(
        surface_id=surface.surface_id,
        reservoir_id=surface.reservoir_id,
        regime=regime,
        N=N,
        n_K=int(mask.sum().item()),
        n_U=int(unknown.sum().item()),
        xy=xy_m,
        faces=surface.F,
        mask=mask,
        d=d,
        z_true=z_true_m,
        z_init=z0 * z_scale,
        z_harmonic=z_harmonic,
        z_model=z_model_m,
        xy_offset=xy_offset,
        z_offset=float(z_offset),
        rmse_harmonic=rmse_harmonic,
        rmse_model=rmse_model,
    )


@dataclass
class RunSpec:
    """Inference settings of a training run, read from its `config.yaml`."""

    run_dir: Path
    cfg: dict[str, Any]
    split_file: str
    normalize_per_surface: bool
    init_method: str
    approach: str
    hybrid_n_passes: int
    rollout_method: str
    rollout_n_multiplier: float
    mask_config: MaskSamplerConfig = field(default_factory=MaskSamplerConfig)

    @classmethod
    def from_dir(cls, run_dir: str | Path) -> "RunSpec":
        run_dir = Path(run_dir)
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml in {run_dir}; cannot recover the run's "
                f"architecture and inference settings."
            )
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        data = cfg.get("data", {})
        mask_cfg = cfg.get("mask")
        return cls(
            run_dir=run_dir,
            cfg=cfg,
            split_file=data.get("split_file", "data/splits/split_v1.json"),
            normalize_per_surface=bool(
                data.get("normalize_per_surface", False)
            ),
            init_method=data.get("init_method", "meanplane"),
            approach=cfg.get("approach", "rollout"),
            hybrid_n_passes=int(cfg.get("hybrid", {}).get("n_passes", 3)),
            rollout_method=cfg.get("rollout", {}).get("method", "standard"),
            rollout_n_multiplier=float(
                cfg.get("rollout", {}).get("n_multiplier", 1)
            ),
            mask_config=(
                MaskSamplerConfig.from_dictconfig(mask_cfg)
                if mask_cfg
                else MaskSamplerConfig()
            ),
        )

    def load_model(
        self, *, ckpt_name: str = "best.pt", device: str | torch.device = "cpu"
    ) -> torch.nn.Module:
        """Load the run's checkpoint into a matching architecture."""
        model_cfg = self.cfg.get("model", {})
        return load_checkpoint(
            self.run_dir / ckpt_name,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            n_message_passing=int(model_cfg.get("n_layers", 2)),
            output_init_scale=float(model_cfg.get("output_init_scale", 0.01)),
            conv_type=model_cfg.get("type", "sage"),
            aggr=model_cfg.get("aggr", "mean"),
            mask_mode=model_cfg.get("mask_mode", "binary"),
            use_mask_feature=bool(model_cfg.get("use_mask_feature", True)),
            device=device,
        ).model

    def mask_sampler(self) -> MaskSampler:
        return MaskSampler(self.mask_config)

    def predict(
        self,
        model: torch.nn.Module | None,
        surface: HorizonSurface,
        seed: int,
        *,
        device: str | torch.device = "cpu",
    ) -> SurfacePrediction:
        """`predict_surface` with this run's settings filled in."""
        return predict_surface(
            model,
            surface,
            self.mask_sampler(),
            seed,
            normalize_per_surface=self.normalize_per_surface,
            init_method=self.init_method,
            approach=self.approach,
            hybrid_n_passes=self.hybrid_n_passes,
            rollout_method=self.rollout_method,
            rollout_n_multiplier=self.rollout_n_multiplier,
            device=device,
        )
