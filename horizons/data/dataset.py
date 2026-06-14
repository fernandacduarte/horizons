"""Dataset for masked-rollout training.

Each item is a dictionary containing everything the rollout needs:
mesh, mask, topological distance, initial z, and metadata. Masks are
resampled per epoch using a deterministic (surface_id, epoch, split)
RNG seed, so train masks vary but val/test masks are stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z


def _make_rng(surface_id: str, epoch: int, split: str) -> torch.Generator:
    """Deterministic RNG: same (surface_id, epoch, split) -> same mask.

    We hash a string seed and use it to seed a fresh Generator. This is
    portable and doesn't depend on the order in which items are accessed.
    """
    seed_str = f"{surface_id}|{epoch}|{split}"
    # Python's built-in hash() is salted by default, so it is not stable
    # across runs. Use SHA-256 for deterministic seeding.
    # Reduce the result to a non-negative 63-bit integer for manual_seed().
    import hashlib
    h = hashlib.sha256(seed_str.encode()).hexdigest()
    seed = int(h[:16], 16) % (2**63)
    return torch.Generator().manual_seed(seed)


class HorizonDataset(Dataset):
    """A collection of HorizonSurfaces with on-the-fly mask sampling.

    Parameters
    ----------
    surfaces : list[HorizonSurface]
        The pre-loaded meshes for this split.
    mask_sampler : MaskSampler
        Configured sampler. The same sampler instance can be used across
        train/val/test (its config is stateless once constructed).
    split : str
        Identifier used in the RNG seed. Typical values: "train", "val", "test".
        Train masks use the current epoch; val/test masks use epoch=0 always.
    initial_epoch : int
        Starting epoch (relevant only for training masks).
    center_per_surface : bool, default True
        Whether to apply per-surface centering of (x, y, z) to the
        coordinates returned by __getitem__. x, y centered by all-vertex
        mean; z centered by the mean over known vertices (using z[U]
        would leak ground truth). See D4.6. Default True; set False for
        tests that compare against uncentered fixtures.
    normalize_per_surface : bool, default False
        Whether to additionally scale (x, y, z) so that the centered
        coordinates lie in [-1, +1] per surface. x, y use the global
        max abs value; z uses the max abs value over known vertices
        only (same no-leakage invariant as centering). When enabled,
        downstream RMSE computations must denormalize by multiplying
        by z_scale to report in physical units. Default False to
        preserve backward compatibility.
    """

    def __init__(
        self,
        surfaces: list[HorizonSurface],
        mask_sampler: MaskSampler,
        split: str,
        initial_epoch: int = 0,
        center_per_surface: bool = True,
        normalize_per_surface: bool = False,
    ) -> None:
        if len(surfaces) == 0:
            raise ValueError("HorizonDataset must be non-empty")
        if normalize_per_surface and not center_per_surface:
            raise ValueError(
                "normalize_per_surface requires center_per_surface=True"
            )
        self.surfaces = surfaces
        self.mask_sampler = mask_sampler
        self.split = split
        self._epoch = initial_epoch
        self.center_per_surface = center_per_surface
        self.normalize_per_surface = normalize_per_surface

    def set_epoch(self, epoch: int) -> None:
        """Called at the start of each training epoch so masks resample.

        For val/test datasets this can be called too but has no effect
        because we ignore the epoch in their seed (always 0).
        """
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.surfaces)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        surface = self.surfaces[idx]

        # Train masks vary by epoch; val/test masks are stable
        epoch_for_seed = self._epoch if self.split == "train" else 0
        rng = _make_rng(surface.surface_id, epoch_for_seed, self.split)

        mask, d, regime = self.mask_sampler.sample(surface, rng)

        # Per-surface centering (resolves D4.6).
        # x, y can use all vertices since they're fully observed.
        # z MUST use only z[K] — using z[U] would leak ground truth into
        # the input the model sees.
        if self.center_per_surface:
            xy_mean = surface.V[:, :2].mean(dim=0)
            z_mean = surface.V[mask, 2].mean()
            V_centered = surface.V.clone()
            V_centered[:, :2] = surface.V[:, :2] - xy_mean
            V_centered[:, 2] = surface.V[:, 2] - z_mean
        else:
            V_centered = surface.V
            xy_mean = torch.zeros(2, dtype=surface.V.dtype)
            z_mean = surface.V.new_zeros(())

        # Per-surface normalization (Candidate 9, Stage 11.6).
        # Scale all coordinates so the centered data lies in roughly
        # [-1, +1]. The xy_scale uses ALL vertices since x, y are known.
        # The z_scale uses ONLY z[K] to maintain the no-leakage invariant.
        # Same float64 promotion trick as fit_mean_plane for stability.
        if self.normalize_per_surface:
            xy_scale = V_centered[:, :2].to(torch.float64).abs().max().item()
            z_scale = V_centered[mask, 2].to(torch.float64).abs().max().item()
            # Guard against degenerate (zero-extent) surfaces by flooring
            # the scale to 1.0. This means truly flat surfaces (z_range=0)
            # don't get divided by zero; their normalized z is just their
            # actual z, which is fine because it's already zero anyway.
            xy_scale = max(xy_scale, 1.0)
            z_scale = max(z_scale, 1.0)
            V_centered = V_centered.clone()
            V_centered[:, :2] = V_centered[:, :2] / xy_scale
            V_centered[:, 2] = V_centered[:, 2] / z_scale
            xy_scale_tensor = torch.tensor(xy_scale, dtype=surface.V.dtype)
            z_scale_tensor = torch.tensor(z_scale, dtype=surface.V.dtype)
        else:
            xy_scale_tensor = torch.tensor(1.0, dtype=surface.V.dtype)
            z_scale_tensor = torch.tensor(1.0, dtype=surface.V.dtype)

        z0 = init_z(V_centered, mask)

        return {
            # Mesh (V is centered if center_per_surface=True)
            "V": V_centered,                 # (n, 3)
            "F": surface.F,                  # (n_faces, 3)
            "edge_index": surface.edge_index,  # (2, n_edges_directed)
            # Mask + topological distance
            "mask": mask,                    # (n,) bool, True=known
            "d": d,                          # (n,) int64
            "N": int(d.max().item()),        # int: rollout depth
            # Initial state
            "z0": z0,                        # (n,) float
            "z_true": V_centered[:, 2],      # (n,) float — supervision target (centered)
            # Centering offsets (for inverting the prediction back to original frame)
            "xy_mean": xy_mean,              # (2,) float
            "z_mean": z_mean if isinstance(z_mean, torch.Tensor) else torch.tensor(z_mean),
            # Normalization scales (for denormalizing predictions back to meters)
            "xy_scale": xy_scale_tensor,     # () float
            "z_scale": z_scale_tensor,       # () float
            # Metadata
            "surface_id": surface.surface_id,
            "reservoir_id": surface.reservoir_id,
            "regime": regime,
        }


def load_fixture_dataset(
    fixture_names: list[str],
    fixtures_dir: str | Path = "tests/fixtures",
    mask_config: MaskSamplerConfig | None = None,
    split: str = "train",
) -> HorizonDataset:
    """Convenience builder: load fixtures and wrap them in a HorizonDataset.

    Used to bootstrap training on synthetic data before the .ts loader exists.
    """
    fixtures_dir = Path(fixtures_dir)
    surfaces = [
        HorizonSurface.from_npz(fixtures_dir / f"{name}.npz")
        for name in fixture_names
    ]
    if mask_config is None:
        mask_config = MaskSamplerConfig()
    sampler = MaskSampler(mask_config)
    return HorizonDataset(surfaces, sampler, split=split,
                          center_per_surface=False)


def load_split_dataset(
    split_name: str,
    mask_config: MaskSamplerConfig | None = None,
    surfaces_dir: str | Path = "data/surfaces",
    split_file: str | Path = "data/splits/split_v1.json",
    center_per_surface: bool = True,
    normalize_per_surface: bool = False,
) -> HorizonDataset:
    """Build a HorizonDataset for one of the real-data splits.

    Loads from data/surfaces/ and data/splits/split_v1.json (produced by
    scripts/build_dataset.py and scripts/build_split.py respectively).
    Applies per-surface centering by default — see D4.6.
    """
    from horizons.data.loaders import load_split as _load_split
    surfaces = _load_split(
        split_name, surfaces_dir=surfaces_dir, split_file=split_file,
    )
    if mask_config is None:
        mask_config = MaskSamplerConfig()
    sampler = MaskSampler(mask_config)
    return HorizonDataset(
        surfaces, sampler, split=split_name,
        center_per_surface=center_per_surface,
        normalize_per_surface=normalize_per_surface,
    )
