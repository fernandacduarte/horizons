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
    """

    def __init__(
        self,
        surfaces: list[HorizonSurface],
        mask_sampler: MaskSampler,
        split: str,
        initial_epoch: int = 0,
        center_per_surface: bool = True,
    ) -> None:
        if len(surfaces) == 0:
            raise ValueError("HorizonDataset must be non-empty")
        self.surfaces = surfaces
        self.mask_sampler = mask_sampler
        self.split = split
        self._epoch = initial_epoch
        self.center_per_surface = center_per_surface

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
    )
