"""Helpers for loading the canonical dataset from disk.

These wrap `HorizonSurface.from_npz` to load surfaces by split name,
using the metadata.json + split_v1.json files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from horizons.data.mesh import HorizonSurface


SplitName = Literal["train", "val", "test_id", "test_ood"]


def load_split(
    split_name: SplitName,
    surfaces_dir: str | Path = "data/surfaces",
    split_file: str | Path = "data/splits/split_v1.json",
) -> list[HorizonSurface]:
    """Load all HorizonSurfaces in a given split.

    Parameters
    ----------
    split_name : one of "train", "val", "test_id", "test_ood"
    surfaces_dir : directory containing <surface_id>.npz files and metadata.json
    split_file : path to the split JSON

    Returns
    -------
    surfaces : list of HorizonSurface, with surface_id and reservoir_id
        populated from metadata.json.
    """
    surfaces_dir = Path(surfaces_dir)
    split_file = Path(split_file)

    with open(split_file) as f:
        split = json.load(f)
    if split_name not in split:
        raise KeyError(
            f"split_name {split_name!r} not found in {split_file}. "
            f"Available: {sorted(split.keys())}"
        )

    # Load metadata to recover reservoir_id per surface
    metadata_path = surfaces_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    sid_to_reservoir = {k["surface_id"]: k["reservoir_id"] for k in metadata["kept"]}

    surfaces = []
    for sid in split[split_name]:
        path = surfaces_dir / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Surface file missing: {path}")
        reservoir_id = sid_to_reservoir.get(sid)
        surf = HorizonSurface.from_npz(
            path, surface_id=sid, reservoir_id=reservoir_id,
        )
        surfaces.append(surf)
    return surfaces
