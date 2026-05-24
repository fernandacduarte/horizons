"""Smoke test: load each fixture as a HorizonSurface and print stats."""
from pathlib import Path
from horizons.data.mesh import HorizonSurface

FIXTURES_DIR = Path("tests/fixtures")

for name in ["plane", "sphere_cap", "anticline"]:
    surf = HorizonSurface.from_npz(FIXTURES_DIR / f"{name}.npz")
    print(f"{name}:")
    print(f"  surface_id : {surf.surface_id}")
    print(f"  vertices   : {surf.n_vertices}")
    print(f"  faces      : {surf.n_faces}")
    print(f"  edges      : {surf.n_edges}  (directed: {surf.edge_index.shape[1]})")
    print(f"  V dtype    : {surf.V.dtype}, shape: {tuple(surf.V.shape)}")
    print(f"  F dtype    : {surf.F.dtype}, shape: {tuple(surf.F.shape)}")
    print(f"  edge_index dtype: {surf.edge_index.dtype}, "
          f"shape: {tuple(surf.edge_index.shape)}")
    print()
