"""Visualize z^0 (mean-plane initialization) vs. z_true on a masked surface.

Three panels:
  - z_true on the surface
  - mask (K in yellow, U in purple)
  - z^0 (= z_true on K, mean-plane evaluation on U)

The middle panel acts as a visual key. The third panel shows the starting
point of every rollout: known vertices at their true depth, unknown vertices
on the regional trend plane.

Usage:
    python scripts/viz_init.py                       # default: anticline
    python scripts/viz_init.py --fixture sphere_cap
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import sample_half_plane_mask
from horizons.data.init import init_z


FIXTURES_DIR = Path("tests/fixtures")


def to_pv_mesh(surface: HorizonSurface) -> pv.PolyData:
    V_np = surface.V.numpy().astype(np.float64)
    F_np = surface.F.numpy()
    n_faces = F_np.shape[0]
    faces_pv = np.column_stack([np.full(n_faces, 3, dtype=np.int64), F_np])
    return pv.PolyData(V_np, faces_pv.ravel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="anticline",
                        choices=["plane", "sphere_cap", "anticline"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi", type=float, default=0.5)
    args = parser.parse_args()

    surface = HorizonSurface.from_npz(FIXTURES_DIR / f"{args.fixture}.npz")
    rng = torch.Generator().manual_seed(args.seed)
    mask = sample_half_plane_mask(surface.V, phi=args.phi, rng=rng)
    z0 = init_z(surface.V, mask)

    z_true = surface.V[:, 2].numpy()
    z0_np = z0.numpy()
    mask_np = mask.to(torch.int32).numpy()

    # Common color scale for z_true and z0 so they're directly comparable
    z_min = float(min(z_true.min(), z0_np.min()))
    z_max = float(max(z_true.max(), z0_np.max()))

    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

    # Panel 1: z_true
    plotter.subplot(0, 0)
    plotter.add_text(f"z_true (ground truth)", font_size=12)
    m1 = to_pv_mesh(surface)
    m1["z"] = z_true
    plotter.add_mesh(m1, scalars="z", cmap="viridis",
                     clim=(z_min, z_max),
                     show_edges=True, edge_color="gray", line_width=0.3)

    # Panel 2: mask
    plotter.subplot(0, 1)
    plotter.add_text(f"mask (yellow=K, purple=U; phi={args.phi})",
                     font_size=12)
    m2 = to_pv_mesh(surface)
    m2["mask"] = mask_np
    plotter.add_mesh(m2, scalars="mask", cmap=["purple", "yellow"],
                     clim=(0, 1), show_edges=True, edge_color="gray",
                     line_width=0.3, show_scalar_bar=False)

    # Panel 3: z^0 — render with the PREDICTED z as geometry, not the true z.
    # This way the shape we see is the actual initialization, not the true
    # surface tinted by z^0.
    plotter.subplot(0, 2)
    plotter.add_text("z^0 (mean-plane init, rendered as geometry)",
                     font_size=12)
    V_z0 = surface.V.clone()
    V_z0[:, 2] = z0  # replace z with the initialization
    surface_z0 = HorizonSurface(
        V=V_z0, F=surface.F, edge_index=surface.edge_index,
        surface_id=surface.surface_id,
    )
    m3 = to_pv_mesh(surface_z0)
    m3["z"] = z0_np
    plotter.add_mesh(m3, scalars="z", cmap="viridis",
                     clim=(z_min, z_max),
                     show_edges=True, edge_color="gray", line_width=0.3)

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    main()
