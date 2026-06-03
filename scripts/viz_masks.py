"""Visualize all three mask regimes on the anticline fixture.

Shows two rows: (top) mask K/U, (bottom) topological distance d.

Usage:
    python scripts/viz_masks.py                  # default: seed 0
    python scripts/viz_masks.py --seed 7         # try a different seed
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import (
    sample_half_plane_mask,
    sample_outward_rectangle_mask,
    sample_outward_rectangle_pinned_mask,
)
from horizons.data.topo_distance import compute_topological_distance


FIXTURES_DIR = Path("tests/fixtures")


def to_pv_mesh(surface: HorizonSurface) -> pv.PolyData:
    V_np = surface.V.numpy().astype(np.float64)
    F_np = surface.F.numpy()
    n_faces = F_np.shape[0]
    faces_pv = np.column_stack([np.full(n_faces, 3, dtype=np.int64), F_np])
    return pv.PolyData(V_np, faces_pv.ravel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi-half-plane", type=float, default=0.5)
    parser.add_argument("--phi-outward", type=float, default=0.6)
    args = parser.parse_args()

    surface = HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")

    # Sample one mask per regime, all with related seeds for visual variety
    rng_hp = torch.Generator().manual_seed(args.seed)
    mask_hp = sample_half_plane_mask(
        surface.V, phi=args.phi_half_plane, rng=rng_hp
    )

    rng_of = torch.Generator().manual_seed(args.seed + 1)
    mask_of = sample_outward_rectangle_mask(
        surface.V, phi=args.phi_outward, rng=rng_of
    )

    rng_op = torch.Generator().manual_seed(args.seed + 1)  # same seed -> nested
    mask_op = sample_outward_rectangle_pinned_mask(
        surface.V, surface.F, phi=args.phi_outward, rng=rng_op
    )

    masks = [
        (f"half_plane (phi={args.phi_half_plane})", mask_hp),
        (f"outward_free (phi={args.phi_outward})", mask_of),
        (f"outward_pinned (phi={args.phi_outward})", mask_op),
    ]

    plotter = pv.Plotter(shape=(2, 3), window_size=(1800, 1000))

    for col, (title, mask) in enumerate(masks):
        d = compute_topological_distance(surface.edge_index, mask)
        # known=1, unknown=0 for display
        mask_int = mask.to(torch.int32).numpy()
        d_np = d.numpy().astype(np.float32)

        # Top row: mask K / U
        plotter.subplot(0, col)
        plotter.add_text(f"{title}\nmask (yellow=K, purple=U)",
                         font_size=10)
        mesh = to_pv_mesh(surface)
        mesh["mask"] = mask_int
        plotter.add_mesh(
            mesh, scalars="mask", cmap=["purple", "yellow"],
            clim=(0, 1), show_edges=True, edge_color="gray", line_width=0.3,
            show_scalar_bar=False,
        )

        # Bottom row: topological distance d
        plotter.subplot(1, col)
        n_known = int(mask.sum().item())
        max_d = int(d.max().item())
        plotter.add_text(
            f"d (max d_i = {max_d}, |K| = {n_known}/{surface.n_vertices})",
            font_size=10,
        )
        mesh2 = to_pv_mesh(surface)
        mesh2["d"] = d_np
        plotter.add_mesh(
            mesh2, scalars="d", cmap="viridis",
            clim=(0, max_d), show_edges=True,
            edge_color="gray", line_width=0.3,
        )

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    main()
