"""Render every mesh in a split as one grid image.

Lays out all surfaces of a split (train / val / test_id / test_ood) in a single
figure — one subplot per surface, coloured by elevation — so the whole split can
be eyeballed at a glance: shapes, sizes, relief, and any odd surface.

Each subplot is centred and framed independently (the surfaces differ in extent
and depth), so the views are deliberately NOT linked and each gets its own
colour range. The subplot title is the surface_id and its vertex count.

Usage:
    python scripts/viz_split_grid.py --split train
    python scripts/viz_split_grid.py --split val --out figures/val_grid.png
    python scripts/viz_split_grid.py --split test           # test_id + test_ood together
    python scripts/viz_split_grid.py --split train --top-down --cols 6
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.loaders import load_split

SPLITS = ("train", "val", "test_id", "test_ood", "test")
Z_CMAP = "viridis"  # elevation colouring, per-surface range


def pv_mesh(xy: torch.Tensor, z: torch.Tensor, F: torch.Tensor) -> pv.PolyData:
    V = torch.column_stack([xy, z]).detach().cpu().numpy().astype(np.float64)
    Fn = F.cpu().numpy()
    faces = np.column_stack([np.full(Fn.shape[0], 3, np.int64), Fn]).ravel()
    return pv.PolyData(V, faces)


def load_surfaces(split: str, split_file: str | Path) -> list:
    """Load a split. 'test' is a convenience alias for test_id + test_ood."""
    if split == "test":
        return (load_split("test_id", split_file=split_file)
                + load_split("test_ood", split_file=split_file))
    return load_split(split, split_file=split_file)


def plot_split_grid(
    surfaces: list,
    *,
    cols: int | None = None,
    cell_size: int = 420,
    cmap: str = Z_CMAP,
    show_edges: bool = False,
    top_down: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Build a grid Plotter with one surface per cell, coloured by elevation.

    Parameters
    ----------
    surfaces : list of HorizonSurface
    cols : columns in the grid (default: ceil(sqrt(n)), i.e. roughly square)
    cell_size : pixel size of each grid cell
    cmap : colormap for the elevation scalars
    show_edges : overlay the triangle edges
    top_down : look straight down (xy) instead of the default 3-D view
    off_screen : render off-screen (for screenshotting without a window)
    """
    n = len(surfaces)
    if n == 0:
        raise ValueError("no surfaces to plot")
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    pl = pv.Plotter(shape=(rows, cols),
                    window_size=(cols * cell_size, rows * cell_size),
                    off_screen=off_screen)
    for i, s in enumerate(surfaces):
        pl.subplot(i // cols, i % cols)
        # Centre each surface on its own centroid so every cell is framed well.
        Vc = s.V.clone()
        Vc[:, :2] -= s.V[:, :2].mean(0)
        Vc[:, 2] -= s.V[:, 2].mean()
        m = pv_mesh(Vc[:, :2], Vc[:, 2], s.F)
        m["z"] = Vc[:, 2].detach().cpu().numpy()
        pl.add_mesh(m, scalars="z", cmap=cmap, show_scalar_bar=False,
                    show_edges=show_edges)
        pl.add_text(f"{s.surface_id}\nV={s.V.shape[0]:,}", font_size=7)
        if top_down:
            pl.view_xy()
    # Any trailing cells (n < rows*cols) are simply left empty.
    return pl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=SPLITS, default="train",
                   help="'test' = test_id + test_ood combined")
    p.add_argument("--split-file", default="data/splits/split_v2.json")
    p.add_argument("--cols", type=int, default=None,
                   help="grid columns (default: roughly square)")
    p.add_argument("--cell-size", type=int, default=420, help="pixel size per grid cell")
    p.add_argument("--out", default=None,
                   help="save a PNG here (off-screen) instead of showing a window")
    p.add_argument("--top-down", action="store_true", help="render straight-down (xy) views")
    p.add_argument("--show-edges", action="store_true", help="draw the triangle edges")
    args = p.parse_args()

    surfaces = load_surfaces(args.split, args.split_file)
    print(f"{args.split}: {len(surfaces)} surfaces "
          f"({sum(s.V.shape[0] for s in surfaces):,} vertices total)")
    for s in surfaces:
        print(f"  {s.surface_id:<28} V={s.V.shape[0]:,}")

    off = args.out is not None
    pl = plot_split_grid(
        surfaces, cols=args.cols, cell_size=args.cell_size,
        show_edges=args.show_edges, top_down=args.top_down, off_screen=off,
    )
    if off:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(args.out)
        print(f"wrote {args.out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
