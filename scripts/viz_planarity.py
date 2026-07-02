"""Show a split's planarity: a small-multiples grid of every surface in the
split, each coloured by its deviation from its own best-fit plane and titled with
the plane-fit residual RMSE.

This is the visual behind the mean-plane finding: a least-squares plane
z = a*x + b*y + c is fit through each surface's vertices; the per-vertex residual
z - plane is the colour. Planar surfaces (test_ood) are flat sheets with a ~0
residual everywhere; structured surfaces (test_id) are terrain with large,
varied residuals.

Usage:
    python scripts/viz_planarity.py --split test_ood --out test_ood_planar.png
    python scripts/viz_planarity.py --split test_id  --out test_id_structured.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv

from horizons.data.loaders import load_split


def plane_residual(V: np.ndarray) -> np.ndarray:
    """Per-vertex residual of the least-squares plane through all vertices."""
    A = np.column_stack([V[:, 0], V[:, 1], np.ones(len(V))])
    coef, *_ = np.linalg.lstsq(A, V[:, 2], rcond=None)
    return V[:, 2] - A @ coef


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_ood")
    p.add_argument("--split-file", default="data/splits/split_v2.json")
    p.add_argument("--out", default=None, help="save a PNG (off-screen) instead of a window")
    p.add_argument("--cols", type=int, default=0, help="grid columns (default: auto)")
    args = p.parse_args()

    surfaces = load_split(args.split, split_file=args.split_file)
    n = len(surfaces)
    cols = args.cols or (n if n <= 5 else math.ceil(n / 2))
    rows = math.ceil(n / cols)

    off = args.out is not None
    pl = pv.Plotter(shape=(rows, cols), window_size=(360 * cols, 380 * rows), off_screen=off)
    for i, s in enumerate(surfaces):
        r, c = divmod(i, cols)
        pl.subplot(r, c)
        V = s.V.numpy().astype(np.float64)
        resid = plane_residual(V)
        rmse = float(np.sqrt((resid ** 2).mean()))
        Vc = V.copy()
        Vc[:, :2] -= V[:, :2].mean(0)
        Vc[:, 2] -= V[:, 2].mean()
        F = s.F.numpy()
        mesh = pv.PolyData(Vc, np.column_stack(
            [np.full(F.shape[0], 3, np.int64), F]).ravel())
        mesh["resid"] = resid
        lim = max(abs(resid).max(), 2.0)  # floor avoids amplifying numeric noise on planes
        tag = "  planar" if rmse < 1.0 else ""
        pl.add_text(f"{s.surface_id}\nplane-residual RMSE = {rmse:.2f} m{tag}",
                    font_size=9)
        pl.add_mesh(mesh, scalars="resid", cmap="coolwarm", clim=(-lim, lim),
                    show_scalar_bar=False)
        pl.view_isometric()
    if off:
        pl.screenshot(args.out)
        print(f"wrote {args.out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
