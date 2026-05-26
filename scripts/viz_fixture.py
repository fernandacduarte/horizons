"""Interactive visualization of a fixture surface with vertex normals.

Usage:
    python scripts/viz_fixture.py                # default: anticline
    python scripts/viz_fixture.py plane          # see the plane
    python scripts/viz_fixture.py sphere_cap     # see the sphere cap
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.features import (
    compute_vertex_normals,
    compute_umbrella_laplacian,
)


FIXTURES_DIR = Path("tests/fixtures")


def to_pv_mesh(surface: HorizonSurface) -> pv.PolyData:
    """Convert a HorizonSurface to a PyVista PolyData mesh.

    PyVista's face format is a flat array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    where the leading 3 is the vertex count for each face.
    """
    V_np = surface.V.numpy().astype(np.float64)
    F_np = surface.F.numpy()
    n_faces = F_np.shape[0]
    faces_pv = np.column_stack([np.full(n_faces, 3, dtype=np.int64), F_np])
    return pv.PolyData(V_np, faces_pv.ravel())


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "anticline"
    if name not in {"plane", "sphere_cap", "anticline"}:
        raise SystemExit(f"Unknown fixture: {name}")

    surface = HorizonSurface.from_npz(FIXTURES_DIR / f"{name}.npz")

    # Compute features we want to visualize
    normals = compute_vertex_normals(surface.V, surface.F).numpy()
    z = surface.V[:, 2]
    kappa = compute_umbrella_laplacian(z, surface.edge_index).numpy()

    mesh = to_pv_mesh(surface)
    mesh["z"] = z.numpy()
    mesh["kappa"] = kappa
    mesh["normals"] = normals

    # Two side-by-side plots: surface colored by z, surface colored by curvature
    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 700))

    # Left: surface colored by z, with normal arrows
    plotter.subplot(0, 0)
    plotter.add_text(f"{name} — colored by z", font_size=12)
    plotter.add_mesh(
        mesh, scalars="z", cmap="viridis", show_edges=True,
        edge_color="gray", line_width=0.5,
    )
    # Subsample normals: showing all 400-900 arrows is visual noise
    n_arrows = min(80, surface.n_vertices)
    arrow_idx = np.linspace(0, surface.n_vertices - 1, n_arrows, dtype=int)
    arrow_centers = surface.V.numpy()[arrow_idx]
    arrow_dirs = normals[arrow_idx]
    arrow_scale = float(surface.V[:, :2].abs().max()) * 0.05
    plotter.add_arrows(arrow_centers, arrow_dirs, mag=arrow_scale, color="red")

    # Right: surface colored by umbrella curvature
    plotter.subplot(0, 1)
    plotter.add_text(f"{name} — colored by umbrella Laplacian", font_size=12)
    # Symmetric color scale clipped to the 98th-percentile of |kappa|.
    # Boundary vertices have very few neighbors and produce extreme umbrella
    # Laplacian values that would otherwise dominate the color scale and
    # hide interior structure. Clipping at p98 makes the interior visible.
    kappa_lim = float(np.percentile(np.abs(kappa), 98))
    plotter.add_mesh(
        mesh, scalars="kappa", cmap="coolwarm",
        clim=(-kappa_lim, kappa_lim),
        show_edges=True, edge_color="gray", line_width=0.5,
    )
    print(f"kappa range: [{kappa.min():.4f}, {kappa.max():.4f}]; "
          f"display clipped to ±{kappa_lim:.4f}")

    plotter.link_views()  # camera moves in sync across both subplots
    plotter.show()


if __name__ == "__main__":
    main()
