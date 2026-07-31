"""Rendering of triangulated horizon surfaces with PyVista.

The viz scripts all need the same two things: turn a horizon's
(vertices, faces, z) into a ``pv.PolyData``, and lay several of those out in
a grid of linked 3-D views. This module holds that shared machinery so the
scripts only describe *what* to show.

Typical use — one row per surface, one column per method:

    from horizons.viz.mesh import SurfacePanel, plot_surface_grid

    rows = [[SurfacePanel.from_mesh(xy, F, z=z_true,     title="ground truth"),
             SurfacePanel.from_mesh(xy, F, z=z_harmonic, title="harmonic"),
             SurfacePanel.from_mesh(xy, F, z=z_model,    title="model")]]
    plot_surface_grid(rows, out="fig.png")

Panels within a row share a camera (they are the same surface under different
z fields), so shapes are directly comparable; rows are independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyvista as pv
import torch

ArrayLike = torch.Tensor | np.ndarray

#: Default look: one flat colour for the whole surface, dark triangle edges.
SURFACE_COLOR = "#9EC1E3"
EDGE_COLOR = "#20313D"


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auto_z_exaggeration(
    vertices: ArrayLike,
    z: ArrayLike | None = None,
    *,
    target_ratio: float = 0.2,
) -> float:
    """Vertical exaggeration that makes a near-planar horizon readable.

    Horizons span kilometres horizontally but only tens to hundreds of metres
    in depth, so at true scale they render as flat sheets. This returns the
    factor that stretches z until its range is `target_ratio` of the wider
    horizontal extent. Never returns less than 1, so a surface that is
    already steep is left alone.
    """
    V = _to_numpy(vertices)
    z_np = V[:, 2] if z is None else _to_numpy(z).reshape(-1)

    xy_extent = max(
        float(V[:, 0].max() - V[:, 0].min()),
        float(V[:, 1].max() - V[:, 1].min()),
    )
    z_extent = float(z_np.max() - z_np.min())
    if z_extent <= 0.0 or xy_extent <= 0.0:
        return 1.0
    return max(1.0, target_ratio * xy_extent / z_extent)


def triangulated_polydata(
    vertices: ArrayLike,
    faces: ArrayLike,
    z: ArrayLike | None = None,
    *,
    z_exaggeration: float = 1.0,
) -> pv.PolyData:
    """Build a PyVista triangle mesh from a horizon's vertices and faces.

    Parameters
    ----------
    vertices : (n, 3) or (n, 2) array
        Vertex coordinates. With shape (n, 2) only x and y are given and `z`
        must be supplied; with shape (n, 3) the third column is used as z
        unless `z` overrides it (the usual case when swapping in a predicted
        depth field).
    faces : (n_faces, 3) integer array
        Triangle vertex indices, 0-based.
    z : (n,) array, optional
        Depth field to attach to the xy coordinates.
    z_exaggeration : float
        Multiplier on z. Horizons span kilometres in xy but only tens of
        metres in z, so a factor > 1 makes the relief readable. Applied to
        z as given, so pass centered z to keep the surface near the origin.

    Returns
    -------
    pv.PolyData
        A mesh of `n_faces` triangles, suitable for `Plotter.add_mesh`.
    """
    V = _to_numpy(vertices)
    if V.ndim != 2 or V.shape[1] not in (2, 3):
        raise ValueError(
            f"vertices must have shape (n, 2) or (n, 3); got {V.shape}"
        )

    if z is None:
        if V.shape[1] != 3:
            raise ValueError("z is required when vertices has shape (n, 2)")
        xy, z_np = V[:, :2], V[:, 2]
    else:
        z_np = _to_numpy(z).reshape(-1)
        if z_np.shape[0] != V.shape[0]:
            raise ValueError(
                f"z has {z_np.shape[0]} values but there are {V.shape[0]} "
                f"vertices"
            )
        xy = V[:, :2]

    points = np.column_stack([xy, z_np * z_exaggeration]).astype(np.float64)

    Fn = _to_numpy(faces)
    if Fn.ndim != 2 or Fn.shape[1] != 3:
        raise ValueError(f"faces must have shape (n_faces, 3); got {Fn.shape}")
    Fn = Fn.astype(np.int64)
    cells = np.column_stack([np.full(Fn.shape[0], 3, np.int64), Fn]).ravel()

    return pv.PolyData(points, cells)


@dataclass
class SurfacePanel:
    """One view in a figure: a triangulated surface plus its caption."""

    mesh: pv.PolyData
    title: str = ""

    @classmethod
    def from_mesh(
        cls,
        vertices: ArrayLike,
        faces: ArrayLike,
        z: ArrayLike | None = None,
        *,
        title: str = "",
        z_exaggeration: float = 1.0,
    ) -> "SurfacePanel":
        """Build a panel straight from a horizon's arrays."""
        return cls(
            mesh=triangulated_polydata(
                vertices, faces, z, z_exaggeration=z_exaggeration
            ),
            title=title,
        )


def _union_bounds(meshes: Sequence[pv.PolyData]) -> tuple[float, ...]:
    """Smallest box containing every mesh, in PyVista's bounds order."""
    boxes = np.array([m.bounds for m in meshes], dtype=np.float64)
    return tuple(
        np.where(
            np.arange(6) % 2 == 0, boxes.min(axis=0), boxes.max(axis=0)
        ).tolist()
    )


def plot_surface_grid(
    rows: Sequence[Sequence[SurfacePanel | None]],
    *,
    out: str | Path | None = None,
    color: str = SURFACE_COLOR,
    show_edges: bool = True,
    edge_color: str = EDGE_COLOR,
    line_width: float = 0.25,
    panel_size: tuple[int, int] = (620, 520),
    font_size: int = 11,
    link_rows: bool = True,
    background: str = "white",
    text_color: str = "black",
    camera_position: str | Iterable = "iso",
    zoom: float = 1.0,
) -> None:
    """Render a grid of triangulated surfaces, one subplot per panel.

    Every surface is drawn in a single flat colour, so the only visual signal
    is geometry and the triangulation itself.

    Parameters
    ----------
    rows : sequence of sequences of SurfacePanel or None
        `rows[r][c]` is the panel at row r, column c. `None` leaves a blank
        subplot, which is how ragged rows are padded.
    out : path, optional
        Where to write a PNG. Rendering is off-screen when given; otherwise
        an interactive window opens.
    color, show_edges, edge_color, line_width :
        Mesh appearance. `line_width` should stay well below 1 for dense
        meshes, otherwise the edges swallow the surface.
    panel_size : (width, height)
        Pixel size of a single subplot; the window is this times the grid.
    link_rows : bool
        Share one camera across each row, so dragging one panel rotates the
        whole row. Rows stay independent. Framing is already identical within
        a row regardless of this flag: every panel is fitted to the row's
        combined bounds, so a prediction that overshoots stays in frame and
        does not shrink its neighbours.
    camera_position : str or camera tuple
        Passed to PyVista, e.g. "iso", "xy", or an explicit camera.
    zoom : float
        Zoom factor applied after the camera is set. > 1 fills more of the
        subplot, which helps because the reset view leaves wide margins.
    """
    if not rows:
        raise ValueError("rows is empty; nothing to plot")

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    off_screen = out is not None

    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        window_size=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        off_screen=off_screen,
    )
    plotter.set_background(background)

    for r, row in enumerate(rows):
        row_bounds = _union_bounds([p.mesh for p in row if p is not None])
        for c in range(n_cols):
            panel = row[c] if c < len(row) else None
            plotter.subplot(r, c)
            if panel is None:
                continue
            if panel.title:
                plotter.add_text(
                    panel.title, font_size=font_size, color=text_color
                )
            plotter.add_mesh(
                panel.mesh,
                color=color,
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=line_width,
                show_scalar_bar=False,
            )
            plotter.camera_position = camera_position
            plotter.reset_camera(bounds=row_bounds)
            if zoom != 1.0:
                plotter.camera.zoom(zoom)

    if link_rows and n_cols > 1:
        for r in range(n_rows):
            plotter.link_views(
                views=[r * n_cols + c for c in range(n_cols)]
            )

    if off_screen:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out))
        plotter.close()
    else:
        plotter.show()


def plot_surface_row(
    panels: Sequence[SurfacePanel | None], **kwargs
) -> None:
    """Render a single row of surfaces. Thin wrapper on plot_surface_grid."""
    plot_surface_grid([panels], **kwargs)
