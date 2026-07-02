"""Illustrate the rollout process by colouring each supervised ring.

The masked rollout fills the unknown region outward from the known set K, one
graph-distance ring at a time: at step t it supervises the frontier
F_t = {i : d_i = t}, where d_i is the topological (BFS) distance from K. This
script makes that visible by colouring every vertex by its ring index d:
  - d = 0  (K, the known input)      -> grey
  - d = 1, 2, ..., N  (rollout steps) -> distinct colours from an ordered map
So the colour sweep from the known region outward *is* the order in which the
rollout predicts the surface. N = max_i d_i is the number of rollout steps.

No trained checkpoint is needed — the rings are determined purely by the mask
and the mesh topology. Pick any surface; optionally force a mask regime.

Usage:
    python scripts/viz_rollout_rings.py --surface TestHorizon4
    python scripts/viz_rollout_rings.py --regime outward_free --top-down
    python scripts/viz_rollout_rings.py --split test_ood --index 0 --out figures/rollout_rings.png
    python scripts/viz_rollout_rings.py --list        # list surfaces in the split and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from matplotlib.colors import LinearSegmentedColormap

from horizons.data.loaders import load_split
from horizons.data.masking import MaskSampler, MaskSamplerConfig

K_COLOR = "#C7CCD3"           # grey = known (input, step 0)
# Ordered muted rainbow: ring d = 1..N sweeps outward. Softer than turbo,
# but more saturated / defined than full pastel.
PASTEL_RING_COLORS = [
    "#E8776A",  # muted coral
    "#EBA45C",  # muted orange
    "#E6C64F",  # muted gold
    "#8FCB7A",  # muted green
    "#5FBAB0",  # muted teal
    "#5E93D1",  # muted blue
    "#9B77C7",  # muted violet
    "#D177A9",  # muted pink
]
RING_CMAP = LinearSegmentedColormap.from_list("muted_rings", PASTEL_RING_COLORS)
REGIMES = ("half_plane", "outward_free", "outward_pinned")


def pv_mesh(xy: torch.Tensor, z: torch.Tensor, F: torch.Tensor) -> pv.PolyData:
    V = torch.column_stack([xy, z]).detach().cpu().numpy().astype(np.float64)
    Fn = F.cpu().numpy()
    faces = np.column_stack([np.full(Fn.shape[0], 3, np.int64), Fn]).ravel()
    return pv.PolyData(V, faces)


def sample_mask(surface, regime: str | None, seed: int):
    """Sample a mask (+ its ring distances d). If regime is given, force it."""
    cfg = MaskSamplerConfig(regime_weights={regime: 1.0}) if regime else MaskSamplerConfig()
    sampler = MaskSampler(cfg)
    return sampler.sample(surface, torch.Generator().manual_seed(seed))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--surface", default=None, help="surface_id to use")
    p.add_argument("--index", type=int, default=0, help="i-th surface if --surface unset")
    p.add_argument("--regime", choices=REGIMES, default=None,
                   help="force a mask regime (default: the trained mixture)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--split-file", default="data/splits/split_v1.json")
    p.add_argument("--out", default=None, help="save a PNG here (off-screen) instead of showing a window")
    p.add_argument("--top-down", action="store_true", help="render a straight-down (xy) view")
    p.add_argument("--show-edges", action="store_true", help="draw the triangle edges on the mesh")
    p.add_argument("--list", action="store_true", help="list surfaces in the split and exit")
    args = p.parse_args()

    surfaces = load_split(args.split, split_file=args.split_file)

    if args.list:
        for i, s in enumerate(surfaces):
            print(f"  [{i}] {s.surface_id:<28} V={s.V.shape[0]:,}")
        return

    if args.surface is not None:
        surface = next((s for s in surfaces if s.surface_id == args.surface), None)
        if surface is None:
            raise SystemExit(f"surface {args.surface!r} not in {args.split}; use --list")
    else:
        surface = surfaces[args.index]

    mask, d, regime = sample_mask(surface, args.regime, args.seed)
    N = int(d.max().item())

    # Centred metres for display (centre z on K, like the prediction views).
    Vc = surface.V.clone()
    Vc[:, :2] -= surface.V[:, :2].mean(0)
    Vc[:, 2] -= surface.V[mask, 2].mean()
    xy, z = Vc[:, :2], Vc[:, 2]

    ring_sizes = torch.bincount(d.clamp(min=0), minlength=N + 1).tolist()
    print(f"{surface.surface_id} | {regime} | N={N} steps | "
          f"|K|={int(mask.sum())}/{surface.V.shape[0]} | "
          f"ring sizes (d=0..N): {ring_sizes}")

    off = args.out is not None
    pl = pv.Plotter(window_size=(1000, 850), off_screen=off)
    pl.add_text(
        f"rollout rings — {regime}   (N={N} steps;  grey=K input,  colour=ring d)",
        font_size=11,
    )
    m = pv_mesh(xy, z, surface.F)
    m["ring"] = d.to(torch.float64).cpu().numpy()
    # Discrete band per ring: clim (0.5, N+0.5) with N colours centres each
    # integer d in its own band; d = 0 (K) falls below the range -> grey.
    pl.add_mesh(m, scalars="ring", cmap=RING_CMAP, clim=(0.5, N + 0.5),
                n_colors=max(N, 1), below_color=K_COLOR, show_edges=args.show_edges,
                scalar_bar_args={"title": "rollout step (ring d)", "fmt": "%.0f",
                                 "n_labels": min(N + 1, 8)})
    if args.top_down:
        pl.view_xy()

    if off:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(args.out)
        print(f"wrote {args.out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
