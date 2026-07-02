"""Render one mask-sample image per masking regime.

Generates a separate PNG for each of the three regimes we use
  1. half_plane      — cut along a random line
  2. outward_free    — unknown frame around a central known rectangle
  3. outward_pinned  — same, plus a pinned (known) outer boundary ring
by forcing the MaskSampler onto a single regime at a time (via regime_weights)
and colouring the surface by region: blue = K (known), orange = U (unknown).

Any surface works — pick one with --surface / --index; the same surface and
camera are reused for all three so the masks are directly comparable.

Usage:
    python scripts/viz_mask_regimes.py                       # val split, first surface
    python scripts/viz_mask_regimes.py --surface TestHorizon4
    python scripts/viz_mask_regimes.py --split test_ood --index 2 --out-dir figures/masks
    python scripts/viz_mask_regimes.py --top-down            # look straight down (clearest mask shape)
    python scripts/viz_mask_regimes.py --list                # list surfaces in the split and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.loaders import load_split
from horizons.data.masking import MaskSampler, MaskSamplerConfig

K_COLOR, U_COLOR = "#377EB8", "#e8553b"  # blue = known (input), orange = unknown
REGIMES = ("half_plane", "outward_free", "outward_pinned")


def pv_mesh(xy: torch.Tensor, z: torch.Tensor, F: torch.Tensor) -> pv.PolyData:
    V = torch.column_stack([xy, z]).detach().cpu().numpy().astype(np.float64)
    Fn = F.cpu().numpy()
    faces = np.column_stack([np.full(Fn.shape[0], 3, np.int64), Fn]).ravel()
    return pv.PolyData(V, faces)


def sample_regime_mask(surface, regime: str, seed: int) -> torch.Tensor:
    """Force the MaskSampler onto a single regime and draw one mask."""
    cfg = MaskSamplerConfig(regime_weights={regime: 1.0})
    sampler = MaskSampler(cfg)
    mask, _d, drawn = sampler.sample(surface, torch.Generator().manual_seed(seed))
    assert drawn == regime, f"expected {regime}, drew {drawn}"
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--surface", default=None, help="surface_id to use")
    p.add_argument("--index", type=int, default=0, help="i-th surface if --surface unset")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--split-file", default="data/splits/split_v1.json")
    p.add_argument("--out-dir", type=Path, default=Path("figures/masks"))
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

    # Centred metres for display (mean-centre so the default camera frames it well).
    Vc = surface.V.clone()
    Vc[:, :2] -= surface.V[:, :2].mean(0)
    Vc[:, 2] -= surface.V[:, 2].mean()
    xy, z = Vc[:, :2], Vc[:, 2]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"surface: {surface.surface_id}  (V={surface.V.shape[0]:,})  split={args.split}")

    for regime in REGIMES:
        mask = sample_regime_mask(surface, regime, args.seed)
        region = mask.to(torch.int32).cpu().numpy()  # 1 = K, 0 = U
        frac_u = float((~mask).float().mean())

        pl = pv.Plotter(window_size=(900, 800), off_screen=True)
        pl.add_text(f"{regime}   (|U| = {frac_u:.0%},  blue=K  orange=U)", font_size=11)
        m = pv_mesh(xy, z, surface.F)
        m["region"] = region
        pl.add_mesh(m, scalars="region", cmap=[U_COLOR, K_COLOR], clim=(0, 1),
                    show_scalar_bar=False, show_edges=args.show_edges)
        if args.top_down:
            pl.view_xy()

        out = args.out_dir / f"mask_{regime}.png"
        pl.screenshot(str(out))
        pl.close()
        print(f"  {regime:<15} |U|={frac_u:.0%}  ->  {out}")


if __name__ == "__main__":
    main()
