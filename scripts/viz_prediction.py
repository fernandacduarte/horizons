"""Visualize a network's extrapolation on a real surface.

For a trained run + a surface + a mask, renders three linked 3-D views:
  1. input        — what the network sees: K at true z, U at the init z0.
  2. extrapolation — the network's prediction (K true, U filled by the model).
  3. ground truth  — the real surface (the "original mesh" to compare against).
Each is coloured by region: blue = K (the input/known part), orange = U (the
extrapolated/unknown part), so it is obvious which part was given and which was
predicted, and you can compare the orange region's shape across the three panels.

The run's config drives everything (architecture, init, approach/n_passes,
normalization, split), so it works for both the rollout and the hybrid runs.

Usage:
    python scripts/viz_prediction.py outputs/tensorboard/run_XXXX --surface TestHorizon4
    python scripts/viz_prediction.py outputs/tensorboard/run_XXXX --split test_ood --index 0
    python scripts/viz_prediction.py ... --out fig.png   # off-screen PNG instead of a window
    python scripts/viz_prediction.py ... --list          # list surfaces in the split and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import yaml

from horizons.data.loaders import load_split
from horizons.data.init import init_z_dispatch
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.eval.checkpoint import load_checkpoint
from horizons.training.rollout import rollout

K_COLOR, U_COLOR = "#377EB8", "#E8743B"  # blue = known (input), orange = extrapolated


def pv_mesh(xy: torch.Tensor, z: torch.Tensor, F: torch.Tensor) -> pv.PolyData:
    V = torch.column_stack([xy, z]).detach().cpu().numpy().astype(np.float64)
    Fn = F.cpu().numpy()
    faces = np.column_stack([np.full(Fn.shape[0], 3, np.int64), Fn]).ravel()
    return pv.PolyData(V, faces)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--surface", default=None, help="surface_id to visualize")
    p.add_argument("--index", type=int, default=0, help="i-th surface if --surface unset")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None, help="save a PNG here (off-screen) instead of showing a window")
    p.add_argument("--list", action="store_true", help="list surfaces in the split and exit")
    p.add_argument("--show-edges", action="store_true", help="draw the triangle edges on the mesh")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.run_dir / "config.yaml"))
    split_file = cfg.get("data", {}).get("split_file", "data/splits/split_v1.json")
    surfaces = load_split(args.split, split_file=split_file)

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

    model = load_checkpoint(
        args.run_dir / "best.pt",
        hidden_dim=int(cfg.get("model", {}).get("hidden_dim", 64)),
        n_message_passing=int(cfg.get("model", {}).get("n_layers", 2)),
        conv_type=cfg.get("model", {}).get("type", "sage"),
        aggr=cfg.get("model", {}).get("aggr", "mean"),
    ).model
    init_method = cfg.get("data", {}).get("init_method", "meanplane")
    normalize = bool(cfg.get("data", {}).get("normalize_per_surface", False))
    approach = cfg.get("approach", "rollout")
    n_passes = int(cfg.get("hybrid", {}).get("n_passes", 3))
    rollout_method = cfg.get("rollout", {}).get("method", "standard")

    # mask + per-surface centering (centred metres for the display geometry)
    mask, d, regime = MaskSampler(MaskSamplerConfig()).sample(
        surface, torch.Generator().manual_seed(args.seed))
    Vc = surface.V.clone()
    Vc[:, :2] -= surface.V[:, :2].mean(0)
    Vc[:, 2] -= surface.V[mask, 2].mean()

    # normalized copy for the model
    if normalize:
        xy_scale = max(Vc[:, :2].to(torch.float64).abs().max().item(), 1.0)
        z_scale = max(Vc[mask, 2].to(torch.float64).abs().max().item(), 1.0)
        Vn = Vc.clone(); Vn[:, :2] /= xy_scale; Vn[:, 2] /= z_scale
    else:
        Vn, z_scale = Vc, 1.0

    z0 = init_z_dispatch(Vn, mask, surface.edge_index, method=init_method)
    N = n_passes if approach == "hybrid" else int(d.max().item())
    res = rollout(model, z0=z0, z_true=Vn[:, 2], V_xy=Vn[:, :2], F=surface.F,
                  edge_index=surface.edge_index, mask=mask, d=d, N=N,
                  rollout_method=rollout_method)

    # back to centred metres for display
    xy = Vc[:, :2]
    z_input = z0 * z_scale
    z_nn = res.z_trajectory[-1] * z_scale
    z_true = Vc[:, 2]
    region = mask.to(torch.int32).cpu().numpy()  # 1 = K, 0 = U

    rmse = ((z_nn[~mask] - z_true[~mask]) ** 2).mean().sqrt().item()
    print(f"{surface.surface_id} | {regime} | N={int(d.max())} | "
          f"|K|={int(mask.sum())}/{surface.V.shape[0]} | U-RMSE={rmse:.1f} m "
          f"(centred units; approach={approach})")

    off = args.out is not None
    pl = pv.Plotter(shape=(1, 3), window_size=(1800, 650), off_screen=off)
    panels = [("input  (blue=K, orange=U)", z_input),
              (f"network extrapolation  (U-RMSE {rmse:.0f} m)", z_nn),
              ("ground truth", z_true)]
    for col, (title, z) in enumerate(panels):
        pl.subplot(0, col)
        pl.add_text(title, font_size=10)
        m = pv_mesh(xy, z, surface.F)
        m["region"] = region
        pl.add_mesh(m, scalars="region", cmap=[U_COLOR, K_COLOR], clim=(0, 1),
                    show_scalar_bar=False, show_edges=args.show_edges)
    pl.link_views()
    if off:
        pl.screenshot(args.out)
        print(f"wrote {args.out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
