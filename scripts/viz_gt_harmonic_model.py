"""Ground truth vs harmonic infill vs trained model, one row per test surface.

For every surface of a test split the same mask is held out and three
triangulated surfaces are rendered side by side:

    ground truth | harmonic infill | model prediction

All three are drawn in a single flat colour with the triangle edges visible,
so the comparison is purely about geometry and mesh structure — the known and
predicted regions are not distinguished. The RMSE over the unknown region is
printed and shown in each panel's caption.

Everything but the split comes from the run's saved config (architecture,
init, approach, normalization, mask mixture), so the figure reflects how that
checkpoint was actually trained.

One PNG per split lands in outputs/evaluation/plots/ by default.

Usage:
    python scripts/viz_gt_harmonic_model.py outputs/tensorboard/run_XXXX
    python scripts/viz_gt_harmonic_model.py RUN --split test_ood --show
    python scripts/viz_gt_harmonic_model.py RUN --surfaces TestHorizon3 FUNDO_DO_MAR
    python scripts/viz_gt_harmonic_model.py RUN --regime half_plane
    python scripts/viz_gt_harmonic_model.py RUN --z-exaggeration 5 --max-vertices 200000
    python scripts/viz_gt_harmonic_model.py RUN --list
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from horizons.data.loaders import load_split
from horizons.eval.predict import RunSpec, load_split_cases, split_seed
from horizons.viz.mesh import (
    SurfacePanel,
    auto_z_exaggeration,
    plot_surface_grid,
)

SPLITS = ("test_id", "test_ood")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="training run directory")
    p.add_argument(
        "--split", default="both", choices=(*SPLITS, "both"),
        help="which test split to render (default: both, one figure each)",
    )
    p.add_argument(
        "--surfaces", nargs="+", default=None,
        help="restrict to these surface_ids",
    )
    p.add_argument(
        "--seed", type=int, default=1000,
        help="base mask seed; each surface uses seed + 100*index + mask-index, "
             "the same scheme as the evaluation driver, so the masks (and "
             "therefore the RMSEs) line up with the eval JSON",
    )
    p.add_argument(
        "--mask-index", type=int, default=0,
        help="which mask draw per surface (0-2 in the driver's eval)",
    )
    p.add_argument(
        "--regime", default=None,
        choices=("half_plane", "outward_free", "outward_pinned"),
        help="force one mask regime instead of drawing from the run's mixture",
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/evaluation/plots"),
        help="output directory, or a .png path when rendering one split",
    )
    p.add_argument(
        "--show", action="store_true",
        help="open an interactive window instead of writing a PNG",
    )
    p.add_argument("--no-edges", action="store_true", help="hide triangle edges")
    p.add_argument(
        "--z-exaggeration", default="auto",
        help="vertical exaggeration, or 'auto' (default) to stretch each "
             "surface until its relief is visible; 1 renders true scale, "
             "where these near-planar horizons look like flat sheets. The "
             "same factor is always used for all three panels of a surface",
    )
    p.add_argument(
        "--max-vertices", type=int, default=0,
        help="skip surfaces above this vertex count (0 = no limit); the "
             "largest horizons take minutes to infill and render",
    )
    p.add_argument("--panel-width", type=int, default=620)
    p.add_argument("--panel-height", type=int, default=520)
    p.add_argument("--zoom", type=float, default=1.3)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--list", action="store_true",
        help="list the surfaces of the selected split(s) and exit",
    )
    return p.parse_args()


def output_path(out: Path, split: str, n_splits: int) -> Path:
    """A .png path for `split`, honouring --out as either a file or a dir."""
    if out.suffix.lower() == ".png":
        if n_splits == 1:
            return out
        return out.with_name(f"{out.stem}_{split}{out.suffix}")
    return out / f"gt_harmonic_model_{split}.png"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    splits = list(SPLITS) if args.split == "both" else [args.split]

    spec = RunSpec.from_dir(args.run_dir)
    if args.regime is not None:
        spec.mask_config = replace(
            spec.mask_config, regime_weights={args.regime: 1.0}
        )

    if args.list:
        for split in splits:
            print(f"{split}:")
            for i, s in enumerate(load_split(split, split_file=spec.split_file)):
                print(f"  [{i}] {s.surface_id:<28} V={s.V.shape[0]:,}")
        return

    model = spec.load_model(device=args.device)
    print(
        f"run {args.run_dir.name}: approach={spec.approach} "
        f"init={spec.init_method} rollout={spec.rollout_method} "
        f"(x{spec.rollout_n_multiplier}) normalize={spec.normalize_per_surface}"
    )

    for split in splits:
        try:
            cases = load_split_cases(
                split, split_file=spec.split_file, surface_ids=args.surfaces
            )
        except KeyError as e:
            raise SystemExit(str(e)) from e

        print(
            f"\n=== {split} ({len(cases)} surfaces, base seed {args.seed}, "
            f"mask index {args.mask_index})"
        )
        rows: list[list[SurfacePanel]] = []
        for index, surface in cases:
            n_v = surface.V.shape[0]
            if args.max_vertices and n_v > args.max_vertices:
                print(f"  {surface.surface_id:<28} skipped (V={n_v:,})")
                continue

            seed = split_seed(args.seed, index, args.mask_index)
            t0 = time.perf_counter()
            pred = spec.predict(model, surface, seed, device=args.device)
            elapsed = time.perf_counter() - t0

            if args.z_exaggeration == "auto":
                z_exag = auto_z_exaggeration(surface.V, pred.z_true)
            else:
                z_exag = float(args.z_exaggeration)

            print(
                f"  {pred.surface_id:<28} {pred.regime:<15} N={pred.N:<4} "
                f"|K|={pred.n_K:,}/{n_v:,}  "
                f"harmonic={pred.rmse_harmonic:8.1f} m  "
                f"model={pred.rmse_model:8.1f} m  "
                f"z x{z_exag:.1f}  [{elapsed:.1f}s]"
            )

            panels = [
                (f"{pred.surface_id}  |  ground truth  |  {pred.regime}  |  "
                 f"K={pred.n_K:,}/{n_v:,}  |  N={pred.N}  |  z x{z_exag:.1f}",
                 pred.z_true),
                (f"harmonic infill  |  U-RMSE {pred.rmse_harmonic:.1f} m",
                 pred.z_harmonic),
                (f"model  |  U-RMSE {pred.rmse_model:.1f} m", pred.z_model),
            ]
            rows.append([
                SurfacePanel.from_mesh(
                    pred.xy, pred.faces, z,
                    title=title,
                    z_exaggeration=z_exag,
                )
                for title, z in panels
            ])

        if not rows:
            print("  nothing to render")
            continue

        out = None if args.show else output_path(args.out, split, len(splits))
        plot_surface_grid(
            rows,
            out=out,
            show_edges=not args.no_edges,
            panel_size=(args.panel_width, args.panel_height),
            zoom=args.zoom,
        )
        if out is not None:
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
