"""Export test-set predictions as .vtp files for ParaView.

One file per surface, holding the triangulation plus every depth field
(z_true, z_init, z_harmonic, z_model), the signed and absolute error of each
method, the K/U mask and the topological distance d. Depths are the models'
own values in the survey's coordinates — nothing is rescaled or exaggerated.

Once open in ParaView:
  - Colour by err_model / abs_err_model to see where the prediction drifts.
  - Warp By Scalar on err_model, direction (0, 0, 1), scale 1, turns the
    geometry into the predicted surface; err_harmonic gives the baseline.
  - Transform with a Z scale is how you exaggerate the relief, leaving the
    stored depths untouched.
  - Threshold on `unknown` isolates the extrapolated region.

Usage:
    python scripts/export_predictions.py outputs/tensorboard/run_XXXX
    python scripts/export_predictions.py RUN --split test_ood
    python scripts/export_predictions.py RUN --surfaces TestHorizon3
    python scripts/export_predictions.py RUN --regime half_plane --mask-index 1
    python scripts/export_predictions.py RUN --centered   # if UTM zoom jitters
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from horizons.eval.predict import RunSpec, load_split_cases, split_seed
from horizons.viz.export import save_prediction

SPLITS = ("test_id", "test_ood")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="training run directory")
    p.add_argument(
        "--split", default="both", choices=(*SPLITS, "both"),
        help="which test split to export (default: both)",
    )
    p.add_argument(
        "--surfaces", nargs="+", default=None,
        help="restrict to these surface_ids",
    )
    p.add_argument(
        "--seed", type=int, default=1000,
        help="base mask seed; each surface uses seed + 100*index + mask-index, "
             "matching the evaluation driver",
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
        "--out", type=Path, default=Path("outputs/paraview"),
        help="output directory; files land in <out>/<run>/<split>/",
    )
    p.add_argument(
        "--centered", action="store_true",
        help="keep the per-surface centered frame instead of restoring the "
             "survey coordinates; useful when large UTM values make the view "
             "jitter as you zoom in",
    )
    p.add_argument(
        "--max-vertices", type=int, default=0,
        help="skip surfaces above this vertex count (0 = no limit)",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    splits = list(SPLITS) if args.split == "both" else [args.split]

    spec = RunSpec.from_dir(args.run_dir)
    if args.regime is not None:
        spec.mask_config = replace(
            spec.mask_config, regime_weights={args.regime: 1.0}
        )

    model = spec.load_model(device=args.device)
    frame = "centered" if args.centered else "survey coordinates"
    print(
        f"run {args.run_dir.name}: approach={spec.approach} "
        f"init={spec.init_method} | {frame}"
    )

    written: list[Path] = []
    for split in splits:
        try:
            cases = load_split_cases(
                split, split_file=spec.split_file, surface_ids=args.surfaces
            )
        except KeyError as e:
            raise SystemExit(str(e)) from e

        out_dir = args.out / args.run_dir.name / split
        print(f"\n=== {split} -> {out_dir}")
        for index, surface in cases:
            n_v = surface.V.shape[0]
            if args.max_vertices and n_v > args.max_vertices:
                print(f"  {surface.surface_id:<28} skipped (V={n_v:,})")
                continue

            t0 = time.perf_counter()
            pred = spec.predict(
                model,
                surface,
                split_seed(args.seed, index, args.mask_index),
                device=args.device,
            )
            path = save_prediction(
                pred,
                out_dir / f"{pred.surface_id}.vtp",
                world_coordinates=not args.centered,
            )
            written.append(path)
            print(
                f"  {pred.surface_id:<28} {pred.regime:<15} "
                f"V={n_v:,}  harmonic={pred.rmse_harmonic:8.1f} m  "
                f"model={pred.rmse_model:8.1f} m  "
                f"{path.stat().st_size / 1e6:6.1f} MB  "
                f"[{time.perf_counter() - t0:.1f}s]"
            )

    if not written:
        print("\nnothing exported")
        return

    print(f"\n{len(written)} file(s) written. In ParaView:")
    print("  colour by err_model or abs_err_model")
    print("  Warp By Scalar (err_model, direction 0 0 1, scale 1) shows the "
          "predicted surface")
    print("  Transform with a Z scale exaggerates the relief")
    print("  Threshold on 'unknown' isolates the extrapolated region")


if __name__ == "__main__":
    main()
