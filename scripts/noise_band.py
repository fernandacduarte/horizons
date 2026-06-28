"""Noise-band calibration: eval-mask variance for a single trained checkpoint.

Re-scores one checkpoint on the val split under several different mask-draw
seeds (base_seed). The spread of the resulting aggregate RMSE numbers is the
EVAL-MASK component of the noise floor: how much our headline metric moves
just because a different random set of masks was drawn, with the model held
fixed.

This is the cheap half of the noise band (no training). The other half is
training-seed variance, measured separately by retraining with different seeds.

Usage:
    python scripts/noise_band.py <run_dir> [--seeds 1000 2000 3000 4000 5000]
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import yaml

from horizons.eval.checkpoint import load_checkpoint
from horizons.eval.driver import evaluate_split, aggregate_by_regime, aggregate_overall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--n-masks", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1000, 2000, 3000, 4000, 5000])
    p.add_argument("--device", default="cpu",
                   help="device for the model rollout (cpu | cuda); the "
                        "mean-plane and harmonic baselines always run on CPU")
    args = p.parse_args()

    # Read arch / data settings from the run's saved config snapshot
    cfg = {}
    cfg_path = args.run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path))
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 64))
    n_layers = int(cfg.get("model", {}).get("n_layers", 2))
    normalize = bool(cfg.get("data", {}).get("normalize_per_surface", False))
    init_method = cfg.get("data", {}).get("init_method", "meanplane")
    conv_type = cfg.get("model", {}).get("type", "sage")
    aggr = cfg.get("model", {}).get("aggr", "mean")
    split_file = cfg.get("data", {}).get("split_file", "data/splits/split_v1.json")
    rollout_method = cfg.get("rollout", {}).get("method", "standard")
    approach = cfg.get("approach", "rollout")
    hybrid_n_passes = int(cfg.get("hybrid", {}).get("n_passes", 3))

    ckpt = load_checkpoint(
        args.run_dir / "best.pt",
        hidden_dim=hidden_dim, n_message_passing=n_layers,
        conv_type=conv_type, aggr=aggr,
        device=args.device,
    )
    print(f"checkpoint: {args.run_dir.name}  (hidden={hidden_dim}, layers={n_layers}, "
          f"conv={conv_type}/{aggr}, normalize={normalize}, init={init_method})")
    print(f"eval: {args.split} split, n_masks={args.n_masks}, "
          f"{len(args.seeds)} mask-draw seeds, device={args.device}\n")

    regimes = ["half_plane", "outward_free", "outward_pinned"]
    # Collect model overall + per-regime means, and the mean-plane and
    # harmonic baseline overalls, per seed
    rows: list[dict] = []
    all_records: list = []  # every (surface, mask) record across seeds, for per-surface stats
    for seed in args.seeds:
        result = evaluate_split(
            ckpt.model, args.split,
            n_masks_per_surface=args.n_masks, base_seed=seed,
            normalize_per_surface=normalize, init_method=init_method,
            split_file=split_file, device=args.device,
            rollout_method=rollout_method,
            approach=approach, hybrid_n_passes=hybrid_n_passes,
        )
        all_records.extend(result.records)
        overall = aggregate_overall(result)
        by_regime = aggregate_by_regime(result)
        row = {
            "seed": seed,
            "model_overall": overall["model"]["mean"],
            "meanplane_overall": overall["meanplane"]["mean"],
            "harmonic_overall": overall["harmonic"]["mean"],
        }
        for r in regimes:
            row[f"model_{r}"] = by_regime.get(r, {}).get("model", {}).get("mean", float("nan"))
        rows.append(row)
        print(f"  seed {seed}: model_overall={row['model_overall']:.2f}  "
              f"meanplane_overall={row['meanplane_overall']:.2f}  "
              f"harmonic_overall={row['harmonic_overall']:.2f}")

    def spread(key: str) -> tuple[float, float, float, float]:
        vals = [r[key] for r in rows]
        return min(vals), max(vals), statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print("\n=== eval-mask spread across seeds (min / max / mean / std) ===")
    for key in ["model_overall", "meanplane_overall", "harmonic_overall",
                "model_half_plane", "model_outward_free", "model_outward_pinned"]:
        lo, hi, mu, sd = spread(key)
        print(f"  {key:<24} {lo:7.2f} / {hi:7.2f} / {mu:7.2f}  (std {sd:.2f}, range {hi-lo:.2f})")

    # Per-surface breakdown: where does the model win/lose vs harmonic?
    by_surface: dict = {}
    for r in all_records:
        by_surface.setdefault(r.surface_id, []).append(r)
    print("\n=== per-surface (mean over masks x seeds), largest first ===")
    print(f"  {'surface':<26} {'V':>9} {'N':>5} {'model':>8} {'mplane':>8} {'harm':>8} {'d(m-h)':>8}")
    surf_rows = []
    for sid, recs in by_surface.items():
        V = recs[0].n_K + recs[0].n_U
        N = statistics.mean(r.N for r in recs)
        m = statistics.mean(r.rmse_model for r in recs)
        mp = statistics.mean(r.rmse_meanplane for r in recs)
        h = statistics.mean(r.rmse_harmonic for r in recs)
        surf_rows.append((V, N, sid, m, mp, h))
    for V, N, sid, m, mp, h in sorted(surf_rows, reverse=True):
        print(f"  {sid:<26} {V:>9,} {N:>5.0f} {m:>8.1f} {mp:>8.1f} {h:>8.1f} {m - h:>+8.1f}")


if __name__ == "__main__":
    main()
