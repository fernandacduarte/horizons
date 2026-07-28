"""Noise-band comparison across multiple trained runs.

Multi-run companion to noise_band.py: re-scores each checkpoint on the same
split under the same set of mask-draw seeds, then prints a side-by-side table
of overall / per-regime RMSE as mean +/- std over seeds — the eval-mask noise
band around every number. The mean-plane and harmonic baselines (computed for
free by the eval driver) are appended as reference rows.

Because every run is scored under identical seeds, the per-seed mask draws are
identical across runs (as long as their data settings match), so differences
between rows are model differences, not mask luck. A config summary highlights
which hyperparameters actually differ between the runs.

Usage:
    python scripts/noise_band_many.py <run_dir> <run_dir> [...] \
        [--seeds 1000 2000 3000 4000 5000] [--n-masks 10] [--split val]
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import yaml

from horizons.eval.checkpoint import load_checkpoint
from horizons.eval.driver import (
    evaluate_split,
    aggregate_by_regime,
    aggregate_overall,
)

REGIMES = ["half_plane", "outward_free", "outward_pinned"]

# Config entries shown in the summary; (label, section, key, default)
CONFIG_FIELDS = [
    ("conv", "model", "type", "sage"),
    ("aggr", "model", "aggr", "mean"),
    ("hidden", "model", "hidden_dim", 64),
    ("layers", "model", "n_layers", 2),
    ("mask_mode", "model", "mask_mode", "binary"),
    ("use_mask_feat", "model", "use_mask_feature", True),
    ("approach", None, "approach", "rollout"),
    ("rollout", "rollout", "method", "standard"),
    ("kN", "rollout", "n_multiplier", 1),
    ("init", "data", "init_method", "meanplane"),
    ("normalize", "data", "normalize_per_surface", False),
    ("split_file", "data", "split_file", "data/splits/split_v1.json"),
]


def read_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(open(cfg_path))
    return {}


def config_value(cfg: dict, section: str | None, key: str, default):
    node = cfg if section is None else cfg.get(section, {}) or {}
    return node.get(key, default)


def mean_std(values: list[float]) -> tuple[float, float]:
    mu = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mu, sd


def fmt_cell(values: list[float]) -> str:
    if not values:
        return f"{'—':>16}"
    mu, sd = mean_std(values)
    return f"{mu:9.2f} ±{sd:5.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", type=Path, nargs="+",
                   help="run_... directories, each containing best.pt and "
                        "config.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--n-masks", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1000, 2000, 3000, 4000, 5000])
    p.add_argument("--device", default="cpu",
                   help="device for the model rollout (cpu | cuda); the "
                        "mean-plane and harmonic baselines always run on CPU")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        if not (run_dir / "best.pt").exists():
            raise SystemExit(f"no best.pt in {run_dir}")

    # ------------------------------------------------------------------
    # Config summary: one line per run, then which fields differ
    # ------------------------------------------------------------------
    configs = {run_dir: read_config(run_dir) for run_dir in args.run_dirs}
    print("=== runs ===")
    for run_dir, cfg in configs.items():
        settings = ", ".join(
            f"{label}={config_value(cfg, section, key, default)}"
            for label, section, key, default in CONFIG_FIELDS[:8]
        )
        print(f"  {run_dir.name}: {settings}")
    differing = [
        label
        for label, section, key, default in CONFIG_FIELDS
        if len({str(config_value(cfg, section, key, default))
                for cfg in configs.values()}) > 1
    ]
    print(f"  differing settings: {', '.join(differing) if differing else 'none'}")

    # Baselines depend only on the data settings; warn if those differ,
    # because then the baseline rows are not a shared reference.
    data_differ = [f for f in differing if f in ("init", "normalize", "split_file")]
    if data_differ:
        print(f"  WARNING: data settings differ across runs ({', '.join(data_differ)}) — "
              f"baseline rows use the FIRST run's settings and are not "
              f"directly comparable to every run")

    print(f"\neval: {args.split} split, n_masks={args.n_masks}, "
          f"{len(args.seeds)} mask-draw seeds, device={args.device}\n")

    # ------------------------------------------------------------------
    # Evaluate every run under every seed
    # ------------------------------------------------------------------
    # per run_dir: {"overall": [per-seed], regime: [per-seed], ...}
    model_stats: dict[Path, dict[str, list[float]]] = {}
    # baselines from the first run's evaluations (identical across runs
    # when data settings match, since seeds and masks are shared)
    baseline_stats: dict[str, dict[str, list[float]]] = {
        "meanplane": {k: [] for k in ["overall", *REGIMES]},
        "harmonic": {k: [] for k in ["overall", *REGIMES]},
    }

    for i, run_dir in enumerate(args.run_dirs):
        cfg = configs[run_dir]
        ckpt = load_checkpoint(
            run_dir / "best.pt",
            hidden_dim=int(config_value(cfg, "model", "hidden_dim", 64)),
            n_message_passing=int(config_value(cfg, "model", "n_layers", 2)),
            conv_type=config_value(cfg, "model", "type", "sage"),
            aggr=config_value(cfg, "model", "aggr", "mean"),
            mask_mode=config_value(cfg, "model", "mask_mode", "binary"),
            use_mask_feature=bool(
                config_value(cfg, "model", "use_mask_feature", True)),
            device=args.device,
        )
        stats: dict[str, list[float]] = {k: [] for k in ["overall", *REGIMES]}
        print(f"[{i + 1}/{len(args.run_dirs)}] {run_dir.name}")
        for seed in args.seeds:
            result = evaluate_split(
                ckpt.model, args.split,
                n_masks_per_surface=args.n_masks, base_seed=seed,
                normalize_per_surface=bool(
                    config_value(cfg, "data", "normalize_per_surface", False)),
                init_method=config_value(cfg, "data", "init_method", "meanplane"),
                split_file=config_value(
                    cfg, "data", "split_file", "data/splits/split_v1.json"),
                device=args.device,
                rollout_method=config_value(cfg, "rollout", "method", "standard"),
                approach=config_value(cfg, None, "approach", "rollout"),
                hybrid_n_passes=int(config_value(cfg, "hybrid", "n_passes", 3)),
                rollout_n_multiplier=float(
                    config_value(cfg, "rollout", "n_multiplier", 1)),
            )
            overall = aggregate_overall(result)
            by_regime = aggregate_by_regime(result)
            stats["overall"].append(overall["model"]["mean"])
            for r in REGIMES:
                stats[r].append(
                    by_regime.get(r, {}).get("model", {}).get("mean", float("nan")))
            if i == 0:
                for method in ("meanplane", "harmonic"):
                    baseline_stats[method]["overall"].append(
                        overall[method]["mean"])
                    for r in REGIMES:
                        baseline_stats[method][r].append(
                            by_regime.get(r, {}).get(method, {})
                            .get("mean", float("nan")))
            print(f"    seed {seed}: model_overall={stats['overall'][-1]:.2f}")
        model_stats[run_dir] = stats

    # ------------------------------------------------------------------
    # Comparison table: mean +/- std over seeds
    # ------------------------------------------------------------------
    name_w = max(len(run_dir.name) for run_dir in args.run_dirs)
    name_w = max(name_w, len("baseline: meanplane"))
    header = (f"{'':<{name_w}}  {'overall':>16} "
              + " ".join(f"{r:>16}" for r in REGIMES))
    print(f"\n=== comparison (RMSE, mean ± std over {len(args.seeds)} seeds) ===")
    print(header)
    ranked = sorted(model_stats.items(),
                    key=lambda kv: statistics.mean(kv[1]["overall"]))
    for run_dir, stats in ranked:
        cells = " ".join(fmt_cell(stats[k]) for k in ["overall", *REGIMES])
        print(f"{run_dir.name:<{name_w}}  {cells}")
    for method in ("meanplane", "harmonic"):
        cells = " ".join(
            fmt_cell(baseline_stats[method][k]) for k in ["overall", *REGIMES])
        print(f"{'baseline: ' + method:<{name_w}}  {cells}")

    # Pairwise verdict vs the eval-mask noise: is best vs each other run
    # separated by more than the combined spread?
    if len(ranked) > 1:
        print("\n=== best run vs the rest (overall) ===")
        best_dir, best = ranked[0]
        best_mu, best_sd = mean_std(best["overall"])
        for run_dir, stats in ranked[1:]:
            mu, sd = mean_std(stats["overall"])
            gap = mu - best_mu
            noise = best_sd + sd
            verdict = "clear" if gap > noise else "WITHIN NOISE BAND"
            print(f"  {best_dir.name} vs {run_dir.name}: "
                  f"gap {gap:+.2f} vs combined std {noise:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
