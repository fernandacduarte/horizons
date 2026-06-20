"""Run the full evaluation suite on a trained checkpoint.

Convenience script that:
1. Loads a checkpoint.
2. Runs evaluate_split() on val.
3. Saves the JSON.
4. Generates the four plots.
5. Prints aggregate summary.

Usage:
    python scripts/eval_run.py <run_dir>
    e.g., python scripts/eval_run.py outputs/tensorboard/run_20260609_092252

The output goes to outputs/evaluation/<run_basename>.json plus
outputs/evaluation/plots/<run_basename>_*.png.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from horizons.eval.checkpoint import load_checkpoint
from horizons.eval.driver import (
    evaluate_split, aggregate_by_regime, aggregate_overall, save_result,
)


def read_run_config(run_dir: Path) -> dict:
    """Read the config.yaml snapshot stored alongside the checkpoint."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--n-masks", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=1000)
    args = p.parse_args()

    ckpt_path = args.run_dir / "best.pt"
    print(f"Loading checkpoint from {ckpt_path}")

    # Read the run's config.yaml to detect architecture and data settings
    # that were used at training time. This lets eval_run handle runs with
    # non-default hidden_dim, n_layers, normalization, init_method, etc.
    cfg = read_run_config(args.run_dir)
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 64))
    n_layers = int(cfg.get("model", {}).get("n_layers", 2))
    normalize_per_surface = (
        cfg.get("data", {}).get("normalize_per_surface", False)
    )
    init_method = cfg.get("data", {}).get("init_method", "meanplane")
    split_file = cfg.get("data", {}).get("split_file", "data/splits/split_v1.json")
    conv_type = cfg.get("model", {}).get("type", "sage")
    aggr = cfg.get("model", {}).get("aggr", "mean")

    if hidden_dim != 64:
        print(f"  detected hidden_dim={hidden_dim} from config.yaml")
    if n_layers != 2:
        print(f"  detected n_layers={n_layers} from config.yaml")

    ckpt = load_checkpoint(
        ckpt_path,
        hidden_dim=hidden_dim,
        n_message_passing=n_layers,
        conv_type=conv_type,
        aggr=aggr,
    )
    print(f"  best_val_loss: {ckpt.best_val_loss:.2f} (epoch {ckpt.epoch})")
    print()

    # Run evaluation
    print(f"Evaluating on {args.split} split "
          f"({args.n_masks} masks per surface)...")
    if normalize_per_surface:
        print("  detected normalization=True from config.yaml")
    if init_method != "meanplane":
        print(f"  detected init_method={init_method!r} from config.yaml")
    if conv_type != "sage":
        print(f"  detected conv_type={conv_type!r} (aggr={aggr}) from config.yaml")
    print()


    result = evaluate_split(
        ckpt.model, args.split,
        n_masks_per_surface=args.n_masks,
        base_seed=args.base_seed,
        normalize_per_surface=normalize_per_surface,
        init_method=init_method,
        split_file=split_file,
    )

    # Save JSON
    eval_name = args.run_dir.name  # e.g., "run_20260609_092252"
    json_path = Path("outputs/evaluation") / f"{eval_name}_{args.split}.json"
    save_result(result, json_path)
    print(f"  saved: {json_path}")
    print()

    # Print summary
    print("=== Overall ===")
    overall = aggregate_overall(result)
    print(f"{'method':<12} {'mean':>10} {'median':>10} {'max':>10} {'n':>5}")
    for method, stats in overall.items():
        print(f"{method:<12} {stats['mean']:>10.2f} "
              f"{stats['median']:>10.2f} {stats['max']:>10.2f} "
              f"{stats['n']:>5}")
    print()

    print("=== By regime ===")
    by_regime = aggregate_by_regime(result)
    for regime in sorted(by_regime.keys()):
        n = by_regime[regime]["meanplane"]["n"]
        print(f"\n{regime} (n={n}):")
        print(f"  {'method':<12} {'mean':>10} {'median':>10} {'max':>10}")
        for method in ["meanplane", "harmonic", "model"]:
            stats = by_regime[regime][method]
            print(f"  {method:<12} {stats['mean']:>10.2f} "
                  f"{stats['median']:>10.2f} {stats['max']:>10.2f}")
    print()

    # Generate plots
    print("Generating plots...")
    subprocess.run(
        ["python", "scripts/eval_plots.py", str(json_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
