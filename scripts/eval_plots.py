"""Generate evaluation plots from a saved EvalResult JSON.

Produces four figures:
1. Per-regime RMSE bar chart (mean RMSE per method, per regime).
2. Per-ring RMSE curves (model, broken down by regime).
3. Per-surface RMSE distribution (strip plot showing variance within regime).
4. Model vs harmonic scatter (one point per (surface, mask), see where each wins).

Usage:
    python scripts/eval_plots.py outputs/evaluation/val_b4.json
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from horizons.eval.driver import load_result, aggregate_by_regime, EvalResult


# Color scheme — consistent across all plots
COLOR_MEANPLANE = "#7f7f7f"  # gray
COLOR_HARMONIC = "#ff7f0e"   # orange
COLOR_MODEL = "#1f77b4"      # blue

METHOD_LABELS = {
    "meanplane": "Mean-plane init",
    "harmonic": "Harmonic infill",
    "model": "GNN model",
}
METHOD_COLORS = {
    "meanplane": COLOR_MEANPLANE,
    "harmonic": COLOR_HARMONIC,
    "model": COLOR_MODEL,
}


def plot_regime_bars(result: EvalResult, out_path: Path) -> None:
    """Bar chart: mean RMSE per (regime, method).

    This is the headline figure. Groups bars by regime; within each
    group, three bars (one per method). Annotates each bar with its
    value in meters.
    """
    by_regime = aggregate_by_regime(result)
    regimes = sorted(by_regime.keys())
    methods = ["meanplane", "harmonic", "model"]

    n_regimes = len(regimes)
    bar_width = 0.25
    x_positions = np.arange(n_regimes)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        means = [by_regime[regime][method]["mean"] for regime in regimes]
        x_offset = (i - 1) * bar_width
        bars = ax.bar(
            x_positions + x_offset, means, bar_width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        # Annotate each bar with its value
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{value:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    # Sample sizes per regime, shown below the regime labels
    n_per_regime = [by_regime[r]["meanplane"]["n"] for r in regimes]
    regime_labels = [
        f"{r}\n(n={n})" for r, n in zip(regimes, n_per_regime)
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Mean RMSE on U (m)")
    ax.set_title(
        f"Mean RMSE by regime and method\n"
        f"({result.split_name} split, {result.n_surfaces} surfaces × "
        f"{result.n_masks_per_surface} masks)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_per_ring(result: EvalResult, out_path: Path) -> None:
    """Per-ring RMSE curves, one subplot per regime.

    For each regime, plot all per-surface curves of model RMSE vs ring
    index. Show median and per-quartile envelope.
    """
    by_regime: dict[str, list] = {}
    for r in result.records:
        by_regime.setdefault(r.regime, []).append(r)

    regimes = sorted(by_regime.keys())
    n_regimes = len(regimes)
    fig, axes = plt.subplots(
        1, n_regimes, figsize=(5 * n_regimes, 4), sharey=True,
    )
    if n_regimes == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        records = by_regime[regime]

        # Plot each record's per-ring curve as a thin transparent line
        max_N = max(r.N for r in records)
        for rec in records:
            x = np.arange(1, rec.N + 1)
            y = np.array(rec.rmse_per_ring_model)
            ax.plot(x, y, color=COLOR_MODEL, alpha=0.25, linewidth=0.8)

        # Overlay median curve, computed up to max_N where data exists
        median_y = []
        x_axis = list(range(1, max_N + 1))
        for k in x_axis:
            ring_rmses = [
                rec.rmse_per_ring_model[k - 1]
                for rec in records
                if rec.N >= k and rec.ring_sizes[k - 1] >= 10
            ]
            if ring_rmses:
                median_y.append(statistics.median(ring_rmses))
            else:
                median_y.append(np.nan)
        ax.plot(
            x_axis, median_y,
            color=COLOR_MODEL, linewidth=2.5,
            label=f"median (rings with ≥10 verts)",
        )

        ax.set_xlabel("ring index d_i")
        ax.set_ylabel("RMSE on ring (m)")
        ax.set_title(f"{regime}\n(n={len(records)} records)")
        ax.grid(True, linewidth=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Per-ring RMSE for the GNN model, by regime "
        f"({result.split_name} split)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_per_surface_distribution(result: EvalResult, out_path: Path) -> None:
    """Strip plot: per-surface RMSE for each method, grouped by regime."""
    by_regime: dict[str, list] = {}
    for r in result.records:
        by_regime.setdefault(r.regime, []).append(r)

    regimes = sorted(by_regime.keys())
    n_regimes = len(regimes)
    methods = ["meanplane", "harmonic", "model"]

    fig, axes = plt.subplots(
        1, n_regimes, figsize=(5 * n_regimes, 4), sharey=True,
    )
    if n_regimes == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        records = by_regime[regime]
        for i, method in enumerate(methods):
            values = [getattr(r, f"rmse_{method}") for r in records]
            # Stripplot: x-position is the method index, with small jitter
            x = np.full(len(values), i) + np.random.uniform(
                -0.1, 0.1, size=len(values),
            )
            ax.scatter(
                x, values,
                color=METHOD_COLORS[method],
                alpha=0.7, s=40, edgecolors="none",
            )
            # Median line for this method
            median = statistics.median(values)
            ax.plot(
                [i - 0.3, i + 0.3], [median, median],
                color=METHOD_COLORS[method], linewidth=2.5,
            )

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15)
        ax.set_ylabel("RMSE on U (m)")
        ax.set_title(f"{regime}\n(n={len(records)} records)")
        ax.grid(True, axis="y", linewidth=0.3)

    fig.suptitle(
        f"Per-surface RMSE distribution by regime and method "
        f"({result.split_name} split)\n"
        f"Each dot is one (surface, mask). Horizontal bar = median.",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_model_vs_harmonic_scatter(
    result: EvalResult, out_path: Path,
) -> None:
    """Scatter: each (surface, mask) point with model RMSE on y, harmonic
    on x. Points below the diagonal mean the model wins; above means
    harmonic wins."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Color points by regime
    regime_colors = {
        "half_plane": "#9467bd",      # purple
        "outward_free": "#2ca02c",    # green
        "outward_pinned": "#d62728",  # red
    }

    for regime in sorted(regime_colors.keys()):
        records = [r for r in result.records if r.regime == regime]
        if not records:
            continue
        x = [r.rmse_harmonic for r in records]
        y = [r.rmse_model for r in records]
        ax.scatter(
            x, y,
            color=regime_colors[regime],
            label=f"{regime} (n={len(records)})",
            alpha=0.7, s=50, edgecolors="black", linewidths=0.5,
        )

    # Diagonal line: y = x
    max_val = max(
        max(r.rmse_model for r in result.records),
        max(r.rmse_harmonic for r in result.records),
    )
    ax.plot(
        [0, max_val], [0, max_val],
        color="gray", linestyle="--", linewidth=1,
        label="y = x (parity)",
    )

    ax.set_xlabel("Harmonic infill RMSE (m)")
    ax.set_ylabel("GNN model RMSE (m)")
    ax.set_title(
        f"Per-(surface, mask) comparison: GNN model vs harmonic infill\n"
        f"({result.split_name} split)\n"
        f"Points BELOW the line: model wins. ABOVE: harmonic wins."
    )
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    result = load_result(args.json_path)

    if args.out_dir is None:
        out_dir = args.json_path.parent / "plots"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a deterministic numpy seed so strip-plot jitter is reproducible
    np.random.seed(0)

    base = args.json_path.stem  # e.g., "val_b4"
    print(f"Generating plots from {args.json_path}")
    print(f"Output directory: {out_dir}")
    plot_regime_bars(result, out_dir / f"{base}_regime_bars.png")
    plot_per_ring(result, out_dir / f"{base}_per_ring.png")
    plot_per_surface_distribution(result, out_dir / f"{base}_distribution.png")
    plot_model_vs_harmonic_scatter(
        result, out_dir / f"{base}_model_vs_harmonic.png",
    )


if __name__ == "__main__":
    main()
