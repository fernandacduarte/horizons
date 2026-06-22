"""Plot the GNN-vs-harmonic RMSE deficit against rollout depth N — the Phase-2
crossover (O19) and the two failed interventions (O20 capacity, O21 harmonic-init).

Deficit = model RMSE − harmonic RMSE (negative = GNN wins). Data is the
per-surface output of `scripts/noise_band.py ... --device cuda` on the split_v2
val (3 seeds, n_masks=10). Update DATA if the runs change.

    python scripts/plot_crossover.py
writes:
    outputs/evaluation/plots/phase2_crossover.png          (O19, baseline)
    outputs/evaluation/plots/phase2_crossover_compare.png  (O20/O21 overlay)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# surface, N, deficit:  baseline (h=64) | O20 (h=128) | O21 (harmonic-init)
DATA = [
    ("TestHorizon4", 11, -26.6, -22.8, -7.9),
    ("TestHorizon7", 11, -21.5, -18.7, -10.4),
    ("09_Horizonte8", 12, 12.9, 11.2, -2.8),
    ("Horizonte5", 19, -2.0, -1.9, -0.2),
    ("horizonte7", 22, -2.0, -1.9, -0.2),
    ("10_BaseModelo", 51, 0.0, 0.0, 0.1),
    ("05_TopoCretaceo", 52, 88.5, 84.5, 73.7),
    ("04BaseOligoMioceno", 69, 41.2, 43.6, 15.8),
    ("02TopoMioceno", 132, 21.4, 55.0, 116.0),
]

WIN, LOSE = "#1D9E75", "#D85A30"
OUT = Path("outputs/evaluation/plots")


def baseline_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.5", ls="--", lw=1, zorder=1)
    for name, N, d, _o20, _o21 in DATA:
        win = d < 0
        ax.scatter(N, d, s=80, marker="^" if win else "o",
                   color=WIN if win else LOSE, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(name, (N, d), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color="0.35")
    ax.set_xlabel("rollout depth  N  (max ring distance)")
    ax.set_ylabel("RMSE deficit:  model − harmonic  (m)")
    ax.set_title("GNN beats harmonic at shallow rollout depth, loses at deep")
    ax.text(0.015, 0.05, "below 0 = GNN wins", transform=ax.transAxes, color=WIN, fontsize=9)
    ax.text(0.015, 0.92, "above 0 = harmonic wins", transform=ax.transAxes, color=LOSE, fontsize=9)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_crossover.png", dpi=150)
    print(f"wrote {OUT / 'phase2_crossover.png'}")


def compare_figure() -> None:
    runs = [
        ("baseline (h=64)", 2, "#378ADD", "o"),
        ("O20 capacity (h=128)", 3, "#BA7517", "s"),
        ("O21 harmonic-init", 4, "#D85A30", "^"),
    ]
    rows = sorted(DATA, key=lambda r: r[1])
    Ns = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.5", ls="--", lw=1, zorder=1)
    for label, col, color, marker in runs:
        ys = [r[col] for r in rows]
        ax.plot(Ns, ys, marker=marker, color=color, lw=1.3, ms=7, label=label, zorder=3)
    ax.set_xlabel("rollout depth  N  (max ring distance)")
    ax.set_ylabel("RMSE deficit:  model − harmonic  (m)")
    ax.set_title("Neither capacity nor harmonic-init bends the depth crossover")
    ax.legend(fontsize=9, frameon=False)
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_crossover_compare.png", dpi=150)
    print(f"wrote {OUT / 'phase2_crossover_compare.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_figure()
    compare_figure()


if __name__ == "__main__":
    main()
