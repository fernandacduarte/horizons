"""Phase-2/3 figures: GNN-vs-harmonic RMSE deficit (model − harmonic; negative =
beats harmonic) per surface, against rollout depth N, for the baseline, the
failed rollout-family interventions, and the hybrid. Data is the per-surface
output of `scripts/noise_band.py ... --device cuda` on the split_v2 val (3 seeds,
n_masks=10). Update DATA if reruns.

    python scripts/plot_crossover.py
writes, in outputs/evaluation/plots/:
    phase2_crossover.png          (O19, baseline crossover)
    phase2_crossover_compare.png  (deficit vs N, baseline + O20–O23)
    phase2_deep_bar.png           (443k-surface deficit across all runs)
    phase2_hybrid.png             (baseline vs hybrid, deficit vs N)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# surface, N, deficit: baseline | O20 cap | O21 harm-init | O22 lambda_r | O23 freeze | O24 hybrid
DATA = [
    ("TestHorizon4", 11, -26.6, -22.8, -7.9, -0.1, -24.9, -40.1),
    ("TestHorizon7", 11, -21.5, -18.7, -10.4, 4.6, -18.9, -42.8),
    ("09_Horizonte8", 12, 12.9, 11.2, -2.8, 10.1, 10.3, -6.6),
    ("Horizonte5", 19, -2.0, -1.9, -0.2, -2.1, -2.3, -1.0),
    ("horizonte7", 22, -2.0, -1.9, -0.2, -2.2, -2.4, -0.9),
    ("10_BaseModelo", 51, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1),
    ("05_TopoCretaceo", 52, 88.5, 84.5, 73.7, 88.2, 73.2, 35.5),
    ("04BaseOligoMioceno", 69, 41.2, 43.6, 15.8, 27.6, 50.6, 2.6),
    ("02TopoMioceno", 132, 21.4, 55.0, 116.0, 76.8, 60.5, -12.9),
]
RUNS = [  # label, column index into DATA, colour, marker  (rollout-family runs)
    ("baseline (h=64)", 2, "#378ADD", "o"),
    ("O20 capacity", 3, "#BA7517", "s"),
    ("O21 harmonic-init", 4, "#D85A30", "^"),
    ("O22 lambda_r", 5, "#534AB7", "D"),
    ("O23 freeze-filled", 6, "#0F6E56", "v"),
]
WIN, LOSE = "#1D9E75", "#D85A30"
OUT = Path("outputs/evaluation/plots")


def baseline_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.5", ls="--", lw=1)
    for name, N, d, *_ in DATA:
        win = d < 0
        ax.scatter(N, d, s=80, marker="^" if win else "o", color=WIN if win else LOSE,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(name, (N, d), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color="0.35")
    ax.set_xlabel("rollout depth  N")
    ax.set_ylabel("RMSE deficit:  model − harmonic  (m)")
    ax.set_title("GNN beats harmonic at shallow rollout depth, loses at deep")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_crossover.png", dpi=150)
    print(f"wrote {OUT / 'phase2_crossover.png'}")


def compare_figure() -> None:
    rows = sorted(DATA, key=lambda r: r[1])
    Ns = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.5", ls="--", lw=1)
    for label, col, color, marker in RUNS:
        ax.plot(Ns, [r[col] for r in rows], marker=marker, color=color, lw=1.2,
                ms=6, label=label)
    ax.set_xlabel("rollout depth  N")
    ax.set_ylabel("RMSE deficit:  model − harmonic  (m)")
    ax.set_title("No rollout-family intervention bends the deep end down")
    ax.legend(fontsize=8, frameon=False)
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_crossover_compare.png", dpi=150)
    print(f"wrote {OUT / 'phase2_crossover_compare.png'}")


def deep_bar() -> None:
    deep = next(r for r in DATA if r[0] == "02TopoMioceno")
    cols = [("baseline", 2), ("O20 cap", 3), ("O21 h-init", 4), ("O22 λ_r", 5),
            ("O23 freeze", 6), ("O24 hybrid", 7)]
    vals = [deep[c] for _, c in cols]
    labels = [lbl for lbl, _ in cols]
    colors = ["#378ADD", "#D85A30", "#D85A30", "#D85A30", "#D85A30", WIN]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("RMSE deficit vs harmonic (m) — below 0 beats harmonic")
    ax.set_title("On the deepest surface (443k, N=132), only the hybrid beats harmonic")
    ax.tick_params(axis="x", labelrotation=18, labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_deep_bar.png", dpi=150)
    print(f"wrote {OUT / 'phase2_deep_bar.png'}")


def hybrid_figure() -> None:
    rows = sorted(DATA, key=lambda r: r[1])
    Ns = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.scatter(Ns, [r[2] for r in rows], s=60, color="#378ADD", label="baseline rollout", zorder=3)
    ax.scatter(Ns, [r[7] for r in rows], s=70, marker="^", color=WIN, label="hybrid (O24)", zorder=3)
    ax.set_xlabel("rollout depth  N")
    ax.set_ylabel("RMSE deficit:  model − harmonic  (m)   (below 0 = beats harmonic)")
    ax.set_title("Hybrid bends the deep end below zero and keeps the shallow wins")
    ax.legend(fontsize=9, frameon=False)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_hybrid.png", dpi=150)
    print(f"wrote {OUT / 'phase2_hybrid.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_figure()
    compare_figure()
    deep_bar()
    hybrid_figure()


if __name__ == "__main__":
    main()
