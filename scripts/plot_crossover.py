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
        ax.scatter(N, d, s=80, marker="o", color=WIN if win else LOSE,
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


def results_figure() -> None:
    """Two-panel test RMSE with the mean-plane baseline: test_ood and test_id."""
    MP, ROLL, HYB, HARM = "#888780", "#378ADD", WIN, "#BA7517"
    ood = [("Mean-plane", 0.0, MP), ("Rollout", 7.8, ROLL),
           ("Hybrid", 51.1, HYB), ("Harmonic", 58.3, HARM)]
    idd = [("Harmonic", 153.6, HARM), ("Hybrid", 167.6, HYB),
           ("Mean-plane", 232.4, MP), ("Rollout", 253.4, ROLL)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, data, title in [(a1, ood, "test_ood"), (a2, idd, "test_id")]:
        ax.bar([d[0] for d in data], [d[1] for d in data], color=[d[2] for d in data])
        for i, d in enumerate(data):
            ax.text(i, d[1], f"{d[1]:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("RMSE (m) — lower better")
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(OUT / "phase_results.png", dpi=150)
    print(f"wrote {OUT / 'phase_results.png'}")


def hybrid_vs_harmonic_figure() -> None:
    """Diverging per-surface deficit (hybrid − harmonic) on test_id."""
    rows = [("TestHorizon3", -38.0), ("07TopoCenomaniano", -36.7), ("02_MCinza", -2.6),
            ("horizonte1-utm", 0.6), ("01_FMar", 11.1),
            ("06TopoCretaceoSuperior", 12.7), ("FUNDO_DO_MAR", 150.7)]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [WIN if v < 0 else LOSE for v in vals]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y, vals, color=colors, zorder=3)
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()                       # first row (best) at top
    ax.set_xlim(-62, 182)                   # padding so value labels never overflow
    ax.set_xlabel("← hybrid wins            harmonic wins →")
    ax.set_title("Hybrid − Harmonic RMSE per test_id surface (m)")
    for yi, v in zip(y, vals):
        ha = "right" if v < 0 else "left"
        ax.text(v + (-3 if v < 0 else 3), yi, f"{'+' if v > 0 else ''}{v:.1f}",
                va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_vs_harmonic.png", dpi=150)
    print(f"wrote {OUT / 'hybrid_vs_harmonic.png'}")


def test_id_vertices_table() -> None:
    """Simple 2-column table (white background): test_id surface + vertex count."""
    rows = [("06TopoCretaceoSuperior", 412260), ("07TopoCenomaniano", 165265),
            ("02_MCinza", 48000), ("01_FMar", 48000), ("FUNDO_DO_MAR", 9976),
            ("horizonte1-utm", 4464), ("TestHorizon3", 2436)]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.93, "Surface", fontsize=12, fontweight="bold", ha="left", va="center")
    ax.text(0.97, 0.93, "Vertices", fontsize=12, fontweight="bold", ha="right", va="center")
    ax.plot([0.02, 0.98], [0.87, 0.87], color="#333333", lw=1.3)
    for i, (name, v) in enumerate(rows):
        y = 0.78 - i * 0.11
        ax.text(0.03, y, name, fontsize=11, ha="left", va="center")
        ax.text(0.97, y, f"{v:,}", fontsize=11, ha="right", va="center")
        ax.plot([0.02, 0.98], [y - 0.055, y - 0.055], color="#e5e5e5", lw=0.7)
    fig.savefig(OUT / "test_id_vertices.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    print(f"wrote {OUT / 'test_id_vertices.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_figure()
    compare_figure()
    deep_bar()
    hybrid_figure()
    results_figure()
    hybrid_vs_harmonic_figure()
    test_id_vertices_table()


if __name__ == "__main__":
    main()
