"""Compare two training runs side by side.

Reads summary.json from two run directories and prints a formatted
comparison table. Useful for A/B experiments.

Usage:
    python scripts/compare_runs.py outputs/tensorboard/run_A outputs/tensorboard/run_B
"""
import argparse
import json
from pathlib import Path


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {run_dir}")
    with open(summary_path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    args = p.parse_args()

    a = load_summary(args.run_a)
    b = load_summary(args.run_b)

    fields = [
        ("best_val_loss", "{:.4f}"),
        ("best_val_epoch", "{}"),
        ("n_epochs_completed", "{}"),
        ("early_stopped", "{}"),
        ("final_train_loss", "{:.4f}"),
        ("final_val_loss", "{:.4f}"),
    ]

    print(f"{'metric':<22} {'A':>20} {'B':>20} {'B - A':>15}")
    print("-" * 80)
    for field, fmt in fields:
        va = a.get(field)
        vb = b.get(field)
        va_str = fmt.format(va) if va is not None else "N/A"
        vb_str = fmt.format(vb) if vb is not None else "N/A"

        # Compute delta where it makes sense
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+}"
        else:
            delta_str = "—"

        print(f"{field:<22} {va_str:>20} {vb_str:>20} {delta_str:>15}")

    print(f"\nRun A: {args.run_a}")
    print(f"Run B: {args.run_b}")


if __name__ == "__main__":
    main()
