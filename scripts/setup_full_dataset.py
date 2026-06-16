"""One-shot setup for the full dataset (Stage 11.13 — V<=1M, all 57 surfaces).

This script:
  1. Runs build_dataset.py with --max-vertices 1000000 to keep all surfaces
     (including the 10 large V>50k surfaces excluded in the default build).
  2. Updates data/splits/split_v1.json to add the 10 new surfaces:
     - 6 to train (full size range, including the 610k surface)
     - 2 to val (110k, 455k)
     - 2 to test_id (195k, 673k)
     - test_ood unchanged
  3. Verifies the final split counts.

Usage:
    python scripts/setup_full_dataset.py --ts-dir <path/to/ts/files>

Environment variable HORIZONS_TS_DIR is read as a fallback for --ts-dir.

After this, run training as usual:
    python scripts/train.py [...standard args...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# The 10 large surfaces and where each goes in the split.
# Sorted by vertex count (smallest fir).
NEW_SURFACES = {
    "04BaseOligoMioceno":     {"V": 110_240, "split": "val"},
    "07TopoCenomaniano":      {"V": 165_265, "split": "train"},
    "16TopoAndarAlagoas":     {"V": 191_977, "split": "train"},
    "15TopoSal":              {"V": 195_059, "split": "test_id"},
    "03TopoOligoMioceno":     {"V": 230_587, "split": "train"},
    "06TopoCretaceoSuperior": {"V": 412_260, "split": "train"},
    "02TopoMioceno":          {"V": 443_782, "split": "train"},
    "01FundoMar":             {"V": 455_081, "split": "val"},
    "18TopoEmbasamento":      {"V": 610_721, "split": "train"},
    "17TopoAndarJiquia":      {"V": 673_793, "split": "test_id"},
}

EXPECTED_FINAL_SIZES = {
    "train":    36,
    "val":       9,
    "test_id":   7,
    "test_ood":  5,
}


def run_build_dataset(ts_dir: str, max_vertices: int = 1_000_000) -> None:
    """Run build_dataset.py with a high V cap to include all surfaces."""
    print("=" * 70)
    print("Step 1/3: Building dataset from .ts files")
    print("=" * 70)
    print(f"  --ts-dir: {ts_dir}")
    print(f"  --max-vertices: {max_vertices}")
    print()

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "build_dataset.py"),
        "--ts-dir", ts_dir,
        "--max-vertices", str(max_vertices),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"build_dataset.py failed with code {result.returncode}"
        )


def update_split(split_path: Path) -> None:
    """Add the 10 new surfaces to split_v1.json."""
    print()
    print("=" * 70)
    print("Step 2/3: Updating split_v1.json with the 10 new large surfaces")
    print("=" * 70)

    with open(split_path) as f:
        split = json.load(f)

    # Make sure none of the new surfaces are already in any split.
    # If they are, skip them — likely the user re-ran this script.
    existing = set()
    for surfaces in split.values():
        existing.update(surfaces)

    additions: dict[str, list[str]] = {"train": [], "val": [], "test_id": []}
  already_present: list[str] = []
    for sid, info in NEW_SURFACES.items():
        if sid in existing:
            already_present.append(sid)
        else:
            additions[info["split"]].append(sid)

    if already_present:
        print(f"  Skipped (already in split): {already_present}")

    for split_name, sids in additions.items():
        if not sids:
            continue
        split[split_name] = sorted(split[split_name] + sids)
        print(f"  Added to {split_name}: {sids}")

    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Saved: {split_path}")


def verify(split_path: Path, surfaces_dir: Path) -> None:
    """Verify the split is what we expect and the .npz files exist."""
    print()
    print("=" * 70)
    print("Step 3/3: Verification")
    print("=" * 70)

    with open(split_path) as f:
        split = json.load(f)

    # Check counts
    print("\nSplit sizes:")
    all_ok = True
    for k, expected in EXPECTED_FINAL_SIZES.items():
        actual = len(split[k])
        ok = actual == expected
        all_ok = all_ok and ok
        marker = "ok" if ok else "MISMATCH"
        print(f"  {k}: {actual} (expected {expected})  [{marker}]")
    if not all_ok:
        print("\nWARNING: split sizes don't match expected. Investigate manually.")

    # Check that each surface has a corresponding .npz file
    all_sids = set()
    for sids in split.values():
        all_sids.update(sids)
    missing = []
    for sid in all_sids:
        if not (surfaces_dir / f"{sid}.npz").is_file():
            missing.append(sid)
    if missing:
        print(f"\nERROR: .npz file missing for {len(missing)} surfaces:")
        for sid in sorted(missing):
            print(f"  - {sid}")
        raise SystemExit(1)
    print(f"\nAll {len(all_sids)} surfaces have .npz files.")

    # Total vertex count summary
    npz_count = len(list(surfaces_dir.glob("*.npz")))
    print(f"\nTotal .npz files in {surfaces_dir}: {npz_count}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ts-dir",
        default=os.environ.get("HORIZONS_TS_DIR"),
        required=os.environ.get("HORIZONS_TS_DIR") is None,
        help=(
            "Directory containing the original .ts files. "
            "Defaults to $HORIZONS_TS_DIR."
        ),
    )
    p.add_argument("--max-vertices", type=int, default=1_000_000)
    p.add_argument("--split-path", type=Path, default=Path("data/splits/split_v1.json"))
    p.add_argument("--surfaces-dir", type=Path, default=Path("data/surfaces"))
    args = p.parse_args()

    if not Path(args.ts_dir).is_dir():
        print(f"ERROR: --ts-dir does not exist: {args.ts_dir}", file=sys.stderr)
        raise SystemExit(1)

    run_build_dataset(args.ts_dir, args.max_vertices)
    update_split(args.split_path)
    verify(args.split_path, args.surfaces_dir)

    print()
    print("=" * 70)
    print("Setup complete. Next step:")
    print("=" * 70)
    print()
    print("Run training. The standard Stage 11.8 / 11.13 configuration:")
    print()
    print("  python scripts/train.py \\")
    print("      train.n_epochs=100 \\")
    print("      train.patience=20 \\")
    print("      optim.accum_steps=4 \\")
    print("      data.normalize_per_surface=true \\")
    print("      data.init_method=meanplane \\")
    print("      data.n_masks_per_epoch=3")
    print()


if __name__ == "__main__":
    main()
