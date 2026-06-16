"""One-shot setup for the full dataset.

Builds the dataset with --max-vertices 1000000 and updates the split
to add the 10 large surfaces (6 train, 2 val, 2 test_id).

Usage:
    python scripts/setup_full_dataset.py --ts-dir <path>
Or set HORIZONS_TS_DIR env var.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


NEW_SURFACES = {
    "04BaseOligoMioceno":     {"V": 110240, "split": "val"},
    "07TopoCenomaniano":      {"V": 165265, "split": "train"},
    "16TopoAndarAlagoas":     {"V": 191977, "split": "train"},
    "15TopoSal":              {"V": 195059, "split": "test_id"},
    "03TopoOligoMioceno":     {"V": 230587, "split": "train"},
    "06TopoCretaceoSuperior": {"V": 412260, "split": "train"},
    "02TopoMioceno":          {"V": 443782, "split": "train"},
    "01FundoMar":             {"V": 455081, "split": "val"},
    "18TopoEmbasamento":      {"V": 610721, "split": "train"},
    "17TopoAndarJiquia":      {"V": 673793, "split": "test_id"},
}

EXPECTED_FINAL_SIZES = {"train": 36, "val": 9, "test_id": 7, "test_ood": 5}


def run_build_dataset(ts_dir, max_vertices=1000000):
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
        raise RuntimeError(f"build_dataset.py failed with code {result.returncode}")


def update_split(split_path):
    print()
    print("=" * 70)
    print("Step 2/3: Updating split_v1.json")
    print("=" * 70)
    with open(split_path) as f:
        split = json.load(f)
    existing = set()
    for surfaces in split.values():
        existing.update(surfaces)
    additions = {"train": [], "val": [], "test_id": []}
    already_present = []
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


def verify(split_path, surfaces_dir):
    print()
    print("=" * 70)
    print("Step 3/3: Verification")
    print("=" * 70)
    with open(split_path) as f:
        split = json.load(f)
    print("
Split sizes:")
    all_ok = True
    for k, expected in EXPECTED_FINAL_SIZES.items():
        actual = len(split[k])
        ok = actual == expected
        all_ok = all_ok and ok
        marker = "ok" if ok else "MISMATCH"
        print(f"  {k}: {actual} (expected {expected})  [{marker}]")
    if not all_ok:
        print("
WARNING: split sizes don't match expected.")
    all_sids = set()
    for sids in split.values():
        all_sids.update(sids)
    missing = []
    for sid in all_sids:
        if not (surfaces_dir / f"{sid}.npz").is_file():
            missing.append(sid)
    if missing:
        print(f"
ERROR: .npz missing for {len(missing)} surfaces:")
        for sid in sorted(missing):
            print(f"  - {sid}")
        raise SystemExit(1)
    print(f"
All {len(all_sids)} surfaces have .npz files.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ts-dir", default=os.environ.get("HORIZONS_TS_DIR"),
                   required=os.environ.get("HORIZONS_TS_DIR") is None)
    p.add_argument("--max-vertices", type=int, default=1000000)
    p.add_argument("--split-path", type=Path, default=Path("data/splits/split_v1.json"))
    p.add_argument("--surfaces-dir", type=Path, default=Path("data/surfaces"))
    args = p.parse_args()
    if not Path(args.ts_dir).is_dir():
        print(f"ERROR: --ts-dir not found: {args.ts_dir}", file=sys.stderr)
        raise SystemExit(1)
    run_build_dataset(args.ts_dir, args.max_vertices)
    update_split(args.split_path)
    verify(args.split_path, args.surfaces_dir)
    print()
    print("=" * 70)
    print("Setup complete. Now run:")
    print("=" * 70)
    print()
    print("  python scripts/train.py train.n_epochs=100 train.patience=20 \")
    print("    optim.accum_steps=4 data.normalize_per_surface=true \")
    print("    data.init_method=meanplane data.n_masks_per_epoch=3 train.device=cuda")
    print()


if __name__ == "__main__":
    main()
