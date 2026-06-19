"""Add all large (V > threshold) surfaces in data/surfaces/ to the TRAIN
split, leaving val / test_id / test_ood untouched.

For the data-diversity experiment (O19): bring back the V>50k surfaces
excluded at Stage 4 (D4.2), now trainable thanks to gradient checkpointing
(D12.2). Adding them to TRAIN only — same 7-surface val — keeps the
comparison to the 30-surface baseline and the O16 noise floor clean.
(Do NOT use build_split.py for this: it re-stratifies everything and would
reshuffle the val set, breaking comparability.)

Run AFTER rebuilding the dataset with the size filter relaxed:
    HORIZONS_TS_DIR=/path/to/ts python scripts/build_dataset.py --max-vertices 1000000
    python scripts/add_large_to_train.py --dry-run   # preview
    python scripts/add_large_to_train.py             # apply (edits split in place)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces-dir", default="data/surfaces")
    ap.add_argument("--split-file", default="data/splits/split_v1.json")
    ap.add_argument("--threshold", type=int, default=50_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split_file))
    held_out = set(split["val"]) | set(split["test_id"]) | set(split["test_ood"])
    train = set(split["train"])

    sdir = Path(args.surfaces_dir)
    added = []
    for p in sorted(sdir.glob("*.npz")):
        sid = p.stem
        try:
            V = np.load(p)["V"].shape[0]
        except Exception:
            continue
        if V > args.threshold and sid not in held_out and sid not in train:
            added.append((V, sid))

    print(f"large surfaces (V > {args.threshold:,}) to add to train: {len(added)}")
    for V, sid in sorted(added, reverse=True):
        print(f"  + {V:>8,}  {sid}")
    if not added:
        print("nothing to add (already in train, held out, or not built yet).")
        return

    if args.dry_run:
        print("\n--dry-run: split file NOT modified.")
        return

    split["train"] = sorted(train | {sid for _, sid in added})
    json.dump(split, open(args.split_file, "w"), indent=2)
    print(
        f"\nupdated {args.split_file}: train now {len(split['train'])} "
        f"(val {len(split['val'])}, test_id {len(split['test_id'])}, "
        f"test_ood {len(split['test_ood'])})"
    )


if __name__ == "__main__":
    main()
