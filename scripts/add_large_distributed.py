"""Distribute large (threshold < V <= max-vertices) surfaces across
train / val / test_id, ADDING them to the existing split (current members
preserved).

Recreates and extends the O15 "smart split": large surfaces present in val
and test_id (not just train), so the data-diversity hypothesis (does adding
the V>50k surfaces help?) can be tested on a val that actually contains large
surfaces. Unlike O15 — whose split lived only on the container and was lost —
this is deterministic (seeded) and writes the split file, so commit it.

Run AFTER rebuilding with the size filter relaxed (capped to exclude the
two >600k giants for tractable epoch time):
    HORIZONS_TS_DIR=/path/to/ts python scripts/build_dataset.py --max-vertices 600000
    python scripts/add_large_distributed.py --dry-run     # preview placement
    python scripts/add_large_distributed.py               # apply, then git add the split
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces-dir", default="data/surfaces")
    ap.add_argument("--split-file", default="data/splits/split_v1.json")
    ap.add_argument("--threshold", type=int, default=50_000)
    ap.add_argument("--max-vertices", type=int, default=600_000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.split_file))
    existing = set().union(
        *[set(split[k]) for k in ["train", "val", "test_id", "test_ood"]]
    )

    sdir = Path(args.surfaces_dir)
    large = []
    for p in sorted(sdir.glob("*.npz")):
        sid = p.stem
        if sid in existing:
            continue
        try:
            V = np.load(p)["V"].shape[0]
        except Exception:
            continue
        if args.threshold < V <= args.max_vertices:
            large.append((V, sid))

    if not large:
        print("no new large surfaces to distribute (none built, or all already in split).")
        return

    # Deterministic: stable size-sorted order, then seeded shuffle, then slice.
    large.sort(reverse=True)
    random.Random(args.seed).shuffle(large)
    n = len(large)
    n_val = round(args.val_frac * n)
    n_test = round(args.test_frac * n)
    to_val = large[:n_val]
    to_test = large[n_val:n_val + n_test]
    to_train = large[n_val + n_test:]

    def show(name: str, items: list) -> None:
        listing = ", ".join(f"{s}({V // 1000}k)" for V, s in sorted(items, reverse=True))
        print(f"  {name:8} += {len(items)}: {listing}")

    print(f"distributing {n} large surfaces "
          f"({args.threshold:,} < V <= {args.max_vertices:,}), seed={args.seed}:")
    show("val", to_val)
    show("test_id", to_test)
    show("train", to_train)

    if args.dry_run:
        print("\n--dry-run: split file NOT modified.")
        return

    split["val"] = sorted(set(split["val"]) | {s for _, s in to_val})
    split["test_id"] = sorted(set(split["test_id"]) | {s for _, s in to_test})
    split["train"] = sorted(set(split["train"]) | {s for _, s in to_train})
    json.dump(split, open(args.split_file, "w"), indent=2)
    print(f"\nupdated {args.split_file}: train {len(split['train'])}, "
          f"val {len(split['val'])}, test_id {len(split['test_id'])}, "
          f"test_ood {len(split['test_ood'])}")


if __name__ == "__main__":
    main()
