"""Build the Phase-2 dataset split: the canonical small-surface split plus the
eight V>50k, V<=600k surfaces, placed by hand to balance mesh magnitude across
train / val / test_id (each set gets one ~400k+ surface and some moderate ones).

Phase 1 = small data (V<=48k), 11.8 the best model. Phase 2 restarts the study
on this fuller, magnitude-balanced split, now that gradient checkpointing
(D12.2) removed the memory wall. The placement is explicit (not seeded) so the
split is reproducible and auditable — fixing O15's "lost split" mistake. It
reads the Phase-1 split (split_v1.json, left untouched) as the small-surface
base and writes a NEW split_v2.json. Commit the result.

Run AFTER building the large surfaces:
    HORIZONS_TS_DIR=/path/to/ts python scripts/build_dataset.py --max-vertices 600000
    python scripts/build_phase2_split.py --dry-run
    python scripts/build_phase2_split.py
    git add data/splits/split_v2.json && git commit -m "O19: Phase-2 split (split_v2)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Hand-assigned placement: one ~400k+ surface per set, the rest spread, so
# train / val / test_id all span the mesh-size range (magnitude-balanced).
LARGE_PLACEMENT = {
    "train":   ["15TopoSal", "16TopoAndarAlagoas", "03TopoOligoMioceno", "01FundoMar"],
    "val":     ["04BaseOligoMioceno", "02TopoMioceno"],
    "test_id": ["07TopoCenomaniano", "06TopoCretaceoSuperior"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces-dir", default="data/surfaces")
    ap.add_argument("--base-file", default="data/splits/split_v1.json",
                    help="Phase-1 small split to read as the base (left untouched)")
    ap.add_argument("--out-file", default="data/splits/split_v2.json",
                    help="Phase-2 split to write")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = json.load(open(args.base_file))
    targets = {sid for ids in LARGE_PLACEMENT.values() for sid in ids}

    # Clean slate for the target surfaces: remove them from wherever they are,
    # so the result is deterministic regardless of the base split's state.
    for k in ["train", "val", "test_id", "test_ood"]:
        split[k] = [s for s in split[k] if s not in targets]
    # Place each large surface in its assigned set.
    for k, ids in LARGE_PLACEMENT.items():
        split[k] = sorted(set(split[k]) | set(ids))

    sdir = Path(args.surfaces_dir)

    def size_of(sid: str):
        p = sdir / f"{sid}.npz"
        if not p.exists():
            return None
        try:
            return int(np.load(p)["V"].shape[0])
        except Exception:
            return None

    print(f"base (read, untouched): {args.base_file}")
    print("Phase-2 large-surface placement:")
    missing = []
    for k, ids in LARGE_PLACEMENT.items():
        parts = []
        for sid in ids:
            v = size_of(sid)
            if v is None:
                missing.append(sid)
                parts.append(f"{sid}(MISSING)")
            else:
                parts.append(f"{sid}({v // 1000}k)")
        print(f"  {k:8}: {', '.join(parts)}")

    print(f"\nsplit sizes: train {len(split['train'])}, val {len(split['val'])}, "
          f"test_id {len(split['test_id'])}, test_ood {len(split['test_ood'])}")
    if missing:
        print(f"\nWARNING: {len(missing)} surface(s) not built yet in {sdir}: "
              f"{', '.join(missing)}")
        print("  run build_dataset.py --max-vertices 600000 first (on the machine with the .ts files)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    json.dump(split, open(args.out_file, "w"), indent=2)
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
