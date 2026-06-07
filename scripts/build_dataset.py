"""Convert GOCAD .ts files to .npz + reservoir/group metadata.

Applies filters from DECISIONS.md (D4.2):
  - Drop V < 500 (degenerate)
  - Drop V > 50,000 (computationally unwieldy)
  - Drop Euler ≠ 1 (non-manifold)
Applies z-sign normalization from D4.3:
  - Flip z if entirely negative, so output is uniformly depth-positive.
Classifies each kept file into one of 8 filename-pattern groups (D4.4).

Outputs:
  data/surfaces/<surface_id>.npz   - one per kept file
  data/surfaces/metadata.json      - bookkeeping (kept, dropped, reservoir map)

Usage:
    python scripts/build_dataset.py
"""
import argparse
import os
import json
import re
from pathlib import Path

import numpy as np

from horizons.data.mesh import HorizonSurface


def classify_reservoir(filename: str) -> str:
    """Classify a .ts filename into one of 8 reservoir groups.

    See DECISIONS.md D4.4 for the rationale. The order of checks matters
    because some patterns are subsets of others.
    """
    name = filename[:-3] if filename.endswith(".ts") else filename

    # R7: HorizonN-OutSpace (test_ood)
    if re.match(r"^Horizon\d+-OutSpace$", name):
        return "R7_HorizonOutSpace"

    # R5: TestHorizonN
    if re.match(r"^TestHorizon\d+$", name):
        return "R5_TestHorizon"

    # R6: horizonteN or horizonteN-utm (lowercase h)
    if re.match(r"^horizonte\d+(-utm)?$", name):
        return "R6_horizonte_utm"

    # R4: HorizonteN or HorizonteN_<suffix> (capital H, no leading number)
    if re.match(r"^Horizonte\d+(_[A-Za-z]+)?$", name):
        return "R4_Horizonte"

    # R1, R2, R3: filenames starting with digits
    m = re.match(r"^(\d+)(_)?(.+)$", name)
    if m:
        digits, underscore, rest = m.group(1), m.group(2), m.group(3)
        if underscore == "_":
            # NN_<rest>: split R1 (lowercase) from R2 (CamelCase)
            # R1 = lowercase + may contain underscores
            if rest.islower() or "_" in rest and rest.split("_")[0].islower():
                return "R1_lowercase_underscore"
            else:
                return "R2_camelcase_underscore"
        else:
            # NN<rest> (no underscore between)
            return "R3_concatenated"

    # R8: anything else (standalone non-numbered files)
    return "R8_standalone"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts-dir",
        default=os.environ.get("HORIZONS_TS_DIR"),
        required=os.environ.get("HORIZONS_TS_DIR") is None,
        help=(
            "Directory containing .ts files. Reads $HORIZONS_TS_DIR if set; "
            "otherwise must be passed explicitly."
        ),
    )
    parser.add_argument("--out-dir", default="data/surfaces")
    parser.add_argument("--min-vertices", type=int, default=500)
    parser.add_argument("--max-vertices", type=int, default=50_000)
    args = parser.parse_args()

    ts_dir = Path(args.ts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean out any prior .npz files in out_dir so we don't carry
    # stale entries from a previous run
    for old_npz in out_dir.glob("*.npz"):
        old_npz.unlink()

    files = sorted(ts_dir.glob("*.ts"))
    print(f"Processing {len(files)} .ts files from {ts_dir}\n")

    kept: list[dict] = []
    dropped: list[dict] = []

    for path in files:
        record = {"filename": path.name}
        try:
            surf = HorizonSurface.from_ts(path)
            V_count, F_count = surf.n_vertices, surf.n_faces
            E_count = surf.n_edges
            euler = V_count - E_count + F_count

            record.update({
                "n_vertices": V_count,
                "n_faces": F_count,
                "euler": euler,
            })

            # Apply filters
            if V_count < args.min_vertices:
                record["reason"] = f"V < {args.min_vertices} (degenerate)"
                dropped.append(record)
                continue
            if V_count > args.max_vertices:
                record["reason"] = f"V > {args.max_vertices} (too large)"
                dropped.append(record)
                continue
            if euler != 1:
                record["reason"] = f"Euler = {euler} (non-manifold)"
                dropped.append(record)
                continue

            # z-sign normalization (D4.3)
            V_np = surf.V.numpy().astype(np.float64)
            F_np = surf.F.numpy()
            z = V_np[:, 2]
            z_flipped = False
            if z.max() <= 0:
                V_np[:, 2] = -V_np[:, 2]
                z_flipped = True

            # Classify reservoir
            reservoir_id = classify_reservoir(path.name)

            # Save .npz with surface_id from the filename stem
            surface_id = path.stem
            out_path = out_dir / f"{surface_id}.npz"
            np.savez(out_path, V=V_np, F=F_np)

            record.update({
                "surface_id": surface_id,
                "reservoir_id": reservoir_id,
                "z_flipped": z_flipped,
                "z_min": float(V_np[:, 2].min()),
                "z_max": float(V_np[:, 2].max()),
            })
            kept.append(record)

        except Exception as e:
            record["reason"] = f"load error: {type(e).__name__}: {e}"
            dropped.append(record)

    # Write metadata.json
    metadata = {
        "kept": kept,
        "dropped": dropped,
        "filter": {
            "min_vertices": args.min_vertices,
            "max_vertices": args.max_vertices,
            "euler": 1,
        },
        "z_sign_normalization": "all_positive (depth convention)",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Console digest
    print(f"Kept:    {len(kept)} files")
    print(f"Dropped: {len(dropped)} files")
    print()
    if dropped:
        print("Dropped:")
        for d in dropped:
            print(f"  {d['filename']:<32} {d['reason']}")
        print()

    # Per-reservoir counts among kept files
    by_reservoir: dict[str, list[str]] = {}
    for k in kept:
        by_reservoir.setdefault(k["reservoir_id"], []).append(
            k["filename"]
        )
    print("Kept files by reservoir group:")
    for rid in sorted(by_reservoir.keys()):
        names = by_reservoir[rid]
        print(f"  {rid}: {len(names)} files")
        for n in sorted(names):
            print(f"    - {n}")

    # z-flip summary
    n_flipped = sum(1 for k in kept if k.get("z_flipped"))
    print(f"\nZ-flipped (negative → positive): {n_flipped} / {len(kept)} files")
    print(f"\nWrote {len(kept)} .npz files to {out_dir}")
    print(f"Wrote metadata to {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
