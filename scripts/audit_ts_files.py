"""Audit all .ts files: report basic mesh stats so we can decide which
to keep for training. Writes a CSV summary and prints a digest.

Usage:
    python scripts/audit_ts_files.py
"""
import argparse
import os
import csv
from pathlib import Path

import torch

from horizons.data.mesh import HorizonSurface


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
    parser.add_argument(
        "--out", default="outputs/ts_audit.csv",
    )
    args = parser.parse_args()

    ts_dir = Path(args.ts_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(ts_dir.glob("*.ts"))
    print(f"Auditing {len(files)} .ts files in {ts_dir}\n")

    rows = []
    for path in files:
        row = {"filename": path.name}
        try:
            surf = HorizonSurface.from_ts(path)
            V, F, E = surf.n_vertices, surf.n_faces, surf.n_edges
            euler = V - E + F
            z = surf.V[:, 2]
            xy = surf.V[:, :2]
            row.update({
                "status": "ok",
                "n_vertices": V,
                "n_faces": F,
                "n_edges": E,
                "euler": euler,
                "z_min": z.min().item(),
                "z_max": z.max().item(),
                "z_range": z.max().item() - z.min().item(),
                "x_range": xy[:, 0].max().item() - xy[:, 0].min().item(),
                "y_range": xy[:, 1].max().item() - xy[:, 1].min().item(),
            })
        except Exception as e:
            row.update({
                "status": f"error: {type(e).__name__}",
                "n_vertices": -1, "n_faces": -1, "n_edges": -1,
                "euler": -1,
                "z_min": float("nan"), "z_max": float("nan"),
                "z_range": float("nan"),
                "x_range": float("nan"), "y_range": float("nan"),
            })
        rows.append(row)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}\n")

    # Console digest
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] != "ok")
    n_disk = sum(1 for r in rows if r["status"] == "ok" and r["euler"] == 1)
    n_nondisk = sum(1 for r in rows if r["status"] == "ok" and r["euler"] != 1)
    print(f"Summary: {n_ok} loaded, {n_err} failed")
    print(f"  Disk-topology (Euler=1): {n_disk}")
    print(f"  Non-disk (Euler!=1):     {n_nondisk}")

    # Per-attribute distributions
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        vs = sorted(r["n_vertices"] for r in ok_rows)
        zs_min = sorted(r["z_min"] for r in ok_rows)
        zs_max = sorted(r["z_max"] for r in ok_rows)
        n = len(ok_rows)
        print()
        print("n_vertices distribution:")
        print(f"  min:    {vs[0]}")
        print(f"  10%:    {vs[n // 10]}")
        print(f"  median: {vs[n // 2]}")
        print(f"  90%:    {vs[9 * n // 10]}")
        print(f"  max:    {vs[-1]}")
        print()
        print("z range across files:")
        print(f"  z_min overall: {min(zs_min):.1f}")
        print(f"  z_max overall: {max(zs_max):.1f}")
        n_positive = sum(1 for r in ok_rows if r["z_min"] >= 0)
        n_negative = sum(1 for r in ok_rows if r["z_max"] <= 0)
        n_mixed = sum(1 for r in ok_rows
                      if r["z_min"] < 0 < r["z_max"])
        print(f"  All-positive z:  {n_positive} files")
        print(f"  All-negative z:  {n_negative} files")
        print(f"  Crosses zero:    {n_mixed} files")

    # Print non-disk files for inspection
    if n_nondisk:
        print()
        print("Non-disk files (will likely need to be excluded):")
        for r in ok_rows:
            if r["euler"] != 1:
                print(f"  {r['filename']:<32} "
                      f"V={r['n_vertices']:>6} F={r['n_faces']:>6} "
                      f"Euler={r['euler']}")

    # Print error files
    err_rows = [r for r in rows if r["status"] != "ok"]
    if err_rows:
        print()
        print("Failed to load:")
        for r in err_rows:
            print(f"  {r['filename']:<32} {r['status']}")


if __name__ == "__main__":
    main()
