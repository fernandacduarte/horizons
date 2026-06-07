"""Generate the canonical train/val/test split for the dataset.

Reads data/surfaces/metadata.json (produced by scripts/build_dataset.py),
puts all R7 surfaces in test_ood, and splits the remaining surfaces
70/15/15 (train/val/test_id) stratified by reservoir group.

Writes data/splits/split_v1.json with four lists of surface_ids.
"""
import argparse
import json
import random
from pathlib import Path


def stratified_split(
    items_by_group: dict[str, list[str]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split each group's items into train/val/test_id by the given fractions.

    For very small groups, the splits may be 0 in one or more partitions.
    test_id gets whatever's left after train and val.
    """
    rng = random.Random(seed)
    train, val, test_id = [], [], []

    for group, items in sorted(items_by_group.items()):
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(train_frac * n)
        n_val = round(val_frac * n)
        # Whatever remains goes to test_id
        n_test = n - n_train - n_val

        # Guard: if rounding pushes us out of bounds, rebalance toward train
        if n_test < 0:
            n_train += n_test
            n_test = 0

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test_id.extend(shuffled[n_train + n_val :])

    return {"train": train, "val": val, "test_id": test_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/surfaces/metadata.json")
    parser.add_argument("--out", default="data/splits/split_v1.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood-group", default="R7_HorizonOutSpace")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Bucket surfaces by reservoir group, separating R7 (test_ood) from the rest
    by_group: dict[str, list[str]] = {}
    test_ood: list[str] = []
    for k in metadata["kept"]:
        sid = k["surface_id"]
        rid = k["reservoir_id"]
        if rid == args.ood_group:
            test_ood.append(sid)
        else:
            by_group.setdefault(rid, []).append(sid)

    # Stratified 70/15/15 on the non-OOD groups
    split = stratified_split(by_group, seed=args.seed)
    split["test_ood"] = sorted(test_ood)
    split = {k: sorted(split[k]) for k in ["train", "val", "test_id", "test_ood"]}

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote {out_path}\n")

    # Digest
    print(f"Split sizes:")
    for name, items in split.items():
        print(f"  {name:<12}: {len(items)}")
    print()

    # Per-group breakdown for the train/val/test_id splits
    surface_to_group = {k["surface_id"]: k["reservoir_id"] for k in metadata["kept"]}
    print(f"{'group':<25} {'train':>6} {'val':>4} {'test_id':>8} {'test_ood':>9}")
    print("-" * 55)
    all_groups = sorted(set(surface_to_group.values()))
    for g in all_groups:
        counts = {"train": 0, "val": 0, "test_id": 0, "test_ood": 0}
        for split_name in ["train", "val", "test_id", "test_ood"]:
            for sid in split[split_name]:
                if surface_to_group[sid] == g:
                    counts[split_name] += 1
        print(f"{g:<25} {counts['train']:>6} {counts['val']:>4} "
              f"{counts['test_id']:>8} {counts['test_ood']:>9}")


if __name__ == "__main__":
    main()
