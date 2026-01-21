#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path
import os 

SPLITS = {"train": 0.85, "val": 0.05, "test": 0.10}

def main():
    parser = argparse.ArgumentParser(description="Randomly move datapoint directories into train/val/test splits.")
    parser.add_argument("src", type=Path, help="Path that currently contains the datapoint directories.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Where to create train/val/test. Defaults to SRC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would move, without moving.")
    parser.add_argument("--mimic", type=str, help='Do the same split as this other dataset', default=None)
    args = parser.parse_args()

    src: Path = args.src.resolve()
    outdir: Path = (args.outdir or src).resolve()

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source path does not exist or is not a directory: {src}")

    # Destination dirs
    dest_dirs = {name: (outdir / name) for name in ("train", "val", "test")}
    for d in dest_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Collect datapoint directories (immediate subdirs), excluding split dirs
    excluded = {p.resolve() for p in dest_dirs.values()}
    datapoints = [p for p in src.iterdir()
                  if p.is_dir() and p.resolve() not in excluded]

    n = len(datapoints)
    if n == 0:
        raise SystemExit(f"No datapoint directories found in {src}")
    
    if args.mimic:
        splits = {
            'train': os.listdir(os.path.join(args.mimic, 'train')),
            'val': os.listdir(os.path.join(args.mimic, 'val')),
            'test': os.listdir(os.path.join(args.mimic, 'test'))
        }
        
        # Summary
        print(f"Found {n} datapoints in {src}")
        print(f"Plan: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        if args.dry_run:
            for name, items in splits.items():
                print(f"\n[{name}] -> {dest_dirs[name]}")
                for p in items[:10]:  # show a sample
                    print(f"  {p}")
                if len(items) > 10:
                    print(f"  ... (+{len(items)-10} more)")
            print("\nDry run only; nothing moved.")
            return

        # Move directories
        for split_name, items in splits.items():
            target_root = dest_dirs[split_name]
            for dp in items:
                dest = target_root / dp
                if dest.exists():
                    raise SystemExit(f"Destination already exists: {dest} (won't overwrite).")
                shutil.move(os.path.join(args.src, dp), dest)
    else:
        # Shuffle deterministically
        rnd = random.Random(args.seed)
        rnd.shuffle(datapoints)

        # Compute split counts: ensure they sum to N
        n_train = int(SPLITS["train"] * n)
        n_val   = int(SPLITS["val"]   * n)
        n_test  = n - n_train - n_val  # remainder goes to test to guarantee sum = N

        splits = {
            "train": datapoints[:n_train],
            "val":   datapoints[n_train:n_train + n_val],
            "test":  datapoints[n_train + n_val:]
        }
        # Summary
        print(f"Found {n} datapoints in {src}")
        print(f"Plan: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        if args.dry_run:
            for name, items in splits.items():
                print(f"\n[{name}] -> {dest_dirs[name]}")
                for p in items[:10]:  # show a sample
                    print(f"  {p.name}")
                if len(items) > 10:
                    print(f"  ... (+{len(items)-10} more)")
            print("\nDry run only; nothing moved.")
            return

        # Move directories
        for split_name, items in splits.items():
            target_root = dest_dirs[split_name]
            for dp in items:
                dest = target_root / dp.name
                if dest.exists():
                    raise SystemExit(f"Destination already exists: {dest} (won't overwrite).")
                shutil.move(str(dp), str(dest))

    

    print("Done.")

if __name__ == "__main__":
    main()
