#!/usr/bin/env python3
"""Verify / summarize a UTKFace folder and its age-group distribution.

Usage:
    python scripts/prepare_utkface.py --data-root ./data/UTKFace

UTKFace download (pick one):
  * Kaggle:  kaggle datasets download -d jangedoo/utkface-new  (then unzip into data/UTKFace)
  * Or place the aligned&cropped jpgs (named like 25_0_0_2017....jpg) under --data-root.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (age_to_group, build_samples, group_labels, list_images,  # noqa: E402
                      parse_age_utkface)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data/UTKFace")
    p.add_argument("--age-bins", type=int, nargs="+", default=[0, 20, 30, 40, 50, 200])
    return p.parse_args()


def main():
    args = parse_args()
    root = args.data_root
    if not os.path.isdir(root):
        print(f"[error] folder not found: {root}")
        print("Download UTKFace first (see header of this file), then re-run.")
        sys.exit(1)

    all_files = list_images(root)
    samples = build_samples(root, args.age_bins)
    labels = group_labels(args.age_bins)
    bad = len(all_files) - len(samples)

    print(f"[scan] root: {root}")
    print(f"[scan] image files found : {len(all_files)}")
    print(f"[scan] valid (parsable)  : {len(samples)}")
    print(f"[scan] skipped/malformed : {bad}")
    if not samples:
        print("[warn] no valid UTKFace-named files. Expected e.g. '25_0_0_20170116.jpg'.")
        sys.exit(1)

    counts = [0] * (len(args.age_bins) - 1)
    for _, g in samples:
        counts[g] += 1
    print("\n[age groups]")
    for lab, c in zip(labels, counts):
        pct = 100.0 * c / len(samples)
        print(f"  {lab:>6} : {c:6d}  ({pct:5.1f}%)  " + "#" * int(pct / 2))
    ages = [parse_age_utkface(f) for f in all_files]
    ages = [a for a in ages if a is not None]
    print(f"\n[age] min={min(ages)} max={max(ages)} mean={sum(ages)/len(ages):.1f}")
    print("\n[ok] dataset looks usable. Train with:")
    print("  python scripts/train.py --config configs/utkface_456.yaml "
          f"--data-root {root}")


if __name__ == "__main__":
    main()
