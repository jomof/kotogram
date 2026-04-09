#!/usr/bin/env python3
"""Compute 5 equal-count token length bins from the dataset bundle.

Usage: .venv/bin/python scratch/token_length_quintiles.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from scripts.dataset import BundledStyleDataset, resolve_dataset


def main():
    bundle, _chive = resolve_dataset()
    ds = BundledStyleDataset.from_bundle(bundle, sample_ratio=1.0)
    gram = ds.filter_by_grammaticality(label=1)

    offsets = gram.offsets
    indices = gram.indices
    lengths = (offsets[indices + 1] - offsets[indices]).numpy()
    lengths.sort()

    n = len(lengths)
    print(f"Grammatical sentences: {n:,}")
    print(f"Token length: min={lengths[0]}  max={lengths[-1]}  "
          f"median={lengths[n//2]}  mean={lengths.mean():.1f}")

    n_bins = 5
    print(f"\n{n_bins} equal-count bins:")
    print(f"  {'bin':>4s}  {'range':>12s}  {'count':>10s}  {'%':>6s}")
    print(f"  {'':->4s}  {'':->12s}  {'':->10s}  {'':->6s}")

    for b in range(n_bins):
        lo_idx = b * n // n_bins
        hi_idx = (b + 1) * n // n_bins - 1
        lo_val = int(lengths[lo_idx])
        hi_val = int(lengths[hi_idx])
        count = hi_idx - lo_idx + 1
        pct = 100 * count / n
        print(f"  {b+1:>4d}  {lo_val:>5d}–{hi_val:<5d}  {count:>10,}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
