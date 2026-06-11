"""
NaN/Inf diagnostic for feature cache directories.

Usage (run from repo root):
    # Check a single split:
    conda run -n vlm_env python VQA_V2/shared/scripts/check_cache_nan.py \
        --cache-dir VQA_V2/feature_cache/static_k288/train

    # Check all splits under a cache root:
    conda run -n vlm_env python VQA_V2/shared/scripts/check_cache_nan.py \
        --cache-root VQA_V2/feature_cache

Exit code 0 = all clean; exit code 1 = NaN/Inf found (do not use cache).
"""

import argparse
import os
import sys

import numpy as np


def check_dir(cache_dir: str) -> bool:
    """Returns True if clean, False if NaN/Inf found."""
    files = {
        "pooled_features.npy": True,
        "per_layer_answer_pos.npy": True,
    }
    any_bad = False
    for fname, do_check in files.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            any_bad = True
            continue
        arr = np.load(path)
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        nan_rows = int(np.sum(np.any(np.isnan(arr.reshape(arr.shape[0], -1)), axis=1)))
        status = "OK" if (nan_count == 0 and inf_count == 0) else "FAIL"
        print(f"  [{status}] {fname}: shape={arr.shape} NaN={nan_count} (rows={nan_rows}) Inf={inf_count}")
        if status == "FAIL":
            any_bad = True
    return not any_bad


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache-dir", help="Check a single split directory (e.g. feature_cache/static_k288/train)")
    group.add_argument("--cache-root", help="Check all train/val splits under a root (e.g. feature_cache)")
    args = parser.parse_args()

    all_clean = True

    if args.cache_dir:
        print(f"\nChecking: {args.cache_dir}")
        clean = check_dir(args.cache_dir)
        all_clean = all_clean and clean
    else:
        root = args.cache_root
        for model_key in sorted(os.listdir(root)):
            key_path = os.path.join(root, model_key)
            if not os.path.isdir(key_path):
                continue
            for split in ["train", "val"]:
                split_path = os.path.join(key_path, split)
                if not os.path.isdir(split_path):
                    continue
                print(f"\nChecking: {model_key}/{split}")
                clean = check_dir(split_path)
                all_clean = all_clean and clean

    print()
    if all_clean:
        print("RESULT: ALL CLEAN — cache is safe to use")
        sys.exit(0)
    else:
        print("RESULT: NaN/Inf DETECTED — discard and re-run caching before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
