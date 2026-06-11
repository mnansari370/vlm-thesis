import argparse
import os
from typing import List


def check_exists(path: str) -> bool:
    return os.path.exists(path)


def format_status(path: str) -> str:
    return "OK" if check_exists(path) else "MISSING"


def main():
    parser = argparse.ArgumentParser(
        description="Check VQA v2 folder structure and required files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/vqav2",
        help="Root directory of the VQA v2 dataset.",
    )
    args = parser.parse_args()

    data_root = args.data_root

    required_paths: List[str] = [
        os.path.join(data_root, "train2014"),
        os.path.join(data_root, "val2014"),
        os.path.join(data_root, "v2_OpenEnded_mscoco_train2014_questions.json"),
        os.path.join(data_root, "v2_OpenEnded_mscoco_val2014_questions.json"),
        os.path.join(data_root, "v2_mscoco_train2014_annotations.json"),
        os.path.join(data_root, "v2_mscoco_val2014_annotations.json"),
    ]

    print(f"Checking VQA v2 data root: {data_root}")
    print("-" * 80)

    all_ok = True
    for path in required_paths:
        status = format_status(path)
        print(f"[{status:7}] {path}")
        if status != "OK":
            all_ok = False

    print("-" * 80)
    if all_ok:
        print("VQA v2 structure check passed.")
    else:
        print("Some required files/folders are missing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()