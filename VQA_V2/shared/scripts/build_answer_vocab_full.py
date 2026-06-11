"""
Build the full VQAv2 answer vocabulary with no top-K cutoff.

Reads the full 443K training annotations, normalizes every majority answer,
and writes ALL unique answers to answer_vocab_full.json — no frequency cutoff.

Usage:
    python VQA_V2/shared/scripts/build_answer_vocab_full.py \
        --annotations-path data/vqav2/v2_mscoco_train2014_annotations.json \
        --output-path data/vqav2/answer_vocab_full.json

The output format matches the existing answer vocab files so that all downstream
code (VQAv2Dataset, LLaVA wrappers) can consume it unchanged.
"""

import argparse
import json
import os
import sys
from collections import Counter

# Allow running from repo root
# Allow running this file directly (repo root = 3 level(s) up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.shared.datasets.vqav2_answers import get_majority_answer


def build_full_vocab(annotations_path: str) -> tuple:
    print(f"Reading annotations from: {annotations_path}", flush=True)
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]
    print(f"Total annotations: {len(annotations)}", flush=True)

    answer_counter = Counter()
    empty_count = 0

    for ann in annotations:
        raw_answers = [a["answer"] for a in ann.get("answers", [])]
        majority = get_majority_answer(raw_answers)
        if majority:
            answer_counter[majority] += 1
        else:
            empty_count += 1

    if empty_count > 0:
        print(f"Skipped {empty_count} annotations with empty majority answer.", flush=True)

    # Sort by frequency descending so id=0 is the most common answer.
    # This matches the convention used by existing vocab files.
    sorted_answers = [ans for ans, _ in answer_counter.most_common()]

    answer_to_id = {ans: idx for idx, ans in enumerate(sorted_answers)}
    id_to_answer = {str(idx): ans for idx, ans in enumerate(sorted_answers)}

    freq_list = [answer_counter[ans] for ans in sorted_answers]

    metadata = {
        "num_unique_answers": len(sorted_answers),
        "total_annotations_processed": len(annotations),
        "empty_majority_skipped": empty_count,
        "top10_answers": sorted_answers[:10],
        "top10_frequencies": freq_list[:10],
        "cutoff": "none (full vocabulary)",
    }

    return answer_to_id, id_to_answer, metadata


def main():
    parser = argparse.ArgumentParser(description="Build full VQAv2 answer vocabulary (no top-K cutoff).")
    parser.add_argument(
        "--annotations-path",
        default="data/vqav2/v2_mscoco_train2014_annotations.json",
    )
    parser.add_argument(
        "--output-path",
        default="data/vqav2/answer_vocab_full.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations_path):
        raise FileNotFoundError(f"Annotations not found: {args.annotations_path}")

    answer_to_id, id_to_answer, metadata = build_full_vocab(args.annotations_path)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    output = {
        "answer_to_id": answer_to_id,
        "id_to_answer": id_to_answer,
        "metadata": metadata,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {args.output_path}")
    print(f"Total unique answers: {metadata['num_unique_answers']}")
    print(f"Top 10 answers: {metadata['top10_answers']}")
    print(f"Top 10 frequencies: {metadata['top10_frequencies']}")


if __name__ == "__main__":
    main()
