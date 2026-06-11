"""
Convert our val2014 10K subset to FasterVLM's JSONL question format.

FasterVLM question format (one JSON per line):
  {"question_id": 123, "image": "COCO_val2014_000000123456.jpg", "text": "Is this a cat?"}

We also write a ground-truth file for scoring:
  scripts/diag1_data/val2014_gt.json
  {question_id: [answer1, answer2, ...]}

Usage (from repo root):
  python VQA_V2_early_proxy/shared/scripts/diag1_convert_data.py [--max-samples 10000] [--seed 42]
"""

import argparse
import json
import os
import random

QUESTIONS_FILE   = "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FILE = "data/vqav2/v2_mscoco_val2014_annotations.json"
IMAGE_DIR        = "data/vqav2/val2014"
OUT_DIR          = "scripts/diag1_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(QUESTIONS_FILE) as f:
        qs_data = json.load(f)
    with open(ANNOTATIONS_FILE) as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}

    # Filter to images that exist
    valid = []
    for q in qs_data["questions"]:
        iid = q["image_id"]
        img = f"COCO_val2014_{iid:012d}.jpg"
        if os.path.exists(os.path.join(IMAGE_DIR, img)):
            valid.append(q)

    # Reproducible subset
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    subset = sorted(valid[: args.max_samples], key=lambda x: x["question_id"])

    print(f"Total valid questions: {len(valid)}")
    print(f"Using subset of:       {len(subset)}")

    # Write JSONL for FasterVLM
    jsonl_path = os.path.join(OUT_DIR, "val2014_questions.jsonl")
    with open(jsonl_path, "w") as f:
        for q in subset:
            iid = q["image_id"]
            record = {
                "question_id": q["question_id"],
                "image":       f"COCO_val2014_{iid:012d}.jpg",
                "text":        q["question"],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(subset)} questions → {jsonl_path}")

    # Write GT for scoring
    gt = {}
    for q in subset:
        qid = q["question_id"]
        ann = anns.get(qid, {})
        gt[qid] = [a["answer"] for a in ann.get("answers", [])]

    gt_path = os.path.join(OUT_DIR, "val2014_gt.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    print(f"Wrote GT answers     → {gt_path}")


if __name__ == "__main__":
    main()
