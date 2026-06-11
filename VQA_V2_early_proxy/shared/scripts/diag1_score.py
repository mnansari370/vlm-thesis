"""
Score FasterVLM predictions against our val2014 ground truth.

FasterVLM outputs one JSONL line per question:
  {"question_id": 123, "text": "cat", "answer_id": ..., ...}

Usage (from repo root):
  python VQA_V2_early_proxy/shared/scripts/diag1_score.py \
      --answers scripts/diag1_data/answers/k128/answers.jsonl \
      --gt      scripts/diag1_data/val2014_gt.json \
      --k       128 \
      --output  scripts/diag1_data/answers/k128/scores.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def vqa_score(pred: str, raw_answers: list) -> float:
    pn = normalize(pred)
    matches = sum(1 for a in raw_answers if normalize(a) == pn)
    return min(1.0, matches / 3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--gt",      required=True)
    parser.add_argument("--k",       type=int, required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)   # {question_id: [answer, ...]}
    # keys may be strings
    gt = {int(k): v for k, v in gt.items()}

    preds = []
    with open(args.answers) as f:
        for line in f:
            preds.append(json.loads(line.strip()))

    scores = []
    for p in preds:
        qid  = int(p["question_id"])
        pred = p.get("text", "").strip()
        raw  = gt.get(qid, [])
        sc   = vqa_score(pred, raw)
        scores.append({"question_id": qid, "pred": pred, "score": sc})

    acc = sum(s["score"] for s in scores) / len(scores) if scores else 0.0

    result = {
        "k":             args.k,
        "n_predictions": len(scores),
        "vqa_accuracy":  acc,
        "pct":           round(acc * 100, 2),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"FasterVLM K={args.k} on val2014:")
    print(f"  n_predictions = {result['n_predictions']}")
    print(f"  vqa_accuracy  = {result['vqa_accuracy']:.4f}  ({result['pct']}%)")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
