"""
TextVQA scoring using the OFFICIAL VQA soft-accuracy metric (m4c_evaluator.py,
copied verbatim from LLaVA). Provides aggregate AND per-sample scores (the
per-sample scores are needed for the C0.4 oracle-headroom diagnostic).

VQA soft accuracy: per question, for the model's processed answer,
score = (leave-one-out averaged) min(#matching humans / 3, 1) over the 10 humans.
Aggregate = mean over questions. This is NOT exact match.

Matching: predictions carry (image_id, question); annotations are keyed by
(image_id, question.lower()) → 10 human answers. This mirrors eval_textvqa.py
(which matches by (question_id==image_id, prompt_processor(prompt)==question)).
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from m4c_evaluator import TextVQAAccuracyEvaluator

ANNOTATION = "data/textvqa/TextVQA_0.5.1_val.json"


def load_annotations(path: str = ANNOTATION) -> dict:
    data = json.load(open(path))["data"]
    return {(a["image_id"], a["question"].lower()): a["answers"] for a in data}


def score_textvqa(predictions: list[dict], annotation_path: str = ANNOTATION,
                  correct_threshold: float = 0.5) -> dict:
    """
    predictions: [{"question_id": image_id, "question": str, "pred_answer": str}, ...]
    Returns aggregate accuracy + per-sample soft scores + binary-correct (>= threshold).
    """
    ann = load_annotations(annotation_path)
    ev = TextVQAAccuracyEvaluator()

    per_sample = []
    missing = 0
    by_type = defaultdict(lambda: {"n": 0, "score_sum": 0.0})
    for p in predictions:
        key = (p["question_id"], p["question"].lower())
        if key not in ann:
            missing += 1
            continue
        gt = ann[key]
        pred = ev.answer_processor(p["pred_answer"])
        scores = ev._compute_answer_scores(gt)
        soft = scores.get(pred, 0.0)
        per_sample.append({
            "question_id": p["question_id"], "question": p["question"],
            "pred_answer": p["pred_answer"], "soft_acc": round(soft, 4),
            "correct": soft >= correct_threshold,
        })

    n = len(per_sample)
    agg = sum(s["soft_acc"] for s in per_sample) / n if n else 0.0
    n_correct_bin = sum(1 for s in per_sample if s["correct"])
    return {
        "accuracy_pct": round(agg * 100, 2),
        "n_evaluated": n,
        "n_missing": missing,
        "binary_correct_pct": round(n_correct_bin / n * 100, 2) if n else 0.0,
        "correct_threshold": correct_threshold,
        "per_sample": per_sample,
    }


def print_textvqa(result: dict, label: str = "", reference: float = 58.2) -> None:
    acc = result["accuracy_pct"]
    diff = acc - reference
    sep = "=" * 56
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(f"  VQA soft-acc : {acc:.2f}%  (n={result['n_evaluated']:,})")
    print(f"  binary@{result['correct_threshold']} : {result['binary_correct_pct']:.2f}%")
    if result.get("n_missing"):
        print(f"  missing      : {result['n_missing']}")
    print(f"  Ref {reference}%   : {diff:+.2f}pp  "
          f"({'MATCH ✓' if abs(diff) <= 1.0 else 'OUT OF OFFSET ✗'})")
    print(sep)


if __name__ == "__main__":
    # quick self-test against a saved predictions.json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotation", default=ANNOTATION)
    args = ap.parse_args()
    raw = json.load(open(args.predictions))
    preds = raw if isinstance(raw, list) else raw.get("predictions", [])
    r = score_textvqa(preds, args.annotation)
    print_textvqa(r, label=args.predictions)
