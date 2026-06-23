"""
Canonical GQA scorer — the single source of truth for all eval in this project.

Normalization matches LLaVA's convert_gqa_for_eval.py exactly:
    text.strip().rstrip('.').lower()

That is ALL. No article removal. No plural stripping. No punctuation removal.
No extract_short_answer. Strict equality after this one transform.

Reference:
  - LLaVA-1.5 convert_gqa_for_eval.py: `text.rstrip('.').lower()`
  - GQA official eval.py: compares lowercased prediction to lowercased gold
  - Published LLaVA-1.5-7B GQA testdev_balanced: 62.0%

Usage (rescore saved predictions):
    python -m GQA.shared.official_score \\
        --predictions outputs/paper_static_cls_attn_k288_*/predictions.json \\
        --questions   data/gqa/val_balanced_questions.json

Usage (score testdev format):
    python -m GQA.shared.official_score \\
        --predictions outputs/testdev_dense_*/testdev_balanced_predictions.json \\
        --questions   data/gqa/testdev_balanced_questions.json \\
        --format testdev
"""

import argparse
import json
from collections import defaultdict


# ── normalization ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Canonical GQA normalization.
    Applied identically to prediction and gold answer.
    Matches LLaVA's convert_gqa_for_eval.py.
    """
    return str(text).strip().rstrip(".").lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


# ── scoring ───────────────────────────────────────────────────────────────────

GQA_TYPES = ["obj", "attr", "rel", "cat", "global"]


def score_val_format(predictions: list[dict], questions: dict) -> dict:
    """
    Score predictions in the val_balanced format (from run_static.py etc.).

    Prediction dict fields:
        question_id, pred_answer, answer, semantic_type
    """
    total   = 0
    correct = 0
    empty   = 0
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})

    for p in predictions:
        pred = p.get("pred_answer", "") or ""
        gold = p.get("answer", "") or ""
        t    = p.get("semantic_type", "unknown")

        if not pred.strip():
            empty += 1

        ok = is_correct(pred, gold)
        total   += 1
        correct += int(ok)
        by_type[t]["n"]       += 1
        by_type[t]["correct"] += int(ok)

    return _make_result(total, correct, empty, by_type)


def score_testdev_format(predictions: list[dict], questions: dict) -> dict:
    """
    Score predictions in the GQA official testdev format:
        [{"questionId": "...", "prediction": "..."}]

    Looks up gold answers and types from the questions JSON.
    """
    pred_map = {str(p["questionId"]): p["prediction"] for p in predictions}

    total   = 0
    correct = 0
    empty   = 0
    missing = 0
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})

    for qid, rec in questions.items():
        if str(qid) not in pred_map:
            missing += 1
            continue

        pred = pred_map[str(qid)]
        gold = rec.get("answer", "") or ""
        t    = rec.get("types", {}).get("semantic", "unknown")

        if not str(pred).strip():
            empty += 1

        ok = is_correct(pred, gold)
        total   += 1
        correct += int(ok)
        by_type[t]["n"]       += 1
        by_type[t]["correct"] += int(ok)

    result = _make_result(total, correct, empty, by_type)
    result["n_missing"] = missing
    return result


def _make_result(total, correct, empty, by_type) -> dict:
    acc = correct / total if total > 0 else 0.0
    per_type = {}
    for t, d in by_type.items():
        n = d["n"]
        per_type[t] = {
            "n":        n,
            "correct":  d["correct"],
            "accuracy": round(d["correct"] / n, 6) if n > 0 else 0.0,
        }
    return {
        "accuracy":     round(acc, 6),
        "accuracy_pct": round(acc * 100, 2),
        "n_correct":    correct,
        "n_total":      total,
        "n_empty":      empty,
        "per_type":     per_type,
    }


# ── pretty print ──────────────────────────────────────────────────────────────

def print_result(result: dict, label: str = "", reference: float = 62.0) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
        print(f"  {'─'*54}")
    acc  = result["accuracy_pct"]
    diff = acc - reference
    print(f"  Overall : {acc:.2f}%  "
          f"({result['n_correct']:,}/{result['n_total']:,})")
    if result.get("n_empty", 0):
        print(f"  Empty   : {result['n_empty']:,}  "
              f"({result['n_empty']/max(result['n_total'],1)*100:.2f}%)")
    if result.get("n_missing", 0):
        print(f"  Missing : {result['n_missing']:,}")
    print(f"  Ref 62% : {diff:+.2f}pp  "
          f"({'MATCH ✓' if abs(diff) <= 0.5 else 'NO MATCH ✗'})")
    print(f"\n  Per semantic type:")
    pt = result.get("per_type", {})
    for t in GQA_TYPES:
        if t in pt:
            d = pt[t]
            print(f"    {t:<8} {d['accuracy']*100:>6.2f}%  (n={d['n']:,})")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions JSON (val format or testdev format).")
    ap.add_argument("--questions", default="data/gqa/testdev_balanced_questions.json",
                    help="Path to GQA questions JSON.")
    ap.add_argument("--format", choices=["val", "testdev"], default="testdev",
                    help="'val' = {predictions: [{question_id, pred_answer, answer, ...}]}; "
                         "'testdev' = [{questionId, prediction}]")
    ap.add_argument("--label", default="",
                    help="Optional label for the results table.")
    ap.add_argument("--reference", type=float, default=62.0,
                    help="Reference accuracy for gap reporting (default 62.0).")
    ap.add_argument("--output", default=None,
                    help="Optional path to save results JSON.")
    args = ap.parse_args()

    with open(args.predictions) as f:
        raw = json.load(f)

    with open(args.questions) as f:
        questions = json.load(f)

    if args.format == "val":
        preds = raw if isinstance(raw, list) else raw.get("predictions", [])
        result = score_val_format(preds, questions)
    else:
        preds = raw if isinstance(raw, list) else raw.get("predictions", [])
        result = score_testdev_format(preds, questions)

    label = args.label or args.predictions.split("/")[-2]
    print_result(result, label=label, reference=args.reference)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Saved] {args.output}")


if __name__ == "__main__":
    main()
