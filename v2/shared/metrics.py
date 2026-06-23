"""
Paper-quality GQA metrics.

Accuracy:  exact match after gqa_exact_match() normalisation.
Per-type:  GQA semantic types — obj / attr / rel / cat / global.
           These are the types printed in every GQA paper table.

The "unknown" bucket is kept for book-keeping but excluded from per-type tables.
"""

import string
from collections import defaultdict
from typing import Any


# ── answer normalisation ──────────────────────────────────────────────────────

def normalize_gqa_answer(answer: str | None) -> str:
    if not answer:
        return ""
    a = str(answer).lower().strip()
    a = a.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in a.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def gqa_exact_match(pred: str | None, gold: str | None) -> bool:
    return normalize_gqa_answer(pred) == normalize_gqa_answer(gold)


# ── answer extraction ─────────────────────────────────────────────────────────

def extract_short_answer(generated_text: str, question: str = "") -> str:
    """
    Extract a short (1-3 word) answer from LLM output.
    Confirmed working on the 67.69% zero-shot run.
    """
    if not generated_text:
        return ""

    text = generated_text.strip()

    for marker in ["ASSISTANT:", "assistant:"]:
        if marker.lower() in text.lower():
            idx = text.lower().find(marker.lower())
            text = text[idx + len(marker):].strip()
            break

    text = text.strip('.,;:!?"')
    words = text.split()
    if not words:
        return ""

    first = words[0].lower().rstrip(".,!?")
    if first in {"yes", "no"}:
        return first

    if len(words) == 1:
        return words[0]

    filler = {"a", "an", "the", "is", "are", "was", "were", "it", "this", "that"}
    if len(words) <= 3:
        for w in words:
            if w.lower().rstrip(".,!?") not in filler:
                return w.rstrip(".,!?")
        return words[-1].rstrip(".,!?")

    strip_starts = [
        "there is a ", "there is an ", "there are ",
        "it is a ", "it is an ", "it is ",
        "this is a ", "this is an ", "this is ",
        "the answer is ", "i think it is ",
        "it looks like ", "it appears to be ",
        "the weather is ", "the color is ", "the colour is ",
    ]
    text_lower = text.lower()
    for prefix in strip_starts:
        if text_lower.startswith(prefix):
            remainder = text[len(prefix):].strip()
            rem_words = remainder.split()
            if rem_words:
                return rem_words[0].rstrip(".,!?")

    skip = {"a", "an", "the", "is", "are", "was", "were", "it", "that", "this"}
    for w in words:
        if w.lower().rstrip(".,!?") not in skip:
            return w.rstrip(".,!?")

    return words[0].rstrip(".,!?")


# ── GQA semantic type ─────────────────────────────────────────────────────────

GQA_SEMANTIC_TYPES = {"obj", "attr", "rel", "cat", "global"}

# Display order and short names used in paper tables
TYPE_ORDER  = ["obj", "attr", "rel", "cat", "global"]
TYPE_LABELS = {"obj": "Object", "attr": "Attribute", "rel": "Relation",
               "cat": "Category", "global": "Global"}


def get_semantic_type(types_dict: dict | None) -> str:
    if not isinstance(types_dict, dict):
        return "unknown"
    t = types_dict.get("semantic") or types_dict.get("structural") or "unknown"
    return t if t in GQA_SEMANTIC_TYPES else "unknown"


# ── accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute overall + per-type GQA accuracy from a list of prediction dicts.

    Each dict must contain:
      pred_answer : str
      answer      : str   (ground truth)
      semantic_type : str  (from get_semantic_type)

    Returns a dict ready for JSON serialisation and LaTeX table generation.
    """
    if not predictions:
        return {}

    total, correct = 0, 0
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})

    for p in predictions:
        pred = p.get("pred_answer", "")
        gold = p.get("answer", "")
        t    = p.get("semantic_type", "unknown")

        is_correct = gqa_exact_match(pred, gold)
        total  += 1
        correct += int(is_correct)

        by_type[t]["n"]       += 1
        by_type[t]["correct"] += int(is_correct)

    overall_acc = correct / total if total > 0 else 0.0

    per_type = {}
    for t, c in by_type.items():
        n = c["n"]
        per_type[t] = {
            "n":           n,
            "correct":     c["correct"],
            "accuracy":    round(c["correct"] / n, 6) if n > 0 else 0.0,
            "pct_of_total": round(100.0 * n / total, 2) if total > 0 else 0.0,
        }

    return {
        "n_evaluated": total,
        "n_correct":   correct,
        "gqa_accuracy": round(overall_acc, 6),
        "per_type": per_type,
    }


def print_accuracy_table(metrics: dict[str, Any], label: str = "") -> None:
    """Print a paper-style accuracy table to stdout."""
    if not metrics:
        print("No metrics to display.")
        return

    header = f"  {label}" if label else ""
    print("=" * 60)
    print(f"GQA Accuracy{header}")
    print("-" * 60)
    print(f"  Overall    : {metrics['gqa_accuracy'] * 100:.2f}%"
          f"  ({metrics['n_correct']:,} / {metrics['n_evaluated']:,})")
    print()
    print(f"  {'Type':<12} {'Acc':>7}   {'N':>7}   {'% of val':>9}")
    print("  " + "-" * 42)
    pt = metrics.get("per_type", {})
    for t in TYPE_ORDER:
        if t in pt:
            r = pt[t]
            print(f"  {TYPE_LABELS[t]:<12} {r['accuracy']*100:>6.2f}%  "
                  f"{r['n']:>8,}  {r['pct_of_total']:>8.1f}%")
    if "unknown" in pt:
        r = pt["unknown"]
        print(f"  {'Unknown':<12} {r['accuracy']*100:>6.2f}%  "
              f"{r['n']:>8,}  {r['pct_of_total']:>8.1f}%")
    print("=" * 60)
