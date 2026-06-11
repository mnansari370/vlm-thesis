"""
per_type_accuracy.py — Break down a generation_eval JSON by question type.

Replicates LlavaDynamicVQAModel._infer_question_type_ids EXACTLY so the per-type
split matches the dynamic model's budget-allocation buckets. Computes per-type VQA
consensus accuracy from the saved predictions (no GPU, no model load).

Types: 0=yes/no, 1=attribute, 2=counting, 3=spatial.

Usage:
    python -m VQA_V2.shared.evaluation.per_type_accuracy \\
        VQA_V2/outputs/static_k265_matched/generation_eval_10k.json [more.json ...]
"""

import json
import os
import sys
from collections import defaultdict
from typing import List

# Allow running this file directly (repo root = 3 levels up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from VQA_V2.shared.datasets.vqav2_answers import normalize_answer

QTYPE_NAMES = {0: "yes/no", 1: "attribute", 2: "counting", 3: "spatial"}

_YES_NO = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
           "will ", "would ", "has ", "have ", "had ", "is there", "are there")
_COUNTING = ("how many", "number of", "count ", "amount of")
_SPATIAL = ("where", "left", "right", "behind", "front", "in front", "on top",
            "under", "above", "below", "next to", "near", "between", "side",
            "position", "located")


def infer_qtype(q: str) -> int:
    """EXACT replica of LlavaDynamicVQAModel._infer_question_type_ids."""
    qn = " ".join(str(q).lower().strip().split())
    if any(qn.startswith(p) for p in _COUNTING) or any(p in qn for p in _COUNTING):
        return 2
    if any(qn.startswith(p) for p in _YES_NO):
        return 0
    if any(p in qn for p in _SPATIAL):
        return 3
    return 1


def vqa_score(pred: str, raw: List[str]) -> float:
    pn = normalize_answer(pred)
    return min(1.0, sum(1 for a in raw if normalize_answer(a) == pn) / 3.0)


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)
    preds = data.get("generation", {}).get("predictions")
    if preds is None:
        print(f"  [no generation predictions in {path}]")
        return
    by_type = defaultdict(list)
    overall = []
    for p in preds:
        s = vqa_score(p["pred_answer"], p["raw_answers"])
        overall.append(s)
        by_type[infer_qtype(p["question"])].append(s)

    print(f"\n=== {path} ===")
    print(f"  overall: {100*sum(overall)/len(overall):.2f}%  (n={len(overall)})")
    print(f"  {'type':12s} {'acc%':>7s} {'count':>7s}")
    for t in [0, 1, 2, 3]:
        v = by_type[t]
        if v:
            print(f"  {QTYPE_NAMES[t]:12s} {100*sum(v)/len(v):>7.2f} {len(v):>7d}")
        else:
            print(f"  {QTYPE_NAMES[t]:12s} {'NA':>7s} {0:>7d}")
    return {
        "overall": sum(overall) / len(overall),
        "n": len(overall),
        "per_type": {QTYPE_NAMES[t]: (sum(by_type[t]) / len(by_type[t]) if by_type[t] else None,
                                      len(by_type[t])) for t in [0, 1, 2, 3]},
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyze(path)
