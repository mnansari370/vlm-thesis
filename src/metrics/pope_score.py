"""
POPE scorer — faithful to LLaVA's official eval_pope.py (copied logic).

Per subset (random/popular/adversarial): yes/no parsing + accuracy, precision,
recall, F1, yes_ratio. Also returns per-sample correctness (for the oracle band).

Pred parsing (verbatim from eval_pope.py): keep first sentence, drop commas, then
'no'/'not'/'no' in words → no, else yes.
"""

import json


def parse_pred(text: str) -> str:
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def score_subset(answers: list[dict], label_file: str) -> dict:
    """answers: [{question_id, text(pred)}] in the SAME order as label_file lines."""
    labels = [json.loads(q)["label"] for q in open(label_file)]
    preds = [parse_pred(a["text"]) for a in answers]
    label_bin = [1 if l == "yes" else 0 for l in labels]
    pred_bin = [1 if p == "yes" else 0 for p in preds]

    TP = TN = FP = FN = 0
    per_sample = []
    for pred, lab, a in zip(pred_bin, label_bin, answers):
        if pred == 1 and lab == 1: TP += 1
        elif pred == 1 and lab == 0: FP += 1
        elif pred == 0 and lab == 0: TN += 1
        elif pred == 0 and lab == 1: FN += 1
        per_sample.append({"question_id": a["question_id"], "correct": pred == lab})

    n = len(pred_bin)
    yes_ratio = pred_bin.count(1) / n
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (TP + TN) / n
    return {
        "accuracy_pct": round(acc * 100, 2), "f1": round(f1 * 100, 2),
        "precision": round(precision * 100, 2), "recall": round(recall * 100, 2),
        "yes_ratio": round(yes_ratio, 4), "n": n,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "per_sample": per_sample,
    }
