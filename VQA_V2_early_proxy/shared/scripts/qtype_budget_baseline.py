import argparse
import json
from collections import Counter, defaultdict


def infer_qtype(question: str) -> str:
    q = question.lower().strip()
    words = q.split()
    first = words[0] if words else ""

    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have"}:
        return "yes_no"
    if first in {"what", "which"}:
        if "color" in q or "colour" in q:
            return "color"
        if "number" in q or "many" in q or "count" in q:
            return "count"
        return "what_which"
    if first == "how":
        if "many" in q or "much" in q:
            return "count"
        return "how"
    if first in {"where", "who", "why", "when"}:
        return first
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    by_qtype = defaultdict(Counter)
    for r in records:
        qtype = infer_qtype(r["question"])
        by_qtype[qtype][int(r["budget_class"])] += 1

    qtype_to_pred = {}
    for qtype, cnt in by_qtype.items():
        qtype_to_pred[qtype] = cnt.most_common(1)[0][0]

    correct = 0
    total = 0
    pred_counts = Counter()
    true_counts = Counter()
    confusion = defaultdict(Counter)

    for r in records:
        qtype = infer_qtype(r["question"])
        y = int(r["budget_class"])
        pred = qtype_to_pred[qtype]

        total += 1
        correct += int(pred == y)
        pred_counts[pred] += 1
        true_counts[y] += 1
        confusion[y][pred] += 1

    names = {
        0: "small_144",
        1: "medium_288_432",
        2: "large_576",
    }

    print("=" * 100)
    print("Question-type budget baseline")
    print("=" * 100)
    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")
    print()
    print("Best class per question type:")
    for qtype, cnt in sorted(by_qtype.items(), key=lambda x: sum(x[1].values()), reverse=True):
        pred = qtype_to_pred[qtype]
        n = sum(cnt.values())
        print(f"{qtype:<14} n={n:<5} pred={pred} {names[pred]:<16} counts={dict(cnt)}")
    print()
    print("True class counts:", {names[k]: v for k, v in sorted(true_counts.items())})
    print("Pred class counts:", {names[k]: v for k, v in sorted(pred_counts.items())})
    print()
    print("Confusion matrix rows=true, cols=pred")
    print(f"{'':<18} {'pred_small':>12} {'pred_medium':>12} {'pred_large':>12}")
    for y in [0, 1, 2]:
        print(
            f"{names[y]:<18} "
            f"{confusion[y][0]:>12} "
            f"{confusion[y][1]:>12} "
            f"{confusion[y][2]:>12}"
        )


if __name__ == "__main__":
    main()
