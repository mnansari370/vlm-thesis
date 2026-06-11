import argparse
import json
from collections import Counter
from pathlib import Path


def budget_to_class(budget: int) -> int:
    if budget == 144:
        return 0  # small
    if budget in {288, 432}:
        return 1  # medium
    if budget == 576:
        return 2  # large
    raise ValueError(f"Unexpected budget: {budget}")


def class_to_name(cls: int) -> str:
    return {0: "small_144", 1: "medium_288_432", 2: "large_576"}[cls]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.oracle, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    out_records = []
    counts = Counter()

    for r in records:
        budget = int(r["oracle_budget"])
        cls = budget_to_class(budget)
        counts[cls] += 1

        out_records.append({
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "question": r["question"],
            "oracle_budget": budget,
            "budget_class": cls,
            "budget_class_name": class_to_name(cls),
        })

    summary = {
        "num_samples": len(out_records),
        "class_mapping": {
            "0": "small_144",
            "1": "medium_288_432",
            "2": "large_576",
        },
        "class_counts": {str(k): counts[k] for k in sorted(counts)},
        "class_percentages": {
            str(k): counts[k] / len(out_records) for k in sorted(counts)
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": out_records}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
