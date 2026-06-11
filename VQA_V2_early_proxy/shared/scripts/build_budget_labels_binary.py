import argparse
import json
from collections import Counter
from pathlib import Path


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

        if budget == 144:
            cls = 0
            name = "small_144"
            mapped_tokens = 144
        else:
            cls = 1
            name = "large_288_432_576"
            mapped_tokens = 576

        counts[cls] += 1

        out_records.append({
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "question": r["question"],
            "oracle_budget": budget,
            "budget_class": cls,
            "budget_class_name": name,
            "mapped_tokens": mapped_tokens,
        })

    summary = {
        "num_samples": len(out_records),
        "class_mapping": {
            "0": "small_144",
            "1": "large_288_432_576",
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
