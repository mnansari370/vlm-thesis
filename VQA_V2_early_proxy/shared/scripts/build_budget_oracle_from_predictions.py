import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any

from VQA_V2_early_proxy.shared.datasets.vqav2_answers import normalize_answer


def vqa_consensus_score(pred_answer: str, raw_answers: List[str]) -> float:
    pred = normalize_answer(pred_answer)
    gt = [normalize_answer(a) for a in raw_answers]
    matches = sum(1 for a in gt if a == pred)
    return min(1.0, matches / 3.0)


def load_predictions(path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", data)

    out = {}
    for item in preds:
        qid = int(item["question_id"])
        out[qid] = item

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k144", type=str, required=True)
    parser.add_argument("--k288", type=str, required=True)
    parser.add_argument("--k432", type=str, required=True)
    parser.add_argument("--dense", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/budget_oracle")
    parser.add_argument("--output_name", type=str, default="val_oracle_static_dense.json")
    parser.add_argument("--correct_threshold", type=float, default=0.3333333333)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    budget_paths = {
        144: args.k144,
        288: args.k288,
        432: args.k432,
        576: args.dense,
    }

    print("Loading predictions...")
    predictions_by_budget = {
        k: load_predictions(path)
        for k, path in budget_paths.items()
    }

    all_qids = sorted(set(predictions_by_budget[576].keys()))

    records = []
    budget_counter = Counter()
    resolved_counter = Counter()
    score_by_budget_sum = defaultdict(float)

    for qid in all_qids:
        base = predictions_by_budget[576][qid]
        raw_answers = base["raw_answers"]

        budget_scores = {}
        budget_pred_answers = {}

        for budget in [144, 288, 432, 576]:
            item = predictions_by_budget[budget].get(qid)

            if item is None:
                score = 0.0
                pred_answer = ""
            else:
                pred_answer = item.get("pred_answer", "")
                score = vqa_consensus_score(pred_answer, raw_answers)

            budget_scores[str(budget)] = score
            budget_pred_answers[str(budget)] = pred_answer
            score_by_budget_sum[budget] += score

        selected_budget = 576
        resolved = False

        for budget in [144, 288, 432, 576]:
            if budget_scores[str(budget)] >= args.correct_threshold:
                selected_budget = budget
                resolved = True
                break

        budget_counter[selected_budget] += 1
        resolved_counter["resolved" if resolved else "unresolved"] += 1

        records.append(
            {
                "question_id": qid,
                "image_id": base["image_id"],
                "question": base["question"],
                "raw_answers": raw_answers,
                "oracle_budget": selected_budget,
                "oracle_keep_ratio": selected_budget / 576.0,
                "resolved_by_any_budget": resolved,
                "scores": budget_scores,
                "pred_answers": budget_pred_answers,
            }
        )

    n = len(records)

    avg_oracle_tokens = sum(r["oracle_budget"] for r in records) / max(1, n)
    avg_oracle_keep_ratio = avg_oracle_tokens / 576.0
    oracle_resolved_acc = resolved_counter["resolved"] / max(1, n)

    per_budget_accuracy = {
        str(k): score_by_budget_sum[k] / max(1, n)
        for k in [144, 288, 432, 576]
    }

    summary = {
        "num_samples": n,
        "budget_order": [144, 288, 432, 576],
        "correct_threshold": args.correct_threshold,
        "oracle_budget_counts": dict(budget_counter),
        "oracle_budget_percentages": {
            str(k): budget_counter[k] / max(1, n)
            for k in [144, 288, 432, 576]
        },
        "resolved_counts": dict(resolved_counter),
        "oracle_resolved_accuracy": oracle_resolved_acc,
        "avg_oracle_tokens": avg_oracle_tokens,
        "avg_oracle_keep_ratio": avg_oracle_keep_ratio,
        "per_budget_accuracy": per_budget_accuracy,
    }

    output_path = os.path.join(args.output_dir, args.output_name)
    summary_path = os.path.join(
        args.output_dir,
        args.output_name.replace(".json", "_summary.json")
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Oracle budget label file saved:")
    print(output_path)
    print()
    print("Summary saved:")
    print(summary_path)
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
