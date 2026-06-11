import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def infer_qtype(question: str) -> str:
    q = question.lower().strip()

    if q.startswith("is ") or q.startswith("are ") or q.startswith("was ") or q.startswith("were ") or q.startswith("do ") or q.startswith("does ") or q.startswith("did ") or q.startswith("can ") or q.startswith("could ") or q.startswith("has ") or q.startswith("have "):
        return "yes_no"

    first = q.split()[0] if q.split() else ""

    if first in {"what", "which"}:
        if "color" in q or "colour" in q:
            return "color"
        if "number" in q or "many" in q or "count" in q:
            return "count"
        return "what_which"

    if first in {"how"}:
        if "many" in q or "much" in q:
            return "count"
        return "how"

    if first in {"where"}:
        return "where"

    if first in {"who"}:
        return "who"

    if first in {"why"}:
        return "why"

    if first in {"when"}:
        return "when"

    return "other"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--summary_out", default="data/budget_oracle/qtype_oracle_summary.json")
    args = parser.parse_args()

    oracle_data = load_json(args.oracle)

    if isinstance(oracle_data, dict) and "samples" in oracle_data:
        samples = oracle_data["samples"]
    elif isinstance(oracle_data, dict) and "oracle_samples" in oracle_data:
        samples = oracle_data["oracle_samples"]
    elif isinstance(oracle_data, dict) and "records" in oracle_data:
        samples = oracle_data["records"]
    elif isinstance(oracle_data, list):
        samples = oracle_data
    else:
        raise ValueError(f"Unknown oracle JSON structure. Top-level keys: {oracle_data.keys() if isinstance(oracle_data, dict) else type(oracle_data)}")

    qtype_stats = defaultdict(lambda: {
        "count": 0,
        "budget_counts": Counter(),
        "resolved": 0,
        "tokens_sum": 0.0,
    })

    global_budget_counts = Counter()

    for ex in samples:
        q = ex.get("question", "")
        qtype = infer_qtype(q)

        budget = (
            ex.get("oracle_budget")
            or ex.get("budget")
            or ex.get("target_budget")
            or ex.get("oracle_tokens")
            or ex.get("tokens")
        )

        if budget is None:
            # Try common nested/alternative names
            for k in ex.keys():
                if "budget" in k.lower() or "token" in k.lower():
                    budget = ex[k]
                    break

        if budget is None:
            raise KeyError(f"Could not find budget/token field in sample keys: {list(ex.keys())}")

        budget = int(budget)

        resolved = ex.get("resolved", None)
        if resolved is None:
            # If oracle uses correctness scores, infer resolved if any model solved it.
            resolved = bool(ex.get("oracle_correct", budget != 576))

        qtype_stats[qtype]["count"] += 1
        qtype_stats[qtype]["budget_counts"][budget] += 1
        qtype_stats[qtype]["tokens_sum"] += budget
        qtype_stats[qtype]["resolved"] += int(bool(resolved))
        global_budget_counts[budget] += 1

    summary = {
        "num_samples": len(samples),
        "global_budget_counts": dict(sorted(global_budget_counts.items())),
        "qtype_summary": {},
    }

    for qtype, st in sorted(qtype_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        count = st["count"]
        budget_counts = dict(sorted(st["budget_counts"].items()))

        summary["qtype_summary"][qtype] = {
            "count": count,
            "percentage": count / len(samples),
            "avg_oracle_tokens": st["tokens_sum"] / count,
            "resolved_rate": st["resolved"] / count,
            "budget_counts": budget_counts,
            "budget_percentages": {
                str(k): v / count for k, v in budget_counts.items()
            },
        }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print(f"Saved question-type oracle summary to: {out_path}")
    print("=" * 100)

    print(f"{'QType':<14} {'Count':>8} {'AvgTok':>10} {'Resolved':>10} {'144%':>8} {'288%':>8} {'432%':>8} {'576%':>8}")
    print("-" * 90)

    for qtype, st in summary["qtype_summary"].items():
        bp = st["budget_percentages"]
        print(
            f"{qtype:<14} "
            f"{st['count']:>8} "
            f"{st['avg_oracle_tokens']:>10.2f} "
            f"{st['resolved_rate']*100:>9.2f}% "
            f"{bp.get('144', 0)*100:>7.2f}% "
            f"{bp.get('288', 0)*100:>7.2f}% "
            f"{bp.get('432', 0)*100:>7.2f}% "
            f"{bp.get('576', 0)*100:>7.2f}%"
        )


if __name__ == "__main__":
    main()
