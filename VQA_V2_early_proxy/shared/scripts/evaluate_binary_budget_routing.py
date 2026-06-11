import argparse
import json
import re
from collections import Counter

import numpy as np


_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens = [_NUMBER_MAP.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in _ARTICLES]
    return " ".join(tokens)


def vqa_score(pred_answer, raw_answers):
    pred = normalize_answer(pred_answer)
    answers = [normalize_answer(a) for a in raw_answers]
    matches = sum(1 for a in answers if a == pred)
    return min(1.0, matches / 3.0)


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data["predictions"] if isinstance(data, dict) and "predictions" in data else data

    out = {}
    for p in preds:
        qid = int(p["question_id"])
        out[qid] = p
    return out


def load_features(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


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


def build_qtype_probability(records, train_indices):
    """
    Estimate P(large | qtype) from training split only.
    This is a simple interpretable router.
    """
    counts = {}
    for i in train_indices:
        r = records[i]
        qtype = r["qtype"] if "qtype" in r else infer_qtype(r["question"])
        y = int(r["budget_class"])
        if qtype not in counts:
            counts[qtype] = [0, 0]
        counts[qtype][y] += 1

    qtype_prob = {}
    global_large = sum(v[1] for v in counts.values())
    global_total = sum(sum(v) for v in counts.values())
    global_prob = global_large / max(1, global_total)

    for qtype, (small, large) in counts.items():
        # Laplace smoothing
        qtype_prob[qtype] = (large + 1) / (small + large + 2)

    return qtype_prob, global_prob


def analytical_attention_flops(seq_len, hidden_size=4096, num_layers=32):
    return 2 * num_layers * (seq_len ** 2) * hidden_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--k144_preds", required=True)
    parser.add_argument("--dense_preds", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/budget_oracle/binary_routing_eval_summary.json")
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import hstack, csr_matrix
    except Exception as e:
        raise SystemExit(
            "Missing sklearn/scipy. Install with: pip install scikit-learn scipy\n"
            f"Original error: {e}"
        )

    records = load_features(args.features)
    k144_preds = load_predictions(args.k144_preds)
    dense_preds = load_predictions(args.dense_preds)

    qids = [int(r["question_id"]) for r in records]
    y = np.array([int(r["budget_class"]) for r in records], dtype=np.int64)

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    questions = [r["question"] for r in records]
    qtypes = [r["qtype"] if "qtype" in r else infer_qtype(r["question"]) for r in records]

    feature_names = sorted(records[0]["features"].keys())
    X_num = np.array(
        [[float(r["features"][k]) for k in feature_names] + [float(r["question_length"])] for r in records],
        dtype=np.float32,
    )

    qtype_vocab = sorted(set(qtypes))
    qtype_to_id = {q: i for i, q in enumerate(qtype_vocab)}
    X_qtype = np.zeros((len(records), len(qtype_vocab)), dtype=np.float32)
    for i, q in enumerate(qtypes):
        X_qtype[i, qtype_to_id[q]] = 1.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
    )

    X_text_train = vectorizer.fit_transform([questions[i] for i in train_idx])
    X_text_test = vectorizer.transform([questions[i] for i in test_idx])

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num[train_idx])
    X_num_test = scaler.transform(X_num[test_idx])

    X_extra_train = np.concatenate([X_num_train, X_qtype[train_idx]], axis=1)
    X_extra_test = np.concatenate([X_num_test, X_qtype[test_idx]], axis=1)

    X_train = hstack([X_text_train, csr_matrix(X_extra_train)])
    X_test = hstack([X_text_test, csr_matrix(X_extra_test)])

    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train, y_train)
    prob_large = clf.predict_proba(X_test)[:, 1]

    qtype_prob, global_qtype_prob = build_qtype_probability(records, train_idx)

    def get_baseline_scores(indices, pred_map, tokens):
        scores = []
        for i in indices:
            qid = qids[i]
            p = pred_map[qid]
            scores.append(vqa_score(p["pred_answer"], p["raw_answers"]))
        return {
            "accuracy": float(np.mean(scores)),
            "avg_tokens": float(tokens),
        }

    k144_base = get_baseline_scores(test_idx, k144_preds, 144)
    dense_base = get_baseline_scores(test_idx, dense_preds, 576)

    results = {
        "num_records": len(records),
        "num_test": len(test_idx),
        "baselines": {
            "k144": k144_base,
            "dense_k576": dense_base,
        },
        "routes": {},
    }

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("=" * 120)
    print("Binary Budget Routing Simulation")
    print("=" * 120)
    print(f"Test samples: {len(test_idx)}")
    print(f"K144 baseline acc:  {k144_base['accuracy']*100:.2f}% | tokens=144")
    print(f"Dense baseline acc: {dense_base['accuracy']*100:.2f}% | tokens=576")
    print()
    print("Router = TFIDF + qtype + visual stats")
    print(f"{'thr':>6} {'acc%':>8} {'avg_tok':>10} {'large%':>9} {'save_vs_dense%':>15} {'gap_to_dense%':>14} {'flopsG':>10}")

    for thr in thresholds:
        route_large = prob_large >= thr

        scores = []
        tokens = []
        route_counts = Counter()

        for local_j, i in enumerate(test_idx):
            qid = qids[i]
            if route_large[local_j]:
                p = dense_preds[qid]
                tok = 576
                route_counts["dense_576"] += 1
            else:
                p = k144_preds[qid]
                tok = 144
                route_counts["k144"] += 1

            scores.append(vqa_score(p["pred_answer"], p["raw_answers"]))
            tokens.append(tok)

        acc = float(np.mean(scores))
        avg_tok = float(np.mean(tokens))
        large_pct = float(np.mean(route_large))
        avg_text_len = float(np.mean([records[i]["question_length"] for i in test_idx]))
        avg_seq_len = avg_tok + avg_text_len
        flops_g = analytical_attention_flops(avg_seq_len) / 1e9
        save_vs_dense = 1.0 - (avg_tok / 576.0)
        gap_to_dense = dense_base["accuracy"] - acc

        key = f"tfidf_visual_thr_{thr:.2f}"
        results["routes"][key] = {
            "accuracy": acc,
            "avg_tokens": avg_tok,
            "large_route_fraction": large_pct,
            "route_counts": dict(route_counts),
            "token_saving_vs_dense": save_vs_dense,
            "accuracy_gap_to_dense": gap_to_dense,
            "analytical_attention_flops_giga": flops_g,
        }

        print(
            f"{thr:>6.2f} "
            f"{acc*100:>8.2f} "
            f"{avg_tok:>10.2f} "
            f"{large_pct*100:>8.2f}% "
            f"{save_vs_dense*100:>14.2f}% "
            f"{gap_to_dense*100:>13.2f}% "
            f"{flops_g:>10.2f}"
        )

    print()
    print("Router = qtype probability only")
    print(f"{'thr':>6} {'acc%':>8} {'avg_tok':>10} {'large%':>9} {'save_vs_dense%':>15} {'gap_to_dense%':>14} {'flopsG':>10}")

    for thr in thresholds:
        scores = []
        tokens = []
        large_flags = []
        route_counts = Counter()

        for i in test_idx:
            r = records[i]
            qid = qids[i]
            qtype = r["qtype"] if "qtype" in r else infer_qtype(r["question"])
            p_large = qtype_prob.get(qtype, global_qtype_prob)

            if p_large >= thr:
                p = dense_preds[qid]
                tok = 576
                large_flags.append(1)
                route_counts["dense_576"] += 1
            else:
                p = k144_preds[qid]
                tok = 144
                large_flags.append(0)
                route_counts["k144"] += 1

            scores.append(vqa_score(p["pred_answer"], p["raw_answers"]))
            tokens.append(tok)

        acc = float(np.mean(scores))
        avg_tok = float(np.mean(tokens))
        large_pct = float(np.mean(large_flags))
        avg_text_len = float(np.mean([records[i]["question_length"] for i in test_idx]))
        avg_seq_len = avg_tok + avg_text_len
        flops_g = analytical_attention_flops(avg_seq_len) / 1e9
        save_vs_dense = 1.0 - (avg_tok / 576.0)
        gap_to_dense = dense_base["accuracy"] - acc

        key = f"qtype_thr_{thr:.2f}"
        results["routes"][key] = {
            "accuracy": acc,
            "avg_tokens": avg_tok,
            "large_route_fraction": large_pct,
            "route_counts": dict(route_counts),
            "token_saving_vs_dense": save_vs_dense,
            "accuracy_gap_to_dense": gap_to_dense,
            "analytical_attention_flops_giga": flops_g,
        }

        print(
            f"{thr:>6.2f} "
            f"{acc*100:>8.2f} "
            f"{avg_tok:>10.2f} "
            f"{large_pct*100:>8.2f}% "
            f"{save_vs_dense*100:>14.2f}% "
            f"{gap_to_dense*100:>13.2f}% "
            f"{flops_g:>10.2f}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 120)
    print(f"Saved summary to: {args.output}")
    print("=" * 120)


if __name__ == "__main__":
    main()
