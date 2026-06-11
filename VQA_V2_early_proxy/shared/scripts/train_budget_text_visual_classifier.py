import argparse
import json
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    records = []
    with open(args.features, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    questions = [r["question"] for r in records]
    qtypes = [r["qtype"] for r in records]
    y = np.array([int(r["budget_class"]) for r in records], dtype=np.int64)

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

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

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

    print("=" * 100)
    print("Budget text + visual-stat classifier")
    print("=" * 100)
    print("Num records:", len(records))
    print("Train class counts:", Counter(y_train.tolist()))
    print("Test class counts:", Counter(y_test.tolist()))
    print("Feature stats used:", feature_names)
    print("Qtype vocab:", qtype_vocab)

    majority_class = Counter(y_train.tolist()).most_common(1)[0][0]
    majority_pred = np.full_like(y_test, majority_class)
    majority_acc = accuracy_score(y_test, majority_pred)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    names = ["small_144", "large_288_432_576"]

    print()
    print(f"Majority baseline accuracy: {majority_acc:.4f}")
    print(f"Text + visual classifier accuracy: {acc:.4f}")
    print()
    print("Predicted class counts:", Counter(pred.tolist()))
    print()
    print("Classification report at default threshold:")
    print(classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=names,
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix rows=true, cols=pred:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]

        print()
        print("=" * 100)
        print("Threshold sweep for large-budget class")
        print("=" * 100)
        print(f"{'thr':>6} {'acc':>8} {'large_prec':>12} {'large_rec':>12} {'large_f1':>12} {'pred_large%':>12}")

        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            p = (probs >= thr).astype(np.int64)
            acc_thr = accuracy_score(y_test, p)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_test,
                p,
                labels=[1],
                average=None,
                zero_division=0,
            )
            pred_large_pct = p.mean() * 100.0
            print(
                f"{thr:>6.2f} "
                f"{acc_thr:>8.4f} "
                f"{pr[0]:>12.4f} "
                f"{rc[0]:>12.4f} "
                f"{f1[0]:>12.4f} "
                f"{pred_large_pct:>11.2f}%"
            )


if __name__ == "__main__":
    main()
