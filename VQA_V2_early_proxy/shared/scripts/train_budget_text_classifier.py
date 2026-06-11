import argparse
import json
from collections import Counter

import numpy as np


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
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from scipy.sparse import hstack
    except Exception as e:
        raise SystemExit(
            "Missing sklearn/scipy. Install with: pip install scikit-learn scipy\n"
            f"Original error: {e}"
        )

    with open(args.labels, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    questions = [r["question"] for r in records]
    y = np.array([int(r["budget_class"]) for r in records], dtype=np.int64)
    qtypes = [infer_qtype(q) for q in questions]

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

    qtype_vocab = sorted(set(qtypes))
    qtype_to_id = {q: i for i, q in enumerate(qtype_vocab)}

    def make_qtype_matrix(indices):
        X = np.zeros((len(indices), len(qtype_vocab)), dtype=np.float32)
        for row, i in enumerate(indices):
            X[row, qtype_to_id[qtypes[i]]] = 1.0
        return X

    X_qtype_train = make_qtype_matrix(train_idx)
    X_qtype_test = make_qtype_matrix(test_idx)

    X_train = hstack([X_text_train, X_qtype_train])
    X_test = hstack([X_text_test, X_qtype_test])

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("=" * 100)
    print("Budget text classifier")
    print("=" * 100)
    print("Train class counts:", Counter(y_train.tolist()))
    print("Test class counts:", Counter(y_test.tolist()))

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

    unique_labels = sorted(set(y.tolist()))
    if unique_labels == [0, 1]:
        names = ["small_144", "large_288_432_576"]
    elif unique_labels == [0, 1, 2]:
        names = ["small_144", "medium_288_432", "large_576"]
    else:
        names = [f"class_{i}" for i in unique_labels]

    print()
    print(f"Majority baseline accuracy: {majority_acc:.4f}")
    print(f"TFIDF + qtype classifier accuracy: {acc:.4f}")
    print()
    print("Predicted class counts:", Counter(pred.tolist()))
    print()
    print("Classification report:")
    print(classification_report(
        y_test,
        pred,
        labels=unique_labels,
        target_names=names,
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix rows=true, cols=pred:")
    print(confusion_matrix(y_test, pred, labels=unique_labels))


if __name__ == "__main__":
    main()
