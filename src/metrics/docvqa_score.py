"""DocVQA official ANLS (Average Normalized Levenshtein Similarity).
Per question: NLS = max over gold answers of 1 - lev(pred,gold)/max(len); if NLS < tau
(0.5) it is zeroed. Aggregate = mean. Case-insensitive, stripped."""


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    p = str(pred).strip().lower()
    best = 0.0
    for g in golds:
        g = str(g).strip().lower()
        m = max(len(p), len(g))
        nls = 1.0 if m == 0 else 1 - _lev(p, g) / m
        best = max(best, nls)
    return best if best >= tau else 0.0


def score_docvqa(predictions: list[dict]) -> dict:
    """predictions: [{"pred_answer": str, "golds": [str,...]}, ...] -> ANLS%."""
    per = [anls(p["pred_answer"], p["golds"]) for p in predictions]
    n = len(per)
    return {"anls_pct": round(100 * sum(per) / n, 2) if n else 0.0,
            "n": n, "per_sample": per}
