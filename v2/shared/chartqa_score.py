"""ChartQA official 'relaxed accuracy': numeric answers correct within 5% relative
tolerance; text answers by exact (case-insensitive) match. One gold answer per question."""


def _to_float(s):
    s = str(s).strip().rstrip("%").replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def relaxed_correct(pred: str, gold: str, max_rel: float = 0.05) -> bool:
    g = _to_float(gold)
    if g is not None:
        p = _to_float(pred)
        if p is None:
            return False
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= max_rel
    return str(pred).strip().lower() == str(gold).strip().lower()


def score_chartqa(predictions: list[dict]) -> dict:
    """predictions: [{"pred_answer": str, "gold": str}, ...]  -> relaxed accuracy."""
    per = [relaxed_correct(p["pred_answer"], p["gold"]) for p in predictions]
    n = len(per)
    return {"accuracy_pct": round(100 * sum(per) / n, 2) if n else 0.0,
            "n": n, "per_sample": per}
