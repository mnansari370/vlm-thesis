"""Official VQA consensus accuracy: min(1, matches/3) over the 10 human answers."""

from typing import Any, Dict, List, Optional

from VQA_V2_early_proxy.shared.datasets.vqav2_answers import normalize_answer


def normalize_predicted_answer(answer: Optional[str]) -> str:
    """
    Normalize predicted answer text before VQA-style comparison.
    """
    if answer is None:
        return ""
    return normalize_answer(answer)


def compute_vqa_consensus_score(
    pred_answer: Optional[str],
    gt_answers: List[str],
) -> float:
    """
    Lightweight VQA-style consensus score:

        score = min(1.0, matches / 3.0)

    where matches is the number of annotator answers equal to the normalized
    predicted answer.

    This is a practical VQA-style approximation and is consistent with the
    current training code.
    """
    pred = normalize_predicted_answer(pred_answer)
    normalized_gt = [normalize_answer(a) for a in gt_answers]

    matches = sum(1 for ans in normalized_gt if ans == pred)
    return min(1.0, matches / 3.0)


def compute_average_vqa_accuracy(predictions: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute mean VQA-style accuracy from a saved predictions list.

    Expected prediction format:
    {
        "pred_answer": ...,
        "raw_answers": [...],
        ...
    }
    """
    if predictions is None or len(predictions) == 0:
        return None

    scores = []
    for item in predictions:
        pred_answer = item.get("pred_answer", "")
        raw_answers = item.get("raw_answers", [])
        scores.append(compute_vqa_consensus_score(pred_answer, raw_answers))

    if len(scores) == 0:
        return None

    return float(sum(scores) / len(scores))