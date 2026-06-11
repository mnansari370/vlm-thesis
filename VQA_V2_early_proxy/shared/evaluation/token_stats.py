"""Token-count bookkeeping helpers (before/after selection, retention)."""

from typing import Any, Dict


def summarize_token_stats(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and standardize token-related statistics from validation/train
    summary metrics.

    Expected keys in metrics:
    - avg_num_visual_tokens_before
    - avg_num_visual_tokens_after
    - avg_retention_ratio
    - avg_text_length
    - avg_multimodal_sequence_length
    """
    return {
        "avg_num_visual_tokens_before": metrics.get("avg_num_visual_tokens_before"),
        "avg_num_visual_tokens_after": metrics.get("avg_num_visual_tokens_after"),
        "avg_retention_ratio": metrics.get("avg_retention_ratio"),
        "avg_text_length": metrics.get("avg_text_length"),
        "avg_multimodal_sequence_length": metrics.get("avg_multimodal_sequence_length"),
    }