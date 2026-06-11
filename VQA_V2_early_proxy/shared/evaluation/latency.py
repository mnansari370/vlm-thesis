"""Wall-clock latency / throughput measurement helpers."""

from typing import Any, Dict, Optional


def summarize_latency_metrics(
    latency_metrics: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Standardize latency summary dictionary.
    """
    if latency_metrics is None:
        return None

    return {
        "avg_batch_time_sec": latency_metrics.get("avg_batch_time_sec"),
        "avg_sample_time_sec": latency_metrics.get("avg_sample_time_sec"),
        "throughput_samples_per_sec": latency_metrics.get("throughput_samples_per_sec"),
    }