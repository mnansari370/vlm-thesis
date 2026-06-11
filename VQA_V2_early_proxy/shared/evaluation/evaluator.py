"""Aggregate evaluation helpers over saved predictions."""

from typing import Any, Dict, List, Optional

from .flops import estimate_attention_flops_from_summary
from .latency import summarize_latency_metrics
from .token_stats import summarize_token_stats
from .vqa_metrics import compute_average_vqa_accuracy


def _infer_backbone_dimensions(model) -> Dict[str, int]:
    """
    Infer hidden size and number of text-transformer layers from the current
    LLaVA backbone config.
    """
    model_config = model.model.config
    text_config = getattr(model_config, "text_config", None)

    if text_config is not None:
        hidden_size = int(getattr(text_config, "hidden_size"))
        num_layers = int(getattr(text_config, "num_hidden_layers"))
    else:
        hidden_size = int(getattr(model_config, "hidden_size"))
        num_layers = int(getattr(model_config, "num_hidden_layers"))

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }


def build_evaluation_report(
    model,
    split_name: str,
    metrics: Dict[str, Any],
    predictions: Optional[List[Dict[str, Any]]] = None,
    latency_metrics: Optional[Dict[str, Any]] = None,
    compute_flops: bool = True,
) -> Dict[str, Any]:
    """
    Build a standardized evaluation report from validation/eval outputs.

    Parameters
    ----------
    model:
        Current instantiated model wrapper (used only for backbone metadata).
    split_name:
        "train" or "val" or another descriptive split label.
    metrics:
        Aggregated epoch/evaluation metrics dictionary.
    predictions:
        Optional saved predictions list.
    latency_metrics:
        Optional latency dictionary.
    compute_flops:
        Whether to include analytical attention FLOPs proxy.
    """
    token_summary = summarize_token_stats(metrics)

    evaluated_vqa_accuracy = metrics.get("vqa_accuracy", None)
    if evaluated_vqa_accuracy is None and predictions is not None:
        evaluated_vqa_accuracy = compute_average_vqa_accuracy(predictions)

    dims = _infer_backbone_dimensions(model)

    flops_summary = None
    if compute_flops:
        flops_summary = estimate_attention_flops_from_summary(
            metrics=metrics,
            hidden_size=dims["hidden_size"],
            num_layers=dims["num_layers"],
        )

    return {
        "split": split_name,
        "loss": metrics.get("loss"),
        "vqa_accuracy": evaluated_vqa_accuracy,
        "token_stats": token_summary,
        "flops": flops_summary,
        "latency": summarize_latency_metrics(latency_metrics),
        "backbone_summary": dims,
    }