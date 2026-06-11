"""Cross-entropy answer loss (ignore_index=-1 for out-of-vocab answers)."""

from typing import Any, Dict

import torch


def extract_model_loss(outputs: Dict[str, Any]) -> torch.Tensor:
    """
    Extract loss from model outputs.

    Expected model output structure:
    outputs["predictions"]["loss"]
    """
    if "predictions" not in outputs:
        raise KeyError("Model outputs must contain a 'predictions' key.")

    loss = outputs["predictions"].get("loss", None)
    if loss is None:
        raise ValueError(
            "Model returned no loss. "
            "This is expected in generation/eval-only mode, "
            "but not in answer-head training mode."
        )

    if not torch.is_tensor(loss):
        raise TypeError(f"Loss must be a torch.Tensor, got {type(loss)}")

    return loss