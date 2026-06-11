"""Answer CE + budget/entropy/diversity regularizers for the dynamic model."""

from typing import Any, Dict

import torch


def extract_model_loss(outputs: Dict[str, Any]) -> torch.Tensor:
    if "predictions" not in outputs:
        raise KeyError("Model outputs must contain a 'predictions' key.")

    loss = outputs["predictions"].get("loss", None)
    if loss is None:
        raise ValueError("Model returned no total loss.")

    if not torch.is_tensor(loss):
        raise TypeError(f"Loss must be a torch.Tensor, got {type(loss)}")

    return loss
