"""AdamW over trainable (non-frozen) parameters only."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def build_optimizer(
    cfg: Dict[str, Any],
    model: nn.Module,
) -> Optional[torch.optim.Optimizer]:
    """
    Build optimizer for trainable parameters only.

    This is especially important in the current thesis setup because:
    - the LLaVA backbone is frozen
    - only the answer head should be trainable in the official dense baseline
    """
    training_mode = cfg["training"]["mode"]

    if training_mode == "eval_only":
        return None

    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "adamw").lower()

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError(
            "No trainable parameters found when building optimizer. "
            "Check freeze settings and answer-head configuration."
        )

    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])

    if optimizer_name == "adamw":
        betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
        eps = float(optimizer_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    raise ValueError(f"Unsupported optimizer name: {optimizer_name}")