"""Linear-warmup + cosine-decay LR schedule."""

import math
from typing import Any, Dict, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    cfg: Dict[str, Any],
    optimizer: Optional[Optimizer],
    num_training_steps: int,
):
    """
    Build LR scheduler.

    Supported:
    - cosine
    - none
    """
    if optimizer is None:
        return None

    if num_training_steps <= 0:
        return None

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "none").lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))
        min_lr = float(scheduler_cfg.get("min_lr", 0.0))
        warmup_steps = int(num_training_steps * warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
            base_lr = float(cfg["training"]["learning_rate"])
            floor_ratio = min_lr / max(base_lr, 1e-12)
            return max(floor_ratio, cosine_value)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler name: {scheduler_name}")