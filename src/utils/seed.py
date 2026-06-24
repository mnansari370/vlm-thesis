import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: Optional[bool] = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed:
            Global random seed.
        deterministic:
            If True, enable deterministic PyTorch/cuDNN behavior where possible.
            This may reduce performance but improves reproducibility.

    Notes:
        - The function is backward compatible with earlier usage:
              set_seed(seed)
        - To fully align with your config, training code can later call:
              set_seed(cfg["seed"], cfg["system"]["deterministic"])
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True