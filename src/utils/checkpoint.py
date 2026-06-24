from typing import Any, Dict, Optional

import torch


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """
    Load checkpoint dictionary from disk.
    """
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a dictionary.")
    return checkpoint
    