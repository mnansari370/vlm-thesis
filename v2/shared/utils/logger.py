import os
from datetime import datetime


def make_output_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create timestamped output directory.

    Example:
      outputs/llava_dense_debug_20260401_153012
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir