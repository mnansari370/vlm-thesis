import torch


def get_default_device(use_cuda: bool = True) -> torch.device:
    """
    Return default device according to availability and config.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")