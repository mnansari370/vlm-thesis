"""AnswerHeadMLP (identical across variants)."""

from typing import Optional

import torch
import torch.nn as nn


def _dtype_from_string(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None

    name = str(dtype_name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype string: {dtype_name}")


class AnswerHeadMLP(nn.Module):
    """
    Small MLP answer head for classification-mode dense baseline.

    This head sits on top of frozen LLaVA backbone features and predicts over
    the fixed answer vocabulary.

    Design notes
    ------------
    - LayerNorm is applied before the MLP to stabilize large frozen-backbone
      features.
    - Input features are cast to the same dtype as the head weights to avoid
      dtype mismatch errors (e.g. float16 backbone output vs float32 head).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        train_dtype: Optional[str] = "float32",
    ):
        super().__init__()

        if input_dim is None or input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim is None or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim is None or output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.input_norm = nn.LayerNorm(input_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        target_dtype = _dtype_from_string(train_dtype)
        if target_dtype is not None:
            self.input_norm = self.input_norm.to(dtype=target_dtype)
            self.net = self.net.to(dtype=target_dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        target_dtype = self.net[0].weight.dtype
        if features.dtype != target_dtype:
            features = features.to(dtype=target_dtype)

        features = self.input_norm(features)
        return self.net(features)