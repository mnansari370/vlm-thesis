from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class DenseTokenSelector(nn.Module):
    """
    Dense token selector for the dense baseline.

    It does not remove any visual tokens.
    It simply reports token statistics and returns the input unchanged.

    Note:
    -----
    The keep_cls_token flag is retained for future extensibility, but in the
    current dense LLaVA setup with vision_feature_select_strategy="default",
    the CLS token is already excluded upstream by the backbone feature
    selection strategy.
    """

    def __init__(self, keep_cls_token: bool = False):
        super().__init__()
        self.keep_cls_token = keep_cls_token

    def forward(
        self,
        visual_features: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must have shape [B, N, D], got {tuple(visual_features.shape)}"
            )

        batch_size, num_tokens, _ = visual_features.shape
        device = visual_features.device

        if token_mask is None:
            token_mask = torch.ones(
                batch_size,
                num_tokens,
                dtype=torch.bool,
                device=device,
            )

        selected_indices = torch.arange(
            num_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, num_tokens)

        num_tokens_before = token_mask.sum(dim=1)
        num_tokens_after = token_mask.sum(dim=1)
        retention_ratio = num_tokens_after.float() / num_tokens_before.clamp(min=1).float()

        return {
            "selected_features": visual_features,
            "selected_indices": selected_indices,
            "num_tokens_before": num_tokens_before,
            "num_tokens_after": num_tokens_after,
            "retention_ratio": retention_ratio,
            "token_mask": token_mask,
        }