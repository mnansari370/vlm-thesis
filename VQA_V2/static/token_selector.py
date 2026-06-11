from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class StaticCLSAttentionTokenSelector(nn.Module):
    """
    Static visual token selector using CLS-attention scores from the CLIP vision encoder.

    Static here means:
    - token selection depends only on the image / vision encoder attention
    - token selection does NOT depend on the question

    Expected attention input shape:
        [B, num_heads, 1 + N, 1 + N]

    where:
        - token 0 is CLS
        - tokens 1..N are visual patch tokens

    Selection procedure:
    1. Take CLS-to-patch attention from the final vision layer
    2. Average across heads
    3. Rank patch tokens by score
    4. Keep top-K patch tokens
    """

    def __init__(self, keep_tokens: int):
        super().__init__()

        if keep_tokens is None or keep_tokens <= 0:
            raise ValueError(f"keep_tokens must be positive, got {keep_tokens}")

        self.keep_tokens = int(keep_tokens)

    def _compute_scores_from_attention(
        self,
        attentions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert final-layer CLS attention into one scalar score per patch token.

        Input:
            attentions: [B, H, 1+N, 1+N]

        Output:
            token_scores: [B, N]
        """
        if attentions.dim() != 4:
            raise ValueError(
                f"Expected attentions with shape [B, H, T, T], got {tuple(attentions.shape)}"
            )

        # CLS attends to patch tokens only: row 0, columns 1:
        cls_to_patch = attentions[:, :, 0, 1:]  # [B, H, N]

        # Average across heads -> one score per patch
        token_scores = cls_to_patch.mean(dim=1)  # [B, N]
        return token_scores

    def forward(
        self,
        visual_features: torch.Tensor,
        final_layer_attentions: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Inputs:
            visual_features:
                [B, N, D]
            final_layer_attentions:
                [B, H, 1+N, 1+N]
            token_mask:
                optional [B, N] boolean mask

        Returns:
            dict containing selected features, indices, scores, and token stats.
        """
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must have shape [B, N, D], got {tuple(visual_features.shape)}"
            )

        batch_size, num_tokens, feat_dim = visual_features.shape
        device = visual_features.device

        token_scores = self._compute_scores_from_attention(final_layer_attentions)

        if token_scores.shape[0] != batch_size or token_scores.shape[1] != num_tokens:
            raise ValueError(
                "Mismatch between visual feature shape and attention-derived token scores: "
                f"visual_features={tuple(visual_features.shape)}, token_scores={tuple(token_scores.shape)}"
            )

        if token_mask is None:
            token_mask = torch.ones(
                batch_size,
                num_tokens,
                dtype=torch.bool,
                device=device,
            )

        # Mask out invalid tokens if ever needed
        masked_scores = token_scores.masked_fill(~token_mask, float("-inf"))

        k = min(self.keep_tokens, num_tokens)

        topk_scores, topk_indices = torch.topk(
            masked_scores,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        )

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        selected_features = torch.gather(visual_features, dim=1, index=gather_indices)

        num_tokens_before = token_mask.sum(dim=1)
        num_tokens_after = torch.full(
            (batch_size,),
            fill_value=k,
            dtype=num_tokens_before.dtype,
            device=device,
        )
        retention_ratio = num_tokens_after.float() / num_tokens_before.clamp(min=1).float()

        return {
            "selected_features": selected_features,
            "selected_indices": topk_indices,
            "selected_scores": topk_scores,
            "all_scores": token_scores,
            "num_tokens_before": num_tokens_before,
            "num_tokens_after": num_tokens_after,
            "retention_ratio": retention_ratio,
            "token_mask": token_mask,
        }