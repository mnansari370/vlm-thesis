"""BudgetController: question embedding + score stats (+ optional type embedding) -> keep ratio in [K_min, K_max]/576."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BudgetController(nn.Module):
    """
    Predicts a continuous keep ratio for dynamic token pruning.

    This module is not used in Stage 1 CLS-only fixed-budget check.
    It is prepared for later stages.

    Output keep_ratio is mapped to:
        [min_keep_tokens / num_visual_tokens, max_keep_tokens / num_visual_tokens]
    """

    def __init__(
        self,
        question_dim: int,
        score_stats_dim: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        min_keep_tokens: int = 64,
        max_keep_tokens: int = 576,
        num_visual_tokens: int = 576,
        question_type_emb_dim: int = 0,
        num_question_types: int = 4,
    ):
        super().__init__()

        self.min_keep_tokens = int(min_keep_tokens)
        self.max_keep_tokens = int(max_keep_tokens)
        self.num_visual_tokens = int(num_visual_tokens)

        self.question_type_emb_dim = int(question_type_emb_dim)

        if self.question_type_emb_dim > 0:
            self.question_type_embedding = nn.Embedding(
                int(num_question_types),
                self.question_type_emb_dim,
            )
        else:
            self.question_type_embedding = None

        input_dim = int(question_dim) + int(score_stats_dim) + self.question_type_emb_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        question_projected: torch.Tensor,
        score_stats: torch.Tensor,
        question_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        target_dtype = self.net[1].weight.dtype

        if question_projected.dtype != target_dtype:
            question_projected = question_projected.to(dtype=target_dtype)

        if score_stats.dtype != target_dtype:
            score_stats = score_stats.to(dtype=target_dtype)

        parts = [question_projected, score_stats]

        if self.question_type_embedding is not None:
            if question_type_ids is None:
                question_type_ids = torch.zeros(
                    question_projected.size(0),
                    dtype=torch.long,
                    device=question_projected.device,
                )
            q_type_vec = self.question_type_embedding(question_type_ids.to(question_projected.device))
            if q_type_vec.dtype != target_dtype:
                q_type_vec = q_type_vec.to(dtype=target_dtype)
            parts.append(q_type_vec)

        combined = torch.cat(parts, dim=-1)

        logits = self.net(combined).squeeze(-1)
        gate = torch.sigmoid(logits)

        min_ratio = self.min_keep_tokens / float(self.num_visual_tokens)
        max_ratio = self.max_keep_tokens / float(self.num_visual_tokens)

        keep_ratio = min_ratio + gate * (max_ratio - min_ratio)

        return {
            "keep_ratio": keep_ratio,
            "budget_gate": gate,
            "budget_logits": logits,
        }
