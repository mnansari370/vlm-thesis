from typing import Any, Dict

import torch
import torch.nn as nn


class QuestionConditionedTokenScorer(nn.Module):
    """
    Question-conditioned visual token scorer.

    Inputs
    ------
    visual_features:
        [B, N, Dv] CLIP patch features.

    question_feature:
        [B, Dq] question representation from the frozen LLM text embeddings.

    Output
    ------
    scores:
        [B, N] learned question-conditioned relevance scores.

    Design
    ------
    The scorer compares each visual patch with the question feature using:
        [v, q, v*q, |v-q|]

    This allows the scorer to learn interactions such as:
        "which patch is relevant to this question?"
    """

    def __init__(
        self,
        visual_dim: int,
        question_dim: int,
        shared_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if visual_dim <= 0:
            raise ValueError(f"visual_dim must be positive, got {visual_dim}")
        if question_dim <= 0:
            raise ValueError(f"question_dim must be positive, got {question_dim}")

        self.visual_proj = nn.Linear(visual_dim, shared_dim)
        self.question_proj = nn.Linear(question_dim, shared_dim)

        interaction_dim = shared_dim * 4

        self.scorer = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        question_feature: torch.Tensor,
    ) -> Dict[str, Any]:
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must be [B, N, Dv], got {tuple(visual_features.shape)}"
            )
        if question_feature.dim() != 2:
            raise ValueError(
                f"question_feature must be [B, Dq], got {tuple(question_feature.shape)}"
            )

        target_dtype = self.visual_proj.weight.dtype

        if visual_features.dtype != target_dtype:
            visual_features = visual_features.to(dtype=target_dtype)

        if question_feature.dtype != target_dtype:
            question_feature = question_feature.to(dtype=target_dtype)

        v = self.visual_proj(visual_features)       # [B, N, D]
        q = self.question_proj(question_feature)    # [B, D]

        q_expand = q.unsqueeze(1).expand(-1, v.size(1), -1)

        interaction = torch.cat(
            [
                v,
                q_expand,
                v * q_expand,
                torch.abs(v - q_expand),
            ],
            dim=-1,
        )

        scores = self.scorer(interaction).squeeze(-1)  # [B, N]

        return {
            "scores": scores,
            "visual_projected": v,
            "question_projected": q,
        }
