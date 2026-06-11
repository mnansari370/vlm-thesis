"""DynamicTokenSelector: learned/cls/cls_prior scoring + fixed/learned budget; soft (train) and hard (eval) selection."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_scorer import QuestionConditionedTokenScorer
from .budget_controller import BudgetController


def _normalize_per_sample(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
    return (scores - mean) / std


def _score_entropy(scores: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(scores, dim=1)
    log_probs = torch.log_softmax(scores, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


def _score_stats(scores: torch.Tensor) -> torch.Tensor:
    """
    Build simple per-sample statistics from token scores.

    Output:
        [B, 7]
    """
    mean = scores.mean(dim=1)
    std = scores.std(dim=1, unbiased=False)
    maxv = scores.max(dim=1).values
    minv = scores.min(dim=1).values

    k5 = min(5, scores.size(1))
    k10 = min(10, scores.size(1))

    top5 = scores.topk(k5, dim=1).values.mean(dim=1)
    top10 = scores.topk(k10, dim=1).values.mean(dim=1)

    entropy = _score_entropy(scores)

    return torch.stack([mean, std, maxv, minv, top5, top10, entropy], dim=1)


class DynamicTokenSelector(nn.Module):
    """
    Dynamic visual token selector.

    Scoring modes
    -------------
    - cls_only:     final_score = CLS attention score
    - learned_only: final_score = question-conditioned learned score
    - cls_prior:    final_score = CLS score + alpha * learned score

    Budget strategies
    -----------------
    - fixed:   keep_ratio fixed (e.g. 0.5 for K288)
    - learned: keep_ratio from BudgetController

    Selection modes
    ---------------
    - soft: differentiable soft weights, keeps all N tokens (training)
    - hard: hard top-K, passes only selected tokens (eval/inference)

    Notes for region grounding
    ---------------------------
    The forward() return dict exposes:
      - "learned_scores":  raw question-conditioned scores [B, N] (pre-CLS-mix)
      - "final_scores":    the scores actually used for ranking [B, N]
    The region-supervision loss (added in the model wrapper) attaches to
    "learned_scores", so grounding directly trains the question-conditioned
    scorer regardless of how CLS is mixed in.
    """

    def __init__(
        self,
        visual_dim: int,
        question_dim: int,
        shared_dim: int = 512,
        scorer_hidden_dim: int = 256,
        budget_hidden_dim: int = 256,
        dropout: float = 0.1,
        min_keep_tokens: int = 64,
        max_keep_tokens: int = 576,
        num_visual_tokens: int = 576,
        train_selection_mode: str = "soft",
        eval_selection_mode: str = "hard",
        scoring_mode: str = "learned_only",
        budget_strategy: str = "learned",
        fixed_keep_ratio: float = 0.5,
        cls_alpha: float = 0.2,
        soft_temperature: float = 0.10,
        question_type_emb_dim: int = 0,
        num_question_types: int = 4,
    ):
        super().__init__()

        self.visual_dim = int(visual_dim)
        self.question_dim = int(question_dim)
        self.num_visual_tokens = int(num_visual_tokens)

        self.min_keep_tokens = int(min_keep_tokens)
        self.max_keep_tokens = int(max_keep_tokens)

        self.train_selection_mode = str(train_selection_mode)
        self.eval_selection_mode = str(eval_selection_mode)

        self.scoring_mode = str(scoring_mode)
        self.budget_strategy = str(budget_strategy)

        self.fixed_keep_ratio = float(fixed_keep_ratio)
        self.cls_alpha = float(cls_alpha)
        self.soft_temperature = float(soft_temperature)

        valid_scoring = {"cls_only", "learned_only", "cls_prior"}
        if self.scoring_mode not in valid_scoring:
            raise ValueError(f"Unsupported scoring_mode={self.scoring_mode}, valid={valid_scoring}")

        valid_budget = {"fixed", "learned"}
        if self.budget_strategy not in valid_budget:
            raise ValueError(f"Unsupported budget_strategy={self.budget_strategy}, valid={valid_budget}")

        valid_select = {"soft", "hard"}
        if self.train_selection_mode not in valid_select:
            raise ValueError(f"Unsupported train_selection_mode={self.train_selection_mode}")
        if self.eval_selection_mode not in valid_select:
            raise ValueError(f"Unsupported eval_selection_mode={self.eval_selection_mode}")

        self.token_scorer = QuestionConditionedTokenScorer(
            visual_dim=visual_dim,
            question_dim=question_dim,
            shared_dim=shared_dim,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )

        self.budget_controller = BudgetController(
            question_dim=shared_dim,
            score_stats_dim=7,
            hidden_dim=budget_hidden_dim,
            dropout=dropout,
            min_keep_tokens=min_keep_tokens,
            max_keep_tokens=max_keep_tokens,
            num_visual_tokens=num_visual_tokens,
            question_type_emb_dim=question_type_emb_dim,
            num_question_types=num_question_types,
        )

    def _choose_selection_mode(self) -> str:
        return self.train_selection_mode if self.training else self.eval_selection_mode

    def _compute_cls_scores(self, final_layer_attentions: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """
        final_layer_attentions: [B, H, 1+N, 1+N]
        Returns CLS-to-patch scores [B, N]
        """
        if final_layer_attentions is None:
            raise ValueError("final_layer_attentions are required for CLS-based dynamic scoring.")

        if final_layer_attentions.dim() != 4:
            raise ValueError(
                f"final_layer_attentions must be [B,H,T,T], got {tuple(final_layer_attentions.shape)}"
            )

        cls_to_patch = final_layer_attentions[:, :, 0, 1 : 1 + num_tokens]
        cls_scores = cls_to_patch.mean(dim=1)
        return cls_scores

    def _compute_final_scores(
        self,
        visual_features: torch.Tensor,
        question_feature: torch.Tensor,
        final_layer_attentions: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        batch_size, num_tokens, _ = visual_features.shape

        learned_scores = None
        question_projected = None
        cls_scores = None

        # Always run the scorer when learned scores are needed for ranking,
        # for the learned budget controller, OR for region grounding.
        need_learned = (
            self.scoring_mode in {"learned_only", "cls_prior"}
            or self.budget_strategy == "learned"
        )

        if need_learned:
            scorer_out = self.token_scorer(
                visual_features=visual_features,
                question_feature=question_feature,
            )
            learned_scores = scorer_out["scores"]
            question_projected = scorer_out["question_projected"]

        if self.scoring_mode in {"cls_only", "cls_prior"}:
            cls_scores = self._compute_cls_scores(
                final_layer_attentions=final_layer_attentions,
                num_tokens=num_tokens,
            )

        if self.scoring_mode == "cls_only":
            final_scores = _normalize_per_sample(cls_scores)
            if question_projected is None:
                scorer_out = self.token_scorer(
                    visual_features=visual_features,
                    question_feature=question_feature,
                )
                question_projected = scorer_out["question_projected"]
                learned_scores = scorer_out["scores"]

        elif self.scoring_mode == "learned_only":
            final_scores = _normalize_per_sample(learned_scores)

        elif self.scoring_mode == "cls_prior":
            cls_norm = _normalize_per_sample(cls_scores)
            learned_norm = _normalize_per_sample(learned_scores)
            final_scores = cls_norm + self.cls_alpha * learned_norm
            final_scores = _normalize_per_sample(final_scores)

        else:
            raise RuntimeError(f"Unexpected scoring_mode={self.scoring_mode}")

        return {
            "final_scores": final_scores,
            "learned_scores": learned_scores,
            "cls_scores": cls_scores,
            "question_projected": question_projected,
        }

    def _compute_keep_ratio(
        self,
        final_scores: torch.Tensor,
        question_projected: torch.Tensor,
        question_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size = final_scores.size(0)
        device = final_scores.device

        if self.budget_strategy == "fixed":
            keep_ratio = torch.full(
                (batch_size,),
                fill_value=self.fixed_keep_ratio,
                dtype=final_scores.dtype,
                device=device,
            )
            budget_gate = keep_ratio.clone()
            budget_logits = torch.zeros_like(keep_ratio)
        else:
            stats = _score_stats(final_scores)
            budget_out = self.budget_controller(
                question_projected=question_projected,
                score_stats=stats,
                question_type_ids=question_type_ids,
            )
            keep_ratio = budget_out["keep_ratio"]
            budget_gate = budget_out["budget_gate"]
            budget_logits = budget_out["budget_logits"]

        min_ratio = self.min_keep_tokens / float(self.num_visual_tokens)
        max_ratio = self.max_keep_tokens / float(self.num_visual_tokens)
        keep_ratio = keep_ratio.clamp(min=min_ratio, max=max_ratio)

        k_float = keep_ratio * float(self.num_visual_tokens)
        k = torch.round(k_float).long()
        k = k.clamp(min=self.min_keep_tokens, max=self.max_keep_tokens)

        return {
            "keep_ratio": keep_ratio,
            "k": k,
            "budget_gate": budget_gate,
            "budget_logits": budget_logits,
        }

    def _soft_select(
        self,
        visual_features: torch.Tensor,
        final_scores: torch.Tensor,
        k: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Soft differentiable selection: keep all N tokens, scale by sigmoid weights.
        """
        batch_size, num_tokens, _ = visual_features.shape
        thresholds = []

        for i in range(batch_size):
            ki = int(k[i].item())
            ki = max(1, min(ki, num_tokens))
            sorted_scores = torch.sort(final_scores[i], descending=True).values
            threshold = sorted_scores[ki - 1]
            thresholds.append(threshold)

        threshold = torch.stack(thresholds, dim=0)  # [B]

        soft_weights = torch.sigmoid(
            (final_scores - threshold.unsqueeze(1)) / max(self.soft_temperature, 1e-6)
        )

        selected_features = visual_features * soft_weights.unsqueeze(-1)

        selected_attention_mask = torch.ones(
            batch_size,
            num_tokens,
            dtype=torch.long,
            device=visual_features.device,
        )

        return {
            "selected_features": selected_features,
            "selected_attention_mask": selected_attention_mask,
            "selected_indices": torch.arange(num_tokens, device=visual_features.device).unsqueeze(0).expand(batch_size, -1),
            "selected_scores": final_scores,
            "soft_weights": soft_weights,
            "threshold": threshold,
            "actual_k": k,
        }

    def _hard_select(
        self,
        visual_features: torch.Tensor,
        final_scores: torch.Tensor,
        k: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Hard top-K selection. Supports variable K by padding to max K in batch.
        """
        batch_size, num_tokens, feat_dim = visual_features.shape
        device = visual_features.device

        max_k = int(k.max().item())
        max_k = max(1, min(max_k, num_tokens))

        selected_list = []
        index_list = []
        score_list = []
        mask_list = []

        for i in range(batch_size):
            ki = int(k[i].item())
            ki = max(1, min(ki, num_tokens))

            topk_scores, topk_idx = torch.topk(
                final_scores[i],
                k=ki,
                largest=True,
                sorted=True,
            )

            sorted_idx, order = torch.sort(topk_idx)
            sorted_scores = topk_scores[order]

            selected = visual_features[i].index_select(dim=0, index=sorted_idx)

            pad_len = max_k - ki
            if pad_len > 0:
                pad_feat = torch.zeros(pad_len, feat_dim, dtype=selected.dtype, device=device)
                pad_idx = torch.full((pad_len,), fill_value=-1, dtype=sorted_idx.dtype, device=device)
                pad_scores = torch.zeros(pad_len, dtype=sorted_scores.dtype, device=device)
                mask = torch.cat(
                    [
                        torch.ones(ki, dtype=torch.long, device=device),
                        torch.zeros(pad_len, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                selected = torch.cat([selected, pad_feat], dim=0)
                sorted_idx = torch.cat([sorted_idx, pad_idx], dim=0)
                sorted_scores = torch.cat([sorted_scores, pad_scores], dim=0)
            else:
                mask = torch.ones(ki, dtype=torch.long, device=device)

            selected_list.append(selected)
            index_list.append(sorted_idx)
            score_list.append(sorted_scores)
            mask_list.append(mask)

        selected_features = torch.stack(selected_list, dim=0)
        selected_indices = torch.stack(index_list, dim=0)
        selected_scores = torch.stack(score_list, dim=0)
        selected_attention_mask = torch.stack(mask_list, dim=0)

        return {
            "selected_features": selected_features,
            "selected_attention_mask": selected_attention_mask,
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
            "soft_weights": None,
            "threshold": torch.zeros(batch_size, dtype=final_scores.dtype, device=device),
            "actual_k": k,
        }

    def forward(
        self,
        visual_features: torch.Tensor,
        question_feature: torch.Tensor,
        final_layer_attentions: Optional[torch.Tensor] = None,
        selection_mode: Optional[str] = None,
        question_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must be [B,N,D], got {tuple(visual_features.shape)}"
            )

        batch_size, num_tokens, _ = visual_features.shape

        if selection_mode is None:
            selection_mode = self._choose_selection_mode()

        score_out = self._compute_final_scores(
            visual_features=visual_features,
            question_feature=question_feature,
            final_layer_attentions=final_layer_attentions,
        )

        final_scores = score_out["final_scores"]
        question_projected = score_out["question_projected"]

        budget_out = self._compute_keep_ratio(
            final_scores=final_scores,
            question_projected=question_projected,
            question_type_ids=question_type_ids,
        )

        keep_ratio = budget_out["keep_ratio"]
        k = budget_out["k"]

        if selection_mode == "soft":
            select_out = self._soft_select(
                visual_features=visual_features,
                final_scores=final_scores,
                k=k,
            )
        elif selection_mode == "hard":
            select_out = self._hard_select(
                visual_features=visual_features,
                final_scores=final_scores,
                k=k,
            )
        else:
            raise ValueError(f"Unsupported selection_mode={selection_mode}")

        num_tokens_before = torch.full(
            (batch_size,),
            fill_value=num_tokens,
            dtype=torch.long,
            device=visual_features.device,
        )

        num_tokens_after = k.to(device=visual_features.device)
        retention_ratio = num_tokens_after.float() / num_tokens_before.float().clamp(min=1)

        entropy = _score_entropy(final_scores)

        return {
            "selected_features": select_out["selected_features"],
            "selected_attention_mask": select_out["selected_attention_mask"],
            "selected_indices": select_out["selected_indices"],
            "selected_scores": select_out["selected_scores"],
            "soft_weights": select_out["soft_weights"],
            "budget_threshold": select_out["threshold"],

            "final_scores": final_scores,
            "learned_scores": score_out["learned_scores"],
            "cls_scores": score_out["cls_scores"],
            "score_entropy": entropy,

            "soft_keep_ratio": keep_ratio,
            "budget_gate": budget_out["budget_gate"],
            "budget_logits": budget_out["budget_logits"],

            "num_tokens_before": num_tokens_before,
            "num_tokens_after": num_tokens_after,
            "retention_ratio": retention_ratio,
            "selection_mode": selection_mode,
        }