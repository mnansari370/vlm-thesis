"""
qwen_ref.py — CLEAN-ROOM Dynamic-WHICH reference for Qwen2.5-VL (independent of qwen_textsim).

Composes the FROZEN QwenPruner (its `_encode` → `generate_keep` seams). Re-derives SCORING and
SELECTION from scratch and routes index selection through the shared pure helpers in
`src.pruning.dynamic_which_ref`. No import of src.pruning.dynamic_which.*. Training-free, no answer
head, and NO dense LM prefill (scores use the ViT visual embeds + the embedding table only).

Selectors: textsim_ref, saliency_mix_ref (norm), static_guarded_textsim_ref.
coverage_guarded_textsim_ref is NOT supported for Qwen: its visual tokens are a variable-length,
per-image merged grid with no stable exposed 2D layout, so an evenly-spaced "spatial" reserve is
not well defined here (a 1-D sequence stride would be misleading). Use static_guarded instead.

Two text inputs: prompt_text (== sample.model_input_text, drives `_encode`) and selector_text
(== sample.question, scored only). K supplied per sample by the caller. Selection = top-K then
SORTED to original order. bs=1. Run in qwen_env. GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.pruning.dynamic_which_ref import (
    allowed_ref_selectors, DEFAULT_ALPHA, DEFAULT_GUARD_FRAC,
    topk_sorted, normalized_mix, guarded_select,
)

_UNSUPPORTED = "coverage_guarded_textsim_ref"


class QwenWhichRef:
    def __init__(self, selector: str = "textsim_ref", alpha: float = DEFAULT_ALPHA,
                 guard_frac: float = DEFAULT_GUARD_FRAC, max_new_tokens: int = 64):
        if selector == _UNSUPPORTED:
            raise ValueError("coverage_guarded_textsim_ref is not available for Qwen "
                             "(variable per-image visual grid; no stable 2D layout). "
                             "Use static_guarded_textsim_ref.")
        if selector not in allowed_ref_selectors("qwen25vl7b"):
            raise ValueError(f"selector must be one of {allowed_ref_selectors('qwen25vl7b')}, got {selector!r}")
        self.selector = selector
        self.alpha = float(alpha)
        self.guard_frac = float(guard_frac)
        self.max_new_tokens = max_new_tokens
        self.model = QwenPruner()

    @property
    def proc(self):
        return self.model.proc

    @torch.no_grad()
    def _textsim(self, visf: torch.Tensor, selector_text: str) -> list:
        m = self.model
        ids = m.proc.tokenizer(selector_text.strip(), return_tensors="pt",
                               add_special_tokens=False).input_ids.to(m.dev)
        q_emb = m.embed(ids)[0].float()                                    # [Lq, D]
        sim = F.normalize(visf, dim=-1) @ F.normalize(q_emb, dim=-1).t()   # [nvis, Lq]
        return sim.max(dim=1).values.tolist()

    def _select(self, k: int, textsim: list, norm_saliency: list) -> list:
        k = max(1, min(int(k), len(textsim)))
        if self.selector == "textsim_ref":
            return topk_sorted(textsim, k)
        if self.selector == "saliency_mix_ref":
            return topk_sorted(normalized_mix(textsim, norm_saliency, self.alpha), k)
        if self.selector == "static_guarded_textsim_ref":
            R = round(self.guard_frac * k)
            reserve = topk_sorted(norm_saliency, max(1, R)) if R > 0 else []
            return guarded_select(k, reserve, textsim)
        raise ValueError(self.selector)

    @torch.no_grad()
    def generate_one(self, image, prompt_text: str, selector_text: str, k: int, instr: bool = True):
        """bs=1 greedy decode keeping the ref-selected top-K visual tokens. Returns (pred, nkeep)."""
        m = self.model
        emb, pos, attn, v0, nvis, vis = m._encode(image, prompt_text, instr)
        visf = vis.float()                                                 # [nvis, D]
        textsim = self._textsim(visf, selector_text)
        norm_saliency = visf.norm(dim=-1).tolist()                         # static Qwen saliency

        keep_idx = self._select(k, textsim, norm_saliency)                 # sorted CPU list
        keep_local = torch.tensor(keep_idx, dtype=torch.long, device=m.dev)
        return m.generate_keep(emb, pos, attn, v0, nvis, keep_local, self.max_new_tokens)
