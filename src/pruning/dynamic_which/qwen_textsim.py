"""
qwen_textsim.py — fresh Dynamic-WHICH selector for Qwen2.5-VL (question-conditioned, fixed K).

COMPOSES QwenPruner to reuse its validated encode + pruned-greedy seams (`_encode` →
`generate_keep`); only the SCORING is new. No training, no answer head, and NO dense LM prefill
(unlike QwenPruner.textattn which forwards the decoder for attention — textsim does NOT).

TWO DISTINCT TEXT INPUTS (do not conflate):
  * prompt_text   — the EXACT text fed to the VLM (== sample.model_input_text). Drives `_encode`.
  * selector_text — the text used ONLY to score WHICH visual tokens (== sample.question, the raw
                    question). Never alters the generation prompt.

Selectors (scored on selector_text):
  * textsim          : score_i = max_j cos(visual_embed_i, q_emb_j)
                       visual_embed = merged Qwen visual embeds (ViT+merger)   [nvis, D]
                       q_emb        = LLM-embedding-table(selector_text tokens) [Lq, D]
  * textsim_norm_mix : alpha · minmax(textsim) + (1-alpha) · minmax(visual L2 norm)

Selection keeps top-K then SORTS indices back into original visual order (critical rule). bs=1.
K is supplied per sample by the caller (clamp(round(budget% · dense_nvis), 1, dense_nvis)). At
K == nvis the top-K is the identity → reproduces dense (structural equivalence hook).

Run in qwen_env (transformers 4.51, qwen_vl_utils). GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner

VALID_SELECTORS = ("textsim", "textsim_norm_mix")


class QwenDynamicWhich:
    def __init__(self, selector: str = "textsim", alpha: float = 0.5, max_new_tokens: int = 64):
        if selector not in VALID_SELECTORS:
            raise ValueError(f"selector must be one of {VALID_SELECTORS}, got {selector!r}")
        self.selector = selector
        self.alpha = float(alpha)
        self.max_new_tokens = max_new_tokens
        self.model = QwenPruner()

    @property
    def proc(self):
        return self.model.proc

    @staticmethod
    def _minmax(x: torch.Tensor) -> torch.Tensor:
        lo = x.min()
        hi = x.max()
        return (x - lo) / (hi - lo + 1e-8)

    @torch.no_grad()
    def _question_embeds(self, selector_text: str) -> torch.Tensor:
        """Selector-text token embeddings from the LLM embedding table (raw text; no template/image).
        `selector_text` is sample.question for the primary method — NOT the VLM prompt."""
        m = self.model
        ids = m.proc.tokenizer(selector_text.strip(), return_tensors="pt",
                               add_special_tokens=False).input_ids.to(m.dev)
        return m.embed(ids)[0].float()                     # [Lq, D]

    @torch.no_grad()
    def generate_one(self, image, prompt_text: str, selector_text: str, k: int, instr: bool = True):
        """bs=1 greedy decode keeping top-K question-conditioned visual tokens. Returns (pred, nkeep).

        prompt_text   : EXACT VLM text (== sample.model_input_text) — drives the encode/prompt.
        selector_text : text scored to choose WHICH tokens (== sample.question) — never in the prompt.
        `k` is the per-sample budget (the caller clamps it to dense_nvis)."""
        m = self.model
        emb, pos, attn, v0, nvis, vis = m._encode(image, prompt_text, instr)
        visf = vis.float()                                 # [nvis, D]

        q_emb = self._question_embeds(selector_text)       # [Lq, D]
        sim = F.normalize(visf, dim=-1) @ F.normalize(q_emb, dim=-1).t()    # [nvis, Lq]
        textsim = sim.max(dim=1).values                    # [nvis]
        if self.selector == "textsim":
            score = textsim
        else:  # textsim_norm_mix
            norm = visf.norm(dim=-1)                        # [nvis]
            score = self.alpha * self._minmax(textsim) + (1.0 - self.alpha) * self._minmax(norm)

        k = max(1, min(int(k), nvis))
        keep_local = score.topk(k).indices.sort().values   # [K], ascending (sorted to original order)
        # reuse the validated pruned-greedy path (no dense LM prefill was done for scoring)
        return m.generate_keep(emb, pos, attn, v0, nvis, keep_local, self.max_new_tokens)
