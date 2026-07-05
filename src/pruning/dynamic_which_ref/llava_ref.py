"""
llava_ref.py — CLEAN-ROOM Dynamic-WHICH reference for LLaVA-1.5 (independent of llava_textsim).

Composes the FROZEN StaticPrunedLlava (backbone, projector, embedding table, the validated
`_build_inputs` physical-removal, honest generate/decode). It re-derives the SCORING and SELECTION
from scratch and routes index selection through the shared pure helpers in
`src.pruning.dynamic_which_ref` (so the tests exercise the exact selection code the runner uses).

No import of src.pruning.dynamic_which.*. Training-free, no answer head, no dense LM prefill.
Selectors: textsim_ref, saliency_mix_ref, static_guarded_textsim_ref, coverage_guarded_textsim_ref.

Two text inputs: prompt_text (== sample.model_input_text, verbatim VLM prompt) and selector_text
(== sample.question, scored only). Selection = top-K then SORTED to original order. bs=1.

Run in vlm_env. GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.static.static import StaticPrunedLlava, HONEST_SUFFIX, N_PATCHES
from src.pruning.dynamic_which_ref import (
    allowed_ref_selectors, DEFAULT_ALPHA, DEFAULT_GUARD_FRAC,
    topk_sorted, normalized_mix, spatial_coverage_indices, guarded_select,
)


class LlavaWhichRef:
    def __init__(self, selector: str = "textsim_ref", keep_k: int = 288, alpha: float = DEFAULT_ALPHA,
                 guard_frac: float = DEFAULT_GUARD_FRAC, add_instruction: bool = True,
                 max_new_tokens: int = 64):
        if selector not in allowed_ref_selectors("llava15"):
            raise ValueError(f"selector must be one of {allowed_ref_selectors('llava15')}, got {selector!r}")
        self.selector = selector
        self.keep_k = int(keep_k)
        self.alpha = float(alpha)
        self.guard_frac = float(guard_frac)
        self.add_instruction = add_instruction
        self.max_new_tokens = max_new_tokens
        self.base = StaticPrunedLlava(method="cls_attn", keep_k=self.keep_k, image_pad=True,
                                      honest=True, append_suffix=add_instruction)
        self.suffix = HONEST_SUFFIX if add_instruction else ""

    # ── scoring (independent re-derivation) ──
    @torch.no_grad()
    def _textsim(self, projected: torch.Tensor, selector_text: str) -> list:
        """max_j cos(projected_visual_i, q_emb_j) as a CPU list [576]."""
        b = self.base
        ids = b.processor.tokenizer(selector_text.strip(), return_tensors="pt",
                                    add_special_tokens=False).input_ids.to(b.backbone.device)
        q_emb = b._embed(ids)[0].float()                                   # [Lq, D]
        sim = F.normalize(projected, dim=-1) @ F.normalize(q_emb, dim=-1).t()   # [576, Lq]
        return sim.max(dim=1).values.tolist()

    def _select(self, textsim: list, cls_saliency: list | None) -> list:
        """Return the kept indices (sorted, len == K) for the chosen selector — pure/CPU."""
        K = max(1, min(self.keep_k, N_PATCHES))
        if self.selector == "textsim_ref":
            return topk_sorted(textsim, K)
        if self.selector == "saliency_mix_ref":
            return topk_sorted(normalized_mix(textsim, cls_saliency, self.alpha), K)
        R = round(self.guard_frac * K)
        if self.selector == "static_guarded_textsim_ref":
            reserve = topk_sorted(cls_saliency, max(1, R)) if R > 0 else []
            return guarded_select(K, reserve, textsim)
        if self.selector == "coverage_guarded_textsim_ref":
            reserve = spatial_coverage_indices(R, N_PATCHES)
            return guarded_select(K, reserve, textsim)
        raise ValueError(self.selector)

    @torch.no_grad()
    def generate_one(self, image, prompt_text: str, selector_text: str) -> str:
        b = self.base
        dev = b.backbone.device
        img = b._pad_to_square(image)
        text = prompt_text.strip() + self.suffix
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = b.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = b.processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
        image_positions = [(inp["input_ids"][0] == b._image_token_id).nonzero(as_tuple=True)[0]]

        pv = inp["pixel_values"].to(b.backbone.dtype)
        need_cls = self.selector in ("saliency_mix_ref", "static_guarded_textsim_ref")
        clip_out = b._vt(pv, output_attentions=need_cls, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[b._vis_layer][:, 1:, :]          # [1, 576, D_clip]
        projected = b._proj(raw_feat)[0].float()                          # [576, D_lm]

        textsim = self._textsim(projected, selector_text)
        cls_saliency = None
        if need_cls:
            attn = clip_out.attentions[b._vis_layer]                      # [1, h, 577, 577]
            cls_saliency = attn[:, :, 0, 1:].mean(dim=1)[0].float().tolist()

        keep_idx = self._select(textsim, cls_saliency)                    # sorted CPU list, len K
        keep = torch.tensor(keep_idx, dtype=torch.long, device=dev)
        b.keep_k = len(keep_idx)                                          # sizes _build_inputs' vis_mask

        inputs_embeds, attention_mask = b._build_inputs(inp, [keep], image_positions, raw_feat)
        out = b.backbone.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=b.processor.tokenizer.eos_token_id,
        )
        text_out = b.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return text_out.strip().rstrip(".").lower()
