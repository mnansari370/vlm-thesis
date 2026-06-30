"""
llava_textsim.py — fresh Dynamic-WHICH selector for LLaVA-1.5 (question-conditioned, fixed K).

COMPOSES StaticPrunedLlava to reuse its loaded backbone, multimodal projector, LLM embedding
table, the validated `_build_inputs` physical-removal path, and the honest generate/decode — so
the ONLY new code is the question-conditioned SCORING. No training, no answer head, and NO dense
LM prefill (scores come from CLIP-level features + the LLM embedding table only).

TWO DISTINCT TEXT INPUTS (do not conflate):
  * prompt_text   — the EXACT text fed to the VLM (== sample.model_input_text; for TextVQA this
                    includes the OCR block + instruction). Builds the chat prompt verbatim.
  * selector_text — the text used ONLY to score WHICH visual tokens to keep (== sample.question,
                    the raw question). Never alters the generation prompt.

Selectors (scored on selector_text):
  * textsim         : score_i = max_j cos(proj_visual_i, q_emb_j)
                      proj_visual = multimodal-projector(CLIP patch features)   [576, D_lm]
                      q_emb       = LLM-embedding-table(selector_text token ids) [Lq, D_lm]
  * textsim_cls_mix : alpha · minmax(textsim) + (1-alpha) · minmax(cls_attn)
                      cls_attn = CLIP CLS→patch attention at vision_feature_layer (image-only).

Selection keeps top-K then SORTS indices back into original visual order (critical rule). bs=1.
At keep_k == 576 the top-K is the identity → reproduces dense (structural equivalence hook).

Run in vlm_env (torch 2.3.0+cu121, transformers 4.46.3). GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.static.static import StaticPrunedLlava, HONEST_SUFFIX, N_PATCHES

VALID_SELECTORS = ("textsim", "textsim_cls_mix")


class LlavaDynamicWhich:
    def __init__(self, selector: str = "textsim", keep_k: int = 288, alpha: float = 0.5,
                 add_instruction: bool = True, max_new_tokens: int = 64):
        if selector not in VALID_SELECTORS:
            raise ValueError(f"selector must be one of {VALID_SELECTORS}, got {selector!r}")
        self.selector = selector
        self.keep_k = int(keep_k)
        self.alpha = float(alpha)
        self.add_instruction = add_instruction
        self.max_new_tokens = max_new_tokens
        # cls_attn base loads the model once; image_pad + honest match the dense/static protocol.
        # keep_k is validated against the (now 86/202-extended) SUPPORTED_K allow-list.
        self.base = StaticPrunedLlava(method="cls_attn", keep_k=self.keep_k, image_pad=True,
                                      honest=True, append_suffix=add_instruction)
        self.suffix = HONEST_SUFFIX if add_instruction else ""

    @staticmethod
    def _minmax(x: torch.Tensor) -> torch.Tensor:
        lo = x.min()
        hi = x.max()
        return (x - lo) / (hi - lo + 1e-8)

    @torch.no_grad()
    def _question_embeds(self, selector_text: str) -> torch.Tensor:
        """Selector-text token embeddings from the LLM embedding table (raw text; no template/image).
        `selector_text` is sample.question for the primary method — NOT the VLM prompt."""
        b = self.base
        ids = b.processor.tokenizer(selector_text.strip(), return_tensors="pt",
                                    add_special_tokens=False).input_ids.to(b.backbone.device)
        return b._embed(ids)[0].float()                    # [Lq, D_lm]

    @torch.no_grad()
    def generate_one(self, image, prompt_text: str, selector_text: str) -> str:
        """bs=1 greedy decode with question-conditioned top-K visual selection.

        prompt_text   : EXACT VLM text (== sample.model_input_text) — builds the chat prompt.
        selector_text : text scored to choose WHICH tokens (== sample.question) — never in the prompt.
        Returns the honest-post-processed answer string."""
        b = self.base
        dev = b.backbone.device

        # ── inputs (pad-to-square + honest suffix), identical preprocessing to dense/static ──
        img = b._pad_to_square(image)
        text = prompt_text.strip() + self.suffix
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = b.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = b.processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        image_positions = [(inp["input_ids"][0] == b._image_token_id).nonzero(as_tuple=True)[0]]

        # ── CLIP features (+ attentions only when the cls mix needs them) ──
        pv = inp["pixel_values"].to(b.backbone.dtype)
        need_attn = (self.selector == "textsim_cls_mix")
        clip_out = b._vt(pv, output_attentions=need_attn, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[b._vis_layer][:, 1:, :]        # [1, 576, D_clip]

        # ── question-conditioned scoring (no LM forward), on selector_text only ──
        projected = b._proj(raw_feat)[0].float()                        # [576, D_lm]
        q_emb = self._question_embeds(selector_text)                    # [Lq, D_lm]
        sim = F.normalize(projected, dim=-1) @ F.normalize(q_emb, dim=-1).t()   # [576, Lq]
        textsim = sim.max(dim=1).values                                 # [576]
        if self.selector == "textsim":
            score = textsim
        else:  # textsim_cls_mix
            attn = clip_out.attentions[b._vis_layer]                    # [1, h, 577, 577]
            cls = attn[:, :, 0, 1:].mean(dim=1)[0].float()              # [576]
            score = self.alpha * self._minmax(textsim) + (1.0 - self.alpha) * self._minmax(cls)

        # ── top-K, then SORT back into original visual order ──
        k = max(1, min(self.keep_k, N_PATCHES))
        keep = score.topk(k).indices.sort().values                      # [K], ascending
        b.keep_k = k                                                    # sizes _build_inputs' vis_mask

        inputs_embeds, attention_mask = b._build_inputs(inp, [keep], image_positions, raw_feat)
        out = b.backbone.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=b.processor.tokenizer.eos_token_id,
        )
        text_out = b.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return text_out.strip().rstrip(".").lower()                     # honest post-processing
