"""
llava_dynamic_count.py — LLaVA-1.5 Dynamic-COUNT wrapper: probe pass with confidence recording +
second pass at an ARBITRARY integer K_i.

Mirrors the FROZEN StaticPrunedLlava cls_attn generation path EXACTLY (same preprocessing, prompt
template, scoring, top-K sorted selection, `_build_inputs` physical removal, honest greedy decode,
honest post-processing). The ONLY additions are (a) `output_scores=True` during generate — which
records per-step probabilities without changing greedy decisions — and (b) a per-call keep_k for
arbitrary K_i (the wrapper pattern validated in the WHICH phase; static.py is NOT modified and
official static results are unaffected).

REPRODUCTION GATE: probe predictions at p15/p25 must equal the frozen static finals byte-for-byte;
the probe runner checks this per sample and refuses to proceed on mismatch.

Run in vlm_env. GPU.
"""

from __future__ import annotations

import torch

from src.models.static.static import StaticPrunedLlava, HONEST_SUFFIX, N_PATCHES
from src.pruning.dynamic_count.confidence import confidence_signals


class LlavaDynamicCount:
    def __init__(self, keep_k_init: int = 288, add_instruction: bool = True,
                 max_new_tokens: int = 64):
        # cls_attn base; image_pad + honest match the dense/static protocol exactly.
        self.base = StaticPrunedLlava(method="cls_attn", keep_k=keep_k_init, image_pad=True,
                                      honest=True, append_suffix=add_instruction)
        self.suffix = HONEST_SUFFIX if add_instruction else ""
        self.max_new_tokens = max_new_tokens
        self._eos = self.base.processor.tokenizer.eos_token_id

    @torch.no_grad()
    def _prepare(self, image, prompt_text: str):
        """Identical preprocessing + CLS-attn scoring to StaticPrunedLlava.generate_answers."""
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
        clip_out = b._vt(pv, output_attentions=True, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[b._vis_layer][:, 1:, :]          # [1, 576, D_clip]
        scores = clip_out.attentions[b._vis_layer][:, :, 0, 1:].mean(dim=1).float()  # [1, 576]
        return inp, image_positions, raw_feat, scores

    @torch.no_grad()
    def _generate(self, inp, image_positions, raw_feat, scores, k: int, *, want_conf: bool):
        """Top-k sorted selection → _build_inputs → greedy generate (frozen kwargs).
        Returns (pred, conf_signals_or_None)."""
        b = self.base
        k = max(1, min(int(k), N_PATCHES))
        keep = scores[0].topk(k).indices.sort().values                    # frozen selection rule
        b.keep_k = k                                                      # sizes _build_inputs
        inputs_embeds, attention_mask = b._build_inputs(inp, [keep], image_positions, raw_feat)
        gen_kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          max_new_tokens=self.max_new_tokens, do_sample=False,
                          pad_token_id=self._eos)
        if not want_conf:
            out = b.backbone.generate(**gen_kwargs)
            text = b.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            return text.strip().rstrip(".").lower(), None
        g = b.backbone.generate(**gen_kwargs, return_dict_in_generate=True, output_scores=True)
        seq = g.sequences[0]                                              # generated ids only
        text = b.processor.tokenizer.batch_decode(g.sequences, skip_special_tokens=True)[0]
        # per-step probabilities of the greedy-chosen tokens (recording only — decisions unchanged)
        step_probs, first_top1, first_top2, first_ent = [], 1.0, 0.0, 0.0
        for t, logit in enumerate(g.scores):
            p = torch.softmax(logit[0].float(), dim=-1)
            tok = int(seq[t])
            if t == 0:
                top2 = p.topk(2)
                first_top1 = float(top2.values[0])
                first_top2 = float(top2.values[1])
                first_ent = float(-(p * (p + 1e-12).log()).sum())
            if tok != self._eos:
                step_probs.append(float(p[tok]))
        conf = confidence_signals(first_top1, first_top2, first_ent, step_probs)
        return text.strip().rstrip(".").lower(), conf

    @torch.no_grad()
    def probe(self, image, prompt_text: str, k_probe: int):
        """Probe pass: returns (pred, conf_signals, saliency_scores[576 floats])."""
        inp, pos, feat, scores = self._prepare(image, prompt_text)
        pred, conf = self._generate(inp, pos, feat, scores, k_probe, want_conf=True)
        return pred, conf, scores[0].tolist()

    @torch.no_grad()
    def generate_at_k(self, image, prompt_text: str, k: int) -> str:
        """Second pass at an ARBITRARY integer K_i ∈ [1, 576]."""
        inp, pos, feat, scores = self._prepare(image, prompt_text)
        pred, _ = self._generate(inp, pos, feat, scores, k, want_conf=False)
        return pred
