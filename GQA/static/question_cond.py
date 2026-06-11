"""
Question-conditioned visual-token selector (frozen, training-free).

Tests the thesis claim: does ranking visual tokens by relevance to the QUESTION beat
image-only saliency (CLS-Attn) at aggressive pruning?

Selector (SparseVLM-style text->visual attention + FastV-style early-layer ranking):
  1. One full prefill of [system | 576 visual | question(+OCR) | instruction] with
     output_attentions (eager).
  2. At LLM layer L, take attention FROM the post-image text query positions TO the 576
     visual key positions, average over heads, average over the text queries -> [576] score.
  3. Keep top-K visual tokens by that score.
We sweep L and also offer last-token-only queries and a CLS+question FUSION.

This subclasses StaticPrunedLlava to reuse the validated CLIP forward, projector,
inputs_embeds assembly (_build_inputs), image_pad, and honest generation. The model is
loaded with eager attention (required for output_attentions). For the PROBE we score
once per sample and generate for several (selector, K) cells from that one scoring pass.

Cite: SparseVLM (2410.04417) text-guided selection; FastV (2403.06764) early-layer attn.
"""

import torch
import torch.nn.functional as F

from GQA.shared.static import (StaticPrunedLlava, HONEST_SUFFIX, PROMPT_SUFFIX,
                               N_PATCHES, MODEL_NAME)


class QuestionCondLlava(StaticPrunedLlava):
    def __init__(self, layers=(2, 5, 8), image_pad=True, honest=True,
                 append_suffix=False, model_name=MODEL_NAME, seed=42):
        # Force eager attention (needed for output_attentions). We pass method='fastv_style'
        # to the parent ONLY to trigger eager loading; we override all scoring/selection here.
        super().__init__(method="fastv_style", keep_k=144, seed=seed,
                         model_name=model_name, image_pad=image_pad, honest=honest,
                         append_suffix=append_suffix)
        self.layers = list(layers)

    @torch.no_grad()
    def score_sample(self, image, prompt):
        """
        One scoring pass for a single (image, prompt).
        Returns dict with:
          q_scores[L]    : [576] question->visual attention score (mean over text queries)
          last_scores[L] : [576] last-token->visual attention score
          cls_scores     : [576] CLS-Attn (VisionZip dominant) score
          raw_feat       : [1,576,D] CLIP features (for projection in generation)
          inp, image_positions : for _build_inputs
        """
        dev = next(self.backbone.parameters()).device
        imgs = [self._pad_to_square(image)] if self.image_pad else [image]
        suffix = (HONEST_SUFFIX if self.honest else PROMPT_SUFFIX) if self.append_suffix else ""
        conv = [[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt.strip() + suffix}]}]]
        text = [self.processor.apply_chat_template(conv[0], add_generation_prompt=True, tokenize=False)]
        inp = self.processor(text=text, images=imgs, return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        vis_pos = (inp["input_ids"][0] == self._image_token_id).nonzero(as_tuple=True)[0]
        seq = inp["input_ids"].shape[1]
        img_end = int(vis_pos[-1].item()) + 1
        q_pos = torch.arange(img_end, seq, device=dev)        # post-image text tokens
        last_pos = torch.tensor([seq - 1], device=dev)

        # CLIP features (for projection) + CLS-Attn score
        pv = inp["pixel_values"].to(self.backbone.dtype)
        clip_out = self._vt(pv, output_attentions=True, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[self._vis_layer][:, 1:, :]          # [1,576,D]
        cls_scores = clip_out.attentions[self._vis_layer][:, :, 0, 1:].mean(1)[0].float()  # [576]

        # LLM forward with attentions
        out = self.backbone(
            input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
            attention_mask=inp["attention_mask"], output_attentions=True, use_cache=False,
        )
        q_scores, last_scores = {}, {}
        for L in self.layers:
            attn = out.attentions[L][0]                       # [heads, seq, seq]
            # text queries -> visual keys
            a_q = attn[:, q_pos][:, :, vis_pos]               # [heads, n_q, 576]
            q_scores[L] = a_q.mean(dim=(0, 1)).float()        # [576]
            a_l = attn[:, last_pos][:, :, vis_pos]            # [heads, 1, 576]
            last_scores[L] = a_l.mean(dim=(0, 1)).float()     # [576]

        return {"q_scores": q_scores, "last_scores": last_scores, "cls_scores": cls_scores,
                "raw_feat": raw_feat, "inp": inp, "image_positions": [vis_pos]}

    @staticmethod
    def _norm(x):
        x = x - x.min()
        d = x.max()
        return x / d if d > 0 else x

    def keep_indices(self, scored, selector, keep_k, layer=None):
        """Return sorted top-K visual indices for a selector ∈
           {random, cls, qcond, qcond_last, fusion}."""
        dev = scored["cls_scores"].device
        if selector == "random":
            import random as _r
            rng = _r.Random(self.seed)
            return torch.tensor(sorted(rng.sample(range(N_PATCHES), keep_k)),
                                dtype=torch.long, device=dev)
        if selector == "cls":
            s = scored["cls_scores"]
        elif selector == "qcond":
            s = scored["q_scores"][layer]
        elif selector == "qcond_last":
            s = scored["last_scores"][layer]
        elif selector == "fusion":
            s = self._norm(scored["cls_scores"]) + self._norm(scored["q_scores"][layer])
        else:
            raise ValueError(selector)
        return s.topk(keep_k).indices.sort().values

    @torch.no_grad()
    def generate_from_keep(self, scored, keep_idx):
        """Physical removal + honest generation given precomputed keep indices."""
        self.keep_k = len(keep_idx)
        inputs_embeds, attn_mask = self._build_inputs(
            scored["inp"], [keep_idx], scored["image_positions"], scored["raw_feat"])
        gen = dict(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                   max_new_tokens=64, do_sample=False,
                   pad_token_id=self.processor.tokenizer.eos_token_id)
        if not self.honest:
            gen["min_new_tokens"] = 1
        out = self.backbone.generate(**gen)
        text = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip().rstrip(".").lower() if self.honest else text.strip()
