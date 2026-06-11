"""
VisionZip (arXiv 2412.04467) — faithful re-implementation on our honest pipeline.

VisionZip = dominant token selection + contextual token MERGING. Our existing
static CLS-Attn is dominant-only; VisionZip additionally merges the non-dominant
("contextual") tokens, so it is NOT already covered.

Algorithm (paper Alg. 1 + 2):
  1. Dominant: rank the 576 patches by CLS->patch attention at CLIP layer -2
     (averaged over heads), keep the top K_dominant.
  2. Contextual: from the remaining (576 - K_dominant) patches, uniformly choose
     K_contextual "target" tokens; assign every other ("merge") token to its most
     similar target and average them in. This yields K_contextual merged tokens.
  3. Feed [K_dominant dominant + K_contextual merged] = keep_k visual tokens to the
     LLM (prune-before-LLM, same FLOPs basis as our static methods).

Split: keep_k = K_dominant + K_contextual. We use K_contextual = round(keep_k*10/64)
(matches the paper's documented 64-token config = 54 dominant + 10 contextual,
~15.6% contextual) and K_dominant = keep_k - K_contextual. Documented, not guessed.

Similarity basis: VisionZip's paper merges by attention KEY vectors. We merge by the
layer-(-2) patch FEATURES (cosine), which are the keys up to a linear projection —
faithful in mechanism, avoids a fragile key-hook. Flagged in docs.

This reuses StaticPrunedLlava's CLIP forward, projector, inputs_embeds assembly,
image_pad, honest prompt, and post-processing — so the only new logic is the merge.
"""

import torch
import torch.nn.functional as F

from GQA.shared.static import StaticPrunedLlava, HONEST_SUFFIX, PROMPT_SUFFIX, N_PATCHES


class VisionZipLlava(StaticPrunedLlava):
    def __init__(self, keep_k: int, k_contextual: int | None = None,
                 image_pad: bool = True, honest: bool = True,
                 model_name: str = "llava-hf/llava-1.5-7b-hf", seed: int = 42):
        # Parent loads with method='cls_attn' (we reuse its CLS-attention scoring).
        # Bypass parent's SUPPORTED_K check by allowing any keep_k for VisionZip.
        self._vz_keep_k = keep_k
        super().__init__(method="cls_attn", keep_k=288, seed=seed,
                         model_name=model_name, image_pad=image_pad, honest=honest)
        self.keep_k = keep_k  # override
        if k_contextual is None:
            k_contextual = max(1, round(keep_k * 10 / 64))
        self.k_contextual = int(k_contextual)
        self.k_dominant = int(keep_k - self.k_contextual)
        assert self.k_dominant > 0, f"k_dominant={self.k_dominant} must be >0"
        print(f"[VisionZip] keep_k={keep_k} = {self.k_dominant} dominant "
              f"+ {self.k_contextual} contextual", flush=True)

    @torch.no_grad()
    def _merge_contextual(self, feats: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        feats:  [576, D] CLIP layer-(-2) patch features for ONE sample.
        scores: [576]    CLS->patch attention for the same sample.
        Returns [keep_k, D] = dominant features ++ merged contextual features.
        """
        N, D = feats.shape
        dev  = feats.device

        # 1. Dominant: top-K_dominant by CLS attention
        dom_idx  = scores.topk(self.k_dominant).indices
        dom_mask = torch.zeros(N, dtype=torch.bool, device=dev)
        dom_mask[dom_idx] = True
        dom_feat = feats[dom_idx]                       # [K_d, D]

        rem_idx  = (~dom_mask).nonzero(as_tuple=True)[0]
        rem_feat = feats[rem_idx]                       # [R, D]
        R = rem_feat.shape[0]
        if self.k_contextual <= 0 or R == 0:
            return dom_feat
        if R <= self.k_contextual:
            # not enough remaining to merge; just append them all
            return torch.cat([dom_feat, rem_feat], dim=0)

        # 2. Uniformly pick K_contextual targets from the remaining tokens
        tgt_pos  = torch.linspace(0, R - 1, self.k_contextual, device=dev).round().long().unique()
        tgt_mask = torch.zeros(R, dtype=torch.bool, device=dev)
        tgt_mask[tgt_pos] = True
        tgt_feat   = rem_feat[tgt_mask]                 # [Kc, D]
        merge_feat = rem_feat[~tgt_mask]                # [M, D]
        Kc = tgt_feat.shape[0]

        if merge_feat.shape[0] == 0:
            return torch.cat([dom_feat, tgt_feat], dim=0)

        # 3. Assign each merge token to most similar target (cosine), average in
        sim    = F.normalize(merge_feat, dim=-1) @ F.normalize(tgt_feat, dim=-1).T  # [M, Kc]
        assign = sim.argmax(dim=-1)                                                  # [M]
        merged = tgt_feat.clone().float()
        counts = torch.ones(Kc, device=dev)
        merged.index_add_(0, assign, merge_feat.float())
        counts.index_add_(0, assign, torch.ones(merge_feat.shape[0], device=dev))
        merged = (merged / counts.unsqueeze(-1)).to(feats.dtype)                     # [Kc, D]

        return torch.cat([dom_feat, merged], dim=0)     # [K_d + Kc, D]

    @torch.no_grad()
    def generate_answers(self, images, questions, sample_offset: int = 0,
                         max_new_tokens: int = 64, return_confidence: bool = False):
        B   = len(questions)
        dev = next(self.backbone.parameters()).device

        if self.image_pad:
            images = [self._pad_to_square(img) for img in images]

        suffix = HONEST_SUFFIX if self.honest else PROMPT_SUFFIX
        convs = [[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q.strip() + suffix}]}]
            for q in questions]
        prompts = [self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                   for c in convs]
        inp = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        image_positions = []
        for i in range(B):
            pos = (inp["input_ids"][i] == self._image_token_id).nonzero(as_tuple=True)[0]
            image_positions.append(pos)

        # CLIP forward with attentions (for dominant scoring) + features (for merge)
        pv = inp["pixel_values"].to(self.backbone.dtype)
        clip_out = self._vt(pv, output_attentions=True, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[self._vis_layer][:, 1:, :]      # [B,576,D]
        attn     = clip_out.attentions[self._vis_layer]                   # [B,h,577,577]
        scores   = attn[:, :, 0, 1:].mean(dim=1).float()                  # [B,576]

        # Build merged visual features per sample → [B, keep_k, D]
        merged = torch.stack(
            [self._merge_contextual(raw_feat[i], scores[i]) for i in range(B)], dim=0
        )
        keep_per_sample = [torch.arange(merged.shape[1], device=dev) for _ in range(B)]

        inputs_embeds, attention_mask = self._build_inputs(
            inp, keep_per_sample, image_positions, merged
        )

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if not self.honest:
            gen_kwargs["min_new_tokens"] = 1

        out = self.backbone.generate(**gen_kwargs)
        texts = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        if self.honest:
            return [t.strip().rstrip(".").lower() for t in texts]
        from GQA.shared.metrics import extract_short_answer
        return [extract_short_answer(t, q) for t, q in zip(texts, questions)]
