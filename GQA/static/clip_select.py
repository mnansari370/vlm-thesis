"""
CLIP-space question-conditioned visual-token selector (frozen, training-free).

Tests the thesis claim with the SINK-FREE formulation (unlike LM-attention, which
FEATHER shows is corrupted by attention sinks/position bias):

  score(patch) = cosine( CLIP_visual_projection(patch_embed) , CLIP_text_projection(question) )

i.e. rank the 576 visual patches by similarity to the QUESTION in CLIP's shared
image-text space. Frozen. Keep top-K. Also a FUSION with CLS-Attn saliency:
  fused = (1-w)*norm(cls_attn) + w*norm(clip_qcond),  w ∈ {0.25,0.5,0.75}.

Uses the full openai/clip-vit-large-patch14-336 (same CLIP LLaVA-1.5 is built on) for
the text encoder + projections. The LLaVA pixel_values are fed directly to the CLIP
vision model (identical normalization), so the 576 patches align 1:1 with the LLaVA
features used for the LLM — selection indices transfer exactly.

Reuses StaticPrunedLlava (sdpa) for the CLIP CLS-attn score, projector, inputs_embeds
assembly and honest generation. Cite: CLIP-space relevance (DivPrune 2503.02175,
SparseVLM 2410.04417 question-guided selection).
"""

import random as _random

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from GQA.shared.static import (StaticPrunedLlava, HONEST_SUFFIX, PROMPT_SUFFIX,
                               N_PATCHES, MODEL_NAME)

CLIP_NAME = "openai/clip-vit-large-patch14-336"


class CLIPSpaceLlava(StaticPrunedLlava):
    def __init__(self, image_pad=True, honest=True, append_suffix=False,
                 model_name=MODEL_NAME, seed=42):
        super().__init__(method="cls_attn", keep_k=144, seed=seed, model_name=model_name,
                         image_pad=image_pad, honest=honest, append_suffix=append_suffix)
        dev = next(self.backbone.parameters()).device
        print(f"[CLIP] Loading {CLIP_NAME} for question->patch scoring ...", flush=True)
        self.clip = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=torch.float16).to(dev).eval()
        self.clip_tok = CLIPTokenizerFast.from_pretrained(CLIP_NAME)
        for p in self.clip.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def score_sample(self, image, prompt, question_text):
        """One scoring pass for a single (image, prompt, question_text)."""
        dev = next(self.backbone.parameters()).device
        imgs = [self._pad_to_square(image)] if self.image_pad else [image]
        suffix = (HONEST_SUFFIX if self.honest else PROMPT_SUFFIX) if self.append_suffix else ""
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt.strip() + suffix}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = self.processor(text=[text], images=imgs, return_tensors="pt", padding=True)
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
        vis_pos = (inp["input_ids"][0] == self._image_token_id).nonzero(as_tuple=True)[0]
        pv = inp["pixel_values"].to(self.backbone.dtype)

        # LLaVA CLIP features (for projection into LLM) + CLS-Attn saliency score
        clip_out = self._vt(pv, output_attentions=True, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[self._vis_layer][:, 1:, :]               # [1,576,D]
        cls_scores = clip_out.attentions[self._vis_layer][:, :, 0, 1:].mean(1)[0].float()  # [576]

        # CLIP-space question->patch relevance (full CLIP, shared space).
        # IMPORTANT: CLIP applies post_layernorm only to the pooled CLS token; for
        # per-patch alignment we must apply post_layernorm to ALL tokens before the
        # visual_projection (else patches are not in the image-text aligned space).
        vout = self.clip.vision_model(pv)                                          # last_hidden_state [1,577,1024]
        hs = self.clip.vision_model.post_layernorm(vout.last_hidden_state)          # [1,577,1024]
        patch_embeds = self.clip.visual_projection(hs[:, 1:, :])                    # [1,576,768]
        patch_embeds = F.normalize(patch_embeds.float(), dim=-1)
        tok = self.clip_tok([question_text], padding=True, truncation=True,
                            max_length=77, return_tensors="pt").to(dev)
        text_embed = self.clip.get_text_features(**tok)                            # [1,768]
        text_embed = F.normalize(text_embed.float(), dim=-1)
        clip_scores = (patch_embeds[0] @ text_embed[0]).float()                    # [576]

        # Fairness fix: with image_pad, black-border patches are uninformative but
        # CLIP-space text-similarity spuriously ranks them high. Mask them out so the
        # CLIP-space selector competes only on real-content patches (cls-attn already
        # avoids them naturally). Padding patch = ~uniform pixel_values (spatial std≈0).
        # per-PATCH, per-CHANNEL spatial std; a padding patch is uniform within every
        # channel (cross-channel std is NOT ~0 because CLIP per-channel means differ).
        p = pv[0].reshape(3, 24, 14, 24, 14).permute(1, 3, 0, 2, 4).reshape(576, 3, 14 * 14).float()
        chan_std = p.std(dim=2)                      # [576, 3]
        pad_mask = chan_std.max(dim=1).values < 1e-3 # uniform in all 3 channels
        clip_scores = clip_scores.masked_fill(pad_mask.to(clip_scores.device), -1e4)
        self._last_pad_mask = pad_mask

        return {"cls_scores": cls_scores, "clip_scores": clip_scores,
                "raw_feat": raw_feat, "inp": inp, "image_positions": [vis_pos]}

    @staticmethod
    def _norm(x):
        x = x - x.min()
        d = x.max()
        return x / d if d > 0 else x

    def keep_indices(self, scored, selector, keep_k, sample_offset=0, mix=0.5):
        dev = scored["cls_scores"].device
        if selector == "random":
            rng = _random.Random(self.seed + sample_offset)   # per-sample (bug fixed)
            return torch.tensor(sorted(rng.sample(range(N_PATCHES), keep_k)),
                                dtype=torch.long, device=dev)
        if selector == "cls":
            s = scored["cls_scores"]
        elif selector == "clip":
            s = scored["clip_scores"]
        elif selector == "fusion":
            s = (1 - mix) * self._norm(scored["cls_scores"]) + mix * self._norm(scored["clip_scores"])
        else:
            raise ValueError(selector)
        return s.topk(keep_k).indices.sort().values

    @torch.no_grad()
    def generate_from_keep(self, scored, keep_idx):
        self.keep_k = len(keep_idx)
        emb, mask = self._build_inputs(scored["inp"], [keep_idx],
                                       scored["image_positions"], scored["raw_feat"])
        gen = dict(inputs_embeds=emb, attention_mask=mask, max_new_tokens=64,
                   do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
        if not self.honest:
            gen["min_new_tokens"] = 1
        out = self.backbone.generate(**gen)
        t = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return t.strip().rstrip(".").lower() if self.honest else t.strip()
