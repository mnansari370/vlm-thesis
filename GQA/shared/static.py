"""
StaticPrunedLlava — training-free visual token pruning via PHYSICAL REMOVAL.

Pipeline (all methods)
----------------------
1. Run CLIP vision tower on pixel_values → scoring signals + raw patch features.
2. Score and rank the 576 patch features by the chosen method.
3. Select top-K features and project with multi_modal_projector → [B, K, D_lm].
4. Manually build inputs_embeds:
       [embed(text_prefix) | K projected visual tokens | embed(text_suffix)]
   where prefix = everything before the 576 image-token placeholders in input_ids,
   suffix = everything after (question + INST end tokens).
5. Build corresponding attention_mask: [prefix_mask | ones×K | suffix_mask].
6. Call backbone.generate(inputs_embeds=…, attention_mask=…).
   With inputs_embeds, the output contains ONLY generated tokens (no input prefix)
   so no slicing is needed for decoding.

This approach:
  - Physically removes (576-K) image tokens — the LM processes a shorter sequence.
  - Is fully compatible with sdpa/FlashAttention for generation.
  - Matches FLOPs numbers (sequence length = K + n_text, not 576 + n_text).
  - Matches VisionZip (arXiv 2412.04467) implementation style.

Methods
-------
none            : K must equal 576. Uses all patch features. Sanity check —
                  must reproduce the 67.73% dense zero-shot result.

spatial_uniform : Deterministic stride-based subsampling of the 24×24 CLIP
                  patch grid. No scoring; same indices every run.
                  ("Are We Solving the Right Problem?", ACL 2025, arXiv 2502.11501)

random          : Uniform random K-subset, seeded as seed+global_sample_index
                  for per-sample reproducibility across batch boundaries.

cls_attn        : VisionZip Dominant scoring — CLS-to-patch attention at CLIP
                  layer vision_feature_layer (=-2), averaged over heads → top-K.
                  Cite: VisionZip (arXiv 2412.04467), Section 3.1.
                  NOTE: CLIPSdpaAttention does not support output_attentions;
                  HuggingFace falls back to the manual implementation automatically.

l2_norm         : L2 norm of CLIP hidden states at vision_feature_layer (patch
                  tokens only, CLS excluded) → top-K.  Ablation baseline.

fastv_style     : LLM-layer-2 average received attention, scored from the original
                  576-token sequence via a full backbone forward pass.
                  Requires attn_implementation='eager' (loaded automatically).
                  Included to show the RoPE position-bias problem: LM-side scoring
                  favours bottom-of-image raster-scan tokens, causing top-of-image
                  pruning (FEATHER, arXiv 2412.13180; FastV, arXiv 2403.06764).

Position-bias note
------------------
LM-side scoring (fastv_style) applies attention computed inside the language model,
which uses Rotary Position Embeddings (RoPE).  RoPE's long-term decay causes tokens
later in the raster-scan order (bottom of image) to receive systematically higher
attention scores, resulting in top-of-image pruning regardless of content.
FEATHER (ICCV 2025) shows this causes near-zero localisation performance
(FastV: 6.7 vs FEATHER: 44.1 on RefCOCO avg at 64% FLOPs reduction).

CLIP-side scoring (cls_attn, l2_norm) operates within the CLIP vision encoder before
the language model.  CLIP uses learned 2D position embeddings — not RoPE — so scores
reflect genuine visual content, not raster-scan position.
"""

import random as _random

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from GQA.shared.metrics import extract_short_answer


MODEL_NAME    = "llava-hf/llava-1.5-7b-hf"
PROMPT_SUFFIX = " Answer with one word or a short phrase only."          # OLD val_balanced
HONEST_SUFFIX = "\nAnswer the question using a single word or phrase."   # LLaVA official (testdev)
N_PATCHES     = 576        # 24×24 CLIP ViT-L/14 @ 336 px
FASTV_LM_LAYER = 2         # LM layer (1-indexed) used for fastv_style scoring


# ── spatial uniform index sets ────────────────────────────────────────────────

def _build_spatial_indices() -> dict[int, list[int]]:
    """
    Pre-compute deterministic stride-based index sets for the 24×24 patch grid.
    Each set contains exactly K sorted patch indices for uniform spatial coverage.

    Pattern (row_stride × col_stride → exact count):
      K=576 : full 24×24  (1×1)
      K=432 : drop every 4th column (c%4 != 3)  → 24×18
      K=288 : col stride 2                        → 24×12
      K=192 : col stride 3                        → 24×8
      K=144 : row stride 2, col stride 2           → 12×12
    """
    idx: dict[int, list[int]] = {
        576: list(range(576)),
        432: [r * 24 + c for r in range(24) for c in range(24) if c % 4 != 3],
        288: [r * 24 + c for r in range(24) for c in range(0, 24, 2)],
        192: [r * 24 + c for r in range(24) for c in range(0, 24, 3)],
        144: [r * 24 + c for r in range(0, 24, 2) for c in range(0, 24, 2)],
    }
    for k, inds in idx.items():
        assert len(inds) == k,           f"spatial_uniform K={k}: {len(inds)} ≠ {k}"
        assert inds == sorted(inds),     f"spatial_uniform K={k}: not sorted"
        assert len(set(inds)) == k,      f"spatial_uniform K={k}: duplicates"
        assert all(0 <= x < 576 for x in inds), f"spatial_uniform K={k}: out of range"
    return idx


SPATIAL_INDICES: dict[int, list[int]] = _build_spatial_indices()

VALID_METHODS = frozenset(
    {"none", "random", "spatial_uniform", "cls_attn", "l2_norm", "fastv_style"}
)
SUPPORTED_K = frozenset({576, 432, 288, 192, 144, 96, 64})


# ── model ─────────────────────────────────────────────────────────────────────

class StaticPrunedLlava(nn.Module):
    """
    LLaVA-1.5-7B with training-free visual token pruning.
    Frozen; no gradients; no weight changes.

    Physical-removal pipeline:
      score → select K of 576 patch features → project → build inputs_embeds
      → backbone.generate(inputs_embeds, attention_mask).
    """

    def __init__(
        self,
        method:     str,
        keep_k:     int,
        seed:       int = 42,
        model_name: str = MODEL_NAME,
        image_pad:  bool = False,
        honest:     bool = False,
        append_suffix: bool = True,
    ) -> None:
        super().__init__()

        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
        if keep_k not in SUPPORTED_K:
            raise ValueError(f"keep_k must be one of {SUPPORTED_K}, got {keep_k}")
        if method == "none" and keep_k != 576:
            raise ValueError("method='none' requires keep_k=576")

        self.method = method
        self.keep_k = keep_k
        self.seed   = seed
        # Honest-protocol knobs (Phase B): pad to square, LLaVA-official suffix,
        # raw rstrip('.').lower() post-processing (no extract_short_answer).
        self.image_pad = image_pad
        self.honest    = honest
        # append_suffix=False: the passed "questions" are already full prompts
        # (e.g. TextVQA text field = question + OCR block + instruction).
        self.append_suffix = append_suffix

        # fastv_style extracts LM attention → needs eager
        attn_impl = "eager" if method == "fastv_style" else "sdpa"

        print(
            f"[Model] Loading {model_name} "
            f"(method={method}, K={keep_k}, attn_impl={attn_impl}, "
            f"image_pad={image_pad}, honest={honest}) …",
            flush=True,
        )
        self.backbone = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = self.backbone.to(dev)
        self.backbone.eval()
        self.backbone.config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        vc = getattr(self.backbone.config, "vision_config", None)
        self.processor.patch_size = getattr(vc, "patch_size", 14)
        self.processor.vision_feature_select_strategy = "default"
        self.processor.num_additional_image_tokens = 0

        for p in self.backbone.parameters():
            p.requires_grad = False

        self._image_token_id = int(
            getattr(self.backbone.config, "image_token_index", 32000)
        )
        self._vis_layer = int(
            getattr(self.backbone.config, "vision_feature_layer", -2)
        )

        # Model sub-modules (confirmed attribute paths for llava-hf/llava-1.5-7b-hf)
        self._vt    = self.backbone.vision_tower          # CLIPVisionModel
        self._proj  = self.backbone.multi_modal_projector # LlavaMultiModalProjector
        self._embed = self.backbone.language_model.model.embed_tokens

        print(
            f"[Model] Ready — device={dev}  "
            f"image_token_id={self._image_token_id}  "
            f"vision_feature_layer={self._vis_layer}",
            flush=True,
        )

    # ── scoring ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _scores_cls_attn(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        VisionZip Dominant: CLS-to-patch attention at CLIP layer vis_layer,
        averaged over heads.  Returns [B, 576] float32 scores.

        Note: CLIPSdpaAttention does not support output_attentions=True;
        HuggingFace automatically falls back to the manual implementation.
        This produces a UserWarning, which is expected and harmless.
        """
        out  = self._vt(pixel_values.to(self.backbone.dtype), output_attentions=True)
        attn = out.attentions[self._vis_layer]  # [B, heads, 577, 577]
        return attn[:, :, 0, 1:].mean(dim=1).float()  # [B, 576]

    @torch.no_grad()
    def _scores_l2_norm(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        L2 norm of CLIP hidden states at vision_feature_layer, patch tokens only.
        Returns [B, 576] float32 scores.
        """
        out = self._vt(pixel_values.to(self.backbone.dtype), output_hidden_states=True)
        hs  = out.hidden_states[self._vis_layer]  # [B, 577, D_clip]
        return hs[:, 1:, :].norm(dim=-1).float()   # [B, 576]

    @torch.no_grad()
    def _scores_fastv_style(
        self,
        inp:             dict,
        image_positions: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        LLM-side scoring (FastV convention, arXiv 2403.06764).
        Full backbone forward on the original 576-token sequence; extracts average
        attention received at LM layer FASTV_LM_LAYER across all heads and queries.

        Subject to RoPE positional bias: bottom-of-image tokens get systematically
        higher scores.  Included for ablation/comparison only.
        Requires attn_implementation='eager'.

        Returns [B, 576] float32 scores.
        """
        B   = inp["input_ids"].shape[0]
        dev = inp["input_ids"].device
        out = self.backbone(
            input_ids      = inp["input_ids"],
            pixel_values   = inp.get("pixel_values"),
            attention_mask = inp["attention_mask"],
            output_attentions = True,
            use_cache      = False,
        )
        # attentions: tuple of T tensors, each [B, heads, seq_len, seq_len]
        # FASTV_LM_LAYER is 1-indexed
        layer_attn = out.attentions[FASTV_LM_LAYER - 1]  # [B, heads, seq, seq]
        # Average received attention per key position over all heads and queries:
        # [B, heads, query, key] → mean over heads → [B, query, key]
        #                        → mean over query → [B, key]
        received = layer_attn.float().mean(dim=1).mean(dim=1)  # [B, seq_len]

        scores = torch.zeros(B, N_PATCHES, dtype=torch.float32, device=dev)
        for i, pos in enumerate(image_positions):
            n = len(pos)
            if n == N_PATCHES:
                scores[i] = received[i, pos]
            elif n > 0:
                k = min(n, N_PATCHES)
                scores[i, :k] = received[i, pos[:k]]
        return scores

    # ── inputs_embeds builder ─────────────────────────────────────────────────

    @torch.no_grad()
    def _build_inputs(
        self,
        inp:             dict,
        keep_per_sample: list[torch.Tensor],
        image_positions: list[torch.Tensor],
        raw_features:    torch.Tensor,         # [B, 576, D_clip] from CLIP
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Physically remove dropped image tokens and build (inputs_embeds, attn_mask)
        for a sequence of length (prefix + K_visual + suffix).

        Since all samples in the batch share the same padded sequence length,
        the pruned sequences also all have the same length (seq_len - 576 + K),
        so NO re-padding is needed.

        Returns
        -------
        inputs_embeds : [B, new_len, D_lm]
        attention_mask: [B, new_len]  (long)
        """
        B   = inp["input_ids"].shape[0]
        K   = self.keep_k
        dev = inp["input_ids"].device

        # Project all 576 features; select K per sample inside the loop
        projected = self._proj(raw_features)  # [B, 576, D_lm]

        embeds_list: list[torch.Tensor] = []
        masks_list:  list[torch.Tensor] = []

        for i in range(B):
            img_pos   = image_positions[i]              # [576] absolute seq positions
            img_start = img_pos[0].item()
            img_end   = img_pos[-1].item() + 1          # exclusive

            # Embed prefix (left-padding + text before image)
            prefix_emb  = self._embed(inp["input_ids"][i, :img_start])
            prefix_mask = inp["attention_mask"][i, :img_start]

            # Selected visual features
            keep_idx = keep_per_sample[i].to(dev)
            vis_emb  = projected[i, keep_idx, :]         # [K, D_lm]
            vis_mask = torch.ones(K, dtype=torch.long, device=dev)

            # Embed suffix (question text + INST end tokens)
            suffix_emb  = self._embed(inp["input_ids"][i, img_end:])
            suffix_mask = inp["attention_mask"][i, img_end:]

            embeds_list.append(torch.cat([prefix_emb, vis_emb, suffix_emb], dim=0))
            masks_list.append(torch.cat([prefix_mask, vis_mask, suffix_mask], dim=0))

        return (
            torch.stack(embeds_list, dim=0),   # [B, new_len, D_lm]
            torch.stack(masks_list,  dim=0),   # [B, new_len]
        )

    # ── main entry point ──────────────────────────────────────────────────────

    @staticmethod
    def _pad_to_square(img):
        """Pad non-square PIL image to square with black border (LLaVA-1.5 default).

        Identical to LlavaTestdevEval._pad_to_square so static and dense use the
        SAME preprocessing — required for the K=576 static==dense sanity check.
        """
        from PIL import Image
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        return padded

    @torch.no_grad()
    def set_keep_k(self, keep_k: int) -> None:
        """Change the token budget K at runtime (for speculative execution)."""
        if keep_k not in SUPPORTED_K and keep_k != 576:
            raise ValueError(f"keep_k must be one of {SUPPORTED_K}")
        self.keep_k = keep_k

    def generate_answers(
        self,
        images:             list,
        questions:          list[str],
        sample_offset:      int  = 0,
        max_new_tokens:     int  = 10,
        return_confidence:  bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        """
        Generate short GQA answers with static visual token pruning.

        Parameters
        ----------
        images        : list of PIL images.
        questions     : list of question strings (same length as images).
        sample_offset : global index of the first sample in this batch.
                        Used to seed per-sample RNG for 'random' so that
                        results are identical regardless of batch size.
        max_new_tokens: generation budget.

        Returns
        -------
        List of short extracted answer strings.
        """
        B   = len(questions)
        dev = next(self.backbone.parameters()).device

        # ── 0. Honest image preprocessing (pad non-square to square) ─────────
        if self.image_pad:
            images = [self._pad_to_square(img) for img in images]

        # ── 1. Processor inputs ──────────────────────────────────────────────
        suffix = (HONEST_SUFFIX if self.honest else PROMPT_SUFFIX) if self.append_suffix else ""
        convs = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": q.strip() + suffix},
            ]}]
            for q in questions
        ]
        prompts = [
            self.processor.apply_chat_template(
                c, add_generation_prompt=True, tokenize=False
            )
            for c in convs
        ]
        inp = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        # ── 2. Find image token positions per sample ─────────────────────────
        image_positions: list[torch.Tensor] = []
        for i in range(B):
            pos = (inp["input_ids"][i] == self._image_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(pos) != N_PATCHES:
                print(
                    f"[Warn] sample {i}: expected {N_PATCHES} image tokens, "
                    f"found {len(pos)}",
                    flush=True,
                )
            image_positions.append(pos)

        # ── 3. Get CLIP features for projection + scoring ────────────────────
        pv = inp["pixel_values"].to(self.backbone.dtype)

        if self.method == "cls_attn":
            # Need attentions; hidden_states also grabbed for projection
            clip_out  = self._vt(pv, output_attentions=True, output_hidden_states=True)
            raw_feat  = clip_out.hidden_states[self._vis_layer][:, 1:, :]  # [B,576,D]
            attn      = clip_out.attentions[self._vis_layer]               # [B,h,577,577]
            scores    = attn[:, :, 0, 1:].mean(dim=1).float()             # [B,576]

        elif self.method == "l2_norm":
            clip_out  = self._vt(pv, output_hidden_states=True)
            raw_feat  = clip_out.hidden_states[self._vis_layer][:, 1:, :]
            scores    = raw_feat.norm(dim=-1).float()                      # [B,576]

        elif self.method == "fastv_style":
            # Score from LM layer 2 attention (full forward pass)
            clip_out  = self._vt(pv, output_hidden_states=True)
            raw_feat  = clip_out.hidden_states[self._vis_layer][:, 1:, :]
            scores    = self._scores_fastv_style(inp, image_positions)    # [B,576]

        else:
            # none / random / spatial_uniform: features only, no scoring pass
            clip_out  = self._vt(pv, output_hidden_states=True)
            raw_feat  = clip_out.hidden_states[self._vis_layer][:, 1:, :]
            scores    = None

        # ── 4. Select K patch indices per sample ────────────────────────────
        keep_per_sample: list[torch.Tensor] = []
        for i in range(B):
            if self.method in ("cls_attn", "l2_norm", "fastv_style"):
                assert scores is not None
                keep = scores[i].topk(self.keep_k).indices.sort().values

            elif self.method == "spatial_uniform":
                keep = torch.tensor(
                    SPATIAL_INDICES[self.keep_k], dtype=torch.long, device=dev
                )

            elif self.method == "random":
                rng  = _random.Random(self.seed + sample_offset + i)
                keep = torch.tensor(
                    sorted(rng.sample(range(N_PATCHES), self.keep_k)),
                    dtype=torch.long, device=dev,
                )

            else:  # none
                keep = torch.arange(N_PATCHES, dtype=torch.long, device=dev)

            keep_per_sample.append(keep)

        # ── 5. Build inputs_embeds with physically removed tokens ────────────
        inputs_embeds, attention_mask = self._build_inputs(
            inp, keep_per_sample, image_positions, raw_feat
        )

        # ── 6. Generate ───────────────────────────────────────────────────────
        # Honest protocol: no min_new_tokens, no repetition_penalty, raw rstrip
        # post-processing. Legacy: min_new_tokens=1 + extract_short_answer.
        gen_kwargs = dict(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = self.processor.tokenizer.eos_token_id,
        )
        if not self.honest:
            gen_kwargs["min_new_tokens"] = 1

        def _post(texts: list[str]) -> list[str]:
            if self.honest:
                return [t.strip().rstrip(".").lower() for t in texts]
            return [extract_short_answer(t, q) for t, q in zip(texts, questions)]

        if return_confidence:
            gen_out = self.backbone.generate(
                **gen_kwargs,
                return_dict_in_generate = True,
                output_scores           = True,
            )
            # First-token max-softmax confidence
            confidences: list[float] = []
            if gen_out.scores:
                probs = gen_out.scores[0].float().softmax(-1)
                confidences = probs.max(-1).values.tolist()
            else:
                confidences = [1.0] * B
            texts = self.processor.tokenizer.batch_decode(
                gen_out.sequences, skip_special_tokens=True
            )
            return _post(texts), confidences

        out = self.backbone.generate(**gen_kwargs)

        # No slicing: with inputs_embeds, out = generated token ids only
        texts = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        return _post(texts)


# ── unit test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Unit test: spatial_uniform index counts ===")
    all_ok = True
    for k, inds in SPATIAL_INDICES.items():
        ok = (len(inds) == k and inds == sorted(inds)
              and len(set(inds)) == k and all(0 <= x < 576 for x in inds))
        if not ok:
            all_ok = False
        print(f"  K={k:4d}: count={len(inds):4d}  sorted=True  → {'OK' if ok else 'FAIL'}")

    print("\n=== Visual layout (row 0 of 24×24 grid) ===")
    for k in [576, 432, 288, 192, 144]:
        row = "".join("█" if (c in set(SPATIAL_INDICES[k])) else "·"
                      for c in range(24))
        print(f"  K={k:3d}: {row}")

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    raise SystemExit(0 if all_ok else 1)
