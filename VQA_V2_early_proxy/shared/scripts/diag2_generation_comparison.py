"""
Diagnostic 2: Side-by-side generation comparison.

Runs 3 paths on the same 10 val2014 samples and prints tokenized
sequences, position ordering, and answers for each.

Path A — Standard LlavaForConditionalGeneration.generate(**inputs)
         Correct path: visual tokens inserted at <image> position
         inside the full model. Expected accuracy: ~77-78%.

Path B — Our current vqa_v2 static pipeline (WRONG ORDERING)
         cat([image_embeds, ALL_text_embeds]) on lm.generate()
         Visual tokens at position 0 before BOS/system prompt.
         Expected accuracy: ~67-68% (confirmed by 1K evals).

Path C — GQA-style correct ordering (FIXED)
         cat([prefix_embeds, image_embeds, suffix_embeds]) on
         backbone.generate(inputs_embeds=...).
         Visual tokens inserted at the <image> placeholder position.
         Expected accuracy: ~77-78% (same as Path A).

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 python VQA_V2_early_proxy/shared/scripts/diag2_generation_comparison.py
    [--n-samples 10]  [--max-new-tokens 10]  [--k 576]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Allow running this file directly (repo root = 3 levels up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
PROMPT_SUFFIX = " Answer the question using a single word or phrase."
N_PATCHES = 576
IMAGE_TOKEN_ID = 32000


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_samples(
    questions_path: str,
    annotations_path: str,
    image_dir: str,
    n: int,
) -> List[Dict[str, Any]]:
    """Load first N samples from val2014 as plain dicts."""
    with open(questions_path) as f:
        qs = json.load(f)["questions"]
    with open(annotations_path) as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}

    samples = []
    for q in qs:
        qid = q["question_id"]
        iid = q["image_id"]
        ann = anns.get(qid, {})
        img_path = os.path.join(image_dir, f"COCO_val2014_{iid:012d}.jpg")
        if not os.path.exists(img_path):
            continue
        samples.append({
            "question_id": qid,
            "image_id":    iid,
            "question":    q["question"],
            "image_path":  img_path,
            "raw_answers": [a["answer"] for a in ann.get("answers", [])],
        })
        if len(samples) == n:
            break
    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def vqa_score(pred: str, raw_answers: List[str]) -> float:
    pn = normalize(pred)
    matches = sum(1 for a in raw_answers if normalize(a) == pn)
    return min(1.0, matches / 3.0)


# ---------------------------------------------------------------------------
# CLIP scoring for token selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_cls_attn_scores(
    vision_tower,
    pixel_values: torch.Tensor,
    feature_layer: int = -2,
) -> torch.Tensor:
    """
    Return [B, 576] CLS-to-patch attention scores from CLIP layer `feature_layer`.
    Uses the penultimate layer (-2) to match LLaVA-1.5 vision_feature_layer.
    """
    out = vision_tower(
        pixel_values,
        output_attentions=True,
        output_hidden_states=True,
    )
    # Use feature_layer for both features AND attention (not -1)
    attn = out.attentions[feature_layer]  # [B, heads, 577, 577]
    scores = attn[:, :, 0, 1:].mean(dim=1).float()  # [B, 576]
    return scores


# ---------------------------------------------------------------------------
# Path A: standard generate  (correct)
# ---------------------------------------------------------------------------

@torch.no_grad()
def path_a_standard_generate(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    samples: List[Dict[str, Any]],
    k: int,
    max_new_tokens: int,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Standard LlavaForConditionalGeneration.generate(**processor_inputs).
    This is the reference / gold-standard path.

    If k < 576: we do NOT prune here — Path A always uses all 576 tokens.
    Its purpose is to establish the baseline, not to test pruning.
    """
    results = []
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": s["question"].strip() + PROMPT_SUFFIX},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = {k2: (v.to(device) if hasattr(v, "to") else v) for k2, v in inputs.items()}

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        # standard path: output includes input tokens; slice them off
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[:, prompt_len:]
        pred = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

        results.append({
            "question_id": s["question_id"],
            "question":    s["question"],
            "pred":        pred,
            "raw_answers": s["raw_answers"],
            "score":       vqa_score(pred, s["raw_answers"]),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "n_image_tokens": int((inputs["input_ids"] == IMAGE_TOKEN_ID).sum()),
        })
    return results


# ---------------------------------------------------------------------------
# Path B: WRONG ORDERING  (current vqa_v2 pipeline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def path_b_wrong_ordering(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    samples: List[Dict[str, Any]],
    k: int,
    max_new_tokens: int,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Reproduces the current vqa_v2 static pipeline bug:
      inputs_embeds = cat([image_embeds_K, ALL_text_embeds])
      → visual tokens at positions 0..K-1, then BOS/system/question tokens

    Calls lm.generate() (language model only, not full LlavaForConditionalGeneration).
    """
    vt      = model.vision_tower
    proj    = model.multi_modal_projector
    embed   = model.language_model.model.embed_tokens
    lm      = model.language_model
    v_layer = int(getattr(model.config, "vision_feature_layer", -2))

    results = []
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": s["question"].strip() + PROMPT_SUFFIX},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = {k2: (v.to(device) if hasattr(v, "to") else v) for k2, v in inputs.items()}

        pv = inputs["pixel_values"].to(model.dtype)
        clip_out = vt(pv, output_attentions=True, output_hidden_states=True)

        # BUG: use final layer attention (-1), not feature layer (-2)
        final_attn = clip_out.attentions[-1]           # [1, h, 577, 577]
        raw_feat   = clip_out.hidden_states[v_layer][:, 1:, :]  # [1, 576, D_clip]

        scores  = final_attn[:, :, 0, 1:].mean(dim=1).float()  # [1, 576]
        topk_idx = scores[0].topk(k).indices.sort().values      # [K] sorted

        selected = raw_feat[0][topk_idx].unsqueeze(0)           # [1, K, D_clip]
        proj_vis = proj(selected)                                # [1, K, D_lm]

        # BUG: strip ALL image tokens from input_ids, embed everything, prepend image
        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        valid_ids  = ids[mask.bool()]
        text_ids   = valid_ids[valid_ids != IMAGE_TOKEN_ID]      # all text tokens, in order
        text_embed = embed(text_ids.unsqueeze(0))                # [1, T_text, D_lm]

        # WRONG: [image_embeds, BOS, SYS, USER, :, \n, question, ASSISTANT]
        lm_embeds = torch.cat([proj_vis, text_embed], dim=1)     # [1, K+T_text, D_lm]
        img_mask  = torch.ones(1, k, dtype=torch.long, device=device)
        txt_mask  = torch.ones(1, text_ids.numel(), dtype=torch.long, device=device)
        lm_mask   = torch.cat([img_mask, txt_mask], dim=1)       # [1, K+T_text]

        gen_ids = lm.generate(
            inputs_embeds  = lm_embeds,
            attention_mask = lm_mask,
            max_new_tokens = max_new_tokens,
            min_new_tokens = 1,
            do_sample      = False,
            pad_token_id   = processor.tokenizer.eos_token_id,
        )
        # With inputs_embeds, output = new tokens only (no input prefix in gen_ids)
        pred = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        results.append({
            "question_id":     s["question_id"],
            "question":        s["question"],
            "pred":            pred,
            "raw_answers":     s["raw_answers"],
            "score":           vqa_score(pred, s["raw_answers"]),
            "seq_len":         int(lm_embeds.shape[1]),
            "ordering":        f"[img×{k}, BOS, SYS..., question, ASST]  ← WRONG",
            "positions_0_to_4": list(range(min(5, k))),  # image tokens at 0..k-1
        })
    return results


# ---------------------------------------------------------------------------
# Path C: CORRECT ORDERING  (GQA-style fix)
# ---------------------------------------------------------------------------

@torch.no_grad()
def path_c_correct_ordering(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    samples: List[Dict[str, Any]],
    k: int,
    max_new_tokens: int,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Correct ordering:
      inputs_embeds = cat([prefix_embeds, image_embeds_K, suffix_embeds])

    prefix = tokens before the first <image> placeholder
    suffix = tokens after the last <image> placeholder

    This matches how LLaVA was trained:
      [BOS, SYS, USER, :, img_1..img_K, \\n, question, ASST]

    Calls backbone.generate(inputs_embeds=...) on the full LlavaForConditionalGeneration.
    Uses penultimate CLIP layer attention for scoring (matching vision_feature_layer).
    """
    vt      = model.vision_tower
    proj    = model.multi_modal_projector
    embed   = model.language_model.model.embed_tokens
    v_layer = int(getattr(model.config, "vision_feature_layer", -2))

    results = []
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": s["question"].strip() + PROMPT_SUFFIX},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inputs = {k2: (v.to(device) if hasattr(v, "to") else v) for k2, v in inputs.items()}

        pv = inputs["pixel_values"].to(model.dtype)
        clip_out = vt(pv, output_attentions=True, output_hidden_states=True)

        # CORRECT: use penultimate layer attention (-2) matching vision_feature_layer
        attn     = clip_out.attentions[v_layer]                  # [1, h, 577, 577]
        raw_feat = clip_out.hidden_states[v_layer][:, 1:, :]     # [1, 576, D_clip]

        scores   = attn[:, :, 0, 1:].mean(dim=1).float()         # [1, 576]
        topk_idx = scores[0].topk(k).indices.sort().values        # [K] sorted

        selected = raw_feat[0][topk_idx].unsqueeze(0)             # [1, K, D_clip]
        proj_vis = proj(selected)                                  # [1, K, D_lm]

        # CORRECT: split at image token position
        ids       = inputs["input_ids"][0]       # [seq_len]
        attn_mask = inputs["attention_mask"][0]  # [seq_len]

        img_positions = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(img_positions) == 0:
            print(f"  [Warn] No image tokens found for q={s['question_id']}")
            continue
        img_start = int(img_positions[0])
        img_end   = int(img_positions[-1]) + 1  # exclusive

        # prefix: tokens before the image placeholder (BOS, system prompt, USER:, etc.)
        prefix_ids  = ids[:img_start]
        prefix_mask = attn_mask[:img_start]
        prefix_emb  = embed(prefix_ids.unsqueeze(0))             # [1, T_pre, D_lm]

        # suffix: tokens after the image placeholder (\n, question text, ASSISTANT:)
        suffix_ids  = ids[img_end:]
        suffix_mask = attn_mask[img_end:]
        suffix_emb  = embed(suffix_ids.unsqueeze(0))             # [1, T_suf, D_lm]

        # CORRECT: [prefix, image_K, suffix]
        lm_embeds = torch.cat([prefix_emb, proj_vis, suffix_emb], dim=1)
        pre_m  = prefix_mask.unsqueeze(0)
        vis_m  = torch.ones(1, k, dtype=torch.long, device=device)
        suf_m  = suffix_mask.unsqueeze(0)
        lm_mask = torch.cat([pre_m, vis_m, suf_m], dim=1)

        gen_ids = model.generate(
            inputs_embeds  = lm_embeds,
            attention_mask = lm_mask,
            max_new_tokens = max_new_tokens,
            min_new_tokens = 1,
            do_sample      = False,
            pad_token_id   = processor.tokenizer.eos_token_id,
        )
        # With inputs_embeds, output = new tokens only
        pred = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        # Decode prefix and suffix for display
        prefix_text = processor.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        suffix_text = processor.tokenizer.decode(suffix_ids, skip_special_tokens=False)

        results.append({
            "question_id":    s["question_id"],
            "question":       s["question"],
            "pred":           pred,
            "raw_answers":    s["raw_answers"],
            "score":          vqa_score(pred, s["raw_answers"]),
            "seq_len":        int(lm_embeds.shape[1]),
            "ordering":       f"[prefix×{prefix_ids.numel()}, img×{k}, suffix×{suffix_ids.numel()}]  ← CORRECT",
            "prefix_text":    repr(prefix_text[-60:]),
            "suffix_text":    repr(suffix_text[:60]),
        })
    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_comparison(
    samples: List[Dict[str, Any]],
    res_a: List[Dict],
    res_b: List[Dict],
    res_c: List[Dict],
) -> None:
    print("\n" + "="*90)
    print(f"{'Q':>4}  {'Question':<40}  {'Path A':>8}  {'Path B':>8}  {'Path C':>8}  {'Gold'}")
    print("="*90)
    for i, s in enumerate(samples):
        a = res_a[i] if i < len(res_a) else {}
        b = res_b[i] if i < len(res_b) else {}
        c = res_c[i] if i < len(res_c) else {}
        gold = s["raw_answers"][0] if s["raw_answers"] else "?"
        print(
            f"  {i+1:>2}  {s['question'][:40]:<40}  "
            f"{a.get('pred','')[:8]:>8}  "
            f"{b.get('pred','')[:8]:>8}  "
            f"{c.get('pred','')[:8]:>8}  "
            f"{gold}"
        )
        # Flag disagreements
        pa = a.get('pred','').lower().strip()
        pb = b.get('pred','').lower().strip()
        pc = c.get('pred','').lower().strip()
        if pa != pb:
            print(f"       ← B differs from A")
        if pa != pc:
            print(f"       ← C differs from A")

    acc_a = sum(r["score"] for r in res_a) / len(res_a) if res_a else 0
    acc_b = sum(r["score"] for r in res_b) / len(res_b) if res_b else 0
    acc_c = sum(r["score"] for r in res_c) / len(res_c) if res_c else 0
    print("="*90)
    print(f"\n  Path A (standard generate, K=576, correct):      {acc_a*100:.1f}%")
    print(f"  Path B (wrong ordering, K={res_b[0].get('seq_len', '?')-len(res_b[0].get('raw_answers',[])), '?'}, current bug): {acc_b*100:.1f}%")
    print(f"  Path C (correct ordering, K={args.k}, GQA-style):     {acc_c*100:.1f}%")
    print()
    print("  Ordering layouts:")
    if res_b:
        print(f"  B: {res_b[0].get('ordering','')}")
    if res_c:
        print(f"  C: {res_c[0].get('ordering','')}")
        print(f"     prefix ends with: {res_c[0].get('prefix_text','')}")
        print(f"     suffix starts:    {res_c[0].get('suffix_text','')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",     type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--k",             type=int, default=576,
                        help="Visual tokens for Paths B and C (A always uses 576).")
    parser.add_argument("--questions",  default="data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--annotations", default="data/vqav2/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--image-dir",   default="data/vqav2/val2014")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}", flush=True)

    print(f"[Info] Loading model: {MODEL_NAME}", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # needed for output_attentions in CLIP
    ).to(device)
    model.eval()
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"
    vc = getattr(model.config, "vision_config", None)
    processor.patch_size = getattr(vc, "patch_size", 14)
    processor.vision_feature_select_strategy = "default"
    processor.num_additional_image_tokens = 0

    print(f"[Info] Loading {args.n_samples} samples from val2014…", flush=True)
    samples = load_samples(args.questions, args.annotations, args.image_dir, args.n_samples)
    print(f"[Info] Loaded {len(samples)} samples.", flush=True)

    print("\n[Path A] Standard LlavaForConditionalGeneration.generate() …", flush=True)
    res_a = path_a_standard_generate(model, processor, samples, args.k, args.max_new_tokens, device)

    print(f"\n[Path B] WRONG ordering: cat([img×{args.k}, ALL_text]) …", flush=True)
    res_b = path_b_wrong_ordering(model, processor, samples, args.k, args.max_new_tokens, device)

    print(f"\n[Path C] CORRECT ordering: cat([prefix, img×{args.k}, suffix]) …", flush=True)
    res_c = path_c_correct_ordering(model, processor, samples, args.k, args.max_new_tokens, device)

    print_comparison(samples, res_a, res_b, res_c)

    # Save full results
    out_path = f"scripts/diag2_results_k{args.k}_n{args.n_samples}.json"
    with open(out_path, "w") as f:
        json.dump({
            "k": args.k,
            "n_samples": args.n_samples,
            "path_a_acc": sum(r["score"] for r in res_a) / len(res_a),
            "path_b_acc": sum(r["score"] for r in res_b) / len(res_b),
            "path_c_acc": sum(r["score"] for r in res_c) / len(res_c),
            "path_a": res_a,
            "path_b": res_b,
            "path_c": res_c,
        }, f, indent=2)
    print(f"\n[Info] Full results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
