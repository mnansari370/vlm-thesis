"""
Phase 1 smoke test for the ordering fix (Section 4.2 of the work order).

Loads LLaVA-1.5-7B once and compares two paths on 20 val2014 samples:
  Path A — standard LlavaForConditionalGeneration.generate(**inputs)
            (always correct; our accuracy baseline)
  Path D — fixed _build_split_embeds logic
            (prefix, visual_K, suffix ordering)
            At K=576 Path D must match Path A within ~1 sample on N=20.
            At K=128 Path D should be clearly above 68% (old buggy level).

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 python VQA_V2_early_proxy/shared/scripts/smoke_test_fixed_wrapper.py
    CUDA_VISIBLE_DEVICES=1 python VQA_V2_early_proxy/shared/scripts/smoke_test_fixed_wrapper.py --k 128 --n-samples 20
"""

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Allow running this file directly (repo root = 3 levels up); `python -m` does not need this.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# ---------------------------------------------------------------------------
MODEL_NAME    = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN_ID = 32000
PROMPT_SUFFIX  = " Answer the question using a single word or phrase."
N_PATCHES      = 576

QUESTIONS_FILE   = "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FILE = "data/vqav2/v2_mscoco_val2014_annotations.json"
IMAGE_DIR        = "data/vqav2/val2014"


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def vqa_score(pred: str, raw_answers: list) -> float:
    pn = normalize(pred)
    return min(1.0, sum(1 for a in raw_answers if normalize(a) == pn) / 3.0)


def load_samples(n: int):
    with open(QUESTIONS_FILE) as f:
        qs = json.load(f)["questions"]
    with open(ANNOTATIONS_FILE) as f:
        anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
    samples = []
    for q in qs:
        qid, iid = q["question_id"], q["image_id"]
        img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{iid:012d}.jpg")
        if not os.path.exists(img_path):
            continue
        samples.append({
            "question_id": qid,
            "question":    q["question"],
            "image_path":  img_path,
            "raw_answers": [a["answer"] for a in anns.get(qid, {}).get("answers", [])],
        })
        if len(samples) == n:
            break
    return samples


# ---------------------------------------------------------------------------
# Path D — fixed split-at-image-placeholder logic
# ---------------------------------------------------------------------------

def run_path_d(model, processor, samples, k: int, max_new_tokens: int = 10):
    """
    Implements the corrected _build_split_embeds logic directly.
    At K=576 this is a pure ordering check (no information loss).
    """
    device = next(model.parameters()).device
    lm     = model.language_model
    embed  = lm.get_input_embeddings()
    vision = model.vision_tower
    proj   = model.multi_modal_projector

    results = []

    for s in samples:
        question  = s["question"] + PROMPT_SUFFIX
        image     = Image.open(s["image_path"]).convert("RGB")
        conv      = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs    = processor(text=text_prompt, images=image, return_tensors="pt")
        inputs    = {k2: v.to(device) for k2, v in inputs.items() if hasattr(v, "to")}

        ids  = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]

        # ── Vision encoder ──────────────────────────────────────────────────
        pv = inputs["pixel_values"]
        try:
            vdtype = next(vision.parameters()).dtype
        except StopIteration:
            vdtype = pv.dtype
        pv = pv.to(dtype=vdtype)

        with torch.no_grad():
            vout = vision(pixel_values=pv, output_attentions=True,
                          output_hidden_states=True, return_dict=True)

        # Features from penultimate layer (matching feature_layer=-2)
        vis_feats = vout.hidden_states[-2][:, 1:, :]  # [1, 576, Dv]

        # CLS-attention scores from penultimate layer (Fix 2)
        attn_scores = vout.attentions[-2][:, :, 0, 1:].mean(dim=1)  # [1, 576]

        # Select top-K by CLS attention
        if k < N_PATCHES:
            topk_idx = torch.topk(attn_scores, k=k, dim=1).indices  # [1, k]
            topk_idx_sorted = topk_idx.sort(dim=1).values
            gather_idx = topk_idx_sorted.unsqueeze(-1).expand(-1, -1, vis_feats.size(-1))
            vis_feats = torch.gather(vis_feats, dim=1, index=gather_idx)  # [1, k, Dv]

        # Project to LM space
        with torch.no_grad():
            try:
                pdtype = next(proj.parameters()).dtype
            except StopIteration:
                pdtype = vis_feats.dtype
            proj_vis = proj(vis_feats.to(dtype=pdtype))  # [1, k, D_lm]

        # ── Split at <image> placeholder (Fix 1) ────────────────────────────
        valid_ids = ids[mask.bool()]  # strip padding

        img_pos = (valid_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        img_start = int(img_pos[0])
        img_end   = int(img_pos[-1]) + 1

        prefix_ids = valid_ids[:img_start]   # [BOS, SYS, USER:]
        suffix_ids = valid_ids[img_end:]      # [\n, question, ASST:]

        with torch.no_grad():
            prefix_emb = embed(prefix_ids.unsqueeze(0))   # [1, P, D]
            suffix_emb = embed(suffix_ids.unsqueeze(0))   # [1, S, D]

        vis_emb = proj_vis.to(dtype=prefix_emb.dtype)    # [1, k, D]
        lm_embeds = torch.cat([prefix_emb, vis_emb, suffix_emb], dim=1)
        lm_mask   = torch.ones(1, lm_embeds.size(1), dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = lm.generate(
                inputs_embeds=lm_embeds,
                attention_mask=lm_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        pred = processor.decode(gen_ids[0], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True).strip()

        score = vqa_score(pred, s["raw_answers"])
        results.append({"question_id": s["question_id"], "pred": pred,
                         "raw_answers": s["raw_answers"], "score": score})

    acc = sum(r["score"] for r in results) / len(results) if results else 0.0
    return acc, results


# ---------------------------------------------------------------------------
# Path A — standard model.generate
# ---------------------------------------------------------------------------

def run_path_a(model, processor, samples, max_new_tokens: int = 10):
    device = next(model.parameters()).device
    results = []

    for s in samples:
        question = s["question"] + PROMPT_SUFFIX
        image    = Image.open(s["image_path"]).convert("RGB")
        conv     = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text_prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs   = processor(text=text_prompt, images=image, return_tensors="pt")
        inputs   = {k2: v.to(device) for k2, v in inputs.items() if hasattr(v, "to")}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        answer_ids = gen_ids[:, prompt_len:]
        pred = processor.decode(answer_ids[0], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True).strip()

        score = vqa_score(pred, s["raw_answers"])
        results.append({"question_id": s["question_id"], "pred": pred,
                         "raw_answers": s["raw_answers"], "score": score})

    acc = sum(r["score"] for r in results) / len(results) if results else 0.0
    return acc, results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",          type=int, default=576, help="Tokens to keep (576 = no pruning)")
    parser.add_argument("--n-samples",  type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    print(f"[Smoke test] K={args.k}, N={args.n_samples}", flush=True)
    print("[Smoke test] Loading model...", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    print("[Smoke test] Loading samples...", flush=True)
    samples = load_samples(args.n_samples)
    print(f"[Smoke test] Loaded {len(samples)} samples.", flush=True)

    print("\n[Smoke test] === Path A (standard generate, full 576 tokens) ===", flush=True)
    acc_a, _ = run_path_a(model, processor, samples, args.max_new_tokens)
    print(f"  Path A accuracy: {acc_a*100:.1f}%  ({acc_a*100:.1f}%/100)", flush=True)

    print(f"\n[Smoke test] === Path D (fixed wrapper, K={args.k}) ===", flush=True)
    acc_d, results_d = run_path_d(model, processor, samples, k=args.k,
                                   max_new_tokens=args.max_new_tokens)
    print(f"  Path D accuracy: {acc_d*100:.1f}%", flush=True)

    diff = abs(acc_a - acc_d) * 100
    status = "PASS" if diff <= 5.0 else "FAIL"  # ≤5pp diff on 20 samples is OK
    print(f"\n[Smoke test] Δ(A-D) = {diff:.1f}pp  →  {status}", flush=True)

    if args.k == N_PATCHES:
        print("  (K=576: no pruning → ordering fix only → expect Δ ≈ 0pp)", flush=True)
    else:
        print(f"  (K={args.k}: expect Path D ≥ 73% if fix is correct; buggy was 67-68%)", flush=True)

    # Save results
    out_path = f"scripts/smoke_test_results_k{args.k}_n{args.n_samples}.json"
    out = {
        "k": args.k, "n_samples": len(samples),
        "path_a_acc": acc_a, "path_d_acc": acc_d,
        "delta_pp": diff, "status": status,
        "path_d_results": results_d,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Smoke test] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
