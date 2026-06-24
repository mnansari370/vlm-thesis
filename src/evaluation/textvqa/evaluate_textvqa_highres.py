"""
Step 0a of Option A — the cheapest possible high-res gate.

Run the STOCK frozen LLaVA-1.6-vicuna-7b (AnyRes, ~2880 visual tokens) at FULL
tokens (NO pruning) on the SAME TextVQA-no-OCR set we ran LLaVA-1.5 on (42.23%).
Question this answers: does high-res even READ better than the 336px postage-stamp
LLaVA-1.5? If yes (a clear lift over 42%), there is real visual information that
pruning could later remove -> the regime is worth the engineering for the K-curve
(Step 0b). If high-res is ALSO ~42%, high-res reads the menu no better than the
stamp -> there is nothing to prune-away -> stop before building the pruner.

Same vision tower (CLIP-336) and same LLM (Vicuna-7B-v1.5) as LLaVA-1.5, so the
ONLY variable is AnyRes high-res tiling -> a clean apples-to-apples contrast.

No custom wrapper, no pruning: stock HF LlavaNext + greedy generation (bs=1, the
locked honest protocol) + our official VQA soft-acc scorer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.textvqa.evaluate_textvqa_highres --max-samples 300
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.metrics.textvqa_score import score_textvqa

MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
TEXTVQA_ANN = "data/textvqa/TextVQA_0.5.1_val.json"
TEXTVQA_IMG = "data/textvqa/train_images"
TEXTVQA_INSTR = "Answer the question using a single word or phrase."
LLAVA15_NOOCR = 42.23   # our measured LLaVA-1.5 full-576 no-OCR soft-acc (the thing to beat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--output", default="results/thesis_main/highres/eval_textvqa_highres_full.json")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hires] loading {MODEL} (bfloat16) ...", flush=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(dev)
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(MODEL)
    eos = processor.tokenizer.eos_token_id

    data = json.load(open(TEXTVQA_ANN))["data"][: args.max_samples]

    preds = []
    n = 0
    n_tokens_seen = []   # record how many visual tokens AnyRes actually produced
    t0 = time.time()
    for rec in data:
        ipath = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
        if not os.path.exists(ipath):
            continue
        img = Image.open(ipath).convert("RGB")
        q_clean = rec["question"].strip()
        q_fmt = q_clean + " " + TEXTVQA_INSTR
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q_fmt}]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inp = processor(images=img, text=prompt, return_tensors="pt").to(dev)

        # how many visual tokens did AnyRes expand this image to? (= image_token_index count)
        n_vis = int((inp["input_ids"][0] == model.config.image_token_index).sum())
        n_tokens_seen.append(n_vis)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1, pad_token_id=eos, use_cache=True)
        gen = out[0, inp["input_ids"].shape[1]:]          # only the newly generated tokens
        pred = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

        preds.append({"question_id": rec["image_id"], "question": q_clean, "pred_answer": pred})
        n += 1
        if n % args.log_every == 0:
            r = score_textvqa(preds)
            print(f"  {n}/{len(data)}  soft-acc={r['accuracy_pct']:.1f}%  "
                  f"(~{sum(n_tokens_seen)//len(n_tokens_seen)} vis-tokens/img)", flush=True)

    res = score_textvqa(preds)
    avg_tok = sum(n_tokens_seen) // max(len(n_tokens_seen), 1)
    out = {
        "model": MODEL, "condition": "no_ocr_full_tokens", "n": res["n_evaluated"],
        "soft_acc": res["accuracy_pct"], "binary_acc": res["binary_correct_pct"],
        "avg_visual_tokens": avg_tok, "max_visual_tokens": max(n_tokens_seen),
        "llava15_noocr_ref": LLAVA15_NOOCR, "lift_vs_llava15": round(res["accuracy_pct"] - LLAVA15_NOOCR, 2),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print("\n" + "=" * 60)
    print(f"  LLaVA-1.6 (AnyRes, ~{avg_tok} vis-tokens) TextVQA no-OCR FULL")
    print(f"  soft-acc        : {out['soft_acc']:.2f}%   (n={out['n']})")
    print(f"  LLaVA-1.5 (576) : {LLAVA15_NOOCR:.2f}%")
    print(f"  LIFT from high-res: {out['lift_vs_llava15']:+.2f} pp")
    gate = "PASS -> room may exist, proceed to 0b K-curve" if out["lift_vs_llava15"] >= 8 \
        else ("WEAK -> small lift, room likely thin" if out["lift_vs_llava15"] >= 3
              else "FAIL -> high-res doesn't help reading; no room to prune")
    print(f"  GATE            : {gate}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({"summary": out, "samples": res["per_sample"]}, f, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
