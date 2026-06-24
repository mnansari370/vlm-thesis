"""
Evaluate the elastic model (or the frozen base) on TextVQA val (generation protocol)
at one or more K, using the OFFICIAL VQA soft-accuracy scorer.

DECISIVE v2 DIAGNOSTIC — "TextVQA WITHOUT OCR":
    On GQA the frozen base degrades GRACEFULLY at low K (58.7 -> 47 from K=576 -> 32),
    so Stage-1 elastic training had nothing to fix and dynamic budgeting has no room.
    Reading is the opposite regime: you cannot read a sign from 32 scattered patches.
    If the frozen base COLLAPSES at low K here (steep drop K=576 -> 32), that is the
    target a question-conditioned selector (Stage 2) could exploit -> v2 has a real
    contribution. If it also degrades gracefully, the v2 premise is weak across the board.

We feed the CLEAN question only (NO "Reference OCR token: ..." hint) so the model must
read the image itself -- that is the regime where the visual-token budget actually bites.

Scoring: official VQA soft accuracy (m4c_evaluator), the standard TextVQA metric. NOT
exact match. v1 reference TextVQA *with* OCR was 57.65%; without OCR the base is expected
to be lower even at K=576 -- the SHAPE of the K-curve is what this diagnostic is about.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.textvqa.evaluate_textvqa \
        --base --k-values 576,128,64,32 --max-samples 300
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.textvqa.evaluate_textvqa \
        --checkpoint results/archived/stage1_quickfix/final.pt --k-values 576,128,64,32
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

from src.models.elastic.elastic_wrapper import ElasticPrunedLlava
from src.metrics.textvqa_score import score_textvqa

TEXTVQA_ANN = "data/textvqa/TextVQA_0.5.1_val.json"
TEXTVQA_IMG = "data/textvqa/train_images"   # TextVQA val images live in the OpenImages pool
V1_OCR_REF = 57.65   # v1 frozen dense TextVQA *with* OCR, for reference (this is no-OCR)
# Same benchmark instruction the LLaVA TextVQA jsonl uses; the elastic model's short-answer
# training samples were formatted with this, and the frozen base expects it too.
TEXTVQA_INSTR = "Answer the question using a single word or phrase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/archived/stage1_quickfix/final.pt")
    ap.add_argument("--base", action="store_true",
                    help="eval the UNTRAINED base model (no checkpoint) — the frozen baseline")
    ap.add_argument("--k-values", default="576,128,64,32")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    Ks = [int(x) for x in args.k_values.split(",")]
    if args.base:
        print("[eval] UNTRAINED BASE model (no checkpoint) — TextVQA no-OCR")
        m = ElasticPrunedLlava(lora_rank=16, dtype="bfloat16", image_pad=True,
                               gradient_checkpointing=False)
        m.eval()
        ckpt_label = "BASE"
    else:
        m = ElasticPrunedLlava.from_checkpoint(args.checkpoint)
        ckpt_label = args.checkpoint

    data = json.load(open(TEXTVQA_ANN))["data"]
    data = data[: args.max_samples]      # deterministic first-N

    out = {"checkpoint": ckpt_label, "condition": "no_ocr", "n": len(data),
           "v1_ocr_ref": V1_OCR_REF, "per_k": {}}
    all_samples = {}
    for K in Ks:
        t0 = time.time()
        preds = []
        n = 0
        for rec in data:
            ipath = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
            if not os.path.exists(ipath):
                continue
            img = Image.open(ipath).convert("RGB")
            q_clean = rec["question"].strip()
            q_fmt = q_clean + " " + TEXTVQA_INSTR
            pred = m.generate_answers([img], [q_fmt], K=K,
                                      max_new_tokens=args.max_new_tokens)[0]
            # scorer keys annotations by (image_id, question.lower()); question must be CLEAN
            preds.append({"question_id": rec["image_id"], "question": q_clean,
                          "pred_answer": pred})
            n += 1
            if n % args.log_every == 0:
                r = score_textvqa(preds)
                print(f"  K={K} {n}/{len(data)}  soft-acc={r['accuracy_pct']:.1f}%", flush=True)

        res = score_textvqa(preds)
        out["per_k"][K] = {"n": res["n_evaluated"], "soft_acc": res["accuracy_pct"],
                           "binary_acc": res["binary_correct_pct"],
                           "n_missing": res["n_missing"],
                           "minutes": round((time.time() - t0) / 60, 1)}
        all_samples[K] = res["per_sample"]
        print(f"\n[TextVQA(no-OCR) K={K}]  soft-acc={res['accuracy_pct']}%   "
              f"binary={res['binary_correct_pct']}%   (n={res['n_evaluated']}, "
              f"v1 dense WITH-OCR ref={V1_OCR_REF}%)", flush=True)

    # collapse summary: drop from the largest K to the smallest
    Ks_sorted = sorted(out["per_k"].keys())
    hi, lo = max(Ks_sorted), min(Ks_sorted)
    drop = out["per_k"][hi]["soft_acc"] - out["per_k"][lo]["soft_acc"]
    out["drop_hiK_to_loK"] = round(drop, 2)
    print(f"\n=== TextVQA no-OCR — {ckpt_label} — soft-acc by K ===")
    for K in sorted(Ks, reverse=True):
        print(f"  K={K:>3}  soft-acc={out['per_k'][K]['soft_acc']:.2f}%")
    print(f"  drop K={hi} -> K={lo}: {drop:+.2f} pp  "
          f"(GQA base drop was ~11.7pp/graceful; a STEEP drop here = Stage-2 target)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": out, "samples": all_samples}, f, indent=2)
        print(f"[saved] {args.output}")
    print("\n" + json.dumps(out["per_k"], indent=2))


if __name__ == "__main__":
    main()
