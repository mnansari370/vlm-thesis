"""
TextVQA evaluation — LLaVA-1.5 protocol on our locked honest pipeline (bs=1).

Prompt (with OCR, the LLaVA-1.5 published setting):
    USER: <image>\n{question}\nReference OCR token: {ocr}\nAnswer the question
    using a single word or phrase. ASSISTANT:
(--use_ocr off drops the "Reference OCR token:" line — the no-OCR / visual-only setting.)

Protocol: image_pad=True · greedy · max_new_tokens=64 · no min_new_tokens · no
repetition_penalty · bs=1. Scoring: GQA/shared/textvqa_score.py (official VQA soft-acc).

Reuses StaticPrunedLlava with append_suffix=False (TextVQA prompts already contain
the instruction). Saves per-sample soft-acc (for the C0.4 oracle headroom).

Usage:
    python -m GQA.eval_runners.run_textvqa --method none --keep_k 576              # dense, validate ~58.2%
    python -m GQA.eval_runners.run_textvqa --method cls_attn --keep_k 288
    python -m GQA.eval_runners.run_textvqa --method none --keep_k 576 --use_ocr off  # no-OCR
    python -m GQA.eval_runners.run_textvqa --method cls_attn --keep_k 144 --return_confidence
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from GQA.shared.textvqa_score import score_textvqa, print_textvqa
from GQA.shared.static import StaticPrunedLlava
from GQA.shared.utils.logger import make_output_dir

JSONL = "data/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_DIR = "data/textvqa/train_images"
INSTR = "Answer the question using a single word or phrase."
DENSE_REF = 58.2


class TextVQADataset(Dataset):
    def __init__(self, jsonl, image_dir, use_ocr=True, max_samples=None):
        self.image_dir = image_dir
        self.use_ocr = use_ocr
        self.records = []
        with open(jsonl) as f:
            for line in f:
                self.records.append(json.loads(line))
        if max_samples:
            self.records = self.records[:max_samples]
        print(f"[Data] {len(self.records):,} TextVQA val samples (use_ocr={use_ocr})", flush=True)

    def __len__(self): return len(self.records)

    def _build_prompt(self, text):
        # text field = "{question}\nReference OCR token: {ocr}\n{INSTR}"
        lines = text.split("\n")
        question = lines[0]
        if self.use_ocr:
            return text, question          # full prompt incl OCR
        # no-OCR: question + instruction only
        return f"{question}\n{INSTR}", question

    def __getitem__(self, idx):
        r = self.records[idx]
        prompt, question = self._build_prompt(r["text"])
        return {
            "question_id": r["question_id"],   # = image_id (LLaVA convention)
            "question": question,
            "prompt": prompt,
            "image": Image.open(os.path.join(self.image_dir, r["image"])).convert("RGB"),
        }


def collate(b):
    return {k: [x[k] for x in b] for k in b[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["none", "random", "spatial_uniform", "cls_attn", "l2_norm"])
    ap.add_argument("--keep_k", required=True, type=int, choices=[576, 432, 288, 192, 144, 96, 64])
    ap.add_argument("--use_ocr", choices=["on", "off"], default="on")
    ap.add_argument("--return_confidence", action="store_true")
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.method == "none" and args.keep_k != 576:
        ap.error("--method none requires --keep_k 576")

    use_ocr = args.use_ocr == "on"
    name = args.output_name or (
        f"textvqa_{args.method}_k{args.keep_k}_{'ocr' if use_ocr else 'noocr'}"
        + (f"_n{args.max_samples}" if args.max_samples else ""))
    out_dir = make_output_dir("outputs", name)
    print(f"[Output] {out_dir}", flush=True)
    print(f"[Config] method={args.method} K={args.keep_k} use_ocr={use_ocr} bs=1 image_pad=True", flush=True)

    ds = TextVQADataset(JSONL, IMAGE_DIR, use_ocr=use_ocr, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = StaticPrunedLlava(method=args.method, keep_k=args.keep_k, seed=args.seed,
                              image_pad=True, honest=True, append_suffix=False)

    preds = []
    t0 = time.time()
    for step, b in enumerate(loader):
        if args.return_confidence:
            ans, conf = model.generate_answers(b["image"], b["prompt"], sample_offset=step,
                                               max_new_tokens=64, return_confidence=True)
            c = conf[0]
        else:
            ans = model.generate_answers(b["image"], b["prompt"], sample_offset=step, max_new_tokens=64)
            c = None
        rec = {"question_id": b["question_id"][0], "question": b["question"][0],
               "pred_answer": ans[0], "prompt": b["prompt"][0]}
        if c is not None:
            rec["confidence"] = c
        preds.append(rec)
        if (step + 1) % args.log_every == 0:
            r = score_textvqa(preds)
            sps = (step + 1) / (time.time() - t0)
            print(f"  {len(preds):>5}/{len(ds):,} acc={r['accuracy_pct']:.2f}% "
                  f"{sps:.1f} samp/s ETA={(len(ds)-len(preds))/max(sps,1e-6)/60:.1f}min", flush=True)

    res = score_textvqa(preds)
    print_textvqa(res, label=f"TextVQA {args.method} K={args.keep_k} use_ocr={use_ocr}", reference=DENSE_REF)
    print(f"  retention vs dense: needs dense ref; elapsed={(time.time()-t0)/3600:.2f}h", flush=True)

    out = {"method": args.method, "keep_k": args.keep_k, "use_ocr": use_ocr,
           "accuracy_pct": res["accuracy_pct"], "binary_correct_pct": res["binary_correct_pct"],
           "n_evaluated": res["n_evaluated"], "n_missing": res["n_missing"],
           "reference_pct": DENSE_REF, "diff_pp": round(res["accuracy_pct"] - DENSE_REF, 2),
           "elapsed_hours": round((time.time() - t0) / 3600, 3)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": preds}, f)
    with open(os.path.join(out_dir, "per_sample_scores.json"), "w") as f:
        json.dump(res["per_sample"], f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
