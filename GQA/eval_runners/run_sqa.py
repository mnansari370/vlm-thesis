"""
ScienceQA-IMG evaluation — LLaVA-1.5 protocol on our locked honest pipeline (bs=1).

Prompt (LLaVA CQM-A, single-pred; convert_sqa_to_llava_base_prompt + model_vqa_science):
  USER: <image>\nContext: {hint or N/A}\nQuestion: {q}\nOptions: (A) .. (B) ..\n
        Answer with the option's letter from the given choices directly. ASSISTANT:
Metric: official eval_science_qa letter extraction → exact match on choice index.
Data: lmms-lab/ScienceQA, ScienceQA-IMG/test (2,017). Published LLaVA-1.5-7B ≈ 66.8%.

Usage:
  python -m GQA.eval_runners.run_sqa --method none --keep_k 576                 # validate ~66.8%
  python -m GQA.eval_runners.run_sqa --method cls_attn --keep_k 144 --return_confidence
"""

import argparse
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from GQA.shared.static import StaticPrunedLlava
from GQA.shared.flops import flops_row
from GQA.shared.utils.logger import make_output_dir

PARQUET = "data/scienceqa/sqa_img_test.parquet"
OPTIONS = ["A", "B", "C", "D", "E"]
INSTR = "Answer with the option's letter from the given choices directly."
DENSE_REF = 66.8
N_TEXT_SQA = 108  # measured mean non-visual prompt tokens (hints are long)


def build_prompt(q, choices, hint):
    context = (hint or "").strip() or "N/A"
    choice = " ".join(f"({OPTIONS[i]}) {c}" for i, c in enumerate(choices))
    body = f"Context: {context}\nQuestion: {q}\nOptions: {choice}"
    return f"{body}\n{INSTR}"


def extract_letter(text, n_choices):
    """Official eval_science_qa extraction (case-insensitive to honest lowercasing)."""
    t = text.strip().upper()
    if t in OPTIONS:
        ans = t
    elif len(t) >= 3 and t[0] in OPTIONS and t[1:3] == ". ":
        ans = t[0]
    elif len(t) >= 2 and t[0] in OPTIONS and not t[1].isalnum():
        ans = t[0]
    else:
        m = re.findall(r"THE ANSWER IS ([A-E])", t)
        ans = m[0] if len(m) == 1 else "FAILED"
    return OPTIONS.index(ans) if ans in OPTIONS[:n_choices] else -1


class SQADataset(Dataset):
    def __init__(self, parquet, max_samples=None):
        self.rows = pq.read_table(parquet).to_pylist()
        if max_samples:
            self.rows = self.rows[:max_samples]
        print(f"[Data] {len(self.rows)} SQA-IMG test samples", flush=True)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        return {"qid": i, "prompt": build_prompt(r["question"], r["choices"], r["hint"]),
                "n_choices": len(r["choices"]), "answer": int(r["answer"]), "image": img}


def collate(b): return {k: [x[k] for x in b] for k in b[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["none", "random", "spatial_uniform", "cls_attn", "l2_norm"])
    ap.add_argument("--keep_k", required=True, type=int, choices=[576, 432, 288, 192, 144, 96, 64])
    ap.add_argument("--return_confidence", action="store_true")
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.method == "none" and args.keep_k != 576:
        ap.error("--method none requires --keep_k 576")

    out_dir = make_output_dir("outputs", args.output_name or f"sqa_{args.method}_k{args.keep_k}")
    print(f"[Output] {out_dir}\n[Config] method={args.method} K={args.keep_k}", flush=True)
    ds = SQADataset(PARQUET, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    model = StaticPrunedLlava(method=args.method, keep_k=args.keep_k, seed=args.seed,
                              image_pad=True, honest=True, append_suffix=False)

    per_sample, n_correct = [], 0
    t0 = time.time()
    for step, b in enumerate(loader):
        if args.return_confidence:
            ans, conf = model.generate_answers(b["image"], b["prompt"], sample_offset=step,
                                               max_new_tokens=64, return_confidence=True)
            c = conf[0]
        else:
            ans = model.generate_answers(b["image"], b["prompt"], sample_offset=step, max_new_tokens=64)
            c = None
        pidx = extract_letter(ans[0], b["n_choices"][0])
        ok = (pidx == b["answer"][0])
        n_correct += int(ok)
        rec = {"question_id": b["qid"][0], "pred_idx": pidx, "answer": b["answer"][0], "correct": ok}
        if c is not None:
            rec["confidence"] = c
        per_sample.append(rec)
        if (step + 1) % args.log_every == 0:
            acc = n_correct / len(per_sample) * 100
            sps = (step + 1) / (time.time() - t0)
            print(f"  {len(per_sample)}/{len(ds)} acc={acc:.2f}% {sps:.1f} samp/s "
                  f"ETA={(len(ds)-len(per_sample))/max(sps,1e-6)/60:.1f}min", flush=True)

    acc = round(n_correct / len(per_sample) * 100, 2)
    flops = flops_row(args.keep_k, n_text=N_TEXT_SQA)
    print(f"\n  SQA-IMG {args.method} K={args.keep_k}: acc={acc}% (ref {DENSE_REF}, diff {acc-DENSE_REF:+.2f}pp) "
          f"n={len(per_sample)} FLOPs={flops['fastv_full_TFLOPs']:.4f}T", flush=True)
    out = {"method": args.method, "keep_k": args.keep_k, "accuracy_pct": acc,
           "n_evaluated": len(per_sample), "reference_pct": DENSE_REF, "diff_pp": round(acc - DENSE_REF, 2),
           "flops": flops, "elapsed_hours": round((time.time() - t0) / 3600, 3)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "per_sample.json"), "w") as f:
        json.dump(per_sample, f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
