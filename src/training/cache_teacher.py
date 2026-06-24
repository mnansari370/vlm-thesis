"""
Step 1 of the cheap-selector distillation pilot.

Cache the mid-layer (L16) question->visual attention "TEACHER" importance map for each
DocVQA sample. This map is the EXPENSIVE signal: it needs a full Qwen2.5-VL forward
(ViT + 28-layer decoder with output_attentions). We cache it ONCE so the student trainer
never re-runs the decoder — the student only ever sees the cheap pre-LLM features.

What the student will predict (Step 2): this teacher map, from cheap pre-LLM features
(ViT visual embeds + question token embeds), which the trainer recomputes on the fly via
QwenPruner._encode (ViT only — no decoder).

LEAKAGE GUARD (pilot): the locally-cached DocVQA config has only validation+test
(the hub 'train' split is not downloaded). So the pilot caches a DETERMINISTIC SLICE of
validation [start:end); the student trains on this slice and the gate (eval_gate.py) uses
a DISJOINT tail of the same split. eval_gate hard-asserts the ranges don't overlap.
For the final paper, download the real DocVQA 'train' split and pass --split train; the
script is otherwise unchanged. We never cache features here (too large); only the
per-sample teacher vector + identity (the row index, so the trainer can reload the image).

Output (torch .pt): {
  "meta": {model, config, split, start, end, textattn_layer, max_pixels, instr},
  "rows": [ {idx, question, answers, nvis, qscores(fp16 cpu [nvis])}, ... ]   # idx = split row index
}

Run in qwen_env, from repo root (pilot TRAIN slice = validation[0:3000]):
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/qwen_env/bin/python -m src.training.cache_teacher \
      --split validation --start 0 --end 3000 --out results/thesis_main/highres/distill/teacher_docvqa_train.pt
Smoke (2 samples):
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/qwen_env/bin/python -m src.training.cache_teacher \
      --split validation --start 0 --end 2 --out /tmp/teacher_smoke.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.data.docvqa import load_docvqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation"],
                    help="DocVQA split. Pilot uses a slice of 'validation' (train not local); "
                         "switch to 'train' for the paper once downloaded.")
    ap.add_argument("--config", default="DocVQA", choices=["DocVQA", "InfographicVQA"])
    ap.add_argument("--n-shards", type=int, default=None,
                    help="train only: number of parquet shards to load (~3289 rows each)")
    ap.add_argument("--start", type=int, default=0, help="first row index (inclusive)")
    ap.add_argument("--end", type=int, default=3000, help="last row index (exclusive)")
    ap.add_argument("--textattn-layer", type=int, default=16,
                    help="Teacher layer (Qwen-7B best mid-layer from the layer sweep).")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28,
                    help="MUST match the resolution used at eval/gate time.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="results/thesis_main/highres/distill/teacher_docvqa_train.pt")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    assert args.end > args.start >= 0, "require 0 <= start < end"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── resume support: reload already-cached rows, skip their indices ──────────
    rows = []
    done = set()
    if os.path.exists(args.out):
        prev = torch.load(args.out, map_location="cpu")
        rows = prev.get("rows", [])
        done = {r["idx"] for r in rows}
        print(f"[cache] resuming: {len(rows)} rows already cached", flush=True)

    print(f"[cache] loading DocVQA split={args.split} (n_shards={args.n_shards}) ...", flush=True)
    ds = load_docvqa(args.split, n_shards=args.n_shards)
    end = min(args.end, len(ds))
    n = end - args.start
    print(f"[cache] split has {len(ds)} rows; caching [{args.start}:{end}) = {n} rows", flush=True)

    pruner = QwenPruner(model_name=args.model, textattn_layer=args.textattn_layer,
                        max_pixels=args.max_pixels)

    meta = {"model": args.model, "config": args.config, "split": args.split,
            "n_shards": args.n_shards, "start": args.start, "end": end,
            "textattn_layer": args.textattn_layer, "max_pixels": args.max_pixels, "instr": True}

    def flush():
        torch.save({"meta": meta, "rows": rows}, args.out)

    t0 = time.time()
    n_new = 0
    for i in range(args.start, end):
        if i in done:
            continue
        ex = ds[i]
        img = ex["image"]
        q = ex["question"]
        answers = [str(a) for a in ex.get("answers", [])]
        try:
            with torch.no_grad():
                # encode_qc runs ViT + full decoder(output_attentions) -> teacher qscores [nvis]
                _emb, _pos, _attn, _v0, nvis, qscores = pruner.encode_qc(img, q, instr=True)
            rows.append({
                "idx": i,
                "question": q,
                "answers": answers,
                "nvis": int(nvis),
                # store compact: cpu fp16 (positive attention masses; fp16 is ample)
                "qscores": qscores.detach().to(torch.float16).cpu(),
            })
            n_new += 1
        except Exception as e:
            print(f"[cache] skip idx={i}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

        if n_new % args.log_every == 0:
            sps = n_new / max(time.time() - t0, 1e-6)
            eta = (n - len(rows)) / max(sps, 1e-6) / 60
            print(f"[cache] {len(rows)}/{n}  ({sps:.2f} samp/s, ETA {eta:.1f} min, "
                  f"last nvis={rows[-1]['nvis']})", flush=True)
        if n_new % args.save_every == 0:
            flush()

    flush()
    print(f"[cache] DONE: {len(rows)} rows -> {args.out}  "
          f"({(time.time()-t0)/60:.1f} min, {n_new} new)", flush=True)


if __name__ == "__main__":
    main()
