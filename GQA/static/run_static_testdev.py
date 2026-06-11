"""
Static visual-token-pruning evaluation on GQA testdev_balanced — LOCKED honest protocol.

Protocol (every run, no exceptions):
  image_pad=True · vicuna_v1 + "Answer the question using a single word or phrase."
  · greedy · max_new_tokens=64 · no min_new_tokens · no repetition_penalty
  · batch_size=1 · scorer = official_score.py (rstrip('.').lower(), strict)
  · testdev_balanced (12,578)

Saves per-sample predictions (needed for the B1d oracle-headroom diagnostic).

Sanity: --method none --keep_k 576 must reproduce the dense bs=1 baseline (61.42%),
because static-none with image_pad builds the SAME [prefix | 576 projected | suffix]
sequence the dense HF path produces internally.

Usage
-----
  # sanity (must match dense 61.42% on the subset):
  python -m GQA.static.run_static_testdev --method none --keep_k 576 --max_samples 200

  # full frontier point:
  python -m GQA.static.run_static_testdev --method cls_attn --keep_k 288
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from GQA.shared.official_score import score_val_format, print_result
from GQA.dense.run_dense_testdev import GQATestdevDataset, collate
from GQA.shared.static import StaticPrunedLlava
from GQA.shared.flops import flops_row_testdev
from GQA.shared.utils.logger import make_output_dir


QUESTIONS = "data/gqa/testdev_balanced_questions.json"
IMAGE_DIR = "data/gqa/images/images"
DENSE_REF = 61.42   # locked honest dense baseline (bs=1, image_pad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["none", "random", "spatial_uniform", "cls_attn", "l2_norm"])
    ap.add_argument("--keep_k", required=True, type=int,
                    choices=[576, 432, 288, 192, 144, 96, 64])
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--image_dir", default=IMAGE_DIR)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.method == "none" and args.keep_k != 576:
        ap.error("--method none requires --keep_k 576")

    name = args.output_name or (
        f"testdev_static_{args.method}_k{args.keep_k}"
        + (f"_n{args.max_samples}" if args.max_samples else "")
    )
    out_dir = make_output_dir("outputs", name)
    print(f"[Output] {out_dir}", flush=True)
    print(f"[Config] method={args.method} K={args.keep_k} bs=1 image_pad=True honest=True",
          flush=True)

    dataset = GQATestdevDataset(args.questions, args.image_dir, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=torch.cuda.is_available())

    model = StaticPrunedLlava(method=args.method, keep_k=args.keep_k,
                              seed=args.seed, image_pad=True, honest=True)

    predictions: list[dict] = []
    n_correct = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        preds = model.generate_answers(
            images=batch["images"], questions=batch["questions"],
            sample_offset=step, max_new_tokens=64,
        )
        for qid, pred, gold, q, st in zip(
            batch["question_ids"], preds, batch["answers"],
            batch["questions"], batch["semantic_types"]
        ):
            predictions.append({
                "question_id": qid, "question": q,
                "pred_answer": pred, "answer": gold, "semantic_type": st,
            })

        if (step + 1) % args.log_every == 0:
            scored = score_val_format(predictions, {})
            sps = (step + 1) / (time.time() - t0)
            eta = (len(loader) - step - 1) / max(sps, 1e-6) / 60
            print(f"  {len(predictions):>6}/{len(dataset):,}  "
                  f"acc={scored['accuracy_pct']:.2f}%  {sps:.1f} samp/s  ETA={eta:.1f}min",
                  flush=True)

    elapsed_h = (time.time() - t0) / 3600
    scored = score_val_format(predictions, {})
    print_result(scored, label=f"static {args.method} K={args.keep_k} (testdev, honest)",
                 reference=DENSE_REF)

    flops = flops_row_testdev(args.keep_k, label=f"{args.method} K={args.keep_k}")
    retention = round(scored["accuracy_pct"] / DENSE_REF * 100, 2)
    print(f"  Retention vs dense({DENSE_REF}%): {retention}%")
    print(f"  FLOPs LM-full(n=K+34): {flops['fastv_full_TFLOPs']:.4f}T "
          f"({flops['fastv_full_reduction_pct']:.1f}% reduction)  "
          f"attn-only: {flops['attention_only_GFLOPs']:.1f}G")
    print(f"  Elapsed: {elapsed_h:.2f}h", flush=True)

    result = {
        "method": args.method, "keep_k": args.keep_k,
        "split": "testdev_balanced", "protocol": "honest bs=1 image_pad",
        "n_evaluated": scored["n_total"], "n_correct": scored["n_correct"],
        "accuracy_pct": scored["accuracy_pct"], "n_empty": scored["n_empty"],
        "per_type": scored["per_type"],
        "retention_pct": retention, "dense_ref_pct": DENSE_REF,
        "flops": flops, "elapsed_hours": round(elapsed_h, 3),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": predictions}, f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
