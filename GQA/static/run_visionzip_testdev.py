"""
B2b — VisionZip (dominant + contextual merge) on GQA testdev, honest protocol bs=1.

Same locked protocol as run_static_testdev.py. VisionZip is prune-before-LLM
(merged token set fed to the LLM), so it shares the static FLOPs basis (n=K+34,
all 32 layers see keep_k tokens).

Validation: run at keep_k=64 (= 54 dominant + 10 contextual, the paper's documented
config) and compare to VisionZip's published GQA testdev number (~57-58, training-free).

Usage:
    python -m GQA.static.run_visionzip_testdev --keep_k 288
    python -m GQA.static.run_visionzip_testdev --keep_k 64 --max_samples 500   # validation
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
from GQA.static.visionzip import VisionZipLlava
from GQA.shared.flops import flops_row_testdev
from GQA.shared.utils.logger import make_output_dir


QUESTIONS = "data/gqa/testdev_balanced_questions.json"
IMAGE_DIR = "data/gqa/images/images"
DENSE_REF = 61.42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_k", required=True, type=int)
    ap.add_argument("--k_contextual", type=int, default=None,
                    help="contextual tokens (default round(keep_k*10/64))")
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    name = args.output_name or (
        f"testdev_visionzip_k{args.keep_k}" + (f"_n{args.max_samples}" if args.max_samples else ""))
    out_dir = make_output_dir("outputs", name)
    print(f"[Output] {out_dir}", flush=True)

    dataset = GQATestdevDataset(QUESTIONS, IMAGE_DIR, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    model = VisionZipLlava(keep_k=args.keep_k, k_contextual=args.k_contextual,
                           image_pad=True, honest=True)

    preds = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        ans = model.generate_answers(batch["images"], batch["questions"],
                                     sample_offset=step, max_new_tokens=64)
        preds.append({
            "question_id": batch["question_ids"][0], "question": batch["questions"][0],
            "pred_answer": ans[0], "answer": batch["answers"][0],
            "semantic_type": batch["semantic_types"][0],
        })
        if (step + 1) % args.log_every == 0:
            scored = score_val_format(preds, {})
            sps = (step + 1) / (time.time() - t0)
            print(f"  {len(preds):>6}/{len(dataset):,} acc={scored['accuracy_pct']:.2f}% "
                  f"{sps:.1f} samp/s ETA={(len(dataset)-len(preds))/max(sps,1e-6)/60:.1f}min", flush=True)

    scored = score_val_format(preds, {})
    print_result(scored, label=f"VisionZip keep_k={args.keep_k} "
                 f"({model.k_dominant}dom+{model.k_contextual}ctx, testdev)", reference=DENSE_REF)
    flops = flops_row_testdev(args.keep_k, method="static")
    print(f"  retention vs dense({DENSE_REF}%): {scored['accuracy_pct']/DENSE_REF*100:.2f}%")
    print(f"  FLOPs LM-full(n=K+34): {flops['fastv_full_TFLOPs']:.4f}T "
          f"({flops['fastv_full_reduction_pct']:.1f}% reduction)")

    result = {
        "method": "visionzip", "keep_k": args.keep_k,
        "k_dominant": model.k_dominant, "k_contextual": model.k_contextual,
        "split": "testdev_balanced", "protocol": "honest bs=1 image_pad",
        "n_evaluated": scored["n_total"], "accuracy_pct": scored["accuracy_pct"],
        "per_type": scored["per_type"],
        "retention_pct": round(scored["accuracy_pct"] / DENSE_REF * 100, 2),
        "flops": flops, "elapsed_hours": round((time.time() - t0) / 3600, 3),
        "merge_basis": "layer-(-2) feature cosine (keys up to linear proj)",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": preds}, f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
