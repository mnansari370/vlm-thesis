"""
POPE evaluation — LLaVA-1.5 protocol on our locked honest pipeline (bs=1).

3 subsets (random/popular/adversarial), yes/no object-hallucination on COCO val2014.
Prompt: the POPE question as-is (bare; --use_suffix to append the instruction).
Metric: official eval_pope logic (acc/precision/recall/F1/yes_ratio) per subset.
Reuses StaticPrunedLlava (image_pad, honest, append_suffix). Saves per-sample correctness.

Validate: dense (method none, K=576) should reproduce LLaVA-1.5-7B POPE (~85.9% acc / ~84-85 F1).

Usage:
  python -m GQA.eval_runners.run_pope --method none --keep_k 576
  python -m GQA.eval_runners.run_pope --method cls_attn --keep_k 288
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from GQA.shared.pope_score import score_subset
from GQA.shared.static import StaticPrunedLlava
from GQA.shared.flops import flops_row, N_TEXT_POPE
from GQA.shared.utils.logger import make_output_dir

COCO_DIR = "data/pope/coco"
IMAGE_DIR = "data/vqav2/val2014"
SUBSETS = ["random", "popular", "adversarial"]
DENSE_REF_ACC = 85.9


class POPESubset(Dataset):
    def __init__(self, subset, image_dir, max_samples=None):
        self.image_dir = image_dir
        self.records = [json.loads(l) for l in open(f"{COCO_DIR}/coco_pope_{subset}.json")]
        if max_samples:
            self.records = self.records[:max_samples]

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {"question_id": r["question_id"], "text": r["text"], "label": r["label"],
                "image": Image.open(os.path.join(self.image_dir, r["image"])).convert("RGB")}


def collate(b): return {k: [x[k] for x in b] for k in b[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["none", "random", "spatial_uniform", "cls_attn", "l2_norm"])
    ap.add_argument("--keep_k", required=True, type=int, choices=[576, 432, 288, 192, 144, 96, 64])
    ap.add_argument("--use_suffix", action="store_true", help="append the single-word instruction")
    ap.add_argument("--save_conf", action="store_true", help="save per-sample first-token confidence")
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.method == "none" and args.keep_k != 576:
        ap.error("--method none requires --keep_k 576")

    out_dir = make_output_dir("outputs", args.output_name or f"pope_{args.method}_k{args.keep_k}")
    print(f"[Output] {out_dir}\n[Config] method={args.method} K={args.keep_k} use_suffix={args.use_suffix}", flush=True)

    model = StaticPrunedLlava(method=args.method, keep_k=args.keep_k, seed=args.seed,
                              image_pad=True, honest=True, append_suffix=args.use_suffix)

    all_results = {}
    per_sample_all = {}
    t0 = time.time()
    for sub in SUBSETS:
        ds = POPESubset(sub, IMAGE_DIR, max_samples=args.max_samples)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        answers, confs = [], []
        for step, b in enumerate(loader):
            if args.save_conf:
                ans, c = model.generate_answers(b["image"], b["text"], sample_offset=step,
                                                max_new_tokens=64, return_confidence=True)
                confs.append(c[0])
            else:
                ans = model.generate_answers(b["image"], b["text"], sample_offset=step, max_new_tokens=64)
            answers.append({"question_id": b["question_id"][0], "text": ans[0]})
        res = score_subset(answers, f"{COCO_DIR}/coco_pope_{sub}.json")
        ps = res.pop("per_sample")
        if args.save_conf:
            for i, p in enumerate(ps):
                p["confidence"] = confs[i]
        per_sample_all[sub] = ps
        all_results[sub] = res
        print(f"  [{sub}] acc={res['accuracy_pct']:.2f}% F1={res['f1']:.2f} "
              f"yes_ratio={res['yes_ratio']:.3f} (n={res['n']})", flush=True)

    # overall = mean over subsets (standard POPE reporting)
    mean_acc = round(sum(all_results[s]["accuracy_pct"] for s in SUBSETS) / 3, 2)
    mean_f1 = round(sum(all_results[s]["f1"] for s in SUBSETS) / 3, 2)
    flops = flops_row(args.keep_k, n_text=N_TEXT_POPE)
    elapsed_h = (time.time() - t0) / 3600
    print(f"\n  MEAN acc={mean_acc}% F1={mean_f1}  (dense ref ~{DENSE_REF_ACC}%)")
    print(f"  diff={mean_acc-DENSE_REF_ACC:+.2f}pp  FLOPs(n=K+21)={flops['fastv_full_TFLOPs']:.4f}T "
          f"({flops['fastv_full_reduction_pct']:.1f}% reduction)  elapsed={elapsed_h:.2f}h", flush=True)

    out = {"method": args.method, "keep_k": args.keep_k, "use_suffix": args.use_suffix,
           "per_subset": all_results, "mean_accuracy_pct": mean_acc, "mean_f1": mean_f1,
           "dense_ref_acc": DENSE_REF_ACC, "diff_pp": round(mean_acc - DENSE_REF_ACC, 2),
           "flops": flops, "elapsed_hours": round(elapsed_h, 3)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "per_sample.json"), "w") as f:
        json.dump(per_sample_all, f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
