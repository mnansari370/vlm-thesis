"""
POPE speculative cascade (the frozen-method BUDGET component) — honest bs=1.

Stage1: CLS-Attn K=144 + first-token confidence for all. Stage2: K=288 if conf<tau.
Per subset: acc/F1 (official metric) + avg_K + honest cascade FLOPs (n=K+21).

Usage: python -m GQA.dynamic.run_pope_speculative --tau 0.55
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torch.utils.data import DataLoader

from GQA.shared.pope_score import score_subset
from GQA.eval_runners.run_pope import POPESubset, collate, SUBSETS, COCO_DIR, IMAGE_DIR, DENSE_REF_ACC
from GQA.shared.static import StaticPrunedLlava
from GQA.shared.flops import flops_row, N_TEXT_POPE
from GQA.shared.utils.logger import make_output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--k_easy", type=int, default=144)
    ap.add_argument("--k_hard", type=int, default=288)
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = make_output_dir("outputs", args.output_name or f"pope_speculative_tau{int(args.tau*100):03d}")
    print(f"[Output] {out_dir}\n[Config] tau={args.tau} K_easy={args.k_easy} K_hard={args.k_hard}", flush=True)
    model = StaticPrunedLlava(method="cls_attn", keep_k=args.k_easy, image_pad=True, honest=True)

    all_res, n_rerun_total, n_total = {}, 0, 0
    t0 = time.time()
    for sub in SUBSETS:
        ds = POPESubset(sub, IMAGE_DIR)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        recs = []
        model.set_keep_k(args.k_easy)
        for step, b in enumerate(loader):
            ans, conf = model.generate_answers(b["image"], b["text"], sample_offset=step,
                                               max_new_tokens=64, return_confidence=True)
            recs.append({"question_id": b["question_id"][0], "image": b["image"][0],
                         "text_q": b["text"][0], "ans_easy": ans[0], "conf": conf[0]})
        rerun = [r for r in recs if r["conf"] < args.tau]
        model.set_keep_k(args.k_hard)
        rr_ids = {r["question_id"] for r in rerun}
        # second pass only for rerun
        for r in recs:
            if r["question_id"] in rr_ids:
                ans = model.generate_answers([r["image"]], [r["text_q"]], max_new_tokens=64)
                r["ans_hard"] = ans[0]
        answers = [{"question_id": r["question_id"],
                    "text": r.get("ans_hard", r["ans_easy"]) if r["question_id"] in rr_ids else r["ans_easy"]}
                   for r in recs]
        res = score_subset(answers, f"{COCO_DIR}/coco_pope_{sub}.json")
        res.pop("per_sample")
        rr = len(rerun)
        res["rerun_rate"] = round(rr / len(recs), 4)
        res["avg_k"] = round((len(recs) * args.k_easy + rr * args.k_hard) / len(recs), 1)
        all_res[sub] = res
        n_rerun_total += rr; n_total += len(recs)
        print(f"  [{sub}] acc={res['accuracy_pct']:.2f}% F1={res['f1']:.2f} "
              f"rerun={res['rerun_rate']*100:.1f}% avgK={res['avg_k']}", flush=True)

    mean_acc = round(sum(all_res[s]["accuracy_pct"] for s in SUBSETS) / 3, 2)
    mean_f1 = round(sum(all_res[s]["f1"] for s in SUBSETS) / 3, 2)
    rerun_rate = n_rerun_total / n_total
    f_easy = flops_row(args.k_easy, n_text=N_TEXT_POPE)["fastv_full_TFLOPs"]
    f_hard = flops_row(args.k_hard, n_text=N_TEXT_POPE)["fastv_full_TFLOPs"]
    cascade = round(f_easy + rerun_rate * f_hard, 4)
    print(f"\n  MEAN acc={mean_acc}% F1={mean_f1} rerun={rerun_rate*100:.1f}% cascade={cascade}T", flush=True)
    out = {"method": "speculative", "tau": args.tau, "per_subset": all_res,
           "mean_accuracy_pct": mean_acc, "mean_f1": mean_f1, "rerun_rate": round(rerun_rate, 4),
           "cascade_TFLOPs": cascade, "dense_ref_acc": DENSE_REF_ACC,
           "elapsed_hours": round((time.time() - t0) / 3600, 3)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
