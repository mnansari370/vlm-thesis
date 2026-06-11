"""
B3 — Speculative cascade on GQA testdev (LOCKED honest protocol, bs=1).

Stage 1: CLS-Attn K=144 for ALL questions → answer + first-token confidence.
Stage 2: CLS-Attn K=288 for questions with confidence < tau (default 0.55).
Merge:   accept K=288 answer for re-run questions, else K=144 answer.

Honest cascade FLOPs: a re-run sample pays for BOTH passes (LM FLOPs, n=K+34):
    per-sample avg = f_LM(144) + rerun_rate * f_LM(288)
(CLIP forward is excluded from the LM-FLOPs convention for ALL methods; for the
cascade it is technically paid twice on re-runs — noted in docs, not in LM FLOPs.)

Also reports the confidence-vs-correctness point-biserial correlation at K=144
on testdev (was r=0.51 on val) — the signal the Phase C dynamic method builds on.

Usage:
    python -m GQA.dynamic.run_speculative_testdev --tau 0.55
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader

from GQA.shared.official_score import is_correct, score_val_format, print_result
from GQA.dense.run_dense_testdev import GQATestdevDataset, collate
from GQA.shared.static import StaticPrunedLlava
from GQA.shared.flops import flops_row_testdev
from GQA.shared.utils.logger import make_output_dir


QUESTIONS = "data/gqa/testdev_balanced_questions.json"
IMAGE_DIR = "data/gqa/images/images"
DENSE_REF = 61.42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--k_easy", type=int, default=144)
    ap.add_argument("--k_hard", type=int, default=288)
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=2000)
    args = ap.parse_args()

    name = args.output_name or f"testdev_speculative_tau{int(args.tau*100):03d}"
    out_dir = make_output_dir("outputs", name)
    print(f"[Output] {out_dir}", flush=True)
    print(f"[Config] tau={args.tau} K_easy={args.k_easy} K_hard={args.k_hard} "
          f"bs=1 image_pad=True honest=True", flush=True)

    dataset = GQATestdevDataset(QUESTIONS, IMAGE_DIR, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    model = StaticPrunedLlava(method="cls_attn", keep_k=args.k_easy,
                              image_pad=True, honest=True)

    # ── Stage 1: K_easy for all ───────────────────────────────────────────────
    model.set_keep_k(args.k_easy)
    rec: dict[str, dict] = {}
    t0 = time.time()
    print(f"[Stage1] K={args.k_easy} + confidence ...", flush=True)
    for step, batch in enumerate(loader):
        ans, conf = model.generate_answers(
            batch["images"], batch["questions"], sample_offset=step,
            max_new_tokens=64, return_confidence=True,
        )
        qid = batch["question_ids"][0]
        rec[qid] = {
            "question": batch["questions"][0], "gold": batch["answers"][0],
            "type": batch["semantic_types"][0],
            "ans_easy": ans[0], "conf": conf[0],
        }
        if (step + 1) % args.log_every == 0:
            n = len(rec)
            acc = sum(is_correct(v["ans_easy"], v["gold"]) for v in rec.values()) / n
            sps = (step + 1) / (time.time() - t0)
            print(f"  [S1] {n:>6}/{len(dataset):,} acc_easy={acc*100:.2f}% "
                  f"{sps:.1f} samp/s ETA={(len(dataset)-n)/max(sps,1e-6)/60:.1f}min", flush=True)

    rerun = {q for q, v in rec.items() if v["conf"] < args.tau}
    rerun_rate = len(rerun) / max(len(rec), 1)
    print(f"[Stage1] done. re-run {len(rerun):,}/{len(rec):,} ({rerun_rate*100:.1f}%)", flush=True)

    # ── Stage 2: K_hard for low-confidence ────────────────────────────────────
    if rerun:
        model.set_keep_k(args.k_hard)
        print(f"[Stage2] K={args.k_hard} for {len(rerun):,} low-confidence ...", flush=True)
        # iterate dataset again, only generate for re-run qids (bs=1)
        for step, batch in enumerate(loader):
            qid = batch["question_ids"][0]
            if qid not in rerun:
                continue
            ans = model.generate_answers(
                batch["images"], batch["questions"], sample_offset=step,
                max_new_tokens=64,
            )
            rec[qid]["ans_hard"] = ans[0]
            if sum(1 for v in rec.values() if "ans_hard" in v) % args.log_every == 0:
                done = sum(1 for v in rec.values() if "ans_hard" in v)
                print(f"  [S2] {done:>6}/{len(rerun):,}", flush=True)

    # ── Merge + score ─────────────────────────────────────────────────────────
    preds = []
    for qid, v in rec.items():
        rr = qid in rerun
        final = v.get("ans_hard", v["ans_easy"]) if rr else v["ans_easy"]
        preds.append({
            "question_id": qid, "question": v["question"],
            "pred_answer": final, "answer": v["gold"], "semantic_type": v["type"],
            "k_used": args.k_hard if rr else args.k_easy,
            "confidence": v["conf"], "rerun": rr,
            "correct_easy": is_correct(v["ans_easy"], v["gold"]),
        })

    scored = score_val_format(preds, {})
    avg_k = sum(p["k_used"] for p in preds) / len(preds)
    elapsed_h = (time.time() - t0) / 3600

    # ── confidence-correctness correlation at K_easy ──────────────────────────
    try:
        import numpy as np
        from scipy.stats import pointbiserialr
        conf = np.array([p["confidence"] for p in preds])
        corr = np.array([int(p["correct_easy"]) for p in preds])
        r, pval = pointbiserialr(corr, conf)
        mean_correct = float(conf[corr == 1].mean())
        mean_wrong   = float(conf[corr == 0].mean())
    except Exception as e:
        r, pval, mean_correct, mean_wrong = None, None, None, None
        print(f"[warn] correlation failed: {e}")

    # ── honest cascade FLOPs (LM, n=K+34) ─────────────────────────────────────
    f_easy = flops_row_testdev(args.k_easy, method="static")["fastv_full_TFLOPs"]
    f_hard = flops_row_testdev(args.k_hard, method="static")["fastv_full_TFLOPs"]
    cascade_tflops = f_easy + rerun_rate * f_hard
    dense_tflops = flops_row_testdev(576, method="static")["fastv_full_TFLOPs"]
    reduction = (1 - cascade_tflops / dense_tflops) * 100

    print_result(scored, label=f"Speculative tau={args.tau} (testdev, honest)", reference=DENSE_REF)
    print(f"  avg_K={avg_k:.1f}  re-run={rerun_rate*100:.1f}%  "
          f"cascade={cascade_tflops:.4f}T ({reduction:.1f}% reduction)")
    print(f"  retention vs dense({DENSE_REF}%): {scored['accuracy_pct']/DENSE_REF*100:.2f}%")
    if r is not None:
        print(f"  confidence-correctness r={r:.4f} (p={pval:.1e})  "
              f"mean_conf correct={mean_correct:.4f} wrong={mean_wrong:.4f}")

    # per-type re-run + acc
    by_t = defaultdict(lambda: {"n": 0, "correct": 0, "rerun": 0, "ksum": 0})
    for p in preds:
        t = p["semantic_type"]
        by_t[t]["n"] += 1
        by_t[t]["correct"] += is_correct(p["pred_answer"], p["answer"])
        by_t[t]["rerun"] += int(p["rerun"])
        by_t[t]["ksum"] += p["k_used"]
    print("  per-type: " + "  ".join(
        f"{t}={by_t[t]['correct']/by_t[t]['n']*100:.1f}%/K{by_t[t]['ksum']/by_t[t]['n']:.0f}/rr{by_t[t]['rerun']/by_t[t]['n']*100:.0f}%"
        for t in ["obj", "attr", "rel", "cat", "global"] if by_t[t]["n"]))

    result = {
        "method": "speculative", "tau": args.tau,
        "k_easy": args.k_easy, "k_hard": args.k_hard,
        "split": "testdev_balanced", "protocol": "honest bs=1 image_pad",
        "n_evaluated": scored["n_total"], "accuracy_pct": scored["accuracy_pct"],
        "per_type": scored["per_type"], "avg_k": round(avg_k, 1),
        "rerun_rate": round(rerun_rate, 4),
        "retention_pct": round(scored["accuracy_pct"] / DENSE_REF * 100, 2),
        "cascade_TFLOPs": round(cascade_tflops, 4),
        "reduction_pct": round(reduction, 2),
        "flops_note": "LM-only n=K+34; re-run pays f(144)+f(288); CLIP excluded (paid 2x on rerun)",
        "confidence_correlation_r": r, "confidence_correlation_p": pval,
        "conf_mean_correct": mean_correct, "conf_mean_wrong": mean_wrong,
        "per_type_detail": {t: dict(v) for t, v in by_t.items()},
        "elapsed_hours": round(elapsed_h, 3),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": preds}, f)
    print(f"[Done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
