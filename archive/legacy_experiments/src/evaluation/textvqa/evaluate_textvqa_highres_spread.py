"""
Step 0c — THE decisive, v1-comparable diagnostic: per-sample token-need SPREAD at high-res.

WHY THIS IS THE REAL GO/NO-GO (and 0b was not enough):
The thesis is "DYNAMIC, question-conditioned" pruning. Two distinct claims:
  (a) better SELECTION  — at a fixed budget K, pick which tokens better.
  (b) DYNAMIC BUDGET    — vary how many tokens PER SAMPLE (easy q -> few, hard -> many).
v1 did NOT die on (a); it died on (b): the per-sample "first-correct-K" band was a thin
6-9%, so a single fixed K beat any per-sample schedule. 0b's K-curve only shows the
AVERAGE budget binds — necessary but NOT sufficient for (b). If every sample needs ~256
tokens, fixed K=256 still wins and dynamic is dead (v1, relocated to a higher token count).

This measures (b) directly, isolating the BUDGET axis: ONE fixed question-blind selector
(CLS-attn, same as 0b), vary ONLY K, record per-sample soft score at each K. Then:
  - oracle-dynamic  = mean_i max_K soft[i][K]   (perfect per-sample budget = upper bound)
  - best static     = max_K mean_i soft[i][K]   (the best single fixed budget)
  - HEADROOM        = oracle-dynamic - best static   (v1's "band"; was 6-9% on frozen 1.5)
  - first-correct-K distribution (binary@0.5): is token-need actually SPREAD across samples?
  - FLOPs-matched   : avg-K the oracle uses vs the static accuracy at that same avg-K.

WIDE band (>~15-20%) + spread first-correct-K  -> dynamic budget has real room -> GO, build it.
NARROW band (like v1)                          -> dynamic still won't beat fixed K -> pivot to (a) only.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.textvqa.evaluate_textvqa_highres_spread --max-samples 300
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image

from src.evaluation.textvqa.evaluate_textvqa_highres_kcurve import HighResPruner, TEXTVQA_ANN, TEXTVQA_IMG
from src.metrics.textvqa_score import score_textvqa

LADDER = [64, 128, 256, 384, 576, 864, 1152, 1728]   # + "full" appended below
THRESH = 0.5                                          # binary-correct threshold (VQA soft)
LLAVA15_REF = 42.23
V1_BAND = "6-9% (frozen LLaVA-1.5, v1)"               # the number to beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--ladder", default=",".join(str(k) for k in LADDER))
    ap.add_argument("--output", default="results/thesis_main/highres/eval_textvqa_highres_spread.json")
    ap.add_argument("--log-every", type=int, default=150)
    args = ap.parse_args()

    ladder = [int(x) for x in args.ladder.split(",")]
    pruner = HighResPruner()
    data = json.load(open(TEXTVQA_ANN))["data"][: args.max_samples]
    qids = [(r["image_id"], r["question"].strip()) for r in data]   # unique per sample

    # soft[K][qid] = VQA soft score for that sample at budget K
    soft = {}
    avg_tok = {}
    levels = ladder + ["full"]
    for K in levels:
        t0 = time.time()
        preds, keeps = [], []
        for i, rec in enumerate(data):
            ip = os.path.join(TEXTVQA_IMG, f"{rec['image_id']}.jpg")
            if not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB")
            sel = "full" if K == "full" else "attn"
            pred, nkeep = pruner.generate(img, rec["question"], selector=sel,
                                          K=(None if K == "full" else K),
                                          max_new_tokens=args.max_new_tokens)
            preds.append({"question_id": rec["image_id"], "question": rec["question"].strip(),
                          "pred_answer": pred})
            keeps.append(nkeep)
            if (i + 1) % args.log_every == 0:
                print(f"    K={K} {i+1}/{len(data)}", flush=True)
        res = score_textvqa(preds)
        # key by (image_id, question) — image_id ALONE is NOT unique (TextVQA has
        # several questions per image); keying by image_id would silently dedup samples.
        soft[K] = {(s["question_id"], s["question"]): s["soft_acc"] for s in res["per_sample"]}
        avg_tok[K] = sum(keeps) // max(len(keeps), 1)
        print(f"  K={K:>5}  mean soft={res['accuracy_pct']:.2f}%  (~{avg_tok[K]} tok, "
              f"{round((time.time()-t0)/60,1)} min)", flush=True)

    # ---- analysis (only samples scored at every level) ----
    common = [q for q in qids if all(q in soft[K] for K in levels)]
    n = len(common)

    curve = {str(K): round(100 * sum(soft[K][q] for q in common) / n, 2) for K in levels}
    best_static_K = max(levels, key=lambda K: curve[str(K)])
    best_static = curve[str(best_static_K)]

    # oracle-dynamic: each sample takes its best budget
    oracle = round(100 * sum(max(soft[K][q] for K in levels) for q in common) / n, 2)
    headroom = round(oracle - best_static, 2)

    # first-correct-K (binary@THRESH); smallest budget that gets the sample right
    first_k, never = [], 0
    for q in common:
        fk = None
        for K in levels:                       # ascending budget
            kk = avg_tok[K] if K != "full" else avg_tok["full"]
            if soft[K][q] >= THRESH:
                fk = K
                break
        if fk is None:
            never += 1
        else:
            first_k.append(fk)
    dist = Counter(first_k)
    ever_correct = round(100 * len(first_k) / n, 2)
    best_static_bin = round(100 * max(
        sum(1 for q in common if soft[K][q] >= THRESH) for K in levels) / n, 2)
    headroom_bin = round(ever_correct - best_static_bin, 2)
    # avg tokens the oracle spends: a sample never correct is charged the FULL budget (honest)
    oracle_tok_per_sample = [avg_tok[fk] for fk in first_k] + [avg_tok["full"]] * never
    oracle_avg_tok = round(sum(oracle_tok_per_sample) / max(n, 1))

    # FLOPs-MATCHED headroom: oracle accuracy vs the STATIC accuracy at the SAME avg budget.
    # static binary acc as a function of (avg_tok at K); interpolate at oracle_avg_tok.
    static_pts = sorted((avg_tok[K], 100 * sum(1 for q in common if soft[K][q] >= THRESH) / n)
                        for K in levels)
    def interp(x):
        if x <= static_pts[0][0]:
            return static_pts[0][1]
        if x >= static_pts[-1][0]:
            return static_pts[-1][1]
        for (x0, y0), (x1, y1) in zip(static_pts, static_pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return static_pts[-1][1]
    static_at_oracle_tok = round(interp(oracle_avg_tok), 2)
    flops_matched_headroom = round(ever_correct - static_at_oracle_tok, 2)

    out = {
        "n": n, "levels": [str(k) for k in levels], "avg_tok": {str(k): avg_tok[k] for k in levels},
        "curve_soft": curve, "best_static_K": str(best_static_K), "best_static_soft": best_static,
        "oracle_dynamic_soft": oracle, "headroom_soft": headroom,
        "binary": {"ever_correct": ever_correct, "best_static": best_static_bin,
                   "headroom": headroom_bin, "never_correct_pct": round(100 * never / n, 2),
                   "first_correct_K_dist": {str(k): dist.get(k, 0) for k in levels},
                   "oracle_avg_tokens": oracle_avg_tok,
                   "static_at_oracle_tok": static_at_oracle_tok,
                   "flops_matched_headroom": flops_matched_headroom},
        "ref": {"llava15_576": LLAVA15_REF, "v1_band": V1_BAND},
        # full matrix for any post-hoc stat; tuple keys -> "image_id|||question" for JSON
        "raw_soft": {str(K): {f"{k[0]}|||{k[1]}": v for k, v in soft[K].items()} for K in levels},
    }

    print("\n" + "=" * 70)
    print("  PER-SAMPLE TOKEN-NEED SPREAD — LLaVA-1.6 high-res, TextVQA no-OCR")
    print("=" * 70)
    print("  static curve (mean soft by budget):")
    for K in levels:
        print(f"     K={str(K):>5} (~{avg_tok[K]:>4} tok)   {curve[str(K)]:6.2f}%")
    print("-" * 70)
    print(f"  best STATIC (fixed K={best_static_K})      : {best_static:.2f}%   soft")
    print(f"  ORACLE dynamic (best K per sample) : {oracle:.2f}%   soft")
    print(f"  >>> SOFT HEADROOM (band)           : {headroom:+.2f} pp   "
          f"[v1 was {V1_BAND}]")
    print("-" * 70)
    print(f"  binary@{THRESH}:  best static {best_static_bin:.1f}%  |  oracle(ever-correct) "
          f"{ever_correct:.1f}%  |  accuracy band {headroom_bin:+.1f}pp")
    print(f"  FLOPs-matched: oracle {ever_correct:.1f}% @ ~{oracle_avg_tok} tok  vs  "
          f"static {static_at_oracle_tok:.1f}% @ same ~{oracle_avg_tok} tok  "
          f"=> {flops_matched_headroom:+.1f}pp")
    print(f"  first-correct-K distribution (how spread is token-need?):")
    for K in levels:
        c = dist.get(K, 0)
        bar = "#" * round(40 * c / max(n, 1))
        print(f"     {str(K):>5} (~{avg_tok[K]:>4} tok): {c:>4}  {bar}")
    print(f"     never        : {never:>4}")
    print("=" * 70)
    # GO if EITHER axis is real: accuracy band (more correct at matched FLOPs) OR
    # FLOPs-matched headroom (same/more correct at fewer FLOPs than best static).
    go = (headroom >= 12) or (flops_matched_headroom >= 8)
    verdict = ("WIDE -> dynamic budget has REAL room -> GO build it" if go
               else "NARROW -> dynamic ~ v1; pivot to selection-only contribution (axis a)")
    print(f"  VERDICT: {verdict}")
    print(f"           (accuracy band {headroom:+.1f}pp | FLOPs-matched {flops_matched_headroom:+.1f}pp)")
    print("=" * 70)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
