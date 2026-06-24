"""
Phase 1c (THE novel centerpiece): per-sample dynamic-BUDGET oracle-noise decomposition
on Qwen2.5-VL. Fix the selector, vary ONLY K, record per-sample score. Then:
  naive oracle    = mean_i max_K score[i][K]   (per-sample best K; exploits noise; UPPER bound)
  best static     = max_K mean_i score[i][K]
  monotone oracle = correct at K AND all larger K (noise-free)
  HONEST gain     = monotone - static @ the monotone avg budget   (FLOPs-matched)
Shows whether the per-sample adaptive-budget premise (ATP-LLaVA, Dynamic-LLaVA, ...) survives
honest, noise-corrected accounting on a strong modern backbone.

Default selector = uniform (the hard-to-beat question-BLIND baseline, per the analysis lit).
Run in qwen_env.

Usage:
  CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/qwen_env/bin/python -m src.pruning.dynamic_budget.qwen_oracle \
      --dataset docvqa --selector uniform --max-samples 200
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import load_dataset
from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.metrics.docvqa_score import anls
from src.metrics.chartqa_score import relaxed_correct

LADDER = [32, 64, 128, 256, 512, 768, 1024]   # + "full"
THRESH = 0.5


def load_bench(name, balanced, n):
    if name == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        idx = list(range(min(n, len(ds))))
        get = lambda i: (ds[i]["image"], ds[i]["question"], [str(a) for a in ds[i]["answers"]])
        per = lambda pred, gold: anls(pred, gold)
    else:
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        half = n // 2
        idx = (list(range(half)) + list(range(1250, 1250 + (n - half)))) if balanced else list(range(min(n, len(ds))))
        get = lambda i: (ds[i]["image"], ds[i]["question"], str(ds[i]["answer"]))
        per = lambda pred, gold: 1.0 if relaxed_correct(pred, gold) else 0.0
    return idx, get, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa"])
    ap.add_argument("--selector", default="uniform", choices=["uniform", "norm", "textattn"])
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--ladder", default=",".join(map(str, LADDER)))
    args = ap.parse_args()

    p = QwenPruner()
    idx, get, per = load_bench(args.dataset, args.balanced, args.max_samples)
    ladder = [int(x) for x in args.ladder.split(",")]
    levels = ladder + ["full"]
    n = len(idx)
    DENSE_TOK = {}    # measured avg tokens at "full"
    soft = {}
    for K in levels:
        t0 = time.time(); scores = []; toks = []
        for c, i in enumerate(idx):
            img, q, gold = get(i)
            sel = "full" if K == "full" else args.selector
            pred, nk = p.generate(img, q, selector=sel, K=(None if K == "full" else K),
                                  max_new_tokens=args.max_new_tokens)
            scores.append(per(pred, gold)); toks.append(nk)
        soft[K] = scores
        DENSE_TOK[K] = sum(toks) // len(toks)   # total kept tokens (text+visual); visual≈K
        print(f"  K={str(K):>5} mean={100*sum(scores)/n:.2f}%  (~{DENSE_TOK[K]} tok, {round((time.time()-t0)/60,1)}m)", flush=True)

    # token budget per level: use VISUAL K (full -> measured avg total as proxy)
    vtok = {K: (DENSE_TOK["full"] if K == "full" else K) for K in levels}
    curve = {K: 100 * sum(soft[K]) / n for K in levels}
    best_static = max(curve.values())
    naive = 100 * sum(max(soft[K][i] for K in levels) for i in range(n)) / n
    corr = {K: [s >= THRESH for s in soft[K]] for K in levels}

    def rfk(i):
        for j, K in enumerate(levels):
            if all(corr[KK][i] for KK in levels[j:]):
                return K
        return None
    rf = [rfk(i) for i in range(n)]
    ok = [k for k in rf if k is not None]
    mono = 100 * len(ok) / n
    mono_tok = (sum(vtok[k] for k in ok) + (n - len(ok)) * vtok["full"]) / n
    pts = sorted((vtok[K], 100 * sum(corr[K]) / n) for K in levels)
    def interp(x):
        if x <= pts[0][0]: return pts[0][1]
        if x >= pts[-1][0]: return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]
    honest = mono - interp(mono_tok)

    print("\n" + "=" * 66)
    print(f"  {args.dataset.upper()} Qwen2.5-VL — dynamic-budget oracle decomposition")
    print(f"  selector={args.selector}, n={n}")
    print(f"  best static       : {best_static:.2f}")
    print(f"  NAIVE oracle       : {naive:.2f}   (band +{naive-best_static:.2f})  <- inflated")
    print(f"  MONOTONE (honest)  : {mono:.2f}  @ ~{mono_tok:.0f} vis-tok")
    print(f"  HONEST matched-FLOP: +{honest:.2f}pp  (vs static {interp(mono_tok):.1f} @ same budget)")
    print("=" * 66)
    json.dump({"dataset": args.dataset, "selector": args.selector, "n": n,
               "curve": curve, "best_static": best_static, "naive_oracle": naive,
               "naive_band": round(naive - best_static, 2), "monotone": mono,
               "mono_tok": mono_tok, "honest_flops_matched": round(honest, 2),
               "raw_soft": {str(K): soft[K] for K in levels}},
              open(f"results/thesis_main/highres/qwen_oracle_{args.dataset}_{args.selector}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
