"""
Finding 3 generalization: monotone oracle-noise decomposition on DocVQA / ChartQA.

Same logic as the TextVQA spread analysis, dataset-general (index-keyed, no id collisions).
ONE fixed question-blind selector (blind CLS-attn), vary ONLY K, record per-sample score:
  - naive oracle   = mean_i max_K score[i][K]            (exploits noise; UPPER bound)
  - best static    = max_K mean_i score[i][K]
  - monotone oracle = correct at K AND all larger K       (noise-free; = full-static acc)
  - FLOPs-matched  = monotone-acc vs static acc at the monotone avg budget
This shows whether the modest, noise-corrected dynamic-budget headroom we found on TextVQA
(naive +9.3pp -> honest +2.6pp) generalizes.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_highres_spread_docchart --dataset docvqa
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import load_dataset
from v2.eval.evaluate_textvqa_highres_kcurve import HighResPruner
from v2.shared.docvqa_score import anls
from v2.shared.chartqa_score import relaxed_correct
from v2.shared.flops import fastv_full_flops

LADDER = [64, 128, 256, 384, 576, 864, 1152, 1728]
N_TEXT, DENSE = 30, 2302
THRESH = 0.5


def load_bench(name):
    if name == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        get = lambda ex: (ex["image"], ex["question"], [str(a) for a in ex["answers"]])
        per_score = lambda pred, gold: anls(pred, gold)          # continuous 0..1
    else:
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        get = lambda ex: (ex["image"], ex["question"], str(ex["answer"]))
        per_score = lambda pred, gold: 1.0 if relaxed_correct(pred, gold) else 0.0
    return ds, get, per_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa"])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    p = HighResPruner()
    ds, get, per_score = load_bench(args.dataset)
    n = min(args.max_samples, len(ds))
    levels = LADDER + ["full"]
    avg_tok = {K: (DENSE if K == "full" else K) for K in levels}

    # soft[K] = list of per-sample scores aligned by index
    soft = {}
    for K in levels:
        t0 = time.time()
        scores = []
        for i in range(n):
            img, q, gold = get(ds[i])
            img = img.convert("RGB")
            sel = "full" if K == "full" else "attn"
            pred, _ = p.generate(img, q, selector=sel, K=(None if K == "full" else K),
                                 max_new_tokens=args.max_new_tokens)
            scores.append(per_score(pred, gold))
        soft[K] = scores
        print(f"  K={str(K):>5} mean={100*sum(scores)/n:.2f}% ({round((time.time()-t0)/60,1)}m)", flush=True)

    # --- decomposition ---
    curve = {K: 100 * sum(soft[K]) / n for K in levels}
    best_static_K = max(levels, key=lambda K: curve[K])
    best_static = curve[best_static_K]
    naive = 100 * sum(max(soft[K][i] for K in levels) for i in range(n)) / n

    def robust_first(i):
        for idx, K in enumerate(levels):
            if all(soft[KK][i] >= THRESH for KK in levels[idx:]):
                return K
        return None
    rfk = [robust_first(i) for i in range(n)]
    mono_correct = [k for k in rfk if k is not None]
    mono_acc = 100 * len(mono_correct) / n          # == full-static binary acc by construction
    mono_tok = (sum(avg_tok[k] for k in mono_correct) + (n - len(mono_correct)) * DENSE) / n

    # static binary acc interpolated at mono avg budget
    pts = sorted((avg_tok[K], 100 * sum(1 for i in range(n) if soft[K][i] >= THRESH) / n) for K in levels)
    def interp(x):
        if x <= pts[0][0]: return pts[0][1]
        if x >= pts[-1][0]: return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]
    static_at_mono = interp(mono_tok)

    print("\n" + "=" * 64)
    print(f"  {args.dataset.upper()} — oracle-noise decomposition (blind selector, n={n})")
    print(f"  best static (fixed K={best_static_K}) : {best_static:.2f}")
    print(f"  NAIVE oracle (per-sample best K)  : {naive:.2f}   (band +{naive-best_static:.2f})")
    print(f"  MONOTONE oracle (noise-free)      : {mono_acc:.2f}   @ ~{mono_tok:.0f} tok")
    print(f"    -> FLOPs-matched honest gain    : +{mono_acc - static_at_mono:.2f}pp "
          f"(static {static_at_mono:.1f} @ ~{mono_tok:.0f} tok)")
    print(f"  (naive band is the inflated number; monotone is the honest one)")
    print("=" * 64)
    out = {"dataset": args.dataset, "n": n, "curve": curve, "best_static": best_static,
           "naive_oracle": naive, "monotone_oracle": mono_acc, "mono_avg_tok": mono_tok,
           "flops_matched_honest": round(mono_acc - static_at_mono, 2),
           "raw_soft": {str(K): soft[K] for K in levels}}
    json.dump(out, open(f"v2/outputs/eval_spread_{args.dataset}.json", "w"), indent=2)
    print(f"[saved] v2/outputs/eval_spread_{args.dataset}.json")


if __name__ == "__main__":
    main()
