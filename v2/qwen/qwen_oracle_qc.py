"""
Phase 1c part 2: the DECISIVE budget test — oracle-noise decomposition with the GOOD
(question-conditioned, mid-layer) selector. If per-sample dynamic budget still helps a lot
here, the budget axis is real; if it COLLAPSES (because good selection already gets samples
right at low K), then the apparent dynamic-budget benefit was a symptom of weak selection.

Efficient: computes the QC attention scores ONCE per sample (one scoring forward), then
evaluates every K from the cached scores (top-K). Run in qwen_env.

Usage:
  CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python -m v2.qwen.qwen_oracle_qc \
      --dataset docvqa --max-samples 200
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datasets import load_dataset
from v2.qwen.qwen_pruner import QwenPruner
from src.metrics.docvqa_score import anls
from src.metrics.chartqa_score import relaxed_correct

LADDER = [32, 64, 128, 256, 512, 768, 1024]
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
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--textattn-layer", type=int, default=16)
    ap.add_argument("--ladder", default=",".join(map(str, LADDER)))
    args = ap.parse_args()

    p = QwenPruner()
    p.textattn_layer = args.textattn_layer
    idx, get, per = load_bench(args.dataset, args.balanced, args.max_samples)
    ladder = [int(x) for x in args.ladder.split(",")]
    levels = ladder + ["full"]
    n = len(idx)
    soft = {K: [] for K in levels}
    nvis_full = []
    t0 = time.time()
    for c, i in enumerate(idx):
        img, q, gold = get(i)
        emb, pos, attn, v0, nvis, qscores = p.encode_qc(img, q)   # ONE scoring forward
        nvis_full.append(nvis)
        for K in levels:
            if K == "full":
                keep_local = torch.arange(nvis, device=p.dev)
            else:
                keep_local = qscores.topk(min(K, nvis)).indices.sort().values
            pred, _ = p.generate_keep(emb, pos, attn, v0, nvis, keep_local, args.max_new_tokens)
            soft[K].append(per(pred, gold))
        if (c + 1) % 50 == 0:
            print(f"  {c+1}/{n} ({round((time.time()-t0)/60,1)}m)  K128={100*sum(soft[128])/(c+1):.1f} full={100*sum(soft['full'])/(c+1):.1f}", flush=True)

    for K in levels:
        print(f"  K={str(K):>5} mean={100*sum(soft[K])/n:.2f}%", flush=True)
    avg_full = sum(nvis_full)//len(nvis_full)
    vtok = {K: (avg_full if K == "full" else K) for K in levels}
    curve = {K: 100*sum(soft[K])/n for K in levels}
    best_static = max(curve.values())
    naive = 100*sum(max(soft[K][i] for K in levels) for i in range(n))/n
    corr = {K: [s >= THRESH for s in soft[K]] for K in levels}
    def rfk(i):
        for j, K in enumerate(levels):
            if all(corr[KK][i] for KK in levels[j:]): return K
        return None
    rf = [rfk(i) for i in range(n)]; ok = [k for k in rf if k is not None]
    mono = 100*len(ok)/n
    mono_tok = (sum(vtok[k] for k in ok) + (n-len(ok))*vtok["full"])/n
    pts = sorted((vtok[K], 100*sum(corr[K])/n) for K in levels)
    def interp(x):
        if x <= pts[0][0]: return pts[0][1]
        if x >= pts[-1][0]: return pts[-1][1]
        for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
            if x0<=x<=x1: return y0+(y1-y0)*(x-x0)/(x1-x0)
        return pts[-1][1]
    honest = mono - interp(mono_tok)
    print("\n"+"="*66)
    print(f"  {args.dataset.upper()} Qwen2.5-VL — budget oracle with GOOD (QC) selection, n={n}")
    print(f"  best static       : {best_static:.2f}")
    print(f"  NAIVE oracle       : {naive:.2f}  (band +{naive-best_static:.2f})")
    print(f"  MONOTONE (honest)  : {mono:.2f}  @ ~{mono_tok:.0f} vis-tok")
    print(f"  HONEST matched-FLOP: +{honest:.2f}pp  (vs static {interp(mono_tok):.1f} @ same budget)")
    print("="*66)
    json.dump({"dataset":args.dataset,"selector":"textattn_qc","n":n,"curve":curve,
               "best_static":best_static,"naive_oracle":naive,"naive_band":round(naive-best_static,2),
               "monotone":mono,"mono_tok":mono_tok,"honest_flops_matched":round(honest,2),
               "raw_soft":{str(K):soft[K] for K in levels}},
              open(f"v2/outputs/qwen_oracle_{args.dataset}_qc.json","w"),indent=2)


if __name__ == "__main__":
    main()
