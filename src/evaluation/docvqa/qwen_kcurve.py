"""
Phase 1 — Qwen2.5-VL pruning K-curve / selection analysis.
Selectors: full (dense ceiling), uniform & norm (question-BLIND baselines; uniform is the
hard-to-beat one per the analysis literature), textattn (question-CONDITIONED, mid LLM layer).
Datasets: docvqa (ANLS) / chartqa (relaxed). Reports per-(selector,K) accuracy + avg visual
tokens kept, and CROSS-CHECKS that full ~ the dense fidelity number.

Run in qwen_env. ChartQA: pass --balanced to sample human+augmented evenly (test set is ordered).

Usage:
  CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/qwen_env/bin/python -m src.evaluation.docvqa.qwen_kcurve \
      --dataset docvqa --selectors uniform,norm,textattn --k-values 64,128,256,512 --max-samples 200
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from datasets import load_dataset
from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.metrics.docvqa_score import anls
from src.metrics.chartqa_score import relaxed_correct

DENSE_REF = {"docvqa": 95.7, "chartqa": 87.3}


def load_bench(name, balanced, n):
    if name == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        idx = list(range(min(n, len(ds))))
        get = lambda i: (ds[i]["image"], ds[i]["question"], [str(a) for a in ds[i]["answers"]])
        score = lambda pred, gold: anls(pred, gold)
        metric = "ANLS"
    else:
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        if balanced:  # test is ordered: first 1250 human, last 1250 augmented
            half = n // 2
            idx = list(range(half)) + list(range(1250, 1250 + (n - half)))
        else:
            idx = list(range(min(n, len(ds))))
        get = lambda i: (ds[i]["image"], ds[i]["question"], str(ds[i]["answer"]))
        score = lambda pred, gold: 1.0 if relaxed_correct(pred, gold) else 0.0
        metric = "relaxed"
    return idx, get, score, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docvqa", "chartqa"])
    ap.add_argument("--selectors", default="uniform,norm,textattn")
    ap.add_argument("--k-values", default="64,128,256,512")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--textattn-layer", type=int, default=16)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    p = QwenPruner()
    p.textattn_layer = args.textattn_layer
    idx, get, score, metric = load_bench(args.dataset, args.balanced, args.max_samples)
    Ks = [int(x) for x in args.k_values.split(",")]
    selectors = args.selectors.split(",")
    n = len(idx)
    out = {"dataset": args.dataset, "metric": metric, "n": n, "dense_ref": DENSE_REF[args.dataset],
           "textattn_layer": args.textattn_layer, "runs": {}}

    def run(tag, selector, K):
        t0 = time.time(); s = 0.0; toks = []
        for c, i in enumerate(idx):
            img, q, gold = get(i)
            pred, nkeep = p.generate(img, q, selector=selector, K=K, max_new_tokens=args.max_new_tokens)
            s += score(pred, gold); toks.append(nkeep)
            if (c + 1) % args.log_every == 0:
                print(f"    {tag} {c+1}/{n} acc={100*s/(c+1):.1f}%", flush=True)
        acc = round(100 * s / n, 2)
        out["runs"][tag] = {"selector": selector, "K": K, "acc": acc,
                            "avg_tokens": sum(toks) // len(toks), "minutes": round((time.time()-t0)/60, 1)}
        print(f"  [{tag}] {metric}={acc}%  (~{sum(toks)//len(toks)} tok)", flush=True)
        return acc

    full = run("full", "full", None)
    print(f"  >>> CROSS-CHECK: full={full} vs dense-ref {DENSE_REF[args.dataset]} "
          f"({'OK' if abs(full - DENSE_REF[args.dataset]) < 6 else 'CHECK SETUP'})", flush=True)
    for K in Ks:
        line = []
        for sel in selectors:
            a = run(f"{sel}_K{K}", sel, K)
            line.append(f"{sel}={a}")
        print(f"  --- K={K}: " + "  ".join(line) + f"  (full={full}) ---", flush=True)

    o = args.output or f"results/thesis_main/highres/qwen_kcurve_{args.dataset}.json"
    json.dump(out, open(o, "w"), indent=2)
    print(f"[saved] {o}")


if __name__ == "__main__":
    main()
