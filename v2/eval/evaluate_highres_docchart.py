"""
Confirm the high-res QC-selection finding on the REAL token-hungry benchmarks
(ChartQA, DocVQA) — does question-conditioned (mid-layer attn) selection beat blind
CLS-attn at low K here too, as it did on TextVQA-no-OCR (+18pp@64, +10.5pp@128)?

Reuses the validated HighResPruner. Selectors: full (ceiling), attn (blind CLS, the
static baseline to beat), textattn (question-conditioned, LLM layer 16). Scorers:
ChartQA relaxed-accuracy, DocVQA ANLS.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_highres_docchart --dataset chartqa --max-samples 300
    CUDA_VISIBLE_DEVICES=0 python -m v2.eval.evaluate_highres_docchart --dataset docvqa  --max-samples 300
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datasets import load_dataset

from v2.eval.evaluate_textvqa_highres_kcurve import HighResPruner
from v2.shared.chartqa_score import score_chartqa
from v2.shared.docvqa_score import score_docvqa

QC_LAYER = 16


def load_bench(name):
    if name == "chartqa":
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        def get(ex): return ex["image"], ex["question"], {"gold": str(ex["answer"])}
        def score(preds): r = score_chartqa(preds); return r["accuracy_pct"], "relaxed-acc"
    elif name == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        def get(ex): return ex["image"], ex["question"], {"golds": [str(a) for a in ex["answers"]]}
        def score(preds): r = score_docvqa(preds); return r["anls_pct"], "ANLS"
    else:
        raise ValueError(name)
    return ds, get, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["chartqa", "docvqa"])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--k-values", default="64,128,256")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--output", default=None)
    ap.add_argument("--log-every", type=int, default=150)
    args = ap.parse_args()

    Ks = [int(x) for x in args.k_values.split(",")]
    p = HighResPruner()
    p.textattn_layer = QC_LAYER
    ds, get, score = load_bench(args.dataset)
    n_total = min(args.max_samples, len(ds))
    out = args.output or f"v2/outputs/eval_highres_{args.dataset}.json"

    results = {"dataset": args.dataset, "qc_layer": QC_LAYER, "n": n_total, "runs": {}}

    def run(tag, selector, K):
        t0 = time.time()
        preds = []
        for i in range(n_total):
            ex = ds[i]
            img, q, goldinfo = get(ex)
            img = img.convert("RGB")
            pred, _ = p.generate(img, q, selector=selector, K=K, max_new_tokens=args.max_new_tokens)
            preds.append({"pred_answer": pred, **goldinfo})
            if (i + 1) % args.log_every == 0:
                acc, _ = score(preds)
                print(f"    {tag} {i+1}/{n_total}  {acc:.1f}%", flush=True)
        acc, metric = score(preds)
        results["runs"][tag] = {"selector": selector, "K": K, "score": acc, "metric": metric,
                                "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  [{tag}] {metric}={acc:.2f}%  (n={n_total})", flush=True)
        return acc

    full = run("full", "full", None)
    print("-" * 56)
    for K in Ks:
        b = run(f"blind_K{K}", "attn", K)
        qc = run(f"qc_K{K}", "textattn", K)
        print(f"  >>> K={K}:  blind={b:.2f}  QC={qc:.2f}  gain={qc-b:+.2f}pp  "
              f"(full={full:.2f})", flush=True)

    print("\n" + "=" * 60)
    print(f"  {args.dataset.upper()} high-res — QC vs blind selection")
    print(f"  full = {full:.2f}%")
    for K in Ks:
        b = results["runs"][f"blind_K{K}"]["score"]
        qc = results["runs"][f"qc_K{K}"]["score"]
        print(f"   K={K:>4}:  blind={b:6.2f}  QC={qc:6.2f}  gain={qc-b:+6.2f}pp")
    print("=" * 60)
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
