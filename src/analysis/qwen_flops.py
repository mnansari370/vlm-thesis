"""
Qwen2.5-VL FLOPs frontier (the 'FLOPs count' deliverable). Same FastV Eq.5 convention as
v1 (prune-before-LLM: all T layers see K+n_text), with Qwen2.5-7B LLM constants.
Reads a qwen_kcurve_*.json and prints accuracy-vs-FLOPs + reduction% per selector.

Honesty note: this is ANALYTICAL prefill FLOPs (clean, comparable). Real latency != FLOPs
(per 'Are We Solving the Right Problem?'); the diagnostic decode is a Python loop, so
latency must be measured separately with an optimized path — flagged as a refinement.
"""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

T, D, M = 28, 3584, 18944          # Qwen2.5-7B LLM
N_TEXT = 40                         # short Q + instruction + chat template (~40)


def per_layer(n):
    return 4 * n * D * D + 2 * n * n * D + 2 * n * D * M


def flops(K):
    return T * per_layer(K + N_TEXT)


def report(kcurve_json):
    d = json.load(open(kcurve_json))
    runs = d["runs"]
    metric = d["metric"]
    dense_K = runs["full"]["avg_tokens"]
    Fd = flops(dense_K)
    print("=" * 76)
    print(f"  {d['dataset'].upper()}  Qwen2.5-VL  (metric={metric}, dense {runs['full']['acc']} @ {dense_K} tok, {Fd/1e12:.1f} TFLOPs)")
    print(f"  {'selector':<10}{'K':>6}{'TFLOPs':>9}{'FLOP-red%':>11}{'acc':>9}{'retain%':>9}")
    print("-" * 76)
    # group by K
    Ks = sorted({r["K"] for tag, r in runs.items() if r["K"] is not None})
    sels = []
    for tag, r in runs.items():
        if r["selector"] not in sels and r["selector"] != "full":
            sels.append(r["selector"])
    dense_acc = runs["full"]["acc"]
    for K in Ks:
        for sel in sels:
            tag = f"{sel}_K{K}"
            if tag not in runs:
                continue
            r = runs[tag]
            F = flops(K)
            red = 100 * (1 - F / Fd)
            ret = 100 * r["acc"] / dense_acc
            print(f"  {sel:<10}{K:>6}{F/1e12:>9.2f}{red:>10.1f}%{r['acc']:>9.2f}{ret:>8.1f}%")
        print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcurve", default="results/thesis_main/highres/qwen_kcurve_docvqa.json")
    args = ap.parse_args()
    report(args.kcurve)
