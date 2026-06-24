"""
Finding 3, consistent: the monotone oracle-noise decomposition across all 3 benchmarks,
computed in ONE metric (binary correct@0.5) so naive vs monotone are directly comparable.

Reads the saved per-sample matrices. Shows: best-static, NAIVE oracle (any-K correct =
upper bound, exploits noise), MONOTONE oracle (correct at K and all larger = noise-free),
and the FLOPs-matched honest gain (monotone vs static at the monotone avg budget).
"""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DENSE, THRESH = 2302, 0.5
FILES = {
    "TextVQA": "results/thesis_main/highres/eval_textvqa_highres_spread.json",
    "DocVQA":  "results/thesis_main/highres/eval_spread_docvqa.json",
    "ChartQA": "results/thesis_main/highres/eval_spread_chartqa.json",
}


def load_matrix(path):
    """Return (levels_sorted_with_full_last, {K: [scores aligned by sample]})."""
    d = json.load(open(path))
    raw = d["raw_soft"]
    # normalize K keys -> ints (+ "full"); values -> aligned lists
    mat, ids = {}, None
    for Ks, v in raw.items():
        K = DENSE if Ks == "full" else int(Ks)
        if isinstance(v, dict):
            if ids is None:
                ids = list(v.keys())
            mat[K] = [v[i] for i in ids]
        else:
            mat[K] = list(v)
    levels = sorted(mat.keys())
    return levels, mat


def decompose(levels, mat):
    n = len(next(iter(mat.values())))
    corr = {K: [s >= THRESH for s in mat[K]] for K in levels}
    static = {K: 100 * sum(corr[K]) / n for K in levels}
    best_static = max(static.values())
    ever = 100 * sum(any(corr[K][i] for K in levels) for i in range(n)) / n   # naive oracle (binary)
    # monotone: correct at K and all larger
    def rfk(i):
        for idx, K in enumerate(levels):
            if all(corr[KK][i] for KK in levels[idx:]):
                return K
        return None
    rf = [rfk(i) for i in range(n)]
    ok = [k for k in rf if k is not None]
    mono = 100 * len(ok) / n
    mono_tok = (sum(ok) + (n - len(ok)) * DENSE) / n
    pts = sorted((K, static[K]) for K in levels)
    def interp(x):
        if x <= pts[0][0]: return pts[0][1]
        if x >= pts[-1][0]: return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1: return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]
    return {"n": n, "best_static": best_static, "naive": ever, "naive_band": ever - best_static,
            "monotone": mono, "mono_tok": mono_tok,
            "honest_flops_matched": mono - interp(mono_tok),
            "static_at_mono": interp(mono_tok)}


if __name__ == "__main__":
    print("=" * 78)
    print(f"  Oracle-noise decomposition (binary correct@{THRESH}, consistent metric)")
    print(f"  {'bench':<9}{'best-stat':>10}{'naive':>8}{'band':>7}{'monotone':>10}"
          f"{'mono-tok':>9}{'HONEST':>8}")
    print("-" * 78)
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"  {name:<9}  (missing {f})"); continue
        r = decompose(*load_matrix(f))
        print(f"  {name:<9}{r['best_static']:>10.1f}{r['naive']:>8.1f}{r['naive_band']:>+7.1f}"
              f"{r['monotone']:>10.1f}{r['mono_tok']:>8.0f}t{r['honest_flops_matched']:>+8.1f}")
    print("-" * 78)
    print("  band = naive oracle - best static (the INFLATED claim)")
    print("  HONEST = monotone - static@mono-budget (noise-free, FLOPs-matched)")
    print("=" * 78)
