"""
Path-A Finding 4: the HONEST accuracy-vs-FLOPs frontier for high-res token pruning.

Uses the v1 FLOPs convention (v2/shared/flops.py): prune-before-LLM = all T=32 layers
see (K + n_text) tokens [fastv_full_flops]; faithful FastV = 3 layers see full, 29 see K
[fastv_layered_flops]. Same Vicuna-7B arch as LLaVA-1.6 (T=32, d=4096, m=11008), so the
formulas apply with the high-res dense token count.

Honesty points this figure makes:
  - blind CLS and QC (mid-layer attn token choice) have IDENTICAL generation FLOPs at a
    given K (both prune-before-LLM) -> the QC>blind gap is a clean matched-FLOPs gap.
  - faithful FastV costs MORE than prune-before-LLM at the same K (3 full-token layers).
  - we report RETENTION (% of our OWN dense), not absolute, because our dense is measured
    on n=300 / 20-tok generation and sits below published LLaVA-NeXT dense.

Populated from our measured n=300 runs (frozen LLaVA-1.6, no training).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.analysis.flops import fastv_full_flops, fastv_layered_flops

N_TEXT = 30          # short Q + "single word/phrase" instruction + chat template (~30)
DENSE = 2302         # measured mean high-res visual tokens (AnyRes), our LLaVA-1.6

# measured soft/relaxed/ANLS accuracy at each (selector, K); None = not measured
DATA = {
    "DocVQA": {  # ANLS, dense(full)=67.19
        "dense": (DENSE, 67.19),
        "blind": {64: 12.77, 128: 27.76, 256: 41.35, 512: 50.72, 768: 57.86, 1152: 63.03},
        "qc":    {64: 49.23, 128: 55.13, 256: 57.44, 512: 61.44, 768: 63.22, 1152: 63.89},
    },
    "ChartQA": {  # relaxed-acc, dense=50.33
        "dense": (DENSE, 50.33),
        "blind": {64: 16.67, 128: 25.67, 256: 29.67},
        "qc":    {64: 39.00, 128: 43.00, 256: 43.00},
    },
    "TextVQA_noOCR": {  # VQA soft, dense=61.40
        "dense": (DENSE, 61.40),
        "blind": {64: 34.80, 128: 44.20, 256: 54.67, 384: 54.80, 576: 57.03,
                  864: 59.57, 1152: 59.43, 1728: 61.33},
        "qc":    {64: 52.73, 128: 54.67, 256: 55.30},   # L16/L20 best-mid-layer
        "fastv": {128: 39.13},                           # early-layer (L2) selection
    },
}


def dense_flops():
    return fastv_full_flops(DENSE, N_TEXT)


def report(bench):
    d = DATA[bench]
    dense_acc = d["dense"][1]
    Fd = dense_flops()
    print("=" * 78)
    print(f"  {bench}  — dense={dense_acc:.2f} @ ~{DENSE} tok ({Fd/1e12:.1f} TFLOPs)")
    print(f"  {'method':<7}{'K':>6}{'TFLOPs':>10}{'FLOP-red%':>11}{'score':>8}{'retain%':>9}")
    print("-" * 78)
    rows = []
    for sel in ("blind", "qc", "fastv"):
        if sel not in d:
            continue
        for K, acc in sorted(d[sel].items()):
            if acc is None:
                continue
            F = fastv_layered_flops(K, N_TEXT) if sel == "fastv" else fastv_full_flops(K, N_TEXT)
            red = 100 * (1 - F / Fd)
            ret = 100 * acc / dense_acc
            rows.append((sel, K, F, red, acc, ret))
            print(f"  {sel:<7}{K:>6}{F/1e12:>10.2f}{red:>10.1f}%{acc:>8.2f}{ret:>8.1f}%")
    # matched-FLOPs gap: at each K where both blind & qc exist
    print("-" * 78)
    print("  matched-FLOPs gap (same K => same prune-before-LLM FLOPs):")
    for K in sorted(set(d.get("blind", {})) & set(d.get("qc", {}))):
        b, q = d["blind"].get(K), d["qc"].get(K)
        if b is not None and q is not None:
            print(f"    K={K:>4} ({100*(1-fastv_full_flops(K,N_TEXT)/Fd):.0f}% FLOP-red): "
                  f"blind={b:.1f}  QC={q:.1f}  gap={q-b:+.1f}")
    print("=" * 78)
    return rows


if __name__ == "__main__":
    for b in ("DocVQA", "ChartQA", "TextVQA_noOCR"):
        report(b)
