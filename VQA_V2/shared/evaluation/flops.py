"""
Canonical FLOPs calculator for the VQA_V2 track — UNIFIED with GQA/shared/flops.py.

Two conventions, reported side by side (identical definitions in all three tracks):

1. PRIMARY — FastV Eq. 5 full-LM prefill (the published convention; FastV ECCV'24,
   followed by the FasterVLM/VisionZip-era comparisons). Per transformer layer:
       4*n*d^2  (QKVO projections)  +  2*n^2*d  (attention)  +  2*n*d*m  (FFN)
   summed over all T layers, with n = K_visual + n_text. Prune-before-LLM:
   every layer sees the already-pruned sequence.

2. SECONDARY — attention-only proxy 2*T*n^2*d. Isolates the quadratic term that
   pruning targets. This was the VQA_V2 track's original primary metric (the
   "12.56G vs 89.41G" style numbers in early reports); keep it only as the
   secondary diagnostic so the thesis/paper compares everything under Eq. 5.

Constants (LLaVA-1.5-7B):
    T = 32 layers · d = 4096 hidden · m = 11008 FFN intermediate · 576 visual tokens

n_text convention: mean FULL non-visual prompt tokens (system + "USER:" + question
+ "Answer the question using a single word or phrase." + "ASSISTANT:"), the same
measured-prompt basis as GQA's n_text=34. For VQAv2 this track's locked constant
is N_TEXT_VQAV2 = 35 (dense S = 611 → the locked 97.864 GFLOPs attention-only
baseline reproduces exactly).

Print the thesis conversion table:
    python -m VQA_V2.shared.evaluation.flops
"""

T: int = 32
D: int = 4096
M: int = 11008
N_VISUAL_DENSE: int = 576
N_TEXT_VQAV2: int = 35

# The locked static curve + study brackets + the dynamic operating point.
THESIS_K_VALUES = [64, 96, 128, 144, 160, 192, 219, 265, 288, 334, 357, 432, 576]
DYNAMIC_AVG_K = 264.3   # type-adaptive dynamic model, avg K (matched static = 265)


def _per_layer(n: float) -> float:
    """One transformer layer prefill FLOPs at sequence length n (FastV Eq. 5)."""
    return 4 * n * D * D + 2 * n * n * D + 2 * n * D * M


def fastv_full_flops(n_visual: float, n_text: int = N_TEXT_VQAV2) -> float:
    """PRIMARY: FastV Eq. 5 full-LM prefill, all T layers at n = K + n_text."""
    return float(T * _per_layer(n_visual + n_text))


def attention_only_flops(n_visual: float, n_text: int = N_TEXT_VQAV2) -> float:
    """SECONDARY: attention-only quadratic proxy 2*T*n^2*d."""
    n = n_visual + n_text
    return float(2 * T * n * n * D)


def flops_row(n_visual: float, label: str = "", n_text: int = N_TEXT_VQAV2) -> dict:
    """One table row: both conventions + reduction vs dense (576)."""
    full = fastv_full_flops(n_visual, n_text)
    attn = attention_only_flops(n_visual, n_text)
    full_d = fastv_full_flops(N_VISUAL_DENSE, n_text)
    attn_d = attention_only_flops(N_VISUAL_DENSE, n_text)
    return {
        "label": label or f"K={n_visual:g}",
        "K_visual": n_visual,
        "n_text": n_text,
        "n_total": n_visual + n_text,
        "fastv_full_TFLOPs": round(full / 1e12, 4),
        "fastv_full_reduction_pct": round((1 - full / full_d) * 100, 2) if n_visual != N_VISUAL_DENSE else 0.0,
        "attention_only_GFLOPs": round(attn / 1e9, 3),
        "attention_only_reduction_pct": round((1 - attn / attn_d) * 100, 2) if n_visual != N_VISUAL_DENSE else 0.0,
    }


def print_thesis_table() -> None:
    """The unified FLOPs table for the thesis/paper (VQAv2, n_text=35)."""
    print("=" * 92)
    print(f"{'Setting':<22} {'n=K+35':>7}  {'LM-full TFLOPs':>15} {'red%':>6}  "
          f"{'attn-only GFLOPs':>17} {'red%':>6}")
    print("-" * 92)
    rows = [flops_row(k, label=("Dense (576)" if k == 576 else f"Static K={k}"))
            for k in THESIS_K_VALUES]
    rows.append(flops_row(DYNAMIC_AVG_K, label=f"Dynamic avg K={DYNAMIC_AVG_K}"))
    for r in sorted(rows, key=lambda x: x["K_visual"]):
        print(f"{r['label']:<22} {r['n_total']:>7.0f}  {r['fastv_full_TFLOPs']:>15.4f} "
              f"{r['fastv_full_reduction_pct']:>5.1f}%  {r['attention_only_GFLOPs']:>17.3f} "
              f"{r['attention_only_reduction_pct']:>5.1f}%")
    print("=" * 92)
    print(f"  FastV Eq.5: T={T} layers x (4nd^2 + 2n^2d + 2ndm), d={D}, m={M}, n=K+{N_TEXT_VQAV2}")


if __name__ == "__main__":
    print_thesis_table()
