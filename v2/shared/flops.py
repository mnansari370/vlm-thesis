"""
FLOPs calculator for LLaVA-1.5-7B on GQA.

Two conventions reported side-by-side in the paper:

1. FastV full (paper Eq. 5) — used for comparison with FastV, PyramidDrop, etc.
   Per-layer:  4*n*d^2  +  2*n^2*d  +  2*n*d*m
   Total LM:   T * per_layer

2. Attention-only — isolates the quadratic n^2 cost that pruning targets.
   Total:      2 * T * n^2 * d

Constants (LLaVA-1.5-7B architecture):
  T = 32    (Vicuna-7B transformer layers)
  d = 4096  (hidden dimension)
  m = 11008 (FFN intermediate dimension)
  n_visual_dense = 576  (LLaVA-1.5 @ 336px, 14-pixel CLIP patches: 24*24)

n_text conventions (measured with LLaVA tokenizer):
  N_QUESTION      = 11  (mean RAW question tokens; val=11.0, testdev=10.67 — comparable)
  N_TEXT_TESTDEV  = 34  (mean FULL non-visual prompt tokens on testdev: system +
                         "USER:" + question + "Answer ... single word or phrase." +
                         "ASSISTANT:"; measured 33.75, rounded to 34)

Use N_TEXT_TESTDEV for testdev FLOPs — it is the physically correct count of
tokens that actually pass through the transformer (everything except the K
visual tokens). The old N_QUESTION=11 undercounts by ignoring system/template
tokens; kept only for backward-comparison with earlier val tables.
"""

T: int   = 32
D: int   = 4096
M: int   = 11008
N_VISUAL_DENSE: int = 576
N_QUESTION: int     = 11
N_TEXT_TESTDEV: int = 34
# TextVQA non-visual prompt lengths (measured on the 5,000-val set, LLaVA tokenizer):
N_TEXT_TEXTVQA_OCR:   int = 86   # with OCR block (mean 86.4) — OCR adds ~54 tokens
N_TEXT_TEXTVQA_NOOCR: int = 32   # question + instruction only (mean 31.9)
N_TEXT_POPE:          int = 21   # POPE yes/no question (mean 20.8)
N_TEXT_SQA:           int = 108  # ScienceQA-IMG CQM-A prompt (mean 108.5, long hints)


def _per_layer(n: int) -> float:
    """One transformer layer prefill FLOPs at sequence length n (FastV Eq.5)."""
    return 4 * n * D * D + 2 * n * n * D + 2 * n * D * M


def fastv_full_flops(n_visual: int, n_text: int = N_QUESTION) -> float:
    """
    Prune-before-LLM convention (our static methods): ALL T layers see n=K+n_text.
    FastV paper Eq. 5 per-layer, summed over all T layers.
    """
    return float(T * _per_layer(n_visual + n_text))


def fastv_layered_flops(n_visual: int, n_text: int = N_TEXT_TESTDEV,
                        prune_after_layer: int = 3) -> float:
    """
    FAITHFUL FastV (arXiv 2403.06764 / lmms-eval fastv_kvcache.py): the first
    `prune_after_layer` layers (0..prune_after_layer-1) see all 576 visual tokens;
    the remaining (T - prune_after_layer) layers see only n_visual=K.

    Default prune_after_layer=3 matches the reference (capture attn at layer_idx 2,
    prune entering layer_idx 3): layers 0,1,2 see 576+n_text; layers 3..31 see K+n_text.

    This is STRICTLY MORE than prune-before-LLM static at the same K (3 full layers),
    so FastV sits to the RIGHT of static on the FLOPs axis.
    """
    full = prune_after_layer * _per_layer(N_VISUAL_DENSE + n_text)
    pruned = (T - prune_after_layer) * _per_layer(n_visual + n_text)
    return float(full + pruned)


def attention_only_flops(n_visual: int, n_text: int = N_QUESTION) -> float:
    """
    Attention-only quadratic cost (VQAv2 convention, also used in the paper
    for the efficiency-vs-accuracy trade-off plots).
    """
    n = n_visual + n_text
    return float(2 * T * n * n * D)


def flops_row(n_visual: int, label: str = "", n_text: int = N_QUESTION) -> dict:
    """
    Build one row of the FLOPs table for a given visual token count.
    Returns a dict ready for JSON serialisation or LaTeX formatting.
    """
    n_dense = N_VISUAL_DENSE + n_text

    full      = fastv_full_flops(n_visual, n_text)
    attn      = attention_only_flops(n_visual, n_text)
    full_d    = fastv_full_flops(N_VISUAL_DENSE, n_text)
    attn_d    = attention_only_flops(N_VISUAL_DENSE, n_text)

    full_red  = 0.0 if n_visual == N_VISUAL_DENSE else (1.0 - full / full_d)
    attn_red  = 0.0 if n_visual == N_VISUAL_DENSE else (1.0 - attn / attn_d)

    return {
        "label":                      label or f"K={n_visual}",
        "K_visual":                   n_visual,
        "n_text":                     n_text,
        "n_total":                    n_visual + n_text,
        "fastv_full_GFLOPs":          round(full  / 1e9, 3),
        "fastv_full_TFLOPs":          round(full  / 1e12, 4),
        "fastv_full_reduction_pct":   round(full_red  * 100.0, 2),
        "attention_only_GFLOPs":      round(attn / 1e9, 3),
        "attention_only_reduction_pct": round(attn_red * 100.0, 2),
    }


def flops_row_testdev(n_visual: int, label: str = "",
                      method: str = "static") -> dict:
    """
    FLOPs row for a testdev run under the locked convention.

    Primary   : n = K + 34  (N_TEXT_TESTDEV, full non-visual prompt).
    Supplement : n = K + 11  (raw-question convention, for cross-paper comparison).

    method='static' : prune-before-LLM (all 32 layers see K) — fastv_full_flops.
    method='fastv'  : layer-split (3 layers see 576, 29 see K) — fastv_layered_flops.
    """
    nt = N_TEXT_TESTDEV
    if method == "fastv":
        full   = fastv_layered_flops(n_visual, nt)
        full_d = fastv_layered_flops(N_VISUAL_DENSE, nt)
    else:
        full   = fastv_full_flops(n_visual, nt)
        full_d = fastv_full_flops(N_VISUAL_DENSE, nt)

    attn   = attention_only_flops(n_visual, nt)
    attn_d = attention_only_flops(N_VISUAL_DENSE, nt)

    full_red = 0.0 if n_visual == N_VISUAL_DENSE else (1.0 - full / full_d)
    attn_red = 0.0 if n_visual == N_VISUAL_DENSE else (1.0 - attn / attn_d)

    # supplementary n = K + 11
    full_supp = (fastv_layered_flops(n_visual, N_QUESTION) if method == "fastv"
                 else fastv_full_flops(n_visual, N_QUESTION))

    return {
        "label":                       label or f"K={n_visual}",
        "method_flops":                method,
        "K_visual":                    n_visual,
        "n_text":                      nt,
        "n_total":                     n_visual + nt,
        "fastv_full_TFLOPs":           round(full / 1e12, 4),
        "fastv_full_reduction_pct":    round(full_red * 100.0, 2),
        "attention_only_GFLOPs":       round(attn / 1e9, 3),
        "attention_only_reduction_pct": round(attn_red * 100.0, 2),
        "supp_n11_TFLOPs":             round(full_supp / 1e12, 4),
    }


def paper_flops_table(k_values: list[int] | None = None) -> list[dict]:
    """
    Return rows for the paper FLOPs table.
    Default K values cover dense + the three static budgets + fine-grained.
    """
    if k_values is None:
        k_values = [576, 432, 288, 192, 144, 96, 64]
    labels = {576: "Dense", 432: "K=432", 288: "K=288",
              192: "K=192", 144: "K=144",  96: "K=96",  64: "K=64"}
    return [flops_row(k, labels.get(k, f"K={k}")) for k in k_values]


def print_flops_table(k_values: list[int] | None = None) -> None:
    rows = paper_flops_table(k_values)
    print("=" * 84)
    print(f"{'Method':<10} {'K':>4}  {'n_total':>7}  "
          f"{'FastV-full TFLOPs':>18} {'Reduce%':>8}  "
          f"{'Attn-only GFLOPs':>17} {'Reduce%':>8}")
    print("-" * 84)
    for r in rows:
        print(f"{r['label']:<10} {r['K_visual']:>4}  {r['n_total']:>7}  "
              f"{r['fastv_full_TFLOPs']:>18.4f} {r['fastv_full_reduction_pct']:>7.1f}%  "
              f"{r['attention_only_GFLOPs']:>17.3f} {r['attention_only_reduction_pct']:>7.1f}%")
    print("=" * 84)
    print(f"  T={T} layers, d={D}, m={M}, n_text={N_QUESTION} (measured mean, GQA val_balanced)")


if __name__ == "__main__":
    print_flops_table()
