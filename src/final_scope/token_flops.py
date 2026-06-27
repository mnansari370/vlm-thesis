"""
token_flops.py — per-sample token stats + per-sample FLOPs scaffold (Phase-2 dense infra).

KEY RULE (docs/DENSE_PROTOCOL.md §9): FLOPs are computed **per sample** from that sample's
(visual + text) token count, then **averaged** — `mean_i FLOPs(n_i)`, NOT `FLOPs(mean_i n_i)`
— because the FastV Eq.5 formula has an `n^2` (convex) term, so plugging the mean undercounts.
This matters for Qwen (visual tokens vary per image); for LLaVA-1.5 every sample has 576
visual tokens so the two agree, but we still go per-sample for one uniform code path.

This module does NOT invent formulas. It REUSES the analytical prefill FLOPs already defined
in the repo:
  - LLaVA-1.5 : src/analysis/flops.py  `fastv_full_flops(n_visual, n_text)`  (T=32, d=4096, m=11008)
  - Qwen-2.5  : src/analysis/qwen_flops.py  `per_layer(n)` + T (T=28, d=3584, m=18944)
We pass the MEASURED per-sample `n_text` (we do NOT reuse qwen_flops's hardcoded N_TEXT=40).
Convention: LLM prefill only — excludes the vision encoder, the projector, and decode steps.

Pure / CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

# Reuse the validated formulas (these imports are dependency-free arithmetic modules).
from src.analysis.flops import fastv_full_flops as _llava_fastv_full_flops
from src.analysis import qwen_flops as _qwen

FLOPS_CONVENTION = "FastV Eq.5 LLM prefill, prune-before-LLM (all layers see visual+text); excludes ViT/projector/decode"

SUPPORTED_MODELS = ("llava15", "qwen25vl7b")
LLAVA_VISUAL_TOKENS = 576


@dataclass
class TokenStats:
    """Per-sample token counts."""
    n_visual_tokens: int
    n_text_tokens: int

    @property
    def seq_len(self) -> int:
        return int(self.n_visual_tokens) + int(self.n_text_tokens)


# ── per-sample FLOPs (wraps the repo formulas) ────────────────────────────────

def per_sample_prefill_flops(model: str, n_visual_tokens: int, n_text_tokens: int) -> float:
    """Analytical LLM-prefill FLOPs for ONE sample, in raw FLOPs (not TFLOPs)."""
    if model == "llava15":
        # fastv_full_flops(n_visual, n_text) = T * (4·n·d² + 2·n²·d + 2·n·d·m), n = visual+text
        return float(_llava_fastv_full_flops(int(n_visual_tokens), int(n_text_tokens)))
    if model == "qwen25vl7b":
        # reuse Qwen constants + per-layer formula, with MEASURED n_text (bypass hardcoded N_TEXT)
        n = int(n_visual_tokens) + int(n_text_tokens)
        return float(_qwen.T * _qwen.per_layer(n))
    raise ValueError(f"unsupported model {model!r}; expected one of {SUPPORTED_MODELS}")


def mean_flops(per_sample_flops: Sequence[float]) -> float:
    """Average FLOPs = mean of per-sample FLOPs (NOT FLOPs of the mean token count)."""
    if not per_sample_flops:
        return 0.0
    return float(sum(per_sample_flops) / len(per_sample_flops))


def summarize_tokens_and_flops(
    model: str,
    stats: Sequence[TokenStats],
) -> Dict[str, float]:
    """
    Roll a list of per-sample TokenStats into the aggregate token + FLOPs fields
    (docs/DENSE_PROTOCOL.md §7). Returns avg/max visual tokens, avg text tokens,
    avg seq_len, and the **per-sample-then-averaged** prefill FLOPs (in TFLOPs).
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model!r}")
    if not stats:
        raise ValueError("no token stats")

    vis = [s.n_visual_tokens for s in stats]
    txt = [s.n_text_tokens for s in stats]
    seq = [s.seq_len for s in stats]
    flops = [per_sample_prefill_flops(model, s.n_visual_tokens, s.n_text_tokens) for s in stats]

    return {
        "visual_tokens_avg": sum(vis) / len(vis),
        "visual_tokens_max": float(max(vis)),
        "text_tokens_avg": sum(txt) / len(txt),
        "seq_len_avg": sum(seq) / len(seq),
        "flops_prefill_TFLOPs_avg": mean_flops(flops) / 1e12,
        "flops_convention": FLOPS_CONVENTION,
    }


def reduction_pct(method_value: float, dense_value: float) -> float:
    """Generic reduction% of `method_value` vs the dense reference (>=0 means a cut)."""
    if dense_value <= 0:
        return 0.0
    return 100.0 * (1.0 - method_value / dense_value)
