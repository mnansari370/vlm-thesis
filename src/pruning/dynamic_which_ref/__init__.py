"""
dynamic_which_ref — CLEAN-ROOM reference re-implementation of Dynamic-WHICH selection.

Purpose: an INDEPENDENT second implementation of the WHICH selection logic, written from scratch,
used to cross-check the original wrappers (src/pruning/dynamic_which/*) and to try rescue selectors.
It deliberately does NOT import llava_textsim / qwen_textsim. It reuses only the FROZEN generators
(StaticPrunedLlava, QwenPruner) and stable final-scope utilities. Training-free, no answer head,
no learned budget, no dense LM prefill for scoring.

This package __init__ holds the PURE (torch-free) selection helpers + selector bookkeeping so they
are CPU-importable and unit-testable without loading any model. The GPU wrappers live in
llava_ref.py / qwen_ref.py (imported lazily by the runner).

Selectors (rescue set):
  * textsim_ref                 — pure max-cosine(visual, question-token embeds).
  * saliency_mix_ref            — normalized mix of textsim and the static saliency
                                  (LLaVA: CLS-attention; Qwen: visual L2 norm).
  * static_guarded_textsim_ref  — reserve round(guard_frac·K) of the static-saliency top tokens,
                                  fill the remaining K with textsim top tokens, sort to original order.
  * coverage_guarded_textsim_ref — like static_guarded but the reserved portion is an evenly-spaced
                                  SPATIAL grid (LLaVA fixed 24×24 only). NOT available for Qwen
                                  (its visual-token→2D layout varies per image; see qwen_ref).
"""

from __future__ import annotations

from typing import List, Sequence

# ── selector bookkeeping ──────────────────────────────────────────────────────

REF_SUFFIX = "dynamic_which_ref"        # every ref output basename starts with this
DEFAULT_ALPHA = 0.5
DEFAULT_GUARD_FRAC = 0.50

LLAVA_REF_SELECTORS = ("textsim_ref", "saliency_mix_ref",
                       "static_guarded_textsim_ref", "coverage_guarded_textsim_ref")
QWEN_REF_SELECTORS = ("textsim_ref", "saliency_mix_ref", "static_guarded_textsim_ref")
GUARDED_SELECTORS = ("static_guarded_textsim_ref", "coverage_guarded_textsim_ref")
MIX_SELECTORS = ("saliency_mix_ref",)


def allowed_ref_selectors(model_name: str):
    if model_name == "llava15":
        return LLAVA_REF_SELECTORS
    if model_name == "qwen25vl7b":
        return QWEN_REF_SELECTORS
    raise ValueError(f"unknown model {model_name!r}")


def default_ref_selector(model_name: str) -> str:
    allowed_ref_selectors(model_name)   # validate
    return "textsim_ref"


def ref_basename(dataset: str, n: int, selector: str, budget_pct: int, *,
                 use_ocr: bool = True, instruction: bool = True) -> str:
    """Pilot-only basename that CANNOT collide with existing dynamic_which outputs (has _ref_)."""
    base = f"{REF_SUFFIX}_pilot_n{n}_{selector}_p{budget_pct}"
    if dataset == "textvqa":
        return f"{base}_ocr_{'on' if use_ocr else 'off'}"
    if dataset == "docvqa":
        return f"{base}_instruction_{'on' if instruction else 'off'}"
    return base


# ── pure selection helpers (torch-free; the GPU wrappers call these on CPU lists) ──

def topk_sorted(scores: Sequence[float], k: int) -> List[int]:
    """Top-K indices by score, then SORTED into original order. Identity at k >= len(scores)."""
    n = len(scores)
    k = max(1, min(int(k), n))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(order)


def minmax(values: Sequence[float]) -> List[float]:
    """Min-max to [0,1] (eps-stabilised; constant input → all zeros)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng <= 0:
        return [0.0 for _ in values]
    return [(v - lo) / (rng + 1e-8) for v in values]


def normalized_mix(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    """alpha·minmax(a) + (1-alpha)·minmax(b)."""
    na, nb = minmax(a), minmax(b)
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(na, nb)]


def spatial_coverage_indices(r: int, n: int) -> List[int]:
    """`r` evenly-spaced indices across [0, n) (a uniform spatial-grid stride). Sorted, unique."""
    r = max(0, min(int(r), int(n)))
    if r == 0:
        return []
    if r >= n:
        return list(range(n))
    # even spacing: floor(i * n / r) for i in 0..r-1 → unique, spread, deterministic
    idx = sorted({(i * n) // r for i in range(r)})
    # rounding collisions can drop a few; top up from the front to reach r if needed
    j = 0
    while len(idx) < r and j < n:
        if j not in idx:
            idx.append(j)
        j += 1
    return sorted(idx[:r])


def guarded_select(k: int, reserve_indices: Sequence[int], fill_scores: Sequence[float]) -> List[int]:
    """Reserve `reserve_indices` (capped at k), then fill the remaining budget with the highest
    `fill_scores` indices NOT already reserved, then SORT to original order. |result| <= k always."""
    n = len(fill_scores)
    k = max(1, min(int(k), n))
    reserve = []
    seen = set()
    for i in reserve_indices:           # unique, in-range, capped at k
        i = int(i)
        if 0 <= i < n and i not in seen:
            reserve.append(i)
            seen.add(i)
            if len(reserve) >= k:
                break
    remaining = k - len(reserve)
    if remaining > 0:
        order = sorted(range(n), key=lambda i: fill_scores[i], reverse=True)
        for i in order:
            if i not in seen:
                reserve.append(i)
                seen.add(i)
                remaining -= 1
                if remaining == 0:
                    break
    return sorted(reserve)
