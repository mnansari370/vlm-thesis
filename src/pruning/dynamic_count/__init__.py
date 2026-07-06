"""
dynamic_count — the REAL Dynamic-COUNT phase: per-sample token budgets.

Two variants (docs: results/tables/dynamic_count_full_matrix_method_plan.md):
  * DC-D (dc_discrete)   — baseline: confidence-gated cascade over the anchor budgets
                           {15,25,35,50,75}%. Composable offline from frozen anchor finals.
  * DC-C (dc_continuous) — THE main method: a calibrated controller predicts a sample-specific
                           INTEGER token count K_i = clamp(round(f(signals_i)), 32, N_i),
                           e.g. 64/97/188/240/391 — NOT restricted to the five anchors.

Execution policy (both variants): cheap PROBE pass at K_probe (p15 or p25) records the answer and
all controller signals for free; if the controller says the sample needed ≤ K_probe, keep the probe
answer; otherwise ONE second pass at the predicted budget. Every executed prefill is charged.

This __init__ holds the PURE (torch/numpy-free) helpers so they are CPU-importable and testable
without any model: budget bookkeeping, basenames, calibration split, saliency statistics, and
question features. GPU wrappers live in llava_dynamic_count.py / qwen_dynamic_count.py; the
controllers in controllers.py; confidence math in confidence.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

METHOD_PROBE = "dynamic_count_probe"
METHOD_DCD = "dynamic_count_dcd"
METHOD_DCC = "dynamic_count_dcc"

PROBE_BUDGETS: Tuple[int, ...] = (15, 25)          # allowed probe budgets (pct)
ANCHOR_BUDGETS: Tuple[int, ...] = (15, 25, 35, 50, 75)
ANCHOR_FRACTIONS: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.50, 0.75, 1.00)  # + dense
MIN_TOKENS = 32                                     # floor for predicted K_i
CALIB_FRAC = 0.20                                   # first 20% of manifest = calibration split

# probe selectors: the LOCKED static selector per model, plus textsim (COUNT-on-WHICH, Qwen TextVQA)
PROBE_SELECTORS = {"llava15": ("cls_attn",), "qwen25vl7b": ("norm", "textsim")}

# canonical signal names (order matters for the regression feature vector)
CONFIDENCE_SIGNALS = ("conf_first_max_prob", "conf_first_margin", "conf_first_entropy",
                      "conf_mean_logprob", "conf_mean_token_prob", "conf_min_token_prob",
                      "conf_answer_len")
SALIENCY_SIGNALS = ("sal_entropy", "sal_participation_ratio", "sal_top32_mass", "sal_top64_mass",
                    "sal_top128_mass", "sal_top256_mass", "sal_max", "sal_mean", "sal_std",
                    "sal_concentration")
QUESTION_SIGNALS = ("q_len_words", "q_len_chars", "q_is_what", "q_is_where", "q_is_who",
                    "q_is_how", "q_is_which", "q_is_yesno", "q_has_number", "nvis_native")
ALL_SIGNALS = CONFIDENCE_SIGNALS + SALIENCY_SIGNALS + QUESTION_SIGNALS


def clamp_k(k: float, min_tokens: int, max_tokens: int) -> int:
    """K_i = clamp(round(k), min_tokens, max_tokens); max wins when max < min (tiny images)."""
    return int(max(min(round(k), max_tokens), min(min_tokens, max_tokens)))


def calibration_split(ids: Sequence, frac: float = CALIB_FRAC) -> Tuple[list, list]:
    """First `frac` of MANIFEST ORDER = calibration; rest = evaluation. Deterministic."""
    n_cal = max(1, int(round(len(ids) * frac)))
    return list(ids[:n_cal]), list(ids[n_cal:])


# ── output basenames (prefix `dynamic_count_` — cannot collide with frozen outputs) ──

def _variant_suffix(dataset: str, use_ocr: bool, instruction: bool) -> str:
    if dataset == "textvqa":
        return f"_ocr_{'on' if use_ocr else 'off'}"
    if dataset == "docvqa":
        return f"_instruction_{'on' if instruction else 'off'}"
    return ""


def probe_basename(dataset: str, selector: str, probe_pct: int, *, n: int = 0, full: bool = True,
                   use_ocr: bool = True, instruction: bool = True) -> str:
    base = (f"dynamic_count_probe_{selector}_p{probe_pct}" if full
            else f"dynamic_count_probe_n{n}_{selector}_p{probe_pct}")
    return base + _variant_suffix(dataset, use_ocr, instruction)


def dcd_basename(dataset: str, selector: str, probe_pct: int, target_pct: int, *,
                 use_ocr: bool = True, instruction: bool = True) -> str:
    return (f"dynamic_count_dcd_{selector}_p{probe_pct}_to_p{target_pct}"
            + _variant_suffix(dataset, use_ocr, instruction))


def dcc_basename(dataset: str, selector: str, probe_pct: int, controller: str, lam: float, *,
                 use_ocr: bool = True, instruction: bool = True) -> str:
    lam_tag = f"{lam:.2f}".replace(".", "_")
    return (f"dynamic_count_dcc_{controller}_{selector}_p{probe_pct}_lam{lam_tag}"
            + _variant_suffix(dataset, use_ocr, instruction))


# ── pure saliency statistics (operate on a python list of scores) ─────────────

def saliency_stats(scores: Sequence[float]) -> Dict[str, float]:
    """Distribution statistics of the selector scores (any length; shift-normalized to a prob.
    distribution). Returns the SALIENCY_SIGNALS dict."""
    n = len(scores)
    if n == 0:
        return {k: 0.0 for k in SALIENCY_SIGNALS}
    lo = min(scores)
    shifted = [s - lo for s in scores]
    tot = sum(shifted)
    p = [s / tot for s in shifted] if tot > 0 else [1.0 / n] * n
    ent = -sum(x * math.log(x) for x in p if x > 0)
    pr = 1.0 / sum(x * x for x in p)                       # participation ratio (eff. support)
    desc = sorted(p, reverse=True)
    mass = lambda k: sum(desc[:min(k, n)])
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    return {
        "sal_entropy": ent,
        "sal_participation_ratio": pr,
        "sal_top32_mass": mass(32), "sal_top64_mass": mass(64),
        "sal_top128_mass": mass(128), "sal_top256_mass": mass(256),
        "sal_max": max(scores), "sal_mean": mean, "sal_std": math.sqrt(var),
        "sal_concentration": (max(p) * n),                 # peak relative to uniform
    }


# ── pure question features ─────────────────────────────────────────────────────

_YESNO_STARTS = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
                 "has ", "have ", "will ", "would ", "should ")


def question_features(question: str, nvis_native: int) -> Dict[str, float]:
    q = str(question).strip().lower()
    words = q.split()
    return {
        "q_len_words": float(len(words)),
        "q_len_chars": float(len(q)),
        "q_is_what": float(q.startswith("what")),
        "q_is_where": float(q.startswith("where")),
        "q_is_who": float(q.startswith("who")),
        "q_is_how": float(q.startswith("how")),
        "q_is_which": float(q.startswith("which")),
        "q_is_yesno": float(q.startswith(_YESNO_STARTS)),
        "q_has_number": float(any(ch.isdigit() for ch in q)),
        "nvis_native": float(nvis_native),
    }
