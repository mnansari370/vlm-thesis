"""
confidence.py — pure confidence-signal math for the Dynamic-COUNT probe pass.

The GPU wrappers record raw per-step quantities during their (unchanged, deterministic) greedy
decode: the first step's top-1/top-2 probabilities and full-distribution entropy, and each step's
chosen-token probability. This module turns those into the canonical CONFIDENCE_SIGNALS dict.
Pure python — CPU-importable/testable without torch.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence


def confidence_signals(first_top1: float, first_top2: float, first_entropy: float,
                       step_probs: Sequence[float]) -> Dict[str, float]:
    """
    first_top1 / first_top2 : probabilities of the top-1 / top-2 tokens at the FIRST decode step
    first_entropy           : entropy (nats) of the full first-step distribution
    step_probs              : probability of the CHOSEN (greedy) token at every generated step
                              (length == number of generated tokens, EOS excluded by the wrapper)
    """
    probs = [max(float(p), 1e-12) for p in step_probs] or [1e-12]
    logs = [math.log(p) for p in probs]
    return {
        "conf_first_max_prob": float(first_top1),
        "conf_first_margin": float(first_top1) - float(first_top2),
        "conf_first_entropy": float(first_entropy),
        "conf_mean_logprob": sum(logs) / len(logs),      # length-normalized answer log-probability
        "conf_mean_token_prob": sum(probs) / len(probs),
        "conf_min_token_prob": min(probs),
        "conf_answer_len": float(len(step_probs)),
    }


def risk_from_signal(value: float, signal: str) -> float:
    """Map a raw signal value to a RISK score (higher = more likely wrong). Probability-like
    signals invert; entropy/length are already risk-increasing."""
    if signal in ("conf_first_entropy", "conf_answer_len"):
        return float(value)
    if signal == "conf_mean_logprob":
        return -float(value)
    return 1.0 - float(value)      # max-prob / margin / mean / min token prob
