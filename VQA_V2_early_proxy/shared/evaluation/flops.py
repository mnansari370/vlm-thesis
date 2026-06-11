"""Analytical attention-FLOPs proxy 2*L*S^2*H (the 60k era's metric).

LEGACY — do NOT use these numbers in the thesis/paper. This proxy (a) counts only
the attention term and (b) uses S = K + raw question length (~8 tokens), which
undercounts the prompt text. The unified, citable convention for all tracks is
FastV Eq. 5 with n = K + full-prompt n_text — see VQA_V2/shared/evaluation/flops.py
and GQA/shared/flops.py. Kept unchanged here only so the retired pipeline still runs.
"""

from typing import Any, Dict, Optional


def estimate_attention_flops_from_summary(
    metrics: Dict[str, Any],
    hidden_size: int,
    num_layers: int,
) -> Optional[Dict[str, Any]]:
    """
    Analytical attention-only FLOPs proxy.

    Uses:
      flops ~= 2 * L * S^2 * H

    where:
      L = number of transformer layers
      S = average multimodal sequence length
      H = hidden size

    This is a comparative proxy, not a full exact FLOPs count.
    """
    seq_len = metrics.get("avg_multimodal_sequence_length", None)
    if seq_len is None:
        return None

    seq_len = float(seq_len)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)

    flops = 2.0 * num_layers * (seq_len ** 2) * hidden_size

    return {
        "method": "analytical_attention_proxy_v1",
        "avg_multimodal_sequence_length": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "attention_flops": flops,
        "attention_flops_giga": flops / 1e9,
    }