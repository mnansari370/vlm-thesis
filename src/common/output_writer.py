"""
output_writer.py — unified per-sample JSONL + aggregate JSON writer (Phase-2 dense infra).

Every final-scope eval (dense first, later static / dynamic) writes the SAME schema via
this writer, so results are comparable and machine-checkable. Schema = docs/DENSE_PROTOCOL.md
§6 (per-sample) and §7 (aggregate). Pure / CPU-only — it does not run any model.

Layout (docs/DENSE_PROTOCOL.md §12):
  results/runs/{model}/{dataset}/dense_100.jsonl   # one per-sample record per line
  results/runs/{model}/{dataset}/dense_100.json    # aggregate
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.common.schema_validator import (
    PER_SAMPLE_REQUIRED,
    validate_aggregate,
    validate_per_sample,
)


def per_sample_record(
    *,
    sample_id: Any,
    dataset: str,
    model: str,
    method: str,
    budget_pct: float,
    question: str,
    prompt: str,
    pred_raw: str,
    pred_norm: str,
    gold: Any,
    per_sample_score: float,
    n_visual_tokens: int,
    n_text_tokens: int,
    image_id: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one schema-complete per-sample record (seq_len derived)."""
    rec = {
        "sample_id": sample_id,
        "dataset": dataset,
        "model": model,
        "method": method,
        "budget_pct": budget_pct,
        "image_id": image_id,
        "question": question,
        "prompt": prompt,
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
        "gold": gold,
        "per_sample_score": per_sample_score,
        "n_visual_tokens": int(n_visual_tokens),
        "n_text_tokens": int(n_text_tokens),
        "seq_len": int(n_visual_tokens) + int(n_text_tokens),
    }
    rec.update(extra)
    return rec


def aggregate_record(
    *,
    model: str,
    dataset: str,
    split: str,
    method: str,
    budget_pct: float,
    n: int,
    metric: str,
    score_pct: float,
    visual_tokens: Dict[str, Any],
    text_tokens_avg: float,
    seq_len_avg: float,
    flops_prefill_TFLOPs_avg: Optional[float],
    flops_convention: str,
    token_reduction_pct: float,
    flop_reduction_pct: float,
    prompt_template: str,
    max_new_tokens: int,
    decoding: str,
    image_pad: Optional[bool],
    sample_ids_path: str,
    sample_ids_sha256: str,
    per_type: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one schema-complete aggregate record."""
    agg = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "method": method,
        "budget_pct": budget_pct,
        "n": int(n),
        "metric": metric,
        "score_pct": score_pct,
        "per_type": per_type or {},
        "visual_tokens": visual_tokens,
        "text_tokens_avg": text_tokens_avg,
        "seq_len_avg": seq_len_avg,
        "flops_prefill_TFLOPs_avg": flops_prefill_TFLOPs_avg,
        "flops_convention": flops_convention,
        "token_reduction_pct": token_reduction_pct,
        "flop_reduction_pct": flop_reduction_pct,
        "prompt_template": prompt_template,
        "max_new_tokens": max_new_tokens,
        "decoding": decoding,
        "image_pad": image_pad,
        "sample_ids_path": sample_ids_path,
        "sample_ids_sha256": sample_ids_sha256,
    }
    agg.update(extra)
    return agg


def write_per_sample_jsonl(
    path: str,
    records: Sequence[Dict[str, Any]],
    *,
    validate: bool = True,
    allow_empty_pred: bool = False,
) -> None:
    if validate:
        problems = validate_per_sample(records, allow_empty_pred=allow_empty_pred)
        if problems:
            raise ValueError(f"per-sample records fail schema: {problems[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def write_aggregate_json(
    path: str,
    agg: Dict[str, Any],
    *,
    validate: bool = True,
    require_flops: bool = True,
) -> None:
    if validate:
        problems = validate_aggregate(agg, require_flops=require_flops)
        if problems:
            raise ValueError(f"aggregate fails schema: {problems}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)


def read_per_sample_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
