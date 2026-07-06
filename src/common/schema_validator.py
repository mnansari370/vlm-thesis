"""
schema_validator.py — schema + fairness-gate checks for final-scope eval outputs.

Defines the canonical per-sample and aggregate field sets (docs/DENSE_PROTOCOL.md §6/§7)
and a fairness gate that a result must pass before it is "accepted". Pure / CPU-only.

A result is accepted only if (docs/DENSE_PROTOCOL.md §13):
  - per-sample records carry all required fields, with no missing predictions,
  - every record has a non-empty `image_id` (the dataset media/document id — image_id for
    VQAv2/GQA/TextVQA, docId/document id for DocVQA),
  - sample IDs match the manifest exactly AND in order,
  - the aggregate carries all required fields incl. the manifest sha,
  - model/dataset/method/budget agree between every record and the aggregate,
  - visual-token counts pass an EXPLICIT, method-specific expected value via
    `validate_visual_tokens(records, expected_visual_tokens)` — the CALLER supplies it:
    LLaVA dense → 576, LLaVA static → K (the kept-token budget), Qwen → None (varies per
    image, so usually unchecked). `seq_len == n_visual_tokens + n_text_tokens` always holds.
  - FLOPs fields are present when required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# ── canonical field sets ─────────────────────────────────────────────────────

PER_SAMPLE_REQUIRED = (
    "sample_id", "image_id", "dataset", "model", "method", "budget_pct",
    "question", "prompt", "pred_raw", "pred_norm", "gold", "per_sample_score",
    "n_visual_tokens", "n_text_tokens", "seq_len",
)

AGG_REQUIRED = (
    "model", "dataset", "split", "method", "budget_pct", "n", "metric", "score_pct",
    "visual_tokens", "text_tokens_avg", "seq_len_avg",
    "flops_prefill_TFLOPs_avg", "flops_convention",
    "token_reduction_pct", "flop_reduction_pct",
    "prompt_template", "max_new_tokens", "decoding", "image_pad",
    "sample_ids_path", "sample_ids_sha256",
)

LLAVA_DENSE_VISUAL_TOKENS = 576


# ── per-sample checks ─────────────────────────────────────────────────────────

def validate_per_sample(
    records: Sequence[Dict[str, Any]],
    *,
    manifest_ids: Optional[Sequence[Any]] = None,
    allow_empty_pred: bool = False,
) -> List[str]:
    """Return a list of problems (empty == OK)."""
    issues: List[str] = []
    if not records:
        return ["no per-sample records"]

    for i, r in enumerate(records):
        missing = [f for f in PER_SAMPLE_REQUIRED if f not in r]
        if missing:
            issues.append(f"record[{i}] missing fields: {missing}")
            continue
        if not allow_empty_pred and (r["pred_raw"] is None or str(r["pred_raw"]).strip() == ""):
            issues.append(f"record[{i}] (id={r.get('sample_id')}) has empty prediction")
        # image_id is the dataset media/document id (image_id / docId / document id) — required
        # to be present AND non-empty for every final dataset.
        if r["image_id"] is None or str(r["image_id"]).strip() == "":
            issues.append(f"record[{i}] (id={r.get('sample_id')}) has empty image_id (media/document id required)")
        if r["seq_len"] != r["n_visual_tokens"] + r["n_text_tokens"]:
            issues.append(
                f"record[{i}] seq_len {r['seq_len']} != visual {r['n_visual_tokens']} + text {r['n_text_tokens']}"
            )

    if manifest_ids is not None:
        rec_ids = [r.get("sample_id") for r in records]
        issues.extend(_check_ids_match(rec_ids, manifest_ids))
    return issues


def _check_ids_match(rec_ids: Sequence[Any], manifest_ids: Sequence[Any]) -> List[str]:
    """Sample IDs must equal the manifest IDs exactly AND in the same order."""
    issues: List[str] = []
    if len(rec_ids) != len(manifest_ids):
        issues.append(f"id count {len(rec_ids)} != manifest n {len(manifest_ids)}")
    a = list(map(str, rec_ids))
    b = list(map(str, manifest_ids))
    if set(a) != set(b):
        only_rec = sorted(set(a) - set(b))[:5]
        only_man = sorted(set(b) - set(a))[:5]
        issues.append(f"id set mismatch (e.g. extra-in-records={only_rec}, missing-from-records={only_man})")
    elif a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        issues.append(f"ids match as a set but ORDER differs (first at index {first})")
    return issues


def validate_visual_tokens(
    records: Sequence[Dict[str, Any]],
    expected_visual_tokens: Optional[int] = None,
) -> List[str]:
    """
    Check `n_visual_tokens` against an EXPLICIT expected value.

    The expected count depends on model + method + budget, so the CALLER supplies it:
      - LLaVA-1.5 dense  → expected_visual_tokens=576
      - LLaVA-1.5 static → expected_visual_tokens=K (the budget's kept-token count)
      - Qwen-2.5-VL      → expected_visual_tokens=None (varies per image; skip)
    If `expected_visual_tokens is None`, NO assertion is made (so static/dynamic outputs are
    never wrongly rejected by a hard-coded 576).
    """
    issues: List[str] = []
    if expected_visual_tokens is None:
        return issues
    for i, r in enumerate(records):
        if r.get("n_visual_tokens") != expected_visual_tokens:
            issues.append(
                f"record[{i}] (id={r.get('sample_id')}) n_visual_tokens="
                f"{r.get('n_visual_tokens')} != expected {expected_visual_tokens}"
            )
    return issues


# ── aggregate checks ─────────────────────────────────────────────────────────

def validate_aggregate(agg: Dict[str, Any], *, require_flops: bool = True) -> List[str]:
    issues: List[str] = []
    required = AGG_REQUIRED if require_flops else tuple(
        f for f in AGG_REQUIRED if not f.startswith("flops_")
    )
    missing = [f for f in required if f not in agg]
    if missing:
        issues.append(f"aggregate missing fields: {missing}")
    if not agg.get("sample_ids_sha256"):
        issues.append("aggregate has no sample_ids_sha256")
    return issues


# ── fairness gate (the accept/reject decision) ───────────────────────────────

def fairness_gate(
    records: Sequence[Dict[str, Any]],
    agg: Dict[str, Any],
    *,
    manifest_ids: Optional[Sequence[Any]] = None,
    manifest_sha: Optional[str] = None,
    require_flops: bool = True,
    allow_empty_pred: bool = False,
    expected_visual_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all checks; return {'ok': bool, 'issues': [...]}.

    `expected_visual_tokens` is the per-method expected visual-token count (e.g. 576 for LLaVA
    dense, K for LLaVA static, None for Qwen). When None, the visual-token count is not asserted.
    """
    issues: List[str] = []
    issues += validate_per_sample(records, manifest_ids=manifest_ids, allow_empty_pred=allow_empty_pred)
    issues += validate_visual_tokens(records, expected_visual_tokens)
    issues += validate_aggregate(agg, require_flops=require_flops)

    # every per-sample record must agree with the aggregate's identity fields
    for key in ("dataset", "model", "method", "budget_pct"):
        bad = [i for i, r in enumerate(records) if r.get(key) != agg.get(key)]
        if bad:
            issues.append(
                f"{len(bad)} record(s) have {key}={records[bad[0]].get(key)!r} "
                f"!= aggregate {key}={agg.get(key)!r} (first at index {bad[0]})"
            )

    if manifest_sha is not None and agg.get("sample_ids_sha256") != manifest_sha:
        issues.append(
            f"aggregate sha {agg.get('sample_ids_sha256')} != manifest sha {manifest_sha}"
        )
    if agg.get("n") is not None and len(records) != agg["n"]:
        issues.append(f"len(records) {len(records)} != aggregate n {agg['n']}")
    return {"ok": len(issues) == 0, "issues": issues}
