"""
static_eval.py — final-scope STATIC pruning runner core + pure helpers.

Mirrors src/final_scope/dense_pilot.py but for STATIC (image-only, training-free) pruning:
  * LLaVA-1.5  → StaticPrunedLlava(method="cls_attn", keep_k=K)  (K fixed per budget; 576 backbone)
  * Qwen-2.5-VL → QwenPruner.generate(selector="norm"|"uniform", K=K_sample)  (K_sample per image)

It REUSES the dense final-scope infrastructure unchanged: the dataset adapters / prompts /
scorers from dense_pilot.py, and output_writer + schema_validator + token_flops. The ONLY
new pieces are (a) static budget→K bookkeeping, (b) loading the accepted DENSE FINAL run as a
reference (for per-sample Qwen K, for token/FLOP reduction over the SAME samples, and for the
accuracy delta), and (c) the static output schema fields.

Budgets (locked): {15, 25, 35, 50, 75} percent.
  LLaVA fixed-576 backbone  → keep_k = {15:86, 25:144, 35:202, 50:288, 75:432}.
  Qwen variable visual count → K_sample = clamp(round(budget_pct/100 · dense_n_visual), 1, dense_n_visual),
    read PER SAMPLE from the dense final JSONL (no extra encode just to count visual tokens).

This module contains NO model/generation code. The actual generation is supplied by a runner
(see scripts/final_scope/run_static_eval.py) with signature
  generate_and_count(sample, dense_record) -> (pred_raw, n_visual_tokens, n_text_tokens, prompt)
The dense_record (the SAME-sample dense final per-sample row) lets the Qwen runner compute its
per-sample K and lets BOTH runners reuse the dense text-token count (pruning removes only visual
tokens, so n_text is identical to dense).

Heavy / optional deps are imported lazily by the adapters, so this module imports under a plain
CPU Python for `compileall` and the self-checks.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.final_scope.sample_ids import load_manifest
from src.final_scope.output_writer import (
    per_sample_record,
    aggregate_record,
    write_per_sample_jsonl,
    write_aggregate_json,
)
from src.final_scope.schema_validator import fairness_gate
from src.final_scope import token_flops as tf
from src.final_scope.dense_pilot import (
    Sample,
    get_adapter,
    describe_prompt,
    default_basename,
    PILOT_N,
    MAX_NEW_TOKENS,
    INSTRUCTION,
    MANIFEST_DIR,
    OUT_ROOT,
    MODELS,
    DATASETS,
)

# ── static budget bookkeeping ──────────────────────────────────────────────────

# LLaVA-1.5 has a FIXED 576-token visual grid → each budget maps to an exact kept-token K.
BUDGET_TO_KEEP_K: Dict[int, int] = {15: 86, 25: 144, 35: 202, 50: 288, 75: 432}
BUDGETS: Tuple[int, ...] = tuple(sorted(BUDGET_TO_KEEP_K))   # (15, 25, 35, 50, 75)

# Selectors per model: LLaVA primary = cls_attn; Qwen primary = norm, optional control = uniform.
# (Qwen textattn is question-conditioned → Dynamic-WHICH, NOT static; VisionZip is a separate method.)
LLAVA_SELECTORS: Tuple[str, ...] = ("cls_attn",)
QWEN_SELECTORS: Tuple[str, ...] = ("norm", "uniform")


def default_selector(model_name: str) -> str:
    """Primary static selector per the locked decision: LLaVA→cls_attn, Qwen→norm."""
    if model_name == "llava15":
        return "cls_attn"
    if model_name == "qwen25vl7b":
        return "norm"
    raise ValueError(f"unknown model {model_name!r}")


def allowed_selectors(model_name: str) -> Tuple[str, ...]:
    if model_name == "llava15":
        return LLAVA_SELECTORS
    if model_name == "qwen25vl7b":
        return QWEN_SELECTORS
    raise ValueError(f"unknown model {model_name!r}")


def budget_keep_k(model_name: str, budget_pct: int) -> Optional[int]:
    """LLaVA: the fixed kept-token count K for this budget. Qwen: None (varies per image)."""
    if budget_pct not in BUDGET_TO_KEEP_K:
        raise ValueError(f"budget_pct must be one of {BUDGETS}, got {budget_pct}")
    if model_name == "llava15":
        return BUDGET_TO_KEEP_K[budget_pct]
    if model_name == "qwen25vl7b":
        return None
    raise ValueError(f"unknown model {model_name!r}")


def clamp_qwen_k(budget_pct: int, dense_n_visual: int) -> int:
    """Per-sample Qwen K = clamp(round(budget_pct/100 · dense_n_visual), 1, dense_n_visual)."""
    n = int(dense_n_visual)
    k = int(round(budget_pct / 100.0 * n))
    return max(1, min(k, n))


def accuracy_deltas(score_pct: float, dense_score_pct: float) -> Tuple[float, float]:
    """Return (accuracy_delta_pp, accuracy_drop_pp), both in percentage points.

    accuracy_delta_pp = score - dense  (SIGNED: positive ⇒ static BEATS dense on the same n);
    accuracy_drop_pp  = dense - score  (positive ⇒ static is WORSE — matches the field name).
    Renamed from the old misleading `accuracy_drop_pp = score - dense`, which was actually a
    signed delta (negative when static won) and therefore mis-named."""
    delta = round(score_pct - dense_score_pct, 2)
    drop = round(dense_score_pct - score_pct, 2)
    return delta, drop


def resolve_n(n: Optional[int], full: bool, total: int) -> int:
    """Pilot/final n resolution. --full uses the whole manifest; --n is a pilot knob; they conflict."""
    if full and n is not None:
        raise ValueError("--n cannot be combined with --full (a final run uses the full manifest length)")
    n0 = total if full else (PILOT_N if n is None else int(n))
    if n0 > total:
        raise ValueError(f"requested n={n0} exceeds manifest length {total}")
    return n0


def static_basename(dataset: str, n: int, selector: str, budget_pct: int, *,
                    use_ocr: bool = True, instruction: bool = True, full: bool = False) -> str:
    """Variant-aware static output basename so budgets/selectors/variants never collide.

      full=False (pilot)                              full=True (final)
      GQA / VQAv2 : static_pilot_n{n}_{sel}_p{b}                  static_final_{sel}_p{b}
      TextVQA     : static_pilot_n{n}_{sel}_p{b}_ocr_{on,off}     static_final_{sel}_p{b}_ocr_{on,off}
      DocVQA      : static_pilot_n{n}_{sel}_p{b}_instruction_{on,off}  static_final_{sel}_p{b}_instruction_{on,off}
    """
    base = (f"static_final_{selector}_p{budget_pct}" if full
            else f"static_pilot_n{n}_{selector}_p{budget_pct}")
    if dataset == "textvqa":
        return f"{base}_ocr_{'on' if use_ocr else 'off'}"
    if dataset == "docvqa":
        return f"{base}_instruction_{'on' if instruction else 'off'}"
    return base


def qwen_retained_sanity(retained_avg: float, dense_visual_avg: float, budget_pct: int, *,
                         tol_frac: float = 0.12, tol_abs: float = 2.0) -> Optional[str]:
    """Soft check: the actual avg retained visual tokens should be close to the requested
    budget% of the dense avg. `uniform` can return slightly FEWER than requested (linspace
    index collisions → .unique()), so equality is NOT required — only a gross mismatch is
    flagged. Returns an issue string, or None when within tolerance."""
    requested = budget_pct / 100.0 * float(dense_visual_avg)
    tol = max(tol_abs, tol_frac * requested)
    if abs(float(retained_avg) - requested) > tol:
        return (f"retained visual avg {retained_avg:.2f} not within ±{tol:.2f} of requested "
                f"{requested:.2f} (= {budget_pct}% of dense visual avg {dense_visual_avg:.2f})")
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dense_reference(model_name: str, dataset: str, *, use_ocr: bool, instruction: bool,
                         out_root: str = OUT_ROOT) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Load the accepted DENSE FINAL per-sample JSONL of the SAME variant, keyed by str(sample_id).

    The dense final covers the entire manifest, so it supplies a reference row for every static
    sample (pilot subset or full). Returns (jsonl_path, {str(sample_id): record})."""
    base = default_basename(dataset, 0, use_ocr=use_ocr, instruction=instruction, full=True)  # dense_final[...]
    jsonl = os.path.join(out_root, model_name, dataset, f"{base}.jsonl")
    if not os.path.exists(jsonl):
        raise FileNotFoundError(
            f"dense reference JSONL not found: {jsonl}\n"
            f"  run the matching dense FINAL ({base}) for {model_name}×{dataset} first."
        )
    lookup: Dict[str, Dict[str, Any]] = {}
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lookup[str(r["sample_id"])] = r
    return jsonl, lookup


# ── static runner core ──────────────────────────────────────────────────────────

@dataclass
class StaticResult:
    out_jsonl: str
    out_json: str
    n: int
    score_pct: float
    dense_score_pct: float
    token_reduction_pct: float
    flop_reduction_pct: float
    gate_ok: bool
    gate_issues: List[str] = field(default_factory=list)


def run_static(
    *,
    model_name: str,
    dataset: str,
    selector: str,
    budget_pct: int,
    generate_and_count: Callable[[Sample, Dict[str, Any]], Tuple[str, int, int, str]],
    n: int = PILOT_N,
    full: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
    image_pad: Optional[bool] = None,
    max_pixels: Optional[int] = None,
    use_ocr: bool = True,
    instruction: bool = True,
    model_instruction_suffix: str = "",
    decode_notes: str = "",
    manifest_dir: str = MANIFEST_DIR,
    out_root: str = OUT_ROOT,
    out_basename: Optional[str] = None,
    log_every: int = 50,
) -> StaticResult:
    """
    Run ONE static cell: (model, dataset, selector, budget_pct).

    `generate_and_count(sample, dense_record) -> (pred_raw, n_visual_tokens, n_text_tokens, prompt)`
    is the GPU runner supplied by the caller. `dense_record` is the SAME-sample dense final row
    (so the Qwen runner can compute its per-sample K from `dense_record["n_visual_tokens"]` and
    BOTH runners reuse `dense_record["n_text_tokens"]`). Everything else — data, scoring, schema,
    FLOPs, the dense-referenced reductions, validation, writing — is handled here. This function
    performs the GPU loop; do not call it until a run is approved.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if selector not in allowed_selectors(model_name):
        raise ValueError(f"selector {selector!r} not allowed for {model_name}; "
                         f"expected one of {allowed_selectors(model_name)}")
    keep_k = budget_keep_k(model_name, budget_pct)            # int (LLaVA) or None (Qwen)

    manifest = load_manifest(os.path.join(manifest_dir, f"{dataset}.json"))
    total = len(manifest.ids)
    n = resolve_n(n, full, total)
    ids = manifest.ids[:n]

    dense_jsonl, dense_lookup = load_dense_reference(
        model_name, dataset, use_ocr=use_ocr, instruction=instruction, out_root=out_root)
    missing = [str(sid) for sid in ids if str(sid) not in dense_lookup]
    if missing:
        raise ValueError(f"{len(missing)} static sample id(s) missing from dense reference "
                         f"{dense_jsonl} (e.g. {missing[:5]})")

    if out_basename is None:
        out_basename = static_basename(dataset, n, selector, budget_pct,
                                       use_ocr=use_ocr, instruction=instruction, full=full)
    adapter = get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)
    decoding = f"greedy bs=1 do_sample=False max_new_tokens={max_new_tokens}"

    records: List[Dict[str, Any]] = []
    stats: List[tf.TokenStats] = []
    dense_stats: List[tf.TokenStats] = []          # dense token stats over the SAME ids
    dense_scores: List[float] = []
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        drec = dense_lookup[str(sid)]
        pred_raw, n_vis, n_txt, prompt = generate_and_count(s, drec)
        pred_norm, score = adapter.score(pred_raw, s)
        records.append(per_sample_record(
            sample_id=s.sample_id, image_id=s.image_id, dataset=dataset, model=model_name,
            method="static", budget_pct=budget_pct, question=s.question, prompt=prompt,
            pred_raw=str(pred_raw).strip(), pred_norm=str(pred_norm), gold=s.gold,
            per_sample_score=float(score), n_visual_tokens=int(n_vis), n_text_tokens=int(n_txt),
            selector=selector,
        ))
        stats.append(tf.TokenStats(int(n_vis), int(n_txt)))
        dense_stats.append(tf.TokenStats(int(drec["n_visual_tokens"]), int(drec["n_text_tokens"])))
        dense_scores.append(float(drec.get("per_sample_score", 0.0)))
        if (i + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name} static {selector} p{budget_pct}] "
                  f"{i+1}/{len(ids)}  running_score={avg*100:.2f}%", flush=True)

    summ = tf.summarize_tokens_and_flops(model_name, stats)
    dense_summ = tf.summarize_tokens_and_flops(model_name, dense_stats)   # SAME samples → fair Δ
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    dense_score_pct = round(sum(dense_scores) / max(len(dense_scores), 1) * 100, 2)
    accuracy_delta_pp, accuracy_drop_pp = accuracy_deltas(score_pct, dense_score_pct)
    token_reduction_pct = round(tf.reduction_pct(summ["seq_len_avg"], dense_summ["seq_len_avg"]), 2)
    flop_reduction_pct = round(
        tf.reduction_pct(summ["flops_prefill_TFLOPs_avg"], dense_summ["flops_prefill_TFLOPs_avg"]), 2)
    expected_vis = keep_k                                # LLaVA: K (exact) ; Qwen: None (varies)

    extra: Dict[str, Any] = {
        "selector_name": selector,
        "keep_k": keep_k,
        "dense_reference_path": dense_jsonl,
        "dense_reference_sha256": _sha256_file(dense_jsonl),
        "dense_reference_score_pct": dense_score_pct,
        "dense_reference_visual_tokens_avg": round(dense_summ["visual_tokens_avg"], 3),
        "dense_reference_seq_len_avg": round(dense_summ["seq_len_avg"], 3),
        "dense_reference_flops_TFLOPs_avg": round(dense_summ["flops_prefill_TFLOPs_avg"], 4),
        # signed delta (static - dense; positive ⇒ static beats dense on the same n) + the
        # literally-named drop (dense - static; positive ⇒ static worse). See accuracy_deltas().
        "accuracy_delta_pp": accuracy_delta_pp,
        "accuracy_drop_pp": accuracy_drop_pp,
        # prompt/decode provenance — TOP-LEVEL, faithful to what the model saw (matches dense schema)
        "protocol_instruction": INSTRUCTION,
        "model_instruction_suffix": model_instruction_suffix,
        "decode_notes": decode_notes,
        "extra_static": {
            "n_requested": n,
            "budget_pct": budget_pct,
            "use_ocr": use_ocr if dataset == "textvqa" else None,
            "instruction": instruction if dataset == "docvqa" else None,
            "max_pixels": max_pixels,
            "qwen_budget_rule": (
                "K_sample = clamp(round(budget_pct/100 * dense_n_visual_tokens), 1, dense_n_visual_tokens)"
                if model_name == "qwen25vl7b" else None),
            "requested_visual_tokens_avg": (
                round(budget_pct / 100.0 * dense_summ["visual_tokens_avg"], 3)
                if model_name == "qwen25vl7b" else None),
            "retained_visual_tokens_avg": round(summ["visual_tokens_avg"], 3),
        },
    }

    agg = aggregate_record(
        model=model_name, dataset=dataset, split=adapter.split, method="static",
        budget_pct=budget_pct, n=len(records), metric=adapter.metric, score_pct=score_pct,
        visual_tokens={"avg": summ["visual_tokens_avg"], "max": summ["visual_tokens_max"]},
        text_tokens_avg=summ["text_tokens_avg"], seq_len_avg=summ["seq_len_avg"],
        flops_prefill_TFLOPs_avg=summ["flops_prefill_TFLOPs_avg"],
        flops_convention=summ["flops_convention"],
        token_reduction_pct=token_reduction_pct, flop_reduction_pct=flop_reduction_pct,
        prompt_template=describe_prompt(dataset, use_ocr=use_ocr, instruction=instruction),
        max_new_tokens=max_new_tokens, decoding=decoding, image_pad=image_pad,
        sample_ids_path=os.path.join(manifest_dir, f"{dataset}.json"),
        sample_ids_sha256=manifest.sha256,
        **extra,
    )

    gate = fairness_gate(records, agg, manifest_ids=ids, manifest_sha=manifest.sha256,
                         expected_visual_tokens=expected_vis)
    # Qwen: soft sanity that the retained tokens track the requested budget (uniform may keep < K).
    if model_name == "qwen25vl7b":
        issue = qwen_retained_sanity(summ["visual_tokens_avg"], dense_summ["visual_tokens_avg"], budget_pct)
        if issue:
            gate = {"ok": False, "issues": gate["issues"] + [f"[retained-token sanity] {issue}"]}

    out_dir = os.path.join(out_root, model_name, dataset)
    out_jsonl = os.path.join(out_dir, f"{out_basename}.jsonl")
    out_json = os.path.join(out_dir, f"{out_basename}.json")
    write_per_sample_jsonl(out_jsonl, records, validate=True)
    write_aggregate_json(out_json, {**agg, "fairness_gate": gate}, validate=True)

    print(f"[{dataset}/{model_name} static {selector} p{budget_pct}] "
          f"score={score_pct}% (dense {dense_score_pct}% on same n, Δ={score_pct - dense_score_pct:+.2f}pp)  "
          f"vis~{summ['visual_tokens_avg']:.0f} (dense~{dense_summ['visual_tokens_avg']:.0f})  "
          f"tok-red={token_reduction_pct}%  flop-red={flop_reduction_pct}%  "
          f"gate={'OK' if gate['ok'] else 'ISSUES'}", flush=True)
    if not gate["ok"]:
        for p in gate["issues"][:8]:
            print(f"    [gate] {p}", flush=True)
    return StaticResult(out_jsonl, out_json, len(records), score_pct, dense_score_pct,
                        token_reduction_pct, flop_reduction_pct, gate["ok"], gate["issues"])
