"""
dynamic_which_eval.py — final-scope DYNAMIC-WHICH runner core + pure helpers.

Dynamic-WHICH = fixed-budget, QUESTION-CONDITIONED visual-token selection. It changes only
WHICH visual tokens are kept; the token BUDGET is IDENTICAL to static:
  budgets = {15, 25, 35, 50, 75}
  LLaVA keep_k  = {15:86, 25:144, 35:202, 50:288, 75:432}        (same map as static)
  Qwen  K_sample = clamp(round(budget_pct/100 · dense_n_visual), 1, dense_n_visual)  (same rule)

It is the methodological sibling of static_eval.py and REUSES static_eval's budget/K bookkeeping
(BUDGET_TO_KEEP_K, budget_keep_k, clamp_qwen_k, resolve_n, accuracy_deltas, qwen_retained_sanity,
_sha256_file, load_dense_reference) plus dense_pilot's adapters/prompts/scorers and the shared
output_writer / schema_validator / token_flops. The new pieces here are:
  * the DYNAMIC selector names per model,
  * loading BOTH a dense reference AND the matching STATIC final (same model/dataset/budget) so the
    runner reports dynamic-vs-dense AND dynamic-vs-static deltas over the SAME samples,
  * the dynamic output schema fields.

This module contains NO model/generation code. The actual question-conditioned scoring +
generation is supplied by a runner (see scripts/dynamic_which/run_dynamic_which.py) via
  generate_and_count(sample, dense_record) -> (pred_raw, n_visual_tokens, n_text_tokens, prompt)
exactly like static_eval. The runner uses fresh textsim wrappers that COMPOSE the frozen
generators (no edits to static.py / qwen_pruner.py), select top-K then SORT indices back into
original visual order, run bs=1 greedy decode, and do NOT train or do a dense LM prefill.

Pure / CPU-only — imports under a plain Python for compileall + the self-checks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.common.output_writer import (
    per_sample_record,
    aggregate_record,
    write_per_sample_jsonl,
    write_aggregate_json,
)
from src.common.schema_validator import fairness_gate
from src.common.sample_ids import load_manifest
from src.common import token_flops as tf
from src.dense.evaluate_dense import (
    Sample,
    get_adapter,
    describe_prompt,
    PILOT_N,
    MAX_NEW_TOKENS,
    INSTRUCTION,
    MANIFEST_DIR,
    OUT_ROOT,
    MODELS,
    DATASETS,
)
# Reuse static_eval bookkeeping verbatim — the budget/K contract is IDENTICAL to static.
from src.static import evaluate_static as se
from src.static.evaluate_static import (
    BUDGET_TO_KEEP_K,
    BUDGETS,
    budget_keep_k,
    clamp_qwen_k,
    resolve_n,
    accuracy_deltas,
    qwen_retained_sanity,
    load_dense_reference,
    _sha256_file,
)

DEFAULT_ALPHA = 0.5
# Which text drives the selector scoring (question-token embeddings). The primary method scores
# on the RAW question — NOT the VLM prompt (which for TextVQA also carries the OCR block).
DEFAULT_SELECTOR_TEXT_SOURCE = "sample.question"

# ── dynamic-WHICH selectors per model ─────────────────────────────────────────
#   LLaVA: textsim (cos(projected-visual, question-token embeds)),
#          textsim_cls_mix (normalized mix of textsim and CLS-attention).
#   Qwen : textsim (cos(visual embeds, question-token embeds)),
#          textsim_norm_mix (normalized mix of textsim and visual L2 norm).
LLAVA_DYNAMIC_SELECTORS: Tuple[str, ...] = ("textsim", "textsim_cls_mix")
QWEN_DYNAMIC_SELECTORS: Tuple[str, ...] = ("textsim", "textsim_norm_mix")


def default_dynamic_selector(model_name: str) -> str:
    """Primary dynamic-WHICH selector: textsim for both models."""
    if model_name in ("llava15", "qwen25vl7b"):
        return "textsim"
    raise ValueError(f"unknown model {model_name!r}")


def allowed_dynamic_selectors(model_name: str) -> Tuple[str, ...]:
    if model_name == "llava15":
        return LLAVA_DYNAMIC_SELECTORS
    if model_name == "qwen25vl7b":
        return QWEN_DYNAMIC_SELECTORS
    raise ValueError(f"unknown model {model_name!r}")


def dynamic_basename(dataset: str, n: int, selector: str, budget_pct: int, *,
                     use_ocr: bool = True, instruction: bool = True, full: bool = False) -> str:
    """Variant-aware dynamic-WHICH output basename (same variant scheme as static).

      full=False (pilot)                                   full=True (final)
      GQA / VQAv2 : dynamic_which_pilot_n{n}_{sel}_p{b}                dynamic_which_final_{sel}_p{b}
      TextVQA     : ..._ocr_{on,off}                                   ..._ocr_{on,off}
      DocVQA      : ..._instruction_{on,off}                           ..._instruction_{on,off}
    """
    base = (f"dynamic_which_final_{selector}_p{budget_pct}" if full
            else f"dynamic_which_pilot_n{n}_{selector}_p{budget_pct}")
    if dataset == "textvqa":
        return f"{base}_ocr_{'on' if use_ocr else 'off'}"
    if dataset == "docvqa":
        return f"{base}_instruction_{'on' if instruction else 'off'}"
    return base


def dynamic_minus_static(dynamic_score_pct: float, static_score_pct: float) -> float:
    """The headline Dynamic-WHICH metric: dynamic − static (pp), positive ⇒ WHICH beats static."""
    return round(float(dynamic_score_pct) - float(static_score_pct), 2)


def select_topk_sorted(scores: Sequence[float], k: int) -> List[int]:
    """Top-K indices by score, then SORTED back into original visual-token order (the critical
    Dynamic-WHICH rule). The GPU runner mirrors this with `scores.topk(k).indices.sort().values`.
    At k == len(scores) this is the identity [0,1,...,N-1] → keep-all reproduces dense."""
    n = len(scores)
    k = max(1, min(int(k), n))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(order)


def minmax_normalize(values: Sequence[float]) -> List[float]:
    """Min-max to [0,1] (eps-stabilised). Mirrors the runner's torch min-max for the mixes."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng <= 0:
        return [0.0 for _ in values]
    return [(v - lo) / (rng + 1e-8) for v in values]


def normalized_mix(score_a: Sequence[float], score_b: Sequence[float], alpha: float) -> List[float]:
    """alpha · norm(a) + (1-alpha) · norm(b) — the textsim_*_mix combination (CPU reference)."""
    na = minmax_normalize(score_a)
    nb = minmax_normalize(score_b)
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(na, nb)]


# ── reference resolution ─────────────────────────────────────────────────────

def static_reference_basename(model_name: str, dataset: str, budget_pct: int, *,
                              use_ocr: bool = True, instruction: bool = True,
                              static_selector: Optional[str] = None) -> str:
    """Basename of the STATIC FINAL to compare against: the PRIMARY static selector at the same
    budget/variant (LLaVA→cls_attn, Qwen→norm), unless `static_selector` overrides it."""
    sel = static_selector if static_selector is not None else se.default_selector(model_name)
    return se.static_basename(dataset, 0, sel, budget_pct,
                              use_ocr=use_ocr, instruction=instruction, full=True)


def load_static_reference(model_name: str, dataset: str, budget_pct: int, *,
                          use_ocr: bool, instruction: bool, static_selector: Optional[str] = None,
                          out_root: str = OUT_ROOT) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Load the matching STATIC FINAL per-sample JSONL (same model/dataset/budget/variant), keyed
    by str(sample_id). Returns (jsonl_path, {str(sample_id): record})."""
    base = static_reference_basename(model_name, dataset, budget_pct, use_ocr=use_ocr,
                                     instruction=instruction, static_selector=static_selector)
    jsonl = os.path.join(out_root, model_name, dataset, f"{base}.jsonl")
    if not os.path.exists(jsonl):
        raise FileNotFoundError(
            f"static reference JSONL not found: {jsonl}\n"
            f"  run the matching STATIC FINAL ({base}) for {model_name}×{dataset} "
            f"budget {budget_pct}% first."
        )
    lookup: Dict[str, Dict[str, Any]] = {}
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lookup[str(r["sample_id"])] = r
    return jsonl, lookup


# ── dynamic-WHICH runner core ─────────────────────────────────────────────────

@dataclass
class DynamicWhichResult:
    out_jsonl: str
    out_json: str
    n: int
    score_pct: float
    dense_score_pct: float
    static_score_pct: float
    dynamic_minus_static_pp: float
    token_reduction_pct: float
    flop_reduction_pct: float
    gate_ok: bool
    gate_issues: List[str] = field(default_factory=list)


def run_dynamic_which(
    *,
    model_name: str,
    dataset: str,
    selector: str,
    budget_pct: int,
    generate_and_count: Callable[[Sample, Dict[str, Any]], Tuple[str, int, int, str]],
    n: int = PILOT_N,
    full: bool = False,
    alpha: float = DEFAULT_ALPHA,
    static_selector: Optional[str] = None,
    selector_text_source: str = DEFAULT_SELECTOR_TEXT_SOURCE,
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
) -> DynamicWhichResult:
    """
    Run ONE dynamic-WHICH cell: (model, dataset, selector, budget_pct).

    Parallel to static_eval.run_static. `generate_and_count(sample, dense_record)` returns
    (pred_raw, n_visual_tokens, n_text_tokens, prompt); the Qwen runner reads its per-sample K
    from `dense_record["n_visual_tokens"]` and both runners reuse `dense_record["n_text_tokens"]`.
    Loads the dense reference (dense-vs-dynamic Δ, reductions, Qwen K) AND the matching static
    final (dynamic-vs-static Δ over the SAME samples). GPU loop — do not call until approved.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if selector not in allowed_dynamic_selectors(model_name):
        raise ValueError(f"selector {selector!r} not allowed for {model_name}; "
                         f"expected one of {allowed_dynamic_selectors(model_name)}")
    keep_k = budget_keep_k(model_name, budget_pct)           # int (LLaVA) or None (Qwen)
    static_sel = static_selector if static_selector is not None else se.default_selector(model_name)

    manifest = load_manifest(os.path.join(manifest_dir, f"{dataset}.json"))
    total = len(manifest.ids)
    n = resolve_n(n, full, total)
    ids = manifest.ids[:n]

    dense_jsonl, dense_lookup = load_dense_reference(
        model_name, dataset, use_ocr=use_ocr, instruction=instruction, out_root=out_root)
    static_jsonl, static_lookup = load_static_reference(
        model_name, dataset, budget_pct, use_ocr=use_ocr, instruction=instruction,
        static_selector=static_sel, out_root=out_root)
    miss_d = [str(sid) for sid in ids if str(sid) not in dense_lookup]
    miss_s = [str(sid) for sid in ids if str(sid) not in static_lookup]
    if miss_d:
        raise ValueError(f"{len(miss_d)} id(s) missing from dense reference {dense_jsonl} (e.g. {miss_d[:5]})")
    if miss_s:
        raise ValueError(f"{len(miss_s)} id(s) missing from static reference {static_jsonl} (e.g. {miss_s[:5]})")

    if out_basename is None:
        out_basename = dynamic_basename(dataset, n, selector, budget_pct,
                                        use_ocr=use_ocr, instruction=instruction, full=full)
    adapter = get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)
    decoding = f"greedy bs=1 do_sample=False max_new_tokens={max_new_tokens}"

    records: List[Dict[str, Any]] = []
    stats: List[tf.TokenStats] = []
    dense_stats: List[tf.TokenStats] = []        # dense token stats over the SAME ids
    dense_scores: List[float] = []
    static_scores: List[float] = []
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        drec = dense_lookup[str(sid)]
        srec = static_lookup[str(sid)]
        pred_raw, n_vis, n_txt, prompt = generate_and_count(s, drec)
        pred_norm, score = adapter.score(pred_raw, s)
        records.append(per_sample_record(
            sample_id=s.sample_id, image_id=s.image_id, dataset=dataset, model=model_name,
            method="dynamic_which", budget_pct=budget_pct, question=s.question, prompt=prompt,
            pred_raw=str(pred_raw).strip(), pred_norm=str(pred_norm), gold=s.gold,
            per_sample_score=float(score), n_visual_tokens=int(n_vis), n_text_tokens=int(n_txt),
            selector=selector,
        ))
        stats.append(tf.TokenStats(int(n_vis), int(n_txt)))
        dense_stats.append(tf.TokenStats(int(drec["n_visual_tokens"]), int(drec["n_text_tokens"])))
        dense_scores.append(float(drec.get("per_sample_score", 0.0)))
        static_scores.append(float(srec.get("per_sample_score", 0.0)))
        if (i + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name} dynamic_which {selector} p{budget_pct}] "
                  f"{i+1}/{len(ids)}  running_score={avg*100:.2f}%", flush=True)

    summ = tf.summarize_tokens_and_flops(model_name, stats)
    dense_summ = tf.summarize_tokens_and_flops(model_name, dense_stats)   # SAME samples → fair Δ
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    dense_score_pct = round(sum(dense_scores) / max(len(dense_scores), 1) * 100, 2)
    static_score_pct = round(sum(static_scores) / max(len(static_scores), 1) * 100, 2)
    accuracy_delta_pp, accuracy_drop_pp = accuracy_deltas(score_pct, dense_score_pct)
    dyn_minus_static_pp = dynamic_minus_static(score_pct, static_score_pct)
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
        "static_reference_path": static_jsonl,
        "static_reference_sha256": _sha256_file(static_jsonl),
        "static_reference_selector": static_sel,
        "static_reference_score_pct": static_score_pct,
        # signed deltas: vs dense (delta = dyn-dense, drop = dense-dyn) and vs static (dyn-static).
        "accuracy_delta_pp": accuracy_delta_pp,
        "accuracy_drop_pp": accuracy_drop_pp,
        "dynamic_minus_static_pp": dyn_minus_static_pp,
        # prompt/decode provenance — TOP-LEVEL (matches dense/static schema)
        "protocol_instruction": INSTRUCTION,
        "model_instruction_suffix": model_instruction_suffix,
        "decode_notes": decode_notes,
        "extra_dynamic_which": {
            "n_requested": n,
            "budget_pct": budget_pct,
            "alpha": alpha if selector.endswith("_mix") else None,
            "selector_text_source": selector_text_source,
            "use_ocr": use_ocr if dataset == "textvqa" else None,
            "instruction": instruction if dataset == "docvqa" else None,
            "max_pixels": max_pixels,
            "selection_rule": "score top-K then sort indices back into original visual order; bs=1; "
                              "no training; no answer head; no dense LM prefill",
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
        model=model_name, dataset=dataset, split=adapter.split, method="dynamic_which",
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
    # Qwen: soft sanity that retained tokens track the requested budget (same rule as static).
    if model_name == "qwen25vl7b":
        issue = qwen_retained_sanity(summ["visual_tokens_avg"], dense_summ["visual_tokens_avg"], budget_pct)
        if issue:
            gate = {"ok": False, "issues": gate["issues"] + [f"[retained-token sanity] {issue}"]}

    out_dir = os.path.join(out_root, model_name, dataset)
    out_jsonl = os.path.join(out_dir, f"{out_basename}.jsonl")
    out_json = os.path.join(out_dir, f"{out_basename}.json")
    write_per_sample_jsonl(out_jsonl, records, validate=True)
    write_aggregate_json(out_json, {**agg, "fairness_gate": gate}, validate=True)

    print(f"[{dataset}/{model_name} dynamic_which {selector} p{budget_pct}] "
          f"score={score_pct}%  vs dense {dense_score_pct}% (Δ={accuracy_delta_pp:+.2f}pp)  "
          f"vs static[{static_sel}] {static_score_pct}% (Δ={dyn_minus_static_pp:+.2f}pp)  "
          f"vis~{summ['visual_tokens_avg']:.0f}  tok-red={token_reduction_pct}%  "
          f"flop-red={flop_reduction_pct}%  gate={'OK' if gate['ok'] else 'ISSUES'}", flush=True)
    if not gate["ok"]:
        for p in gate["issues"][:8]:
            print(f"    [gate] {p}", flush=True)
    return DynamicWhichResult(out_jsonl, out_json, len(records), score_pct, dense_score_pct,
                              static_score_pct, dyn_minus_static_pp, token_reduction_pct,
                              flop_reduction_pct, gate["ok"], gate["issues"])
