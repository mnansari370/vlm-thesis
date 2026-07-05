"""
run_dynamic_which_ref_eval.py — run ONE CLEAN-ROOM Dynamic-WHICH REFERENCE pilot
(model × dataset × ref-selector × budget) under the final-scope schema. GPU job.

Uses the fresh src/pruning/dynamic_which_ref/* wrappers (NOT the original dynamic_which/*). Keeps
the SAME fairness schema, manifests, prompts, scorers, and the SAME dense/static references as the
original path, so ref-vs-current is a controlled comparison. Outputs are named
`dynamic_which_ref_pilot_n{n}_{selector}_p{b}[variant]` — they CANNOT overwrite existing
dynamic_which outputs. Pilot only: there is NO --full here (by design).

Rescue selectors:
  textsim_ref, saliency_mix_ref, static_guarded_textsim_ref  (+ coverage_guarded_textsim_ref, LLaVA only)

Envs / GPU:
  LLaVA : CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python \
              -m scripts.final_scope.run_dynamic_which_ref_eval --model llava15 --dataset textvqa \
              --budget-pct 25 --selector static_guarded_textsim_ref --n 200
  Qwen  : CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python \
              -m scripts.final_scope.run_dynamic_which_ref_eval --model qwen25vl7b --dataset gqa \
              --budget-pct 25 --selector saliency_mix_ref --n 200
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import static_eval as se
from src.final_scope import dynamic_which_eval as dw
from src.final_scope import token_flops as tf
from src.final_scope.sample_ids import load_manifest
from src.final_scope.output_writer import (
    per_sample_record, aggregate_record, write_per_sample_jsonl, write_aggregate_json)
from src.final_scope.schema_validator import fairness_gate
from src.pruning import dynamic_which_ref as ref

METHOD = "dynamic_which_ref"


# ── model runners (lazy torch imports) ────────────────────────────────────────

def build_llava_ref_runner(*, selector, keep_k, alpha, guard_frac, add_instruction, max_new_tokens):
    import torch  # noqa: F401
    from src.pruning.dynamic_which_ref.llava_ref import LlavaWhichRef
    from src.models.static.static import HONEST_SUFFIX
    model = LlavaWhichRef(selector=selector, keep_k=keep_k, alpha=alpha, guard_frac=guard_frac,
                          add_instruction=add_instruction, max_new_tokens=max_new_tokens)
    suffix = HONEST_SUFFIX if add_instruction else ""

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        pred = model.generate_one(s.image, s.model_input_text, s.question)
        return pred, keep_k, int(dense_rec["n_text_tokens"]), s.model_input_text.strip() + suffix

    meta = {"model_instruction_suffix": suffix, "image_pad": True, "max_pixels": None,
            "decode_notes": (f"LlavaWhichRef selector={selector} K={keep_k} alpha={alpha} "
                             f"guard_frac={guard_frac}; clean-room selection; VLM prompt = "
                             "sample.model_input_text; scored on sample.question; honest greedy; "
                             "no dense LM prefill")}
    return generate_and_count, meta


def build_qwen_ref_runner(*, selector, budget_pct, alpha, guard_frac, add_instruction, max_new_tokens):
    import torch  # noqa: F401
    from src.pruning.dynamic_which_ref.qwen_ref import QwenWhichRef
    from src.pruning.question_conditioned_selection.qwen_pruner import INSTR
    model = QwenWhichRef(selector=selector, alpha=alpha, guard_frac=guard_frac,
                         max_new_tokens=max_new_tokens)
    max_pixels = model.proc.image_processor.max_pixels if hasattr(model.proc, "image_processor") else None
    suffix = (" " + INSTR) if add_instruction else ""

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        k = dw.clamp_qwen_k(budget_pct, int(dense_rec["n_visual_tokens"]))
        pred, nkeep = model.generate_one(s.image, s.model_input_text, s.question, k, instr=s.add_instruction)
        return pred, int(nkeep), int(dense_rec["n_text_tokens"]), s.model_input_text.strip() + suffix

    meta = {"model_instruction_suffix": suffix, "image_pad": None, "max_pixels": max_pixels,
            "decode_notes": (f"QwenWhichRef selector={selector} alpha={alpha} guard_frac={guard_frac}; "
                             f"K_sample=clamp(round({budget_pct}%·dense_n_visual),1,dense_n_visual); "
                             "clean-room selection; no dense LM prefill")}
    return generate_and_count, meta


# ── ref eval core (pilot only; reuses dense/static references + fairness schema) ──

def run_ref(*, model_name, dataset, selector, budget_pct, generate_and_count, n, alpha, guard_frac,
            static_selector=None, max_new_tokens=dp.MAX_NEW_TOKENS, image_pad=None, max_pixels=None,
            use_ocr=True, instruction=True, model_instruction_suffix="", decode_notes="",
            manifest_dir=dp.MANIFEST_DIR, out_root=dp.OUT_ROOT, log_every=50):
    if selector not in ref.allowed_ref_selectors(model_name):
        raise ValueError(f"selector {selector!r} not allowed for {model_name}; "
                         f"expected {ref.allowed_ref_selectors(model_name)}")
    keep_k = dw.budget_keep_k(model_name, budget_pct)          # int (LLaVA) / None (Qwen)
    static_sel = static_selector or se.default_selector(model_name)

    manifest = load_manifest(os.path.join(manifest_dir, f"{dataset}.json"))
    n = dw.resolve_n(n, False, len(manifest.ids))             # full always disallowed here
    ids = manifest.ids[:n]

    dense_jsonl, dense_lookup = dw.load_dense_reference(
        model_name, dataset, use_ocr=use_ocr, instruction=instruction, out_root=out_root)
    static_jsonl, static_lookup = dw.load_static_reference(
        model_name, dataset, budget_pct, use_ocr=use_ocr, instruction=instruction,
        static_selector=static_sel, out_root=out_root)
    for sid in ids:
        if str(sid) not in dense_lookup:
            raise ValueError(f"id {sid} missing from dense reference {dense_jsonl}")
        if str(sid) not in static_lookup:
            raise ValueError(f"id {sid} missing from static reference {static_jsonl}")

    out_basename = ref.ref_basename(dataset, n, selector, budget_pct,
                                    use_ocr=use_ocr, instruction=instruction)
    adapter = dp.get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)

    records: List[Dict[str, Any]] = []
    stats: List[tf.TokenStats] = []
    dense_stats: List[tf.TokenStats] = []
    dense_scores: List[float] = []
    static_scores: List[float] = []
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        drec, srec = dense_lookup[str(sid)], static_lookup[str(sid)]
        pred_raw, n_vis, n_txt, prompt = generate_and_count(s, drec)
        pred_norm, score = adapter.score(pred_raw, s)
        records.append(per_sample_record(
            sample_id=s.sample_id, image_id=s.image_id, dataset=dataset, model=model_name,
            method=METHOD, budget_pct=budget_pct, question=s.question, prompt=prompt,
            pred_raw=str(pred_raw).strip(), pred_norm=str(pred_norm), gold=s.gold,
            per_sample_score=float(score), n_visual_tokens=int(n_vis), n_text_tokens=int(n_txt),
            selector=selector))
        stats.append(tf.TokenStats(int(n_vis), int(n_txt)))
        dense_stats.append(tf.TokenStats(int(drec["n_visual_tokens"]), int(drec["n_text_tokens"])))
        dense_scores.append(float(drec.get("per_sample_score", 0.0)))
        static_scores.append(float(srec.get("per_sample_score", 0.0)))
        if (i + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name} {METHOD} {selector} p{budget_pct}] "
                  f"{i+1}/{len(ids)} running={avg*100:.2f}%", flush=True)

    summ = tf.summarize_tokens_and_flops(model_name, stats)
    dense_summ = tf.summarize_tokens_and_flops(model_name, dense_stats)
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    dense_score_pct = round(sum(dense_scores) / max(len(dense_scores), 1) * 100, 2)
    static_score_pct = round(sum(static_scores) / max(len(static_scores), 1) * 100, 2)
    delta_pp, drop_pp = dw.accuracy_deltas(score_pct, dense_score_pct)
    dyn_minus_static = dw.dynamic_minus_static(score_pct, static_score_pct)
    token_red = round(tf.reduction_pct(summ["seq_len_avg"], dense_summ["seq_len_avg"]), 2)
    flop_red = round(tf.reduction_pct(summ["flops_prefill_TFLOPs_avg"],
                                      dense_summ["flops_prefill_TFLOPs_avg"]), 2)
    expected_vis = keep_k

    extra = {
        "selector_name": selector, "keep_k": keep_k,
        "dense_reference_path": dense_jsonl, "dense_reference_score_pct": dense_score_pct,
        "static_reference_path": static_jsonl, "static_reference_selector": static_sel,
        "static_reference_score_pct": static_score_pct,
        "accuracy_delta_pp": delta_pp, "accuracy_drop_pp": drop_pp,
        "dynamic_minus_static_pp": dyn_minus_static,
        "protocol_instruction": dw.INSTRUCTION, "model_instruction_suffix": model_instruction_suffix,
        "decode_notes": decode_notes,
        "extra_dynamic_which_ref": {
            "clean_room": True, "n_requested": n, "budget_pct": budget_pct,
            "alpha": alpha if selector == "saliency_mix_ref" else None,
            "guard_frac": guard_frac if selector in ref.GUARDED_SELECTORS else None,
            "selector_text_source": "sample.question",
            "use_ocr": use_ocr if dataset == "textvqa" else None,
            "instruction": instruction if dataset == "docvqa" else None,
            "max_pixels": max_pixels,
            "retained_visual_tokens_avg": round(summ["visual_tokens_avg"], 3),
        },
    }
    agg = aggregate_record(
        model=model_name, dataset=dataset, split=adapter.split, method=METHOD,
        budget_pct=budget_pct, n=len(records), metric=adapter.metric, score_pct=score_pct,
        visual_tokens={"avg": summ["visual_tokens_avg"], "max": summ["visual_tokens_max"]},
        text_tokens_avg=summ["text_tokens_avg"], seq_len_avg=summ["seq_len_avg"],
        flops_prefill_TFLOPs_avg=summ["flops_prefill_TFLOPs_avg"],
        flops_convention=summ["flops_convention"], token_reduction_pct=token_red,
        flop_reduction_pct=flop_red,
        prompt_template=dp.describe_prompt(dataset, use_ocr=use_ocr, instruction=instruction),
        max_new_tokens=max_new_tokens, decoding=f"greedy bs=1 do_sample=False max_new_tokens={max_new_tokens}",
        image_pad=image_pad, sample_ids_path=os.path.join(manifest_dir, f"{dataset}.json"),
        sample_ids_sha256=manifest.sha256, **extra)

    gate = fairness_gate(records, agg, manifest_ids=ids, manifest_sha=manifest.sha256,
                         expected_visual_tokens=expected_vis)
    if model_name == "qwen25vl7b":
        issue = se.qwen_retained_sanity(summ["visual_tokens_avg"], dense_summ["visual_tokens_avg"], budget_pct)
        if issue:
            gate = {"ok": False, "issues": gate["issues"] + [f"[retained-token sanity] {issue}"]}

    out_dir = os.path.join(out_root, model_name, dataset)
    out_jsonl = os.path.join(out_dir, f"{out_basename}.jsonl")
    out_json = os.path.join(out_dir, f"{out_basename}.json")
    if os.path.exists(out_jsonl) or os.path.exists(out_json):
        raise FileExistsError(f"refusing to overwrite existing ref output: {out_basename}.*")
    write_per_sample_jsonl(out_jsonl, records, validate=True)
    write_aggregate_json(out_json, {**agg, "fairness_gate": gate}, validate=True)
    print(f"[{dataset}/{model_name} {METHOD} {selector} p{budget_pct}] score={score_pct}% "
          f"vs static[{static_sel}] {static_score_pct}% (Δ={dyn_minus_static:+.2f}pp) "
          f"vs dense {dense_score_pct}% gate={'OK' if gate['ok'] else 'ISSUES'}", flush=True)
    if not gate["ok"]:
        for p in gate["issues"][:8]:
            print(f"    [gate] {p}", flush=True)
    return out_json, out_jsonl, score_pct, static_score_pct, dyn_minus_static, gate["ok"]


def build_parser() -> argparse.ArgumentParser:
    """Pilot-only CLI. By design there is NO --full flag (ref eval is n-only until enabled)."""
    ap = argparse.ArgumentParser(description="Clean-room Dynamic-WHICH REFERENCE pilot (n only; no --full).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--budget-pct", required=True, type=int, choices=list(dw.BUDGETS), dest="budget_pct")
    ap.add_argument("--selector", default=None,
                    choices=sorted(set(ref.LLAVA_REF_SELECTORS + ref.QWEN_REF_SELECTORS)))
    ap.add_argument("--alpha", type=float, default=ref.DEFAULT_ALPHA)
    ap.add_argument("--guard-frac", type=float, default=ref.DEFAULT_GUARD_FRAC, dest="guard_frac")
    ap.add_argument("--static-selector", default=None, dest="static_selector",
                    choices=["cls_attn", "norm", "uniform"])
    ap.add_argument("--n", type=int, default=None, help="pilot size (default 200). No --full here.")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-instruction", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    selector = args.selector or ref.default_ref_selector(args.model)
    if selector not in ref.allowed_ref_selectors(args.model):
        ap.error(f"--selector {selector} not valid for {args.model}; allowed: "
                 f"{ref.allowed_ref_selectors(args.model)}")
    n = args.n if args.n is not None else dp.PILOT_N
    use_ocr = not args.no_ocr
    instruction = not args.no_instruction
    if args.dataset in ("gqa", "vqav2"):
        add_instruction = True
    elif args.dataset == "textvqa":
        add_instruction = False
    else:
        add_instruction = instruction

    keep_k = dw.budget_keep_k(args.model, args.budget_pct)
    if args.model == "llava15":
        gen, meta = build_llava_ref_runner(selector=selector, keep_k=keep_k, alpha=args.alpha,
                                           guard_frac=args.guard_frac, add_instruction=add_instruction,
                                           max_new_tokens=args.max_new_tokens)
    else:
        gen, meta = build_qwen_ref_runner(selector=selector, budget_pct=args.budget_pct, alpha=args.alpha,
                                          guard_frac=args.guard_frac, add_instruction=add_instruction,
                                          max_new_tokens=args.max_new_tokens)

    out_json, out_jsonl, score, static, dms, ok = run_ref(
        model_name=args.model, dataset=args.dataset, selector=selector, budget_pct=args.budget_pct,
        generate_and_count=gen, n=n, alpha=args.alpha, guard_frac=args.guard_frac,
        static_selector=args.static_selector, max_new_tokens=args.max_new_tokens,
        image_pad=meta["image_pad"], max_pixels=meta["max_pixels"], use_ocr=use_ocr,
        instruction=instruction, model_instruction_suffix=meta["model_instruction_suffix"],
        decode_notes=meta["decode_notes"])
    print(f"\n[done] ref pilot: {args.model} × {args.dataset} {selector} p{args.budget_pct}: "
          f"score={score}% dyn−static={dms:+.2f}pp gate={'OK' if ok else 'ISSUES'}\n  {out_json}\n  {out_jsonl}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
