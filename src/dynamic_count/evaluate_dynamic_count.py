"""
dynamic_count_eval.py — final-scope Dynamic-COUNT runner cores (probe / DC-D / DC-C).

Three cores, all writing the standard final-scope schema (method = dynamic_count_probe |
dynamic_count_dcd | dynamic_count_dcc), sharing the locked manifests, adapters, prompts, scorers,
fairness gate, and honest multi-pass FLOPs accounting:

  * run_probe  (GPU)  — cheap pass at K_probe with confidence/saliency/question signal recording.
                        BUILT-IN REPRODUCTION GATE: predictions must equal the frozen reference
                        finals (static cls_attn/norm, or WHICH textsim) per sample; mismatches
                        fail the gate and the run exits non-zero. Probe outputs are the substrate
                        for both DC variants.
  * compose_dcd (CPU) — DC-D discrete cascade composed offline: keep probe answer if confident,
                        else take the FROZEN target-anchor answer; both prefills charged.
  * run_dcc    (GPU)  — DC-C, THE main method: a calibrated controller predicts a per-sample
                        INTEGER K_i (continuous fractions × native token count, clamped to
                        [MIN_TOKENS, N_i]); K_i ≤ probe keeps the probe answer, K_i > probe does
                        ONE real second pass at exactly K_i. Never restricted to the anchors.

Fair comparison: the static accuracy-vs-FLOPs curve is rebuilt from the frozen per-sample anchor
JSONLs restricted to EXACTLY the evaluated ids (plus the dense point), so Δcurve at matched
average FLOPs is computed on identical samples. Labels: dynamic_win > +0.50pp, near_tie within
±0.50pp, dynamic_loss < −0.50pp.

CPU-importable (GPU deps only inside the runner loops via the supplied model wrappers).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.dense import evaluate_dense as dp
from src.static import evaluate_static as se
from src.common import token_flops as tf
from src.common.sample_ids import load_manifest
from src.common.output_writer import (per_sample_record, aggregate_record,
                                           write_per_sample_jsonl, write_aggregate_json)
from src.common.schema_validator import fairness_gate
from src.pruning import dynamic_count as dc
from src.pruning.dynamic_count.confidence import risk_from_signal

WIN_TH = 0.50   # pp vs interpolated static curve

# split/metric per dataset (mirrors the dense_pilot adapters; avoids constructing adapters — and
# for DocVQA, loading the HF dataset — during CPU-only composition/finalization)
SPLIT_METRIC = {"gqa": ("testdev_balanced", "gqa_exact"), "vqav2": ("validation", "vqa_consensus"),
                "textvqa": ("val", "m4c_soft"), "docvqa": ("validation", "anls")}


def label_delta(delta_pp: float) -> str:
    if delta_pp > WIN_TH:
        return "dynamic_win"
    if delta_pp < -WIN_TH:
        return "dynamic_loss"
    return "near_tie"


# ── shared loading helpers ─────────────────────────────────────────────────────

def _load_jsonl(path: str) -> Dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["sample_id"])] = r
    return out


def frozen_reference_path(model: str, dataset: str, selector: str, pct: int, *,
                          use_ocr: bool = True, instruction: bool = True) -> str:
    """The frozen final whose predictions the probe must reproduce exactly."""
    if selector == "textsim":
        if not (model == "qwen25vl7b" and dataset == "textvqa"):
            raise ValueError("textsim probe is only defined for qwen25vl7b × textvqa")
        base = f"dynamic_which_final_textsim_p{pct}_ocr_on"
    else:
        base = se.static_basename(dataset, 0, selector, pct, use_ocr=use_ocr,
                                  instruction=instruction, full=True)
    return os.path.join(dp.OUT_ROOT, model, dataset, f"{base}.jsonl")


def anchor_reference_path(model: str, dataset: str, selector: str, pct: int, *,
                          use_ocr: bool = True, instruction: bool = True) -> str:
    """Frozen anchor outcomes for a given probe SELECTOR: textsim probes escalate/calibrate within
    the frozen WHICH textsim finals (COUNT-on-WHICH stays self-contained); static probes use the
    locked static selector's finals. Never mixes the two."""
    if selector == "textsim":
        return frozen_reference_path(model, dataset, "textsim", pct,
                                     use_ocr=use_ocr, instruction=instruction)
    return frozen_reference_path(model, dataset, se.default_selector(model), pct,
                                 use_ocr=use_ocr, instruction=instruction)


def dense_reference_rows(model: str, dataset: str, *, use_ocr: bool = True,
                         instruction: bool = True) -> Dict[str, dict]:
    path = os.path.join(dp.OUT_ROOT, model, dataset,
                        dp.default_basename(dataset, 0, use_ocr=use_ocr,
                                            instruction=instruction, full=True) + ".jsonl")
    return _load_jsonl(path)


def static_curve_for_ids(model: str, dataset: str, ids: Sequence[str], *,
                         use_ocr: bool = True, instruction: bool = True) -> List[Tuple[float, float]]:
    """(avg prefill TFLOPs, score_pct) of every frozen anchor + dense, over EXACTLY these ids."""
    idset = [str(i) for i in ids]
    pts = []
    sel = se.default_selector(model)
    sources = [(os.path.join(dp.OUT_ROOT, model, dataset,
                             se.static_basename(dataset, 0, sel, b, use_ocr=use_ocr,
                                                instruction=instruction, full=True) + ".jsonl"))
               for b in dc.ANCHOR_BUDGETS]
    dense_base = dp.default_basename(dataset, 0, use_ocr=use_ocr, instruction=instruction, full=True)
    sources.append(os.path.join(dp.OUT_ROOT, model, dataset, f"{dense_base}.jsonl"))
    for path in sources:
        rows = _load_jsonl(path)
        sub = [rows[i] for i in idset if i in rows]
        if len(sub) != len(idset):
            raise ValueError(f"{path}: only {len(sub)}/{len(idset)} evaluated ids present")
        fl = [tf.per_sample_prefill_flops(model, r["n_visual_tokens"], r["n_text_tokens"]) for r in sub]
        pts.append((sum(fl) / len(fl) / 1e12,
                    100.0 * sum(float(r["per_sample_score"]) for r in sub) / len(sub)))
    return sorted(pts)


def interp_curve(pts: Sequence[Tuple[float, float]], x: float) -> float:
    pts = sorted(pts)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-12)
    return pts[-1][1]


def _k_range_hist(ks: Sequence[int]) -> Dict[str, float]:
    edges = [(0, 64), (65, 128), (129, 192), (193, 288), (289, 432), (433, 10 ** 9)]
    n = max(len(ks), 1)
    return {f"k_{lo}_{'plus' if hi > 10**8 else hi}": round(100.0 * sum(1 for k in ks if lo <= k <= hi) / n, 2)
            for lo, hi in edges}


# ── PROBE (GPU) ────────────────────────────────────────────────────────────────

def run_probe(*, model_name: str, dataset: str, selector: str, probe_pct: int,
              probe_fn: Callable, n: Optional[int] = None, full: bool = False,
              max_new_tokens: int = dp.MAX_NEW_TOKENS, use_ocr: bool = True,
              instruction: bool = True, model_instruction_suffix: str = "",
              decode_notes: str = "", out_root: str = dp.OUT_ROOT, log_every: int = 100):
    """
    probe_fn(sample, k_probe_sample) -> (pred_raw, n_kept, conf_signals dict, saliency_scores list)
    K_probe per sample: LLaVA = fixed anchor K; Qwen = clamp(round(pct·dense_nvis),1,dense_nvis).
    """
    if probe_pct not in dc.PROBE_BUDGETS:
        raise ValueError(f"probe_pct must be one of {dc.PROBE_BUDGETS}")
    if selector not in dc.PROBE_SELECTORS[model_name]:
        raise ValueError(f"selector {selector!r} not allowed for {model_name}")
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"))
    n_eff = se.resolve_n(n, full, len(manifest.ids))
    ids = manifest.ids[:n_eff]
    adapter = dp.get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)
    dense_path = os.path.join(dp.OUT_ROOT, model_name, dataset,
                              dp.default_basename(dataset, 0, use_ocr=use_ocr,
                                                  instruction=instruction, full=True) + ".jsonl")
    dense = _load_jsonl(dense_path)
    ref = _load_jsonl(frozen_reference_path(model_name, dataset, selector, probe_pct,
                                            use_ocr=use_ocr, instruction=instruction))
    fixed_k = se.budget_keep_k(model_name, probe_pct)      # int (LLaVA) / None (Qwen)

    records, mismatches = [], 0
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        drec = dense[str(sid)]
        k_probe = fixed_k if fixed_k is not None else se.clamp_qwen_k(probe_pct, drec["n_visual_tokens"])
        pred_raw, n_kept, conf, sal_scores = probe_fn(s, k_probe)
        pred_norm, score = adapter.score(pred_raw, s)
        match = str(pred_raw).strip() == str(ref[str(sid)]["pred_raw"]).strip()
        mismatches += (not match)
        extra = {"k_probe": int(k_probe), "probe_selector": selector,
                 "match_frozen": bool(match), **conf,
                 **dc.saliency_stats(sal_scores),
                 **dc.question_features(s.question, drec["n_visual_tokens"])}
        records.append(per_sample_record(
            sample_id=s.sample_id, image_id=s.image_id, dataset=dataset, model=model_name,
            method=dc.METHOD_PROBE, budget_pct=probe_pct, question=s.question,
            prompt=str(ref[str(sid)]["prompt"]), pred_raw=str(pred_raw).strip(),
            pred_norm=str(pred_norm), gold=s.gold, per_sample_score=float(score),
            n_visual_tokens=int(n_kept), n_text_tokens=int(drec["n_text_tokens"]), **extra))
        if (i + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name} probe {selector} p{probe_pct}] {i+1}/{len(ids)} "
                  f"score={avg*100:.2f}% frozen-mismatch={mismatches}", flush=True)

    stats = [tf.TokenStats(r["n_visual_tokens"], r["n_text_tokens"]) for r in records]
    summ = tf.summarize_tokens_and_flops(model_name, stats)
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    basename = dc.probe_basename(dataset, selector, probe_pct, n=n_eff, full=full,
                                 use_ocr=use_ocr, instruction=instruction)
    equivalence_ok = (mismatches == 0)
    agg = aggregate_record(
        model=model_name, dataset=dataset, split=adapter.split, method=dc.METHOD_PROBE,
        budget_pct=probe_pct, n=len(records), metric=adapter.metric, score_pct=score_pct,
        visual_tokens={"avg": summ["visual_tokens_avg"], "max": summ["visual_tokens_max"]},
        text_tokens_avg=summ["text_tokens_avg"], seq_len_avg=summ["seq_len_avg"],
        flops_prefill_TFLOPs_avg=summ["flops_prefill_TFLOPs_avg"],
        flops_convention=summ["flops_convention"], token_reduction_pct=0.0, flop_reduction_pct=0.0,
        prompt_template=dp.describe_prompt(dataset, use_ocr=use_ocr, instruction=instruction),
        max_new_tokens=max_new_tokens, decoding=f"greedy bs=1 do_sample=False max_new_tokens={max_new_tokens}",
        image_pad=(True if model_name == "llava15" else None),
        sample_ids_path=os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"),
        sample_ids_sha256=manifest.sha256,
        protocol_instruction=dp.INSTRUCTION, model_instruction_suffix=model_instruction_suffix,
        decode_notes=decode_notes, selector_name=selector,
        frozen_reference_path=frozen_reference_path(model_name, dataset, selector, probe_pct,
                                                    use_ocr=use_ocr, instruction=instruction),
        frozen_match_count=len(records) - mismatches, frozen_mismatch_count=mismatches,
        equivalence_ok=equivalence_ok,
        extra_dynamic_count={"k_probe_fixed": fixed_k, "signals": list(dc.ALL_SIGNALS)})
    gate = fairness_gate(records, agg, manifest_ids=ids, manifest_sha=manifest.sha256,
                         expected_visual_tokens=fixed_k)
    if not equivalence_ok:
        gate = {"ok": False, "issues": gate["issues"] + [
            f"[reproduction gate] {mismatches} prediction(s) differ from the frozen reference — STOP AND DEBUG"]}
    out_dir = os.path.join(out_root, model_name, dataset)
    jl, js = os.path.join(out_dir, f"{basename}.jsonl"), os.path.join(out_dir, f"{basename}.json")
    if os.path.exists(jl) or os.path.exists(js):
        raise FileExistsError(f"refusing to overwrite existing probe output: {basename}.*")
    write_per_sample_jsonl(jl, records, validate=True)
    write_aggregate_json(js, {**agg, "fairness_gate": gate}, validate=True)
    print(f"[{dataset}/{model_name} probe {selector} p{probe_pct}] score={score_pct}% "
          f"frozen-match={len(records)-mismatches}/{len(records)} gate={'OK' if gate['ok'] else 'ISSUES'}",
          flush=True)
    return js, jl, score_pct, equivalence_ok, gate["ok"]


# ── DC-D (CPU compose) ─────────────────────────────────────────────────────────

def compose_dcd(*, model_name: str, dataset: str, selector: str, probe_pct: int, target_pct: int,
                risk_signal: str, tau: float, use_ocr: bool = True, instruction: bool = True,
                eval_split_only: bool = True, out_root: str = dp.OUT_ROOT):
    """DC-D: keep probe answer if risk ≤ τ, else take the FROZEN target-anchor answer.
    Both prefills charged for escalated samples. Composed offline (both passes deterministic)."""
    probe_path = os.path.join(out_root, model_name, dataset,
                              dc.probe_basename(dataset, selector, probe_pct,
                                                use_ocr=use_ocr, instruction=instruction) + ".jsonl")
    probe = _load_jsonl(probe_path)
    # target follows the PROBE selector: textsim probe → frozen WHICH textsim anchor (COUNT-on-WHICH
    # stays self-contained); static probe → the locked static selector's anchor. Never mixed.
    target = _load_jsonl(anchor_reference_path(model_name, dataset, selector, target_pct,
                                               use_ocr=use_ocr, instruction=instruction))
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"))
    all_ids = [str(i) for i in manifest.ids]
    calib_ids, eval_ids = dc.calibration_split(all_ids)
    ids = eval_ids if eval_split_only else all_ids
    missing = [i for i in ids if i not in probe or i not in target]
    if missing:
        raise ValueError(f"{len(missing)} id(s) missing from probe/target (e.g. {missing[:3]})")

    records, flops, esc_n = [], [], 0
    for sid in ids:
        p = probe[sid]
        risk = risk_from_signal(p[risk_signal], risk_signal)
        escalate = risk > tau
        src = target[sid] if escalate else p
        esc_n += escalate
        f = tf.per_sample_prefill_flops(model_name, p["n_visual_tokens"], p["n_text_tokens"])
        if escalate:
            f += tf.per_sample_prefill_flops(model_name, src["n_visual_tokens"], src["n_text_tokens"])
        flops.append(f)
        records.append(per_sample_record(
            sample_id=p["sample_id"], image_id=p["image_id"], dataset=dataset, model=model_name,
            method=dc.METHOD_DCD, budget_pct=probe_pct, question=p["question"], prompt=p["prompt"],
            pred_raw=src["pred_raw"], pred_norm=src["pred_norm"], gold=src["gold"],
            per_sample_score=float(src["per_sample_score"]),
            n_visual_tokens=int(src["n_visual_tokens"]), n_text_tokens=int(p["n_text_tokens"]),
            escalated=bool(escalate), n_passes=2 if escalate else 1, risk=round(risk, 6),
            k_probe=p["k_probe"], k_executed=int(src["n_visual_tokens"]),
            probe_visual_tokens=int(p["n_visual_tokens"])))
    return _finalize_dc(model_name=model_name, dataset=dataset, method=dc.METHOD_DCD,
                        basename=dc.dcd_basename(dataset, selector, probe_pct, target_pct,
                                                 use_ocr=use_ocr, instruction=instruction),
                        records=records, flops=flops, ids=ids, manifest=manifest,
                        probe_pct=probe_pct, use_ocr=use_ocr, instruction=instruction,
                        out_root=out_root,
                        extra={"variant": "dc_discrete", "selector_name": selector,
                               "risk_signal": risk_signal, "tau": tau, "target_pct": target_pct,
                               "escalation_rate_pct": round(100 * esc_n / len(ids), 2),
                               "stopped_at_probe_pct": round(100 * (1 - esc_n / len(ids)), 2),
                               "eval_split_only": eval_split_only,
                               "calibration_n": len(calib_ids)})


# ── DC-C (GPU second passes) ───────────────────────────────────────────────────

def run_dcc(*, model_name: str, dataset: str, selector: str, probe_pct: int, controller_cfg: dict,
            lam: float, second_pass_fn: Callable, controller_tag: str,
            use_ocr: bool = True, instruction: bool = True, eval_split_only: bool = True,
            smoke_n: Optional[int] = None, out_root: str = dp.OUT_ROOT, log_every: int = 50):
    """
    DC-C: per sample K_i = clamp(round(fraction(risk)·N_i), MIN_TOKENS, N_i).
    second_pass_fn(sample, k_i) -> pred_raw  (real GPU generation at exactly K_i).
    controller_cfg: {"controller": <rule|ridge dict>, "risk_signal": str, "rule": <rule dict>}
      - rule controller: risk = risk_from_signal(probe[risk_signal])
      - ridge: risk = ridge.predict_risk(features); mapped through the SAME rule dict.
    """
    from src.pruning.dynamic_count.controllers import load_controller
    probe_path = os.path.join(out_root, model_name, dataset,
                              dc.probe_basename(dataset, selector, probe_pct,
                                                use_ocr=use_ocr, instruction=instruction) + ".jsonl")
    probe = _load_jsonl(probe_path)
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"))
    all_ids = [str(i) for i in manifest.ids]
    calib_ids, eval_ids = dc.calibration_split(all_ids)
    ids = eval_ids if eval_split_only else all_ids
    if smoke_n is not None:                      # SMOKE: first N eval ids, distinct basename
        ids = ids[:int(smoke_n)]
    adapter = dp.get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)

    rule = load_controller(controller_cfg["rule"])
    ridge = load_controller(controller_cfg["controller"]) if controller_cfg["controller"]["type"] == "ridge" else None
    risk_signal = controller_cfg.get("risk_signal", "conf_first_max_prob")

    records, flops, esc_n, k_preds = [], [], 0, []
    for j, sid in enumerate(ids):
        p = probe[sid]
        if ridge is not None:
            x = [float(p.get(f, 0.0)) for f in ridge.feature_names]
            risk = ridge.predict_risk(x)
        else:
            risk = risk_from_signal(p[risk_signal], risk_signal)
        n_native = int(p["nvis_native"]) if model_name == "qwen25vl7b" else 576
        frac = rule.predict_fraction(risk, lam)
        k_i = dc.clamp_k(frac * n_native, dc.MIN_TOKENS, n_native)
        k_preds.append(k_i)
        escalate = k_i > int(p["k_probe"])
        f = tf.per_sample_prefill_flops(model_name, p["n_visual_tokens"], p["n_text_tokens"])
        if escalate:
            s = adapter.sample(sid)
            pred_raw = second_pass_fn(s, k_i)
            pred_norm, score = adapter.score(pred_raw, s)
            k_exec = k_i
            f += tf.per_sample_prefill_flops(model_name, k_i, p["n_text_tokens"])
            esc_n += 1
        else:
            pred_raw, pred_norm, score = p["pred_raw"], p["pred_norm"], float(p["per_sample_score"])
            k_exec = int(p["n_visual_tokens"])
        flops.append(f)
        records.append(per_sample_record(
            sample_id=p["sample_id"], image_id=p["image_id"], dataset=dataset, model=model_name,
            method=dc.METHOD_DCC, budget_pct=probe_pct, question=p["question"], prompt=p["prompt"],
            pred_raw=str(pred_raw).strip(), pred_norm=str(pred_norm), gold=p["gold"],
            per_sample_score=float(score), n_visual_tokens=int(k_exec),
            n_text_tokens=int(p["n_text_tokens"]), escalated=bool(escalate),
            n_passes=2 if escalate else 1, risk=round(float(risk), 6),
            k_probe=p["k_probe"], k_predicted=int(k_i), k_executed=int(k_exec),
            probe_visual_tokens=int(p["n_visual_tokens"])))
        if (j + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name} dcc {controller_tag} lam={lam}] {j+1}/{len(ids)} "
                  f"score={avg*100:.2f}% esc={esc_n}", flush=True)

    basename = dc.dcc_basename(dataset, selector, probe_pct, controller_tag, lam,
                               use_ocr=use_ocr, instruction=instruction)
    if smoke_n is not None:
        basename += f"_smoke_n{int(smoke_n)}"   # smoke outputs can never collide with real runs
    return _finalize_dc(model_name=model_name, dataset=dataset, method=dc.METHOD_DCC,
                        basename=basename,
                        records=records, flops=flops, ids=ids, manifest=manifest,
                        probe_pct=probe_pct, use_ocr=use_ocr, instruction=instruction,
                        out_root=out_root,
                        extra={"variant": "dc_continuous", "selector_name": selector,
                               "controller": controller_tag, "lam": lam,
                               "risk_signal": risk_signal, "smoke_n": smoke_n,
                               "k_predicted_avg": round(sum(k_preds) / max(len(k_preds), 1), 2),
                               "k_predicted_min": min(k_preds), "k_predicted_max": max(k_preds),
                               "k_predicted_unique": len(set(k_preds)),
                               "k_predicted_unique_pct": round(100 * len(set(k_preds)) / max(len(k_preds), 1), 2),
                               "k_histogram_pct": _k_range_hist(k_preds),
                               "escalation_rate_pct": round(100 * esc_n / len(ids), 2),
                               "stopped_at_probe_pct": round(100 * (1 - esc_n / len(ids)), 2),
                               "avg_passes": round(1 + esc_n / len(ids), 4),
                               "eval_split_only": eval_split_only,
                               "calibration_n": len(calib_ids)})


# ── shared finalization: curve comparison + gate + write ──────────────────────

def _finalize_dc(*, model_name, dataset, method, basename, records, flops, ids, manifest,
                 probe_pct, use_ocr, instruction, out_root, extra):
    split, metric = SPLIT_METRIC[dataset]
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    flops_avg_T = sum(flops) / max(len(flops), 1) / 1e12
    curve = static_curve_for_ids(model_name, dataset, ids, use_ocr=use_ocr, instruction=instruction)
    static_at = interp_curve(curve, flops_avg_T)
    delta = round(score_pct - static_at, 2)
    dense_pt = max(curve)   # dense = highest-flops point
    vis = [r["n_visual_tokens"] for r in records]
    txt = [r["n_text_tokens"] for r in records]
    # HONEST token accounting: total visual tokens EXECUTED per sample (probe + second pass when
    # escalated) vs the dense visual average over the SAME ids — distinct from the FLOPs ratio.
    total_vis = [r["probe_visual_tokens"] + (r["k_executed"] if r.get("escalated") else 0)
                 for r in records]
    dense_rows = dense_reference_rows(model_name, dataset, use_ocr=use_ocr, instruction=instruction)
    dense_vis_avg = sum(dense_rows[str(i)]["n_visual_tokens"] for i in ids) / len(ids)
    total_vis_avg = sum(total_vis) / len(total_vis)
    extra = {**extra,
             "total_visual_tokens_executed_avg": round(total_vis_avg, 2),
             "dense_visual_tokens_avg": round(dense_vis_avg, 2),
             "answer_pass_visual_tokens_avg": round(sum(vis) / len(vis), 2)}
    agg = aggregate_record(
        model=model_name, dataset=dataset, split=split, method=method, budget_pct=probe_pct,
        n=len(records), metric=metric, score_pct=score_pct,
        visual_tokens={"avg": sum(vis) / len(vis), "max": float(max(vis))},
        text_tokens_avg=sum(txt) / len(txt),
        seq_len_avg=sum(v + t for v, t in zip(vis, txt)) / len(vis),
        flops_prefill_TFLOPs_avg=flops_avg_T,
        flops_convention=tf.FLOPS_CONVENTION + "; MULTI-PASS: per-sample sum of ALL executed prefills",
        token_reduction_pct=round(100 * (1 - total_vis_avg / dense_vis_avg), 2),
        flop_reduction_pct=round(100 * (1 - flops_avg_T / dense_pt[0]), 2),
        prompt_template=dp.describe_prompt(dataset, use_ocr=use_ocr, instruction=instruction),
        max_new_tokens=dp.MAX_NEW_TOKENS, decoding="greedy bs=1 do_sample=False max_new_tokens=64",
        image_pad=(True if model_name == "llava15" else None),
        sample_ids_path=os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"),
        sample_ids_sha256=manifest.sha256,
        protocol_instruction=dp.INSTRUCTION, model_instruction_suffix="", decode_notes="",
        static_curve_points=[[round(x, 4), round(y, 2)] for x, y in curve],
        static_at_matched_flops=round(static_at, 2),
        delta_vs_static_curve_pp=delta, interpretation_label=label_delta(delta),
        dense_score_pct=round(dense_pt[1], 2),
        delta_vs_dense_pp=round(score_pct - dense_pt[1], 2),
        extra_dynamic_count=extra)
    gate = fairness_gate(records, agg, manifest_ids=[str(i) for i in ids],
                         manifest_sha=manifest.sha256, expected_visual_tokens=None)
    out_dir = os.path.join(out_root, model_name, dataset)
    jl, js = os.path.join(out_dir, f"{basename}.jsonl"), os.path.join(out_dir, f"{basename}.json")
    if os.path.exists(jl) or os.path.exists(js):
        raise FileExistsError(f"refusing to overwrite existing output: {basename}.*")
    write_per_sample_jsonl(jl, records, validate=True)
    write_aggregate_json(js, {**agg, "fairness_gate": gate}, validate=True)
    print(f"[{dataset}/{model_name} {method}] score={score_pct}% @ {flops_avg_T:.3f}T "
          f"(static@same={static_at:.2f}) Δ={delta:+.2f}pp [{label_delta(delta)}] "
          f"gate={'OK' if gate['ok'] else 'ISSUES'}", flush=True)
    return js, jl, score_pct, delta, gate["ok"]
