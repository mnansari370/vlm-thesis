"""
make_dynamic_count_controller_calibration.py — CPU calibration: fit DC-D thresholds and DC-C
controllers on the FIRST 20% of the manifest, per cell. No GPU; no thresholds touched by eval data.

Per cell (model × dataset), for each available probe (selector × p15/p25):
  1. DC-D: sweep escalation rates × target anchors {50, 75} × confidence signals on the CALIBRATION
     split; pick the config with the best Δ vs the static curve at matched FLOPs.
  2. DC-C rule controller: risks (chosen signal) + per-sample anchor outcomes {0.15..1.0} →
     RuleController (risk-bin sufficiency fractions, anchor-interpolated, isotonic).
  3. DC-C ridge controller: tiny ridge on ALL_SIGNALS → risk; shares the rule mapping.
  4. Signal quality report: wrong-capture at 15% false-escalation per signal (the pre-registered
     gate quantity).

Writes per-cell config JSON to results/configs/dynamic_count/{model}_{dataset}.json
and a calibration report to results/tables/dynamic_count_calibration_report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dense import evaluate_dense as dp
from src.static import evaluate_static as se
from src.common import token_flops as tf
from src.dynamic_count import evaluate_dynamic_count as dce
from src.common.sample_ids import load_manifest
from src.pruning import dynamic_count as dc
from src.pruning.dynamic_count.confidence import risk_from_signal
from src.pruning.dynamic_count.controllers import (RuleController, RidgeRiskModel,
                                                   threshold_for_rate)

CFG_DIR = os.path.join("results", "configs", "dynamic_count")
REPORT = os.path.join("results", "tables", "dynamic_count_calibration_report.md")
VAR = {"gqa": (True, True), "vqav2": (True, True), "textvqa": (True, True), "docvqa": (True, True)}
ESC_RATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
TARGETS = (50, 75)
LAM_GRID = [0.6, 0.8, 1.0, 1.2, 1.5]


def anchor_rows(model, dataset, selector, use_ocr, instruction):
    """{fraction: rows} for the 5 anchors + dense, SELECTOR-AWARE: textsim probes calibrate against
    the frozen WHICH textsim finals (COUNT-on-WHICH self-contained); static probes against the
    locked static selector's finals. Never mixed."""
    out = {}
    for b, fr in zip(dc.ANCHOR_BUDGETS, dc.ANCHOR_FRACTIONS[:-1]):
        out[fr] = dce._load_jsonl(dce.anchor_reference_path(
            model, dataset, selector, b, use_ocr=use_ocr, instruction=instruction))
    dense_p = os.path.join(dp.OUT_ROOT, model, dataset,
                           dp.default_basename(dataset, 0, use_ocr=use_ocr,
                                               instruction=instruction, full=True) + ".jsonl")
    out[1.0] = dce._load_jsonl(dense_p)
    return out


def capture_at_false_rate(risks_wrong, risks_right, false_rate=0.15):
    """Wrong-capture when the threshold is set to falsely escalate `false_rate` of rights."""
    if not risks_right or not risks_wrong:
        return 0.0
    tau = threshold_for_rate(risks_right, false_rate)
    return sum(1 for r in risks_wrong if r > tau) / len(risks_wrong)


def calibrate_cell(model, dataset, selector, use_ocr=True, instruction=True):
    """Calibrate ONE (cell, probe-selector). The pure Dynamic-COUNT config uses the locked static
    selector; textsim (COUNT-on-WHICH, qwen×textvqa) is calibrated SEPARATELY with textsim anchors."""
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"))
    all_ids = [str(i) for i in manifest.ids]
    calib_ids, _ = dc.calibration_split(all_ids)
    anchors = anchor_rows(model, dataset, selector, use_ocr, instruction)
    curve = dce.static_curve_for_ids(model, dataset, calib_ids, use_ocr=use_ocr, instruction=instruction)

    best = None
    report_rows = []
    for _sel in (selector,):
        for pct in dc.PROBE_BUDGETS:
            ppath = os.path.join(dp.OUT_ROOT, model, dataset,
                                 dc.probe_basename(dataset, selector, pct, use_ocr=use_ocr,
                                                   instruction=instruction) + ".jsonl")
            if not os.path.exists(ppath):
                continue
            probe = dce._load_jsonl(ppath)
            cal = [probe[i] for i in calib_ids if i in probe]
            if len(cal) < 50:
                continue
            # signal quality on calibration
            for sig in dc.CONFIDENCE_SIGNALS:
                rw = [risk_from_signal(r[sig], sig) for r in cal if float(r["per_sample_score"]) < 0.5]
                rr = [risk_from_signal(r[sig], sig) for r in cal if float(r["per_sample_score"]) >= 0.5]
                cap = capture_at_false_rate(rw, rr)
                report_rows.append((selector, pct, sig, round(100 * cap, 1)))
            # DC-D sweep on calibration
            for sig in dc.CONFIDENCE_SIGNALS:
                risks = {r["sample_id"]: risk_from_signal(r[sig], sig) for r in cal}
                rvals = list(risks.values())
                for rate in ESC_RATES:
                    tau = threshold_for_rate(rvals, rate)
                    for tgt in TARGETS:
                        tfrac = tgt / 100.0
                        ts = fl = 0.0
                        for r in cal:
                            sid = str(r["sample_id"])
                            f = tf.per_sample_prefill_flops(model, r["n_visual_tokens"], r["n_text_tokens"])
                            if risks[r["sample_id"]] > tau:
                                a = anchors[tfrac][sid]
                                ts += float(a["per_sample_score"])
                                f += tf.per_sample_prefill_flops(model, a["n_visual_tokens"], a["n_text_tokens"])
                            else:
                                ts += float(r["per_sample_score"])
                            fl += f
                        acc = 100 * ts / len(cal)
                        favg = fl / len(cal) / 1e12
                        delta = acc - dce.interp_curve(curve, favg)
                        cand = dict(selector=selector, probe_pct=pct, risk_signal=sig,
                                    tau=tau, esc_rate=rate, target_pct=tgt,
                                    calib_acc=round(acc, 2), calib_flops_T=round(favg, 4),
                                    calib_delta_pp=round(delta, 2))
                        if best is None or cand["calib_delta_pp"] > best["calib_delta_pp"]:
                            best = cand
    if best is None:
        return None, report_rows

    # fit DC-C controllers using the chosen probe/signal
    ppath = os.path.join(dp.OUT_ROOT, model, dataset,
                         dc.probe_basename(dataset, best["selector"], best["probe_pct"],
                                           use_ocr=use_ocr, instruction=instruction) + ".jsonl")
    probe = dce._load_jsonl(ppath)
    cal = [probe[i] for i in calib_ids if i in probe]
    risks = [risk_from_signal(r[best["risk_signal"]], best["risk_signal"]) for r in cal]
    anchor_scores = [{fr: float(anchors[fr][str(r["sample_id"])]["per_sample_score"])
                      for fr in dc.ANCHOR_FRACTIONS} for r in cal]
    rule = RuleController().fit(risks, anchor_scores)
    X = [[float(r.get(f, 0.0)) for f in dc.ALL_SIGNALS] for r in cal]
    y = [1.0 if float(r["per_sample_score"]) < 0.5 else 0.0 for r in cal]
    ridge = RidgeRiskModel(dc.ALL_SIGNALS).fit(X, y)
    # a rule mapping in ridge-risk space too (ridge risk has its own scale)
    ridge_risks = [ridge.predict_risk(x) for x in X]
    rule_for_ridge = RuleController().fit(ridge_risks, anchor_scores)

    def lam_default(rc, sample_risks):
        """λ chosen ON CALIBRATION ONLY: predicted DC-C average FLOPs (K_i known → cost known,
        accuracy not needed) closest to the DC-D calibration operating cost — compute-matched,
        never tuned on evaluation data."""
        target_T = best["calib_flops_T"]
        best_lam, best_gap = LAM_GRID[0], float("inf")
        for lam in LAM_GRID:
            tot = 0.0
            for r, rk in zip(cal, sample_risks):
                n_native = int(r["nvis_native"]) if model == "qwen25vl7b" else 576
                k_i = dc.clamp_k(rc.predict_fraction(rk, lam) * n_native, dc.MIN_TOKENS, n_native)
                f = tf.per_sample_prefill_flops(model, r["n_visual_tokens"], r["n_text_tokens"])
                if k_i > int(r["k_probe"]):
                    f += tf.per_sample_prefill_flops(model, k_i, r["n_text_tokens"])
                tot += f
            gap = abs(tot / len(cal) / 1e12 - target_T)
            if gap < best_gap:
                best_lam, best_gap = lam, gap
        return best_lam

    cfg = {"model": model, "dataset": dataset, "selector": selector,
           "count_on_which": selector == "textsim", "dcd": best,
           "dcc_rule": {"risk_signal": best["risk_signal"], "controller": {"type": "rule"},
                        "rule": rule.to_dict(), "lam_grid": LAM_GRID,
                        "lam_default": lam_default(rule, risks)},
           "dcc_ridge": {"risk_signal": best["risk_signal"], "controller": ridge.to_dict(),
                         "rule": rule_for_ridge.to_dict(), "lam_grid": LAM_GRID,
                         "lam_default": lam_default(rule_for_ridge, ridge_risks)},
           "calibration_n": len(cal), "calibration_frac": dc.CALIB_FRAC}
    return cfg, report_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit Dynamic-COUNT controllers on the calibration split (CPU).")
    ap.add_argument("--model", default=None, choices=list(dp.MODELS))
    ap.add_argument("--dataset", default=None, choices=list(dp.DATASETS))
    args = ap.parse_args()
    models = [args.model] if args.model else list(dp.MODELS)
    datasets = [args.dataset] if args.dataset else list(dp.DATASETS)
    os.makedirs(CFG_DIR, exist_ok=True)
    lines = ["# Dynamic-COUNT calibration report (first-20% split; CPU-only)", "",
             "Pure Dynamic-COUNT uses the locked static selector; COUNT-on-WHICH (textsim) is "
             "calibrated and reported SEPARATELY (own config, textsim anchors).", ""]
    n_ok = 0
    for m in models:
        for d in datasets:
            # one calibration per probe selector; textsim → its own config file
            for selector in dc.PROBE_SELECTORS[m]:
                if selector == "textsim" and d != "textvqa":
                    continue
                cow = selector == "textsim"
                tag = f"{m}_{d}" + ("_textsim" if cow else "")
                title = f"{m} × {d}" + (" [COUNT-on-WHICH]" if cow else "")
                cfg, rows = calibrate_cell(m, d, selector)
                if cfg is None:
                    lines += [f"## {title}", "", "probe outputs not found yet — run the probe launcher first.", ""]
                    print(f"[skip] {tag}: no probe outputs")
                    continue
                path = os.path.join(CFG_DIR, f"{tag}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                n_ok += 1
                b = cfg["dcd"]
                lines += [f"## {title}", "",
                          f"- chosen probe: **{b['selector']} p{b['probe_pct']}**, signal "
                          f"**{b['risk_signal']}**, esc-rate {b['esc_rate']}, target p{b['target_pct']} "
                          f"→ calib Δcurve **{b['calib_delta_pp']:+.2f}pp**",
                          f"- DC-C λ_default (compute-matched on calibration): "
                          f"rule={cfg['dcc_rule']['lam_default']}, ridge={cfg['dcc_ridge']['lam_default']}",
                          f"- signal capture@15%false-esc (calibration): "
                          + ", ".join(f"{s}={c}%" for sel, pc, s, c in rows
                                      if sel == b["selector"] and pc == b["probe_pct"]),
                          f"- config: `{path}`", ""]
                print(f"[ok] {tag}: probe={b['selector']} p{b['probe_pct']} sig={b['risk_signal']} "
                      f"calibΔ={b['calib_delta_pp']:+.2f}pp lam={cfg['dcc_rule']['lam_default']} -> {path}")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {REPORT}\nCELLS_CALIBRATED={n_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
