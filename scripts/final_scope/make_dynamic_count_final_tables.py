"""
make_dynamic_count_final_tables.py — build all Dynamic-COUNT report tables (CPU, read-only over
result files; writes only under results/final_scope/tables/).

Produces (gracefully partial when stages are incomplete):
  dynamic_count_probe_summary.md                       — probe scores + reproduction gate + signals
  dynamic_count_dc_d_summary.md                        — DC-D vs static curve, per cell
  dynamic_count_dc_c_summary.md                        — DC-C (main) vs static curve + K_i stats
  dynamic_count_dense_static_which_count_comparison.md — the five-way thesis table
  dynamic_count_win_loss_summary.md                    — labels across DC-D/DC-C
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import static_eval as se

TABLES = os.path.join("results", "final_scope", "tables")


def _agg(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def _cell_aggs(pattern):
    out = {}
    for m in dp.MODELS:
        for d in dp.DATASETS:
            hits = sorted(glob.glob(os.path.join(dp.OUT_ROOT, m, d, pattern)))
            hits = [h for h in hits if not h.endswith(".jsonl")]
            if hits:
                out[(m, d)] = [_agg(h) for h in hits if _agg(h)]
    return out


def _w(name, lines):
    path = os.path.join(TABLES, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[written] {path}")


def probe_summary():
    lines = ["# Dynamic-COUNT probe summary", "",
             "| Model | Dataset | Selector | p | n | Score | Frozen match | Equiv |",
             "|---|---|---|--:|--:|--:|--:|:--:|"]
    for (m, d), aggs in _cell_aggs("dynamic_count_probe_*.json").items():
        for a in aggs:
            lines.append(f"| {m} | {d} | {a.get('selector_name')} | {a.get('budget_pct')} | "
                         f"{a.get('n')} | {a.get('score_pct')} | "
                         f"{a.get('frozen_match_count')}/{a.get('n')} | "
                         f"{'OK' if a.get('equivalence_ok') else '**FAIL**'} |")
    if len(lines) == 4:
        lines.append("| — | — | — | — | — | — | — | probes not run yet |")
    _w("dynamic_count_probe_summary.md", lines)


def _dc_table(pattern, title, extra_cols=()):
    lines = [f"# {title}", "",
             "PURE Dynamic-COUNT rows use the locked static selector; rows with selector `textsim` "
             "are COUNT-on-WHICH (qwen×textvqa) and are reported separately — never mixed.", "",
             "| Model | Dataset | Sel | Variant tag | n | Score | avgTFLOPs | static@same | Δcurve | "
             "Label | Esc% |" + "".join(f" {c} |" for c in extra_cols),
             "|---|---|---|---|--:|--:|--:|--:|--:|---|--:|" + "---|" * len(extra_cols)]
    rows = []
    for (m, d), aggs in _cell_aggs(pattern).items():
        for a in aggs:
            ex = a.get("extra_dynamic_count", {})
            row = (f"| {m} | {d} | {ex.get('selector_name', '')} | "
                   f"{ex.get('controller', ex.get('variant', ''))}"
                   f"{(' λ=' + str(ex.get('lam'))) if ex.get('lam') is not None else ''} | "
                   f"{a.get('n')} | {a.get('score_pct')} | "
                   f"{round(a.get('flops_prefill_TFLOPs_avg', 0), 3)} | "
                   f"{a.get('static_at_matched_flops')} | "
                   f"{a.get('delta_vs_static_curve_pp'):+.2f} | {a.get('interpretation_label')} | "
                   f"{ex.get('escalation_rate_pct', '')} |")
            for c in extra_cols:
                row += f" {ex.get(c, '')} |"
            lines.append(row)
            rows.append((m, d, a))
    if not rows:
        lines.append("| — | — | — | not composed/run yet | | | | | | | |" + " |" * len(extra_cols))
    return lines, rows


def comparison(dcd_rows, dcc_rows):
    # PURE rows only — COUNT-on-WHICH lives in its own tables, never in the headline comparison
    def _pure(rows):
        return {(m, d): a for m, d, a in rows
                if a.get("extra_dynamic_count", {}).get("selector_name") != "textsim"}
    dcd = _pure(dcd_rows)
    dcc = _pure(dcc_rows)
    lines = ["# Dense / Static / WHICH / DC-D / DC-C — thesis comparison", "",
             "Static reference = frozen anchors (interpolated curve at each method's average "
             "FLOPs, same evaluated ids). WHICH = frozen textsim finals (full manifest).", "",
             "| Model | Dataset | Dense | Static p25/p75 | WHICH best (p) | DC-D Δcurve | "
             "DC-C Δcurve | DC-C label |",
             "|---|---|--:|--:|--:|--:|--:|---|"]
    for m in dp.MODELS:
        for d in dp.DATASETS:
            dense_a = _agg(os.path.join(dp.OUT_ROOT, m, d,
                           dp.default_basename(d, 0, full=True) + ".json"))
            sel = se.default_selector(m)
            s25 = _agg(os.path.join(dp.OUT_ROOT, m, d,
                       se.static_basename(d, 0, sel, 25, full=True) + ".json"))
            s75 = _agg(os.path.join(dp.OUT_ROOT, m, d,
                       se.static_basename(d, 0, sel, 75, full=True) + ".json"))
            which = [_agg(p) for p in glob.glob(os.path.join(
                dp.OUT_ROOT, m, d, "dynamic_which_final_textsim_p*.json"))
                if not p.endswith(".jsonl")]
            which = [w for w in which if w]
            wbest = max(which, key=lambda w: w.get("dynamic_minus_static_pp", -99)) if which else None
            a_dcd, a_dcc = dcd.get((m, d)), dcc.get((m, d))
            lines.append(
                f"| {m} | {d} | {dense_a.get('score_pct') if dense_a else '—'} | "
                f"{s25.get('score_pct') if s25 else '—'}/{s75.get('score_pct') if s75 else '—'} | "
                f"{(str(wbest.get('score_pct')) + ' (p' + str(wbest.get('budget_pct')) + ')') if wbest else '—'} | "
                f"{format(a_dcd.get('delta_vs_static_curve_pp'), '+.2f') if a_dcd else '—'} | "
                f"{format(a_dcc.get('delta_vs_static_curve_pp'), '+.2f') if a_dcc else '—'} | "
                f"{a_dcc.get('interpretation_label') if a_dcc else '—'} |")
    return lines


def win_loss(dcd_rows, dcc_rows):
    lines = ["# Dynamic-COUNT win/loss summary (labels at ±0.50pp vs static curve)", "",
             "PURE Dynamic-COUNT (static selector) is the headline; COUNT-on-WHICH (textsim) is "
             "listed separately so WHICH-vs-COUNT attribution stays clean.", ""]

    def _is_cow(a):
        return a.get("extra_dynamic_count", {}).get("selector_name") == "textsim"

    for tag, rows in (("DC-D (baseline)", dcd_rows), ("DC-C (MAIN)", dcc_rows)):
        for scope, sel_rows in (("pure", [r for r in rows if not _is_cow(r[2])]),
                                ("COUNT-on-WHICH", [r for r in rows if _is_cow(r[2])])):
            counts = {}
            for _, _, a in sel_rows:
                lab = a.get("interpretation_label", "?")
                counts[lab] = counts.get(lab, 0) + 1
            lines.append(f"## {tag} — {scope}: "
                         + (", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                            if sel_rows else "no outputs yet"))
            for m, d, a in sorted(sel_rows, key=lambda r: -(r[2].get("delta_vs_static_curve_pp") or 0)):
                lines.append(f"- {m} × {d}: Δ={a.get('delta_vs_static_curve_pp'):+.2f}pp "
                             f"[{a.get('interpretation_label')}] score={a.get('score_pct')}")
            lines.append("")
    return lines


def main() -> int:
    os.makedirs(TABLES, exist_ok=True)
    probe_summary()
    dcd_lines, dcd_rows = _dc_table("dynamic_count_dcd_*.json", "DC-D (discrete cascade baseline)")
    _w("dynamic_count_dc_d_summary.md", dcd_lines)
    dcc_lines, dcc_rows = _dc_table("dynamic_count_dcc_*.json",
                                    "DC-C (continuous Dynamic-COUNT — MAIN method)",
                                    extra_cols=("k_predicted_avg", "stopped_at_probe_pct",
                                                "avg_passes", "k_histogram_pct"))
    _w("dynamic_count_dc_c_summary.md", dcc_lines)
    _w("dynamic_count_dense_static_which_count_comparison.md", comparison(dcd_rows, dcc_rows))
    _w("dynamic_count_win_loss_summary.md", win_loss(dcd_rows, dcc_rows))
    print(f"\nDC-D outputs: {len(dcd_rows)}   DC-C outputs: {len(dcc_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
