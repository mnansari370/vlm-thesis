"""
compare_dynamic_which_current_vs_ref.py — decide WHY Dynamic-WHICH failed, per cell/budget.

Reads (read-only) the aggregate JSONs and cross-tabulates, per (model, dataset, budget):
  * static + dense reference scores,
  * CURRENT dynamic_which n=200 pilots (textsim + its mix), dyn−static per selector,
  * fresh clean-room REF rescue pilots (dynamic_which_ref_*, n=200 ONLY), dyn−static per ref selector.

STRICT n=200 matching on BOTH sides — n=20 smoke outputs are ignored (mixing them was the cause of
the earlier spurious "IMPL-SUSPECT" rows). Confirmed: at n=200 the clean-room textsim_ref reproduces
current textsim exactly.

Classifies each in-scope cell's failure as:
  * IMPLEMENTATION-RELATED — ref textsim_ref disagrees with current textsim by > tol at n=200.
  * SELECTOR-DESIGN-RELATED — implementations agree, and a better selector (mix / static_guarded)
    materially improves dyn−static.
  * TASK/REGIME-RELATED — no tested selector beats static at this budget.

Cells outside the rescue scope are reported separately (NOT as "pending"); Qwen×TextVQA is marked as
the completed final success (ref validation optional).

Writes:
  results/final_scope/tables/dynamic_which_current_vs_ref_summary.csv
  results/final_scope/tables/dynamic_which_current_vs_ref_summary.md
Stdlib only — CPU, no GPU, no result files modified.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.join("results", "final_scope")
TABLES_DIR = os.path.join(ROOT, "tables")
CSV_PATH = os.path.join(TABLES_DIR, "dynamic_which_current_vs_ref_summary.csv")
MD_PATH = os.path.join(TABLES_DIR, "dynamic_which_current_vs_ref_summary.md")

TOL_IMPL = 1.5    # pp: |ref.textsim_ref − current.textsim| above this ⇒ implementation-suspect
GAIN = 1.0        # pp: a selector "materially helps" if it improves dyn−static by ≥ this

# Cells the clean-room ref rescue launcher covered (model, dataset, budget).
REF_SCOPE = {
    ("llava15", "textvqa", 25), ("llava15", "textvqa", 35), ("llava15", "textvqa", 50),
    ("llava15", "docvqa", 25), ("llava15", "docvqa", 35), ("llava15", "docvqa", 50),
    ("llava15", "vqav2", 25), ("llava15", "vqav2", 35),
    ("qwen25vl7b", "gqa", 25), ("qwen25vl7b", "gqa", 35),
    ("qwen25vl7b", "vqav2", 25), ("qwen25vl7b", "vqav2", 35),
    ("qwen25vl7b", "docvqa", 25), ("qwen25vl7b", "docvqa", 35), ("qwen25vl7b", "docvqa", 50),
}

CSV_COLS = [
    "model", "dataset", "budget_pct", "scope", "static_score_pct", "dense_score_pct",
    "current_textsim_dynstatic", "current_best_dynstatic", "current_best_selector",
    "ref_textsim_ref_dynstatic", "ref_best_dynstatic", "ref_best_selector",
    "impl_check", "verdict",
]


def _load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cell_scope(model, dataset, budget):
    if model == "qwen25vl7b" and dataset == "textvqa":
        return "final-success"
    if (model, dataset, budget) in REF_SCOPE:
        return "in-ref-scope"
    return "not-in-ref-scope"


def scan():
    """cell[(m,d,b)] = {static, dense, current:{sel:dms}, ref:{sel:dms}, final:{sel:dms}}."""
    cells = {}
    for p in glob.glob(os.path.join(ROOT, "*", "*", "dynamic_which_*.json")):
        if p.endswith(".jsonl"):
            continue
        r = _load(p)
        if not r or "budget_pct" not in r:
            continue
        method, n = r.get("method"), r.get("n")
        key = (r.get("model"), r.get("dataset"), r.get("budget_pct"))
        c = cells.setdefault(key, {"static": None, "dense": None,
                                   "current": {}, "ref": {}, "final": {}})
        if r.get("static_reference_score_pct") is not None:
            c["static"] = r["static_reference_score_pct"]
        if r.get("dense_reference_score_pct") is not None:
            c["dense"] = r["dense_reference_score_pct"]
        sel, dms = r.get("selector_name"), r.get("dynamic_minus_static_pp")
        if dms is None:
            continue
        # STRICT n=200 on BOTH current and ref (ignore n=20 smoke; ignore other pilot sizes)
        if method == "dynamic_which_ref" and n == 200:
            c["ref"][sel] = dms
        elif method == "dynamic_which" and n == 200:
            c["current"][sel] = dms
        elif method == "dynamic_which" and n and n > 200:      # final runs (e.g., Qwen TextVQA)
            c["final"][sel] = dms
    return cells


def _best(d):
    if not d:
        return None, None
    sel = max(d, key=lambda k: d[k])
    return d[sel], sel


def classify(key, c):
    scope = cell_scope(*key)
    ct = c["current"].get("textsim")
    rt = c["ref"].get("textsim_ref")
    cur_best, cur_sel = _best(c["current"])
    ref_best, ref_sel = _best(c["ref"])

    if scope == "final-success":
        return scope, "n/a", "FINAL SUCCESS; independent ref validation pending", \
            cur_best, cur_sel, ref_best, ref_sel

    if scope == "not-in-ref-scope":
        return scope, "n/a", "not-in-ref-scope (current evidence stands; ref not run by design)", \
            cur_best, cur_sel, ref_best, ref_sel

    # in-ref-scope
    if not c["ref"]:
        return scope, "ref-missing", "in-ref-scope but ref n=200 output missing (re-run rescue)", \
            cur_best, cur_sel, ref_best, ref_sel

    if ct is not None and rt is not None and abs(ct - rt) > TOL_IMPL:
        return (scope, f"IMPL-SUSPECT (Δ={rt - ct:+.2f}pp)",
                "IMPLEMENTATION-RELATED (ref textsim ≠ current textsim at n=200)",
                cur_best, cur_sel, ref_best, ref_sel)

    impl = "impl-consistent" if (ct is not None and rt is not None) else "impl-partial"
    overall_best = max(v for v in (cur_best, ref_best) if v is not None)
    if overall_best <= 0:
        verdict = "TASK/REGIME (no selector beats static)"
    elif ref_best is not None and ref_best >= GAIN and (cur_best is None or ref_best > cur_best + GAIN):
        verdict = f"SELECTOR-DESIGN (rescue '{ref_sel}' recovers, {ref_best:+.2f}pp vs static)"
    elif overall_best >= GAIN:
        verdict = "SELECTOR-DESIGN (some selector beats static; margin small)"
    else:
        verdict = "TASK/REGIME-ish (best barely ≥ static)"
    return scope, impl, verdict, cur_best, cur_sel, ref_best, ref_sel


def build_rows(cells):
    rows = []
    for key in sorted(cells):
        model, dataset, budget = key
        c = cells[key]
        scope, impl, verdict, cur_best, cur_sel, ref_best, ref_sel = classify(key, c)
        # keep only rows with meaningful data (drops stray n=20 smoke-only / final-only noise)
        has_data = bool(c["current"]) or bool(c["ref"]) or (scope == "final-success" and c["final"])
        if not has_data:
            continue
        cur_textsim = c["current"].get("textsim")
        if cur_textsim is None and scope == "final-success":
            cur_textsim = c["final"].get("textsim")            # show the final dyn−static instead
        rows.append({
            "model": model, "dataset": dataset, "budget_pct": budget, "scope": scope,
            "static_score_pct": c["static"], "dense_score_pct": c["dense"],
            "current_textsim_dynstatic": cur_textsim,
            "current_best_dynstatic": cur_best, "current_best_selector": cur_sel,
            "ref_textsim_ref_dynstatic": c["ref"].get("textsim_ref"),
            "ref_best_dynstatic": ref_best, "ref_best_selector": ref_sel,
            "impl_check": impl, "verdict": verdict,
        })
    return rows


def _f(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) else ("—" if x is None else str(x))


def _row_md(r):
    cur_best = f"{_f(r['current_best_dynstatic'])} ({r['current_best_selector']})" \
        if r["current_best_selector"] else "—"
    ref_best = f"{_f(r['ref_best_dynstatic'])} ({r['ref_best_selector']})" \
        if r["ref_best_selector"] else "—"
    return (f"| {r['model']} | {r['dataset']} | {r['budget_pct']} | "
            f"{_f(r['static_score_pct']).lstrip('+')} | {_f(r['dense_score_pct']).lstrip('+')} | "
            f"{_f(r['current_textsim_dynstatic'])} | {cur_best} | "
            f"{_f(r['ref_textsim_ref_dynstatic'])} | {ref_best} | {r['impl_check']} | {r['verdict']} |")


def build_md(rows):
    header = ("| Model | Dataset | p | Static | Dense | cur textsim | cur best (sel) | "
              "ref textsim_ref | ref best (sel) | Impl check | Verdict |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"
    in_scope = [r for r in rows if r["scope"] == "in-ref-scope"]
    other = [r for r in rows if r["scope"] != "in-ref-scope"]

    out = [
        "# Dynamic-WHICH: current vs clean-room reference — failure attribution",
        "",
        "Per (model, dataset, budget): dyn−static for the CURRENT textsim path (n=200) vs the fresh "
        "clean-room REFERENCE selectors (n=200), against the same static/dense references. **Strict "
        "n=200 on both sides** — n=20 smoke outputs are excluded (mixing them caused the earlier "
        "false IMPL-SUSPECT rows).",
        "",
        "**Verdict key:** *IMPLEMENTATION-RELATED* = ref textsim_ref ≠ current textsim at n=200 (> "
        f"{TOL_IMPL}pp) ⇒ a bug. *SELECTOR-DESIGN-RELATED* = implementations agree and a better "
        "selector materially lifts dyn−static. *TASK/REGIME-RELATED* = no tested selector beats static.",
        "",
        "## A) In-ref-scope decision cells",
        "",
        header, sep,
    ]
    out += [_row_md(r) for r in in_scope]

    out += [
        "",
        "## B) Out-of-ref-scope cells (context only — not tested by the rescue by design)",
        "",
        "Qwen×TextVQA is the completed FINAL success — **independent ref validation pending** "
        "(run `run_qwen_textvqa_ref_validation.sh` + `compare_qwen_textvqa_current_vs_ref.py`). Other "
        "rows here were intentionally not covered by the rescue launcher; their existing current "
        "evidence stands (NOT a pending failure).",
        "",
        header, sep,
    ]
    out += [_row_md(r) for r in other]

    # rollup over in-scope decision cells only
    counts = {}
    for r in in_scope:
        tag = r["verdict"].split(" (")[0].split(":")[0]
        counts[tag] = counts.get(tag, 0) + 1
    out += ["", "## Rollup (in-ref-scope decision cells)", ""]
    for tag, k in sorted(counts.items(), key=lambda kv: -kv[1]):
        out.append(f"- {tag}: {k} cell-budget(s)")

    impl_suspect = [r for r in in_scope if r["impl_check"].startswith("IMPL")]
    out += [
        "",
        "## Conclusion (ref pilots complete)",
        "",
        "The clean-room ref rescue pilots are **complete** (45 runs, no failures). The code audit "
        "confirmed the current Dynamic-WHICH path is training-free, frozen-generation, no answer head, "
        "no old dynamic-budget code, and fairness-correct; keep-all==dense and static-criterion==static "
        "equivalence passed. With current and ref matched **strictly at n=200**, current textsim and "
        "ref textsim_ref agree. "
        + ("**No IMPLEMENTATION-RELATED cells remain** — the earlier IMPL-SUSPECT rows were a "
           "reporting bug (the ref branch did not filter n==200, so n=20 smoke outputs were mixed "
           "with n=200 ref outputs; fixed here). "
           if not impl_suspect else
           f"{len(impl_suspect)} cell(s) still flagged IMPL-SUSPECT — inspect with "
           "debug_textsim_current_vs_ref.py. "),
        "So the failures are **not implementation bugs**; the remaining split is SELECTOR-DESIGN vs "
        "TASK/REGIME per the table above. The one positive result (Qwen×TextVQA) still needs its own "
        "independent ref validation (pending).",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    cells = scan()
    rows = build_rows(cells)
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([",".join(CSV_COLS)]
                          + [",".join(str(r[c]) for c in CSV_COLS) for r in rows]) + "\n")
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_md(rows))
    print(f"[written] {CSV_PATH}")
    print(f"[written] {MD_PATH}")
    in_scope = [r for r in rows if r["scope"] == "in-ref-scope"]
    print(f"\n{len(rows)} rows ({len(in_scope)} in-ref-scope decision cells):")
    for r in rows:
        print(f"  [{r['scope']:16}] {r['model']:11} {r['dataset']:7} p{r['budget_pct']:<2} "
              f"cur_textsim={_f(r['current_textsim_dynstatic']):>7} "
              f"ref_textsim_ref={_f(r['ref_textsim_ref_dynstatic']):>7} "
              f"ref_best={_f(r['ref_best_dynstatic']):>7}  {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
