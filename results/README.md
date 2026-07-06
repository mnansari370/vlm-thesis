# `results/` — experiment results

> **For a per-method results summary, start from [`dense/RESULTS.md`](../dense/RESULTS.md),
> [`static/RESULTS.md`](../static/RESULTS.md), [`dynamic_which/RESULTS.md`](../dynamic_which/RESULTS.md),
> or [`dynamic_count/RESULTS.md`](../dynamic_count/RESULTS.md).** This file describes the full result
> tree; the citable tables live under `results/tables/`.

```
results/
  runs/                           THE final thesis evidence (git-ignored local outputs)
    llava15/{gqa,vqav2,textvqa,docvqa}/      per cell: one aggregate .json + one per-sample .jsonl
    qwen25vl7b/{gqa,vqav2,textvqa,docvqa}/     (basenames are load-bearing references; never rename)
  tables/                         all final tables & reports — TRACKED in git
  configs/dynamic_count/          fitted DC-D/DC-C controllers per cell — TRACKED
  README.md  INDEX.md  legacy_index.md
```

Out-of-scope and legacy evidence (the pre-final `thesis_main/` trees and `paper_candidates/`) was
moved to `archive/legacy_results/` during the 2026-07-05 restructure.

**Every citable thesis number lives under `results/runs/`.** The run outputs cover the full
matrix: 8 dense finals, 40 static finals (LLaVA `cls_attn`, Qwen `norm` × 5 budgets), 40
Dynamic-WHICH textsim finals, and the complete Dynamic-COUNT set (18 reproduction-gated probes,
DC-D cascades, DC-C rule+ridge runs, plus the separate COUNT-on-WHICH variant). Each aggregate
records its sample-manifest sha256, its dense/static reference paths with file sha256s, and its
fairness-gate verdict.

Start with `tables/final_thesis_results_summary.md` (per-cell verdicts), then
`tables/final_dense_static_dynamic_comparison.md` (every method setting),
`tables/dynamic_which_final_report.md` (the WHICH phase freeze), and the DC summaries
(`tables/dynamic_count_{dc_d,dc_c,win_loss,oracle}_summary.md`). Validation:
`python -m scripts.validation.validate_dynamic_which` and
`validate_dynamic_count_final` (CPU, read-only).

The per-sample JSONLs are git-ignored (large); a compressed backup exists at
`~/vlm-thesis-backups/final_scope_backup_20260705.tar.gz`. Legacy result trees under `thesis_main/`
are historical evidence for archived analyses — do not cite them as final numbers.
