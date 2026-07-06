# `results/` — experiment results

> **For a per-method results summary, start from [`dense/RESULTS.md`](../dense/RESULTS.md),
> [`static/RESULTS.md`](../static/RESULTS.md), [`dynamic_which/RESULTS.md`](../dynamic_which/RESULTS.md),
> or [`dynamic_count/RESULTS.md`](../dynamic_count/RESULTS.md).** This file describes the full result
> tree; the citable tables live under `results/final_scope/tables/`.

```
results/
  final_scope/                    THE final thesis evidence
    llava15/{gqa,vqav2,textvqa,docvqa}/      per cell: one aggregate .json + one per-sample .jsonl
    qwen25vl7b/{gqa,vqav2,textvqa,docvqa}/     (git-ignored local evidence — basenames are
                                                load-bearing references; never rename)
    tables/                       all 37 final tables & reports — TRACKED in git
    dynamic_count_configs/        fitted DC-D/DC-C controllers per cell — TRACKED
  thesis_main/{gqa,vqav2,highres}/   legacy pre-final-scope evidence (kept in place until
                                     thesis submission; mapped by legacy_index.md)
  paper_candidates/               old budget-generality raw inputs (out of scope, kept)
  README.md  INDEX.md  legacy_index.md
```

**Every citable thesis number lives under `results/final_scope/`.** The run outputs cover the full
matrix: 8 dense finals, 40 static finals (LLaVA `cls_attn`, Qwen `norm` × 5 budgets), 40
Dynamic-WHICH textsim finals, and the complete Dynamic-COUNT set (18 reproduction-gated probes,
DC-D cascades, DC-C rule+ridge runs, plus the separate COUNT-on-WHICH variant). Each aggregate
records its sample-manifest sha256, its dense/static reference paths with file sha256s, and its
fairness-gate verdict.

Start with `tables/final_thesis_results_summary.md` (per-cell verdicts), then
`tables/final_dense_static_dynamic_comparison.md` (every method setting),
`tables/dynamic_which_final_report.md` (the WHICH phase freeze), and the DC summaries
(`tables/dynamic_count_{dc_d,dc_c,win_loss,oracle}_summary.md`). Validation:
`python -m scripts.final_scope.validate_dynamic_which_final` and
`validate_dynamic_count_final` (CPU, read-only).

The per-sample JSONLs are git-ignored (large); a compressed backup exists at
`~/vlm-thesis-backups/final_scope_backup_20260705.tar.gz`. Legacy result trees under `thesis_main/`
are historical evidence for archived analyses — do not cite them as final numbers.
