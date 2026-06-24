# Results index (SCAFFOLD — points to current locations)

> No results have been migrated yet. This index maps the future `results/` classes to where the evidence
> **currently** lives. The authoritative paths are in `docs/THESIS_EVIDENCE_LEDGER.md`.

| Future class | Currently located at | Evidence |
|---|---|---|
| `thesis_main/` | `outputs/results_frozen/`, `outputs/{testdev_frontier_analysis,cascade_sweep,week1_all_numbers}.json`, `VQA_V2/outputs/dynamic_150k_clsonly/`, `v2/outputs/qwen_*.json`, `v2/outputs/eval_highres_*.json`, `v2/outputs/llava_latency.json`, `v2/outputs/qwen_flops_summary.json`, `v2/outputs/distill/{gate,control}_*.json` | L1–L12, E1–E3 |
| `appendix/` | `outputs/{testdev,textvqa,pope,sqa}_*` run dirs, `VQA_V2/outputs/static_k*_{pertype,matched}/`, `v2/outputs/{qwen_kcurve_chartqa,eval_*chartqa}.json` | L2/L3/L4 support |
| `paper_candidates/` | `v2/outputs/qwen_budget_data_*.json` | L10 (recompute) |
| `archived/` | `archive/failed_experiments/`, `archive/checkpoints/`, `archive/retired_code/` | none (history) |

**Nothing here is the source of truth yet.** Migration of these into `results/` happens in Phase 3E,
atomically with the code-path and ledger updates (see `docs/METHOD_BASED_MIGRATION_PLAN.md` §7).
