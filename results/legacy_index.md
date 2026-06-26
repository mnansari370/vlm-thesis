# Legacy Results Index (post scope-lock, 2026-06-27)

*The existing evidence in `results/thesis_main/` was **left in place** (ledger-pinned; several active reader
scripts hardcode these paths). New standardized runs go to `results/final_scope/`. This index marks which
legacy results are in-scope (reusable as final evidence) vs out-of-scope (archive-class, do not cite as
final). Nothing here was moved or deleted. Elastic Stage-1 checkpoints were moved to
`archive/legacy_results/stage1_elastic/`.*

## `results/thesis_main/gqa/` — LLaVA-1.5 low-res track
| In scope (keep) | Out of scope (archive-class, ignore for final) |
|---|---|
| GQA dense `testdev_dense_honest_bs1_*` (61.42) | POPE: `pope_*` (all) |
| GQA static `testdev_static_cls_attn_k{144,192,288,432}_*` | ScienceQA: `sqa_*` |
| GQA VisionZip `testdev_visionzip_k*` | TextVQA-CLIP-probe `clip_textvqa_*` (CLIP-space, archived method) |
| GQA QC probe `qcond_gqa_*` (negative, K64) | `pope_analysis.json`, `sqa_analysis.json`, `pope_speculative_*` |
| GQA cascade `testdev_speculative_*`, `cascade_sweep.json` | `clip_gqa_*` (CLIP-space probe) |
| TextVQA dense/static `textvqa_dense_full_*`, `textvqa_cls_attn_k*` (+`_noocr`) | — |
| TextVQA QC probe `qcond_textvqa_*`, `textvqa_random_*` | — |
| `testdev_frontier_analysis.json`, `textvqa_analysis_{ocr,noocr}.json`, `week1_all_numbers.json`, `results_frozen/` | (these aggregate both; read the in-scope rows only) |

## `results/thesis_main/vqav2/` — LLaVA-1.5 VQAv2 track
| In scope (keep) | Out of scope |
|---|---|
| `dense_pad/` (76.44), `static_k{64,128,144,192,288,432}_pad/` | the `_fixed`/`_v1`/`_pertype`/`_matched` preprocessing variants (non-canonical; keep `_pad`) |
| `dynamic_150k_clsonly/` (budget ties static, +0.05pp — dynamic-COUNT evidence) | `static_baseline_locked_centercrop_archived.json` |
| `gate_real/`, `static_baseline_locked_expand2square.json` (locked curve) | `cascade/` (confidence cascade — keep numbers if Idea-2 discussed) |

## `results/thesis_main/highres/` — MIXED (read carefully)
This folder holds **both** in-scope Qwen-2.5-VL-7B/DocVQA results **and** out-of-scope LLaVA-1.6 / ChartQA /
3B / 32B results. It was **not** split physically (a bulk move would break ledger-pinned paths). Classification:

| In scope — KEEP (Qwen-7B × DocVQA) | Out of scope — archive-class (do not cite as final) |
|---|---|
| `qwen_kcurve_docvqa.json` | `qwen_kcurve_chartqa.json`, `qwen25_dense_chartqa.json`, `qwen_oracle_chartqa_qc.json` (ChartQA) |
| `qwen25_dense_docvqa.json` | `qwen_layer_sweep_3b.json`, `*_32b.*` (Qwen-3B/32B) |
| `qwen_layer_sweep.json` (7B) | all `eval_*highres*`, `eval_base.json`, `eval_quickfix.json`, `eval_control_qcond.json`, `eval_gqa_*`, `eval_textvqa_*`, `eval_spread_*`, `eval_textattn_layer_sweep*` (LLaVA-1.6) |
| `qwen_control_docvqa.json` | `llava_latency.json` (LLaVA-1.6 — keep the **3.31× number**, code archived) |
| `qwen_oracle_docvqa_qc.json`, `qwen_oracle_docvqa_uniform.json` | `llava_budget_data.log` (LLaVA-1.6) |
| `qwen_budget_eval.json`, `qwen_budget_robust.json`, `qwen_budget_data.json` (DocVQA) | `figures/` (LLaVA-1.6 figures) |
| `qwen_flops_summary.json` | — |
| `distill/*` (student on Qwen-DocVQA — in scope) | — |

## `results/paper_candidates/` — KEEP ALL
`qwen_budget_data_docvqa{,_7b_n1000}.json` (in scope, Qwen-7B/DocVQA), and `*_3b/_32b/_infovqa/_docvqa_llava`
(out of scope but **raw inputs for the L10 generality recompute** — do not delete).

## Moved out of `results/` (now archived)
`results/archived/stage1_full{,.log}`, `stage1_quickfix{,.log}` → `archive/legacy_results/stage1_elastic/`.

## Convention going forward
Final, citable numbers come **only** from `results/final_scope/{llava15,qwen25vl7b}/<dataset>/<method>_<pct>.json`
once the runs in `docs/TODO_NEXT_RUNS.md` are executed. Legacy in-scope results above may be lifted directly
where the protocol already matches (e.g. LLaVA-1.5 dense/static), but re-express static to the 15–75% grid.
