# Phase 3E — Real Migration Map (batch-sequenced, with status)

> The concrete, executable old→new map for moving `GQA/ VQA_V2/ v2/ outputs/` into the method-based
> `src/ … results/` structure. Each group is assigned to a **batch** (dependency-safe order) with a
> **status**. Companion to `METHOD_BASED_MIGRATION_PLAN.md` (rationale, risks, smoke tests). Branch:
> `method-migration`. Created 2026-06-24.
>
> **Rule:** a batch is done only when (a) files moved, (b) imports rewritten, (c) `compileall` + targeted
> import smoke pass, (d) committed. Result moves (Batch 9) also update `THESIS_EVIDENCE_LEDGER.md` paths
> in the same commit. Never leave the tree in a broken state between commits.

---

## Batch status overview

| Batch | Scope | Risk | Status |
|---|---|---|---|
| **1** | `utils/` → `src/utils/` (dedup 3→1) | low | **DONE (2026-06-24)** |
| 2 | scorers → `src/metrics/` + externalize archive | low–med | **DONE (2026-06-24)** |
| 3 | FLOPs/analysis → `src/analysis/` | low–med | pending |
| 4 | data loaders → `src/data/` | med | pending |
| 5 | models + pruning per method | high | pending |
| 6 | evaluation runners + training | high | pending |
| 7 | configs + experiments (`.sh`) | med | pending |
| 8 | results → `results/` + ledger update | high | pending |
| 9 | drop empty `GQA/ VQA_V2/ v2/ outputs/`; root README/docs refresh | med | pending |

---

## Batch 1 — utils (DONE)
| From | To | Note |
|---|---|---|
| `GQA/shared/utils/{__init__,config,seed,logger,io,device,checkpoint}.py` | `src/utils/` | canonical copy |
| `VQA_V2/shared/utils/*`, `v2/shared/utils/*` | (removed) | byte-identical duplicates; preserved in git history + backup |
| 18 `.py` + 2 `.sh` import sites | rewritten `(GQA\|VQA_V2\|v2).shared.utils` → `src.utils` | 0 stale refs remain |
| (added) `src/__init__.py` | — | makes `src` a package |
| (fixed) `.gitignore` `data/` → `/data/` | — | so `src/data`, `scripts/data` are not ignored |

## Batch 2 — metrics / scorers → `src/metrics/`
| From | To |
|---|---|
| `GQA/shared/{official_score,m4c_evaluator,textvqa_score,pope_score,eval_pope_official,metrics}.py` | `src/metrics/` (canonical) |
| `v2/shared/{official_score,m4c_evaluator,textvqa_score,pope_score,eval_pope_official,metrics}.py` | (remove dups) + add `v2/shared/{docvqa_score,chartqa_score}.py` → `src/metrics/` |
| `VQA_V2/shared/datasets/vqav2_answers.py` (answer normalize/score) | `src/metrics/` or `src/data/` (decide at batch time) |
| importers of `*.shared.{official_score,m4c_evaluator,…}` | rewrite → `src.metrics.*` |

## Batch 3 — FLOPs / analysis → `src/analysis/`
| From | To |
|---|---|
| `GQA/shared/flops.py` (canonical; `v2/shared/flops.py` identical → drop dup) | `src/analysis/flops.py` |
| `v2/qwen/qwen_flops.py`, `v2/analysis/{flops_frontier,oracle_decomposition,make_figures,verify_dense}.py`, `v2/qwen/make_qwen_figures.py`, `VQA_V2/shared/evaluation/{flops,make_figures}.py` | `src/analysis/` (+ `scripts/analysis/` for the runnable entry points) |
| importers + `qwen_flops_summary.json` provenance line | rewrite to `src.analysis.*` |

## Batch 4 — data loaders → `src/data/`
| From | To |
|---|---|
| `GQA/shared/dataset.py` | `src/data/gqa.py` |
| `VQA_V2/shared/datasets/*` | `src/data/vqa/` |
| `v2/data_setup/mix_dataset.py` | `src/data/llava_mix.py` |
| `v2/data_setup/download_*.py` | `scripts/data/` |
| importers | rewrite to `src.data.*` (keep `data/...` dataset paths unchanged) |

## Batch 5 — models + pruning (per method)
| From | To |
|---|---|
| `GQA/shared/static.py`, `GQA/static/visionzip.py`, `VQA_V2/static/*` | `src/models/static/`, `src/pruning/static/` |
| `GQA/static/{clip_select,question_cond}.py`; `v2/qwen/qwen_pruner.py`; selector in `v2/eval/evaluate_textvqa_highres_kcurve.py` | `src/pruning/question_conditioned_selection/`, `src/models/question_conditioned_selection/` |
| `VQA_V2/dynamic/{budget_controller,token_scorer,token_selector,llava_wrapper}.py`; `GQA/dynamic/*`; `v2/qwen/{qwen_budget_*,qwen_oracle*}.py` | `src/models/dynamic_budget/`, `src/pruning/dynamic_budget/` |
| `v2/distill/{student_selector,docvqa_data}.py` | `src/models/distillation/` |
| `v2/model/elastic_wrapper.py` (+tests) | `src/models/elastic/` (auxiliary) |
| `GQA/dense/run_dense_testdev.py` (`GQATestdevDataset`) | split: dataset → `src/data/`, dense model → `src/models/dense/` |

## Batch 6 — evaluation runners + training
| From | To |
|---|---|
| `GQA/eval_runners/run_{textvqa,pope,sqa}.py` | `src/evaluation/{textvqa,pope,scienceqa}/` (+ `scripts/evaluation/`) |
| `GQA/static/run_static*.py`, `run_visionzip_testdev.py`, `dense/run_dense_testdev.py` | `src/evaluation/gqa/` (+ `scripts/evaluation/`) |
| `VQA_V2/shared/evaluation/{generate_and_score,per_type_accuracy,instance_headroom,cascade_pass,cascade_analyze}.py` | `src/evaluation/vqa/` |
| `v2/qwen/{qwen_kcurve,qwen_control,qwen_layer_sweep,qwen25_dense_eval}.py`, `v2/eval/*highres*`, `v2/distill/{eval_gate,eval_control}.py` | `src/evaluation/{docvqa,chartqa}/` |
| `VQA_V2/dynamic/train_dynamic.py`, `v2/distill/{train_student,cache_teacher}.py`, `v2/training/train_stage1.py`, `VQA_V2/shared/training/cached/train_cached.py` | `src/training/` (+ `scripts/training/` launchers) |

## Batch 7 — configs + experiments
| From | To |
|---|---|
| `VQA_V2/{dense,dynamic}/*.yaml`, `VQA_V2/static/*.yaml` (21), `v2/configs/stage1_elastic.yaml` | `configs/{dense,dynamic_budget,static,distillation}/` |
| all `*.sh` (`VQA_V2/shared/experiments/*`, `v2/training/*.sh`, `v2/distill/*.sh`) | `experiments/<method>/` + `scripts/` |
| `VQA_V2/shared/scripts/*` | `scripts/data/` |

## Batch 8 — results → `results/` (with ledger update, atomic)
| From | To | Class |
|---|---|---|
| `outputs/results_frozen/`, `outputs/{testdev_frontier_analysis,cascade_sweep,week1_all_numbers,*_analysis}.json` | `results/thesis_main/` | L1,L3,L4 |
| `outputs/{testdev,textvqa,pope,sqa}_*` run dirs | `results/appendix/` | L3/L4 support |
| `VQA_V2/outputs/dynamic_150k_clsonly/` | `results/thesis_main/` | L2 |
| `VQA_V2/outputs/static_k*_{pertype,matched}/`, `cascade/`, `figures/` | `results/appendix/` | L2 support |
| `v2/outputs/qwen_*.json`, `eval_*.json`, `llava_latency.json`, `qwen_flops_summary.json`, `distill/{gate,control}_*.json` | `results/thesis_main/` | L5–L12,E |
| `v2/outputs/qwen_budget_data_*.json` | `results/paper_candidates/` | L10 |
| `v2/outputs/stage1_*` (kept ckpts) | `results/archived/` or stay in `archive/` | none |
> After each result group: edit its `THESIS_EVIDENCE_LEDGER.md` `File:` path + the reader-script glob in the **same commit**; re-run the ledger path-existence check.

## Batch 9 — finalize
- Confirm `GQA/ VQA_V2/ v2/ outputs/` are empty of real files → remove the now-empty dirs.
- Refresh root `README.md` folder map (drop the historical-names note); update `docs/` + `MIGRATION_REPORT`.
- Add a `results/` ignore rule for large result contents if needed (keep small frozen tables/index tracked).

---

## Phase status
- Batch 1: **DONE**. Batches 2–9: pending (recommended after the thesis draft; resume on `method-migration`).
