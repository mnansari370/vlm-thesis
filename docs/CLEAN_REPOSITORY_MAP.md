# CLEAN REPOSITORY MAP (post scope-lock, 2026-06-27)

*What is active vs archived after the final-scope reorganization. Active code = imports cleanly
(`compileall` exit 0, 0 in-scope import errors). Archived = preserved on disk under `archive/`, not deleted.*

```
vlm-thesis/
├── README.md  requirements.txt  .gitignore
├── configs/
│   ├── dense/          (LLaVA-1.5 dense, 150k/443k)             ACTIVE
│   ├── static/         (LLaVA-1.5 static, 21 K-values)          ACTIVE (re-express to 15–75% later)
│   ├── dynamic_budget/ (LLaVA-1.5 BudgetController + gate)      ACTIVE (dynamic-COUNT redesign target)
│   └── final_scope/{llava15,qwen25vl7b}/                        NEW, EMPTY (new standardized configs go here)
├── src/
│   ├── data/           gqa.py, docvqa.py, vqav2/                ACTIVE   (llava_mix.py → archived)
│   ├── metrics/        official/GQA, textvqa/m4c, docvqa ANLS   ACTIVE   (pope_score, chartqa_score: shared, unused-by-scope)
│   ├── models/         dense/ static/ dynamic_budget/ distillation/   ACTIVE   (elastic/ → archived)
│   ├── pruning/        static/visionzip, question_conditioned_selection/, dynamic_budget/qwen_*   ACTIVE (llava_budget_data → archived)
│   ├── evaluation/     gqa/ vqa/ textvqa/ docvqa/               ACTIVE   (pope/ scienceqa/ highres → archived)
│   ├── analysis/       flops, flops_vqav2, qwen_flops, cascade_sweep, make_*figures, verify_dense   ACTIVE (highres analysis → archived)
│   ├── training/       train_cached, train_dynamic, train_student, cache_teacher   ACTIVE (train_stage1 → archived)
│   ├── utils/          config/seed/logger/io/checkpoint/device  ACTIVE
│   └── final_scope/                                             NEW, EMPTY (new wrappers/scripts later)
├── scripts/
│   ├── data/           cache_features, vocab, download_qwen25vl, download_docchart, …   ACTIVE (dl_qwen3b/32b, download_llava16/mix → archived)
│   ├── training/       generation_eval, run_* launchers          ACTIVE (launch_stage1_full → archived)
│   └── final_scope/                                              NEW, EMPTY
├── results/
│   ├── thesis_main/{gqa,vqav2,highres}/   LEGACY EVIDENCE — kept in place, ledger-pinned, see legacy_index.md
│   ├── final_scope/{llava15,qwen25vl7b}/{gqa,textvqa,docvqa,vqav2}/   NEW, EMPTY (standardized runs land here)
│   ├── paper_candidates/                  L10 raw inputs — kept
│   ├── legacy_index.md                    NEW — maps legacy results, flags in-scope vs out-of-scope in highres/
│   ├── README.md  INDEX.md                kept
│   └── (archived/stage1_* → moved to archive/legacy_results/)
├── data/               vqav2/ gqa/ textvqa/   ACTIVE   |  llava_mix/ (34G, out-of-scope, left for size)  |  budget_oracle → archived
├── docs/               THESIS_*, PAPER_PLAN, FULL_/METHOD_ reports, + NEW: FINAL_SCOPE_LOCK, CLEAN_EXPERIMENT_MATRIX,
│                       CLEAN_REPOSITORY_MAP, TODO_NEXT_RUNS, FINAL_CONFIG_INVENTORY
└── archive/            legacy_{models,datasets,experiments,results,docs,scripts}/ + migration_manifests/
```

## Active code (the final-scope working set)
- **Models:** `src/models/{dense,static,dynamic_budget,distillation}`. Dense + `StaticPrunedLlava`
  (training-free physical pruning) are the LLaVA-1.5 engines; the Qwen engine is
  `src/pruning/question_conditioned_selection/qwen_pruner.py`.
- **Datasets/loaders:** `src/data/{gqa.py, docvqa.py, vqav2/}`.
- **Metrics:** `src/metrics/{official_score, metrics, textvqa_score, m4c_evaluator, docvqa_score}`.
- **Evaluation entrypoints:** `src/evaluation/gqa/{run_dense_testdev, run_static, run_static_testdev,
  run_visionzip_testdev, run_qcond_probe, run_clip_probe, run_speculative_testdev}`,
  `src/evaluation/vqa/{generate_and_score, cascade_pass, cascade_analyze, instance_headroom, per_type_accuracy}`,
  `src/evaluation/textvqa/run_textvqa`, `src/evaluation/docvqa/{qwen_kcurve, qwen_control, qwen_layer_sweep,
  qwen25_dense_eval, eval_gate, eval_control}`.
- **Analysis:** `src/analysis/{flops, flops_vqav2, qwen_flops, cascade_sweep, make_qwen_figures,
  make_figures_vqav2, verify_dense}`.
- **Training:** `src/training/{train_cached, train_dynamic, train_student, cache_teacher}`.

## Active configs
`configs/{dense,static,dynamic_budget}` (LLaVA-1.5). **No Qwen configs exist yet** (Qwen scripts are
argparse-driven). New standardized configs go under `configs/final_scope/`.

## Active datasets (on disk)
`data/{vqav2,gqa,textvqa}` (local); DocVQA via HF hub. `data/llava_mix` left in place (out of scope, 34G).

## Active results
`results/thesis_main/*` (legacy evidence, indexed) + new empty `results/final_scope/*`.

## Archived areas (preserved, not deleted)
`archive/legacy_models/` (elastic), `archive/legacy_experiments/` (highres, POPE, ScienceQA, elastic
trainers/evals), `archive/legacy_datasets/` (llava_mix loader, budget_oracle), `archive/legacy_results/`
(stage1 elastic), `archive/legacy_scripts/` (3B/32B/llava16/mix downloads, stage1 launcher),
`archive/migration_manifests/archive_manifest_20260627.md`.

## Shared utilities held back (clean during method rewrite)
See `archive/migration_manifests/archive_manifest_20260627.md` §C: `models/{static,dynamic_budget}`, the Qwen
budget files, `metrics/{chartqa,pope}_score`, `analysis/flops.py`, `cascade_sweep.py`, `download_docchart.py`,
`budget_variance_gate.py`.
