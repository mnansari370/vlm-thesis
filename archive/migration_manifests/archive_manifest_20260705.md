# Archive Manifest — Cleanup Pass 1 (low-risk archive)

- **cleanup pass:** Pass 1 low-risk archive
- **date:** 2026-07-05
- **branch:** method-migration
- **commit before cleanup:** 37631e9 ("Add final-scope thesis reports and validation tables")
- **plan:** /tmp/vlm_cleanup_archive_plan_20260705.md (SAFE_TO_ARCHIVE_NOW items only)
- **backup taken first:** ~/vlm-thesis-backups/final_scope_backup_20260705.tar.gz (108 MB, 595 entries — full copy of results/final_scope, which is git-ignored local evidence; results/final_scope itself was NOT touched)

**Nothing was deleted.** Tracked files moved with `git mv` (121 renames, 0 content changes); git-ignored logs moved with plain `mv`; stray `__pycache__` directories were relocated alongside their packages (regenerable caches, kept not deleted). Four empty, untracked directories were removed with `rmdir` (no content existed).

**Restore rule (unchanged from the 2026-06-27 manifest):** every archived path mirrors its original relative path under its bucket, so restoring any item is
`git mv archive/<bucket>/<original/relative/path> <original/relative/path>` (tracked) or plain `mv` (logs).

**NOT moved in this pass (explicitly):** the Pass-2 trapped files
`src/models/static/{llava_wrapper,answer_head,token_selector}.py` and
`src/data/vqav2/{vqav2,collate,image_transforms}.py` (blocked on the two `__init__.py` imports);
everything under `results/final_scope/`, `configs/final_scope/sample_ids/`, `src/final_scope/`,
`scripts/final_scope/`, the active pruning packages, `src/models/static/`, `src/data/vqav2/`,
`src/metrics/`, `src/analysis/{flops,qwen_flops,__init__}.py`, `data/`, `docs/`,
`results/thesis_main/`, `results/paper_candidates/`, and the seven final-run log folders
(`logs/{final_dense,final_static,dynamic_which_final,dynamic_which_textsim_full_missing,dynamic_which_ref_textvqa_validation,dynamic_count_probe,dynamic_count_dcc}`).

**Nesting correction during execution:** four destination directories already existed from the
2026-06-27 archive (`legacy_scripts/scripts/training`, `legacy_experiments/src/evaluation/gqa`,
`legacy_experiments/src/evaluation/textvqa`, `legacy_experiments/src/pruning/dynamic_budget`), so the
initial directory `git mv` nested the new content one level too deep; the contents were flattened up
one level before commit (verified: no name collisions with the June contents; final layout is the flat
mirror path recorded below).

---

## Cluster summary (reason per cluster)

| Cluster | What moved | New location | Reason |
|---|---|---|---|
| 2g | `configs/dense/` (2), `configs/static/` (21), `configs/dynamic_budget/` (3) | `archive/legacy_experiments/configs/{dense,static,dynamic_budget}/` | classification-head-era YAMLs (mnt=10, answer-head, missing `answer_vocab_full.json`); final-scope runs are argparse-driven |
| 2f | `src/models/dense/`, `src/models/dynamic_budget/` | `archive/legacy_models/src/models/{dense,dynamic_budget}/` | retired classification wrapper + old BudgetController (L2 evidence); no active importers (their importers moved in the same pass) |
| 2a | `src/models/distillation/`, `src/training/{cache_teacher,train_student}.py`, `src/evaluation/docvqa/{eval_gate,eval_control}.py` | `archive/legacy_models/...` + `archive/legacy_experiments/...` | Qwen DocVQA distillation era (L11/L12 evidence); not part of the final matrix |
| 2b | `src/pruning/dynamic_budget/` (5+init), `src/evaluation/docvqa/{qwen25_dense_eval,qwen_kcurve,qwen_control,qwen_layer_sweep,__init__}.py` | `archive/legacy_experiments/src/{pruning/dynamic_budget,evaluation/docvqa}/` | old Qwen budget-mirage study (L5–L9 evidence); Dynamic-COUNT re-implemented the methodology fresh and does not import it |
| 2c | `src/evaluation/{gqa,textvqa,vqa}/` + `src/evaluation/__init__.py`; QC probes `question_cond.py`, `clip_select.py`, `clip_visual_check.py` | `archive/legacy_experiments/src/{evaluation,pruning/question_conditioned_selection}/` | pre-final-scope eval harnesses (produced `results/thesis_main`) and the L4 QC probes; superseded by `src/final_scope/` runners |
| 2d | `src/analysis/{flops_vqav2,cascade_sweep,make_figures_vqav2,make_qwen_figures,verify_dense}.py` | `archive/legacy_experiments/src/analysis/` | legacy figures/cascade; `verify_dense.py` already had a broken (archived) import. KEPT: `flops.py`, `qwen_flops.py`, `__init__.py` (used by `token_flops.py`) |
| 2e | `src/training/{train_cached,train_dynamic,__init__}.py`; `scripts/training/` (10 launchers); `scripts/data/` legacy (30 files) | `archive/legacy_experiments/src/training/`, `archive/legacy_scripts/scripts/{training,data}/` | answer-head/feature-cache/HPC era. KEPT ACTIVE: `scripts/data/download_docchart.py` (rebuilds the DocVQA HF cache the final evals need), `scripts/data/download_qwen25vl.py` |
| 2h | `logs/{dynamic_which_missing_pilots,dynamic_which_ref_rescue,dynamic_which_ref_smoke,dynamic_which_confirm,dynamic_which_equiv_extended}/` | `archive/legacy_logs/logs/` (git-ignored, plain `mv`) | pilot/rescue/debug logs whose conclusions are fully captured in the committed tables |
| 2i | `configs/final_scope/{llava15,qwen25vl7b}/`, `logs/dynamic_which_pilots/`, `results/archived/` | removed (`rmdir`) | empty, untracked placeholder directories — no content existed |

## Untracked moves (plain `mv`, not in git)

| Old path | New archive path |
|---|---|
| `logs/dynamic_which_missing_pilots/` | `archive/legacy_logs/logs/dynamic_which_missing_pilots/` |
| `logs/dynamic_which_ref_rescue/` | `archive/legacy_logs/logs/dynamic_which_ref_rescue/` |
| `logs/dynamic_which_ref_smoke/` | `archive/legacy_logs/logs/dynamic_which_ref_smoke/` |
| `logs/dynamic_which_confirm/` | `archive/legacy_logs/logs/dynamic_which_confirm/` |
| `logs/dynamic_which_equiv_extended/` | `archive/legacy_logs/logs/dynamic_which_equiv_extended/` |
| `src/evaluation/__pycache__/`, `src/evaluation/docvqa/__pycache__/`, `src/training/__pycache__/`, `scripts/data/__pycache__/` (partial) | relocated beside their archived packages (regenerable caches) |

Restore (logs): `mv archive/legacy_logs/logs/<name> logs/<name>`

## Verification (run after every cluster and at the end)

`python3 -m compileall -q src scripts` → exit 0 · `python3 -m src.final_scope.test_final_scope` → ALL PASSED
(full end-of-pass suite incl. audits, validators, and import smokes recorded in the Pass-1 commit message context).

---

## Complete tracked move list (121 `git mv` renames: old path → new archive path)

- `configs/dense/llava_dense_150k_10k_fullvocab.yaml` → `archive/legacy_experiments/configs/dense/llava_dense_150k_10k_fullvocab.yaml`
- `configs/dense/llava_dense_443k_10k_fullvocab.yaml` → `archive/legacy_experiments/configs/dense/llava_dense_443k_10k_fullvocab.yaml`
- `configs/dynamic_budget/llava_dynamic_150k_10k_fullvocab.yaml` → `archive/legacy_experiments/configs/dynamic_budget/llava_dynamic_150k_10k_fullvocab.yaml`
- `configs/dynamic_budget/llava_dynamic_gate_2k.yaml` → `archive/legacy_experiments/configs/dynamic_budget/llava_dynamic_gate_2k.yaml`
- `configs/dynamic_budget/llava_dynamic_gate_smoke.yaml` → `archive/legacy_experiments/configs/dynamic_budget/llava_dynamic_gate_smoke.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k128.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k128.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k144.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k144.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k160.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k160.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k192.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k192.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k219.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k219.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k255.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k255.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k265.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k265.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k275.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k275.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k276.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k276.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k334.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k334.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k357.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k357.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k432.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k432.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k64.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k64.yaml`
- `configs/static/llava_static_clsattn_150k_10k_fullvocab_k96.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_150k_10k_fullvocab_k96.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k128.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k128.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k144.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k144.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k192.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k192.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k288.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k288.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k432.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k432.yaml`
- `configs/static/llava_static_clsattn_443k_10k_fullvocab_k64.yaml` → `archive/legacy_experiments/configs/static/llava_static_clsattn_443k_10k_fullvocab_k64.yaml`
- `scripts/data/budget_variance_gate.py` → `archive/legacy_scripts/scripts/data/budget_variance_gate.py`
- `scripts/data/build_all_caches.sh` → `archive/legacy_scripts/scripts/data/build_all_caches.sh`
- `scripts/data/build_answer_vocab_full.py` → `archive/legacy_scripts/scripts/data/build_answer_vocab_full.py`
- `scripts/data/build_cache_k288.sh` → `archive/legacy_scripts/scripts/data/build_cache_k288.sh`
- `scripts/data/cache_dense_443k_pipeline.sh` → `archive/legacy_scripts/scripts/data/cache_dense_443k_pipeline.sh`
- `scripts/data/cache_dense_train.sh` → `archive/legacy_scripts/scripts/data/cache_dense_train.sh`
- `scripts/data/cache_dense_val.sh` → `archive/legacy_scripts/scripts/data/cache_dense_val.sh`
- `scripts/data/cache_features.py` → `archive/legacy_scripts/scripts/data/cache_features.py`
- `scripts/data/cache_features.sh` → `archive/legacy_scripts/scripts/data/cache_features.sh`
- `scripts/data/cache_static_443k_gpu0.sh` → `archive/legacy_scripts/scripts/data/cache_static_443k_gpu0.sh`
- `scripts/data/cache_static_443k_gpu1.sh` → `archive/legacy_scripts/scripts/data/cache_static_443k_gpu1.sh`
- `scripts/data/cache_static_gpu0.sh` → `archive/legacy_scripts/scripts/data/cache_static_gpu0.sh`
- `scripts/data/cache_static_gpu1.sh` → `archive/legacy_scripts/scripts/data/cache_static_gpu1.sh`
- `scripts/data/cache_static_k128_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k128_train.sh`
- `scripts/data/cache_static_k128_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k128_val.sh`
- `scripts/data/cache_static_k144_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k144_train.sh`
- `scripts/data/cache_static_k144_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k144_val.sh`
- `scripts/data/cache_static_k192_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k192_train.sh`
- `scripts/data/cache_static_k192_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k192_val.sh`
- `scripts/data/cache_static_k288_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k288_train.sh`
- `scripts/data/cache_static_k288_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k288_val.sh`
- `scripts/data/cache_static_k432_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k432_train.sh`
- `scripts/data/cache_static_k432_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k432_val.sh`
- `scripts/data/cache_static_k64_train.sh` → `archive/legacy_scripts/scripts/data/cache_static_k64_train.sh`
- `scripts/data/cache_static_k64_val.sh` → `archive/legacy_scripts/scripts/data/cache_static_k64_val.sh`
- `scripts/data/check_cache_nan.py` → `archive/legacy_scripts/scripts/data/check_cache_nan.py`
- `scripts/data/download_fast.sh` → `archive/legacy_scripts/scripts/data/download_fast.sh`
- `scripts/data/rsync_from_hpc.sh` → `archive/legacy_scripts/scripts/data/rsync_from_hpc.sh`
- `scripts/data/smoke_cache_20.sh` → `archive/legacy_scripts/scripts/data/smoke_cache_20.sh`
- `scripts/data/train_mlp_cached.sh` → `archive/legacy_scripts/scripts/data/train_mlp_cached.sh`
- `scripts/training/generation_eval.sh` → `archive/legacy_scripts/scripts/training/generation_eval.sh`
- `scripts/training/run_dense_sequential.sh` → `archive/legacy_scripts/scripts/training/run_dense_sequential.sh`
- `scripts/training/run_followups.sh` → `archive/legacy_scripts/scripts/training/run_followups.sh`
- `scripts/training/run_gate_smoke.sh` → `archive/legacy_scripts/scripts/training/run_gate_smoke.sh`
- `scripts/training/run_gpu1_tasks.sh` → `archive/legacy_scripts/scripts/training/run_gpu1_tasks.sh`
- `scripts/training/run_option_a.sh` → `archive/legacy_scripts/scripts/training/run_option_a.sh`
- `scripts/training/train_all_heads.sh` → `archive/legacy_scripts/scripts/training/train_all_heads.sh`
- `scripts/training/train_mlp_dense.sh` → `archive/legacy_scripts/scripts/training/train_mlp_dense.sh`
- `scripts/training/train_mlp_static.sh` → `archive/legacy_scripts/scripts/training/train_mlp_static.sh`
- `scripts/training/vonasah_pipeline.sh` → `archive/legacy_scripts/scripts/training/vonasah_pipeline.sh`
- `src/analysis/cascade_sweep.py` → `archive/legacy_experiments/src/analysis/cascade_sweep.py`
- `src/analysis/flops_vqav2.py` → `archive/legacy_experiments/src/analysis/flops_vqav2.py`
- `src/analysis/make_figures_vqav2.py` → `archive/legacy_experiments/src/analysis/make_figures_vqav2.py`
- `src/analysis/make_qwen_figures.py` → `archive/legacy_experiments/src/analysis/make_qwen_figures.py`
- `src/analysis/verify_dense.py` → `archive/legacy_experiments/src/analysis/verify_dense.py`
- `src/evaluation/docvqa/eval_control.py` → `archive/legacy_experiments/src/evaluation/docvqa/eval_control.py`
- `src/evaluation/docvqa/eval_gate.py` → `archive/legacy_experiments/src/evaluation/docvqa/eval_gate.py`
- `src/evaluation/docvqa/__init__.py` → `archive/legacy_experiments/src/evaluation/docvqa/__init__.py`
- `src/evaluation/docvqa/qwen25_dense_eval.py` → `archive/legacy_experiments/src/evaluation/docvqa/qwen25_dense_eval.py`
- `src/evaluation/docvqa/qwen_control.py` → `archive/legacy_experiments/src/evaluation/docvqa/qwen_control.py`
- `src/evaluation/docvqa/qwen_kcurve.py` → `archive/legacy_experiments/src/evaluation/docvqa/qwen_kcurve.py`
- `src/evaluation/docvqa/qwen_layer_sweep.py` → `archive/legacy_experiments/src/evaluation/docvqa/qwen_layer_sweep.py`
- `src/evaluation/gqa/__init__.py` → `archive/legacy_experiments/src/evaluation/gqa/__init__.py`
- `src/evaluation/gqa/run_clip_probe.py` → `archive/legacy_experiments/src/evaluation/gqa/run_clip_probe.py`
- `src/evaluation/gqa/run_dense_testdev.py` → `archive/legacy_experiments/src/evaluation/gqa/run_dense_testdev.py`
- `src/evaluation/gqa/run_qcond_probe.py` → `archive/legacy_experiments/src/evaluation/gqa/run_qcond_probe.py`
- `src/evaluation/gqa/run_speculative_testdev.py` → `archive/legacy_experiments/src/evaluation/gqa/run_speculative_testdev.py`
- `src/evaluation/gqa/run_static.py` → `archive/legacy_experiments/src/evaluation/gqa/run_static.py`
- `src/evaluation/gqa/run_static_testdev.py` → `archive/legacy_experiments/src/evaluation/gqa/run_static_testdev.py`
- `src/evaluation/gqa/run_visionzip_testdev.py` → `archive/legacy_experiments/src/evaluation/gqa/run_visionzip_testdev.py`
- `src/evaluation/__init__.py` → `archive/legacy_experiments/src/evaluation/__init__.py`
- `src/evaluation/textvqa/__init__.py` → `archive/legacy_experiments/src/evaluation/textvqa/__init__.py`
- `src/evaluation/textvqa/run_textvqa.py` → `archive/legacy_experiments/src/evaluation/textvqa/run_textvqa.py`
- `src/evaluation/vqa/cascade_analyze.py` → `archive/legacy_experiments/src/evaluation/vqa/cascade_analyze.py`
- `src/evaluation/vqa/cascade_pass.py` → `archive/legacy_experiments/src/evaluation/vqa/cascade_pass.py`
- `src/evaluation/vqa/generate_and_score.py` → `archive/legacy_experiments/src/evaluation/vqa/generate_and_score.py`
- `src/evaluation/vqa/__init__.py` → `archive/legacy_experiments/src/evaluation/vqa/__init__.py`
- `src/evaluation/vqa/instance_headroom.py` → `archive/legacy_experiments/src/evaluation/vqa/instance_headroom.py`
- `src/evaluation/vqa/per_type_accuracy.py` → `archive/legacy_experiments/src/evaluation/vqa/per_type_accuracy.py`
- `src/models/dense/answer_head.py` → `archive/legacy_models/src/models/dense/answer_head.py`
- `src/models/dense/__init__.py` → `archive/legacy_models/src/models/dense/__init__.py`
- `src/models/dense/llava_wrapper.py` → `archive/legacy_models/src/models/dense/llava_wrapper.py`
- `src/models/dense/token_selector.py` → `archive/legacy_models/src/models/dense/token_selector.py`
- `src/models/distillation/__init__.py` → `archive/legacy_models/src/models/distillation/__init__.py` *
- `src/models/distillation/student_selector.py` → `archive/legacy_models/src/models/distillation/student_selector.py`
- `src/models/dynamic_budget/answer_head.py` → `archive/legacy_models/src/models/dynamic_budget/answer_head.py`
- `src/models/dynamic_budget/budget_controller.py` → `archive/legacy_models/src/models/dynamic_budget/budget_controller.py`
- `src/models/dynamic_budget/__init__.py` → `archive/legacy_models/src/models/dynamic_budget/__init__.py`
- `src/models/dynamic_budget/llava_wrapper.py` → `archive/legacy_models/src/models/dynamic_budget/llava_wrapper.py`
- `src/models/dynamic_budget/token_scorer.py` → `archive/legacy_models/src/models/dynamic_budget/token_scorer.py`
- `src/models/dynamic_budget/token_selector.py` → `archive/legacy_models/src/models/dynamic_budget/token_selector.py`
- `src/pruning/dynamic_budget/__init__.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/__init__.py` *
- `src/pruning/dynamic_budget/qwen_budget_data.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/qwen_budget_data.py`
- `src/pruning/dynamic_budget/qwen_budget_eval.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/qwen_budget_eval.py`
- `src/pruning/dynamic_budget/qwen_budget_robust.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/qwen_budget_robust.py`
- `src/pruning/dynamic_budget/qwen_oracle.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/qwen_oracle.py`
- `src/pruning/dynamic_budget/qwen_oracle_qc.py` → `archive/legacy_experiments/src/pruning/dynamic_budget/qwen_oracle_qc.py`
- `src/pruning/question_conditioned_selection/clip_select.py` → `archive/legacy_experiments/src/pruning/question_conditioned_selection/clip_select.py`
- `src/pruning/question_conditioned_selection/clip_visual_check.py` → `archive/legacy_experiments/src/pruning/question_conditioned_selection/clip_visual_check.py`
- `src/pruning/question_conditioned_selection/question_cond.py` → `archive/legacy_experiments/src/pruning/question_conditioned_selection/question_cond.py`
- `src/training/cache_teacher.py` → `archive/legacy_experiments/src/training/cache_teacher.py`
- `src/training/__init__.py` → `archive/legacy_experiments/src/training/__init__.py` *
- `src/training/train_cached.py` → `archive/legacy_experiments/src/training/train_cached.py`
- `src/training/train_dynamic.py` → `archive/legacy_experiments/src/training/train_dynamic.py`
- `src/training/train_student.py` → `archive/legacy_experiments/src/training/train_student.py`

*\* Corrected rows: these three `__init__.py` files are byte-identical empty files, so git's rename detection paired them arbitrarily across packages in `diff --cached -M`; the rows above record the LOGICAL move actually performed (verified on disk). Content and final locations are identical either way.*
