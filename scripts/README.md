# `scripts/` — runnable launchers

```
scripts/
  final_scope/   the ACTIVE layer — everything that ran the final matrix:
                   run_dense_pilot.py · run_static_eval.py · run_dynamic_which_eval.py
                   run_dynamic_which_ref_eval.py (clean-room validation pilots)
                   run_dynamic_count_{probe,discrete,continuous}.py
                   batch launchers: run_{llava,qwen}_{dense,static}_final.sh,
                     run_dynamic_which_textsim_full_missing.sh,
                     run_dynamic_count_{probe,continuous}_full_matrix.sh, ...
                   audits/validators: audit_dynamic_which_full_final_matrix.py,
                     validate_dynamic_which_final.py, audit_dynamic_count_full_matrix.py,
                     validate_dynamic_count_final.py, check_dynamic_which_equivalence*.py,
                     compare_*_current_vs_ref.py
                   table generators: make_final_thesis_tables.py, make_dynamic_which_*.py,
                     make_dynamic_count_*.py, analyze_dynamic_count_oracle.py
                   manifest builder: build_sample_manifests.py
  data/          setup only: download_docchart.py (caches the DocVQA HF dataset the final
                 evals read) · download_qwen25vl.py (fetches the Qwen model snapshot)
```

Conventions: GPU runs are per-cell, bs=1 greedy; LLaVA under `vlm_env` on GPU 0, Qwen under
`qwen_env` on GPU 1; batch launchers are skip-safe (never overwrite an existing final). Static runs
require the matching dense final; Dynamic-WHICH additionally requires the matching static final;
Dynamic-COUNT requires the frozen finals its probes must reproduce. The audits/validators and table
generators are CPU-only and read-only over results.

The legacy launchers (feature-cache building, answer-head training, HPC/Slurm pipelines, the old
download helpers) were archived in cleanup Pass 1 (2026-07-05) to
`archive/legacy_scripts/scripts/{data,training}/`; see
`archive/migration_manifests/archive_manifest_20260705.md`.
