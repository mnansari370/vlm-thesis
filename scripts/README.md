# `scripts/` — runnable launchers

> **For method-specific commands, start from [`dense/`](../dense/), [`static/`](../static/),
> [`dynamic_which/`](../dynamic_which/), or [`dynamic_count/`](../dynamic_count/)** — each folder's
> `COMMANDS.md` gives the CPU validation, table, and GPU-rerun commands for that method, and a safe
> wrapper under its `scripts/`. This file is the full launcher reference.

All active launchers live under `scripts/final_scope/` and are invoked as
`python -m scripts.final_scope.<name>`. They are grouped by method family below (the files are not
physically foldered — the `python -m` module paths are the documented interface and are referenced
by the shell launchers, the README validation commands, and the run logs).

```
scripts/
  final_scope/   the ACTIVE layer — everything that ran the final matrix (grouped below)
  data/          setup only: download_docchart.py (caches the DocVQA HF dataset the final
                 evals read) · download_qwen25vl.py (fetches the Qwen model snapshot)
```

### Shared foundation
- `build_sample_manifests.py` — builds the sha256-locked `configs/final_scope/sample_ids/*.json`
  that every method and both models read (run once, before any evaluation).

### Dense
- `run_dense_pilot.py` — one dense cell (`--n 200` pilot or `--full`); wraps the frozen engines.
- `run_llava_dense_final.sh` (GPU 0, vlm_env) · `run_qwen_dense_final.sh` (GPU 1, qwen_env).

### Static
- `run_static_eval.py` — one static cell (`--budget-pct`, `--selector`); loads the dense final as reference.
- `run_llava_static_final.sh` · `run_qwen_static_final.sh`.

### Dynamic-WHICH
- `run_dynamic_which_eval.py` — one WHICH cell (`--selector textsim`); loads dense + same-budget static.
- `run_dynamic_which_ref_eval.py` — clean-room reference pilots (independent implementation).
- batch/pilot launchers: `run_dynamic_which_textsim_full_missing.sh` (the 35 full finals) ·
  `run_dynamic_which_missing_pilots.sh` · `run_dynamic_which_ref_rescue_pilots.sh` ·
  `run_qwen_textvqa_ref_validation.sh`.
- equivalence/comparison: `check_dynamic_which_equivalence{,_extended}.py` ·
  `compare_dynamic_which_current_vs_ref.py` · `compare_qwen_textvqa_current_vs_ref.py` ·
  `debug_textsim_current_vs_ref.py`.

### Dynamic-COUNT (5-stage pipeline)
1. `run_dynamic_count_probe.py` / `run_dynamic_count_probe_full_matrix.sh` — reproduction-gated probes.
2. `make_dynamic_count_controller_calibration.py` — fit DC-D/DC-C controllers on the first-20% split.
3. `run_dynamic_count_discrete.py` / `compose_dynamic_count_discrete.py` — DC-D (CPU compose).
4. `run_dynamic_count_continuous.py` / `run_dynamic_count_continuous_full_matrix.sh` — DC-C (GPU).
5. `analyze_dynamic_count_oracle.py` — the oracle upper bound (CPU; not a deployable method).

### Validation, audits, and tables (CPU, read-only over results)
- WHICH: `validate_dynamic_which_final.py` · `audit_dynamic_which_full_final_matrix.py` ·
  `audit_dynamic_which_coverage.py`.
- COUNT: `validate_dynamic_count_final.py` · `audit_dynamic_count_full_matrix.py`.
- tables: `make_final_thesis_tables.py` · `make_dynamic_which_*.py` · `make_dynamic_count_final_tables.py`.

Conventions: GPU runs are per-cell, bs=1 greedy; LLaVA under `vlm_env` on GPU 0, Qwen under
`qwen_env` on GPU 1; batch launchers are skip-safe (never overwrite an existing final). Static runs
require the matching dense final; Dynamic-WHICH additionally requires the matching static final;
Dynamic-COUNT requires the frozen finals its probes must reproduce.

The legacy launchers (feature-cache building, answer-head training, HPC/Slurm pipelines, the old
download helpers) were archived in cleanup Pass 1 (2026-07-05) to
`archive/legacy_scripts/scripts/{data,training}/`; see
`archive/migration_manifests/archive_manifest_20260705.md`.
