# Deep structure cleanup plan (2026-07-05)

Goal: a clean, method-oriented, thesis-ready layout. The tested runtime is fully validated, so every
group below is executed in isolation, tested, and **reverted if it breaks validation**. Coupling was
measured before planning (134 module-import references to `src.final_scope`/`scripts.final_scope`;
~26 path-string constant sites; the run trees `results/final_scope/{llava15,qwen25vl7b}` are
git-ignored, 536 files / 966 MB, while `tables/` and `dynamic_count_configs/` are tracked).

## logs/ stays in the root

`logs/` is **not** moved, renamed, or archived in this pass. The final-run logs are experiment
provenance and debugging evidence and remain visible at the repository root.

## Stop condition

If moving `src/`, `scripts/`, or `results/` breaks the CPU validation suite (`test_evaluation_core`,
the four validators/audits) and the break cannot be fixed by a reference update, **revert that group
and report it** rather than forcing the move.

---

## Group A — configs/final_scope/sample_ids → configs/sample_ids  (LOW risk)

Files that move (tracked → `git mv`): the four `{gqa,textvqa,docvqa,vqav2}.json` manifests.
Import/constant updates required:
- `src/final_scope/dense_pilot.py`: `MANIFEST_DIR = "configs/final_scope/sample_ids"` → `configs/sample_ids`
- `scripts/final_scope/build_sample_manifests.py`: `OUT_DIR` → `configs/sample_ids`
- docstring/string mentions in `sample_ids.py`, `test_final_scope.py` (string fields only, not loaded)
Then `rmdir configs/final_scope` if empty.
Tests: `compileall`; `test_final_scope`; a manifest-load check (`load_manifest(MANIFEST_DIR/gqa.json)`).
Rollback: `git mv configs/sample_ids configs/final_scope/sample_ids` + `git checkout -- <edited files>`.

## Group B — results/final_scope → results/{runs,tables,configs/dynamic_count}  (HIGH risk)

Moves:
- `git mv results/final_scope/tables results/tables`  (tracked)
- `git mv results/final_scope/dynamic_count_configs results/configs/dynamic_count`  (tracked)
- `mv results/final_scope/llava15 results/runs/llava15`  (ignored; plain mv)
- `mv results/final_scope/qwen25vl7b results/runs/qwen25vl7b`  (ignored; plain mv)
Constant updates (every one must flip together — a miss makes an audit read an empty dir):
- `dense_pilot.py`: `OUT_ROOT` → `results/runs`
- all `CFG_DIR = results/final_scope/dynamic_count_configs` (5 files) → `results/configs/dynamic_count`
- all `TABLES`/`TABLES_DIR = results/final_scope/tables` → `results/tables`
- the audit/compare scripts that use `ROOT = results/final_scope` for BOTH runs and tables must be
  **split** into `RUNS_ROOT = results/runs` (for `{model}/{dataset}`) and `TABLES_DIR = results/tables`
- shell `OUTROOT="results/final_scope"` (3 files) and `OUTDIR` (1) → `results/runs`
- hardcoded globs in `make_final_thesis_tables.py` (`results/final_scope/qwen25vl7b/...`) → `results/runs/...`
New: `results/{dense,static,dynamic_which,dynamic_count}/README.md` (index to `results/runs/` + `results/tables/`).
`.gitignore` update: negate the tracked new locations (`results/tables`, `results/configs`) so they stay tracked and the run trees stay ignored under the new path.
Tests: full suite — `validate_dynamic_{count,which}`, `audit_dynamic_{count,which}` — these read the
run JSONLs and re-verify shas, so green means the constant flips are all correct.
Rollback: reverse the four moves + `git checkout -- <edited files>` + revert `.gitignore`.

## Group C — scripts/final_scope → scripts/{dense,static,dynamic_which,dynamic_count,validation,tables,data}  (MEDIUM risk)

`git mv` per the role mapping (with renames). Scripts import only from `src`, so moving a script does
not break its own imports. Updates:
- the shell launchers' `python -m scripts.final_scope.run_*` module paths → new module paths
- the four method-folder wrappers (`dense/scripts/validate_*.sh` …) `python -m scripts.final_scope.*`
- the method `COMMANDS.md` files, `scripts/README.md`, root `README.md`
Backward compatibility: old `python -m scripts.final_scope.*` module paths are **not** preserved with
wrappers (all callers are updated; result JSONs record no module paths; only historical logs/tables
reference the old paths). `scripts/final_scope/README.md` (if the folder remains) documents the move.
Tests: `python -m scripts.validation.{validate,audit}_*` (new paths).
Rollback: reverse the `git mv`s + `git checkout -- <edited callers>`.

## Group D — src/final_scope → src/{common,dense,static,dynamic_which,dynamic_count}  (MEDIUM-HIGH risk)

`git mv` per mapping (common: sample_ids/output_writer/schema_validator/token_flops/test; method
cores: dense_pilot→dense/evaluate_dense, static_eval→static/evaluate_static, etc.). Update all
`from src.final_scope.X import …` references in `src/` and `scripts/` (measured: 134 sites) via a
mechanical sweep. **Compatibility shims**: `src/final_scope/*.py` become thin re-export modules
(`from src.common.sample_ids import *`) + a README stating the folder is a backward-compatibility
layer, so `python -m src.final_scope.test_final_scope` and any missed reference keep working.
Tests: `python -m src.common.test_evaluation_core` (new) AND `python -m src.final_scope.test_final_scope`
(shim) must both pass; both-env import smokes on the new `src.{method}.evaluate_*` paths.
Rollback: reverse the `git mv`s, drop the shims, `git checkout -- <edited files>`.

## Group E — docs restructure (LOCAL-ONLY; docs/ is git-ignored)

New clean set: `README, overview, protocol, dense, static, dynamic_which, dynamic_count,
final_results, reproducibility, limitations`. The current docs move (plain `mv`, not deleted) to
`archive/local_only/docs_replaced_by_clean_docs_20260705/`. Not force-added; `docs/` stays local.

## Group F — archive out-of-scope results/data + pycache  (LOW risk)

- `results/thesis_main/`, `results/paper_candidates/` (ignored) → `archive/legacy_results/…` (plain mv)
- `data/{llava_mix,pope,scienceqa}` (ignored) → `archive/legacy_datasets/data/…` (plain mv).
  **Not moved:** `data/{gqa,textvqa,vqav2}` (active).
- `__pycache__` dirs (not under archive/) → `archive/local_only/pycache_20260705/…` preserving paths.

## Files that stay in place / documented only

`logs/` (root, provenance); `data/{gqa,textvqa,vqav2}`; the frozen engines
`src/models/static/static.py`, `src/pruning/question_conditioned_selection/qwen_pruner.py`; the
method implementation packages `src/pruning/{dynamic_which,dynamic_which_ref,dynamic_count,static}`,
`src/metrics`, `src/analysis`, `src/data`, `src/utils`, `src/models` (already method/role shaped);
`results/{README,INDEX,legacy_index}.md`; sample-manifest and result JSON/JSONL **contents** and
**basenames** (never edited or renamed).

## Recommendation

Execute A → F in order, testing after each. B and D are the risk-bearing groups; both are reverted
rather than forced if the validation suite goes red. `logs/` remains in the root throughout.
