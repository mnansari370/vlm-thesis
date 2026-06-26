# Method-Based Migration Plan (Phase 3E)

> **Plan only — nothing moved/renamed/deleted/changed.** The detailed, batch-executable plan to turn the
> historical `GQA/ VQA_V2/ v2/ outputs/` tree into a clean method-based `src/ … results/` layout.
> Authorities: `THESIS_MASTER_PLAN.md`, `THESIS_EVIDENCE_LEDGER.md`, `REPOSITORY_CLEANUP_LOG.md`.
> Created 2026-06-24. **Recommendation: execute only after the thesis draft exists** (high risk, low
> thesis value — the thesis is written from `docs/` + results, not from code layout).

---

## 0. Scale & honest framing

| Surface | Count | Impact |
|---|---|---|
| Python package imports (`from {GQA,VQA_V2,v2}…`) | **150** (GQA 49, VQA_V2 32, v2 69) | every one must be rewritten |
| Files using `sys.path.insert(...)` | **61** | remove during move (replace with a real package) |
| YAML configs | **27** (VQA_V2 26, v2 1) | path/module keys to update |
| Hardcoded **output** paths in code | **77** (GQA→`outputs/` 14, `VQA_V2/outputs` 21, `v2/outputs` 42) + 11 `make_output_dir("outputs",…)` | update **with** the results move |
| Hardcoded **`data/`** paths | **270** (46 py, 210 yaml, 14 sh) | **DO NOT CHANGE** — `data/` stays flat |
| Shell scripts | **38** (1 uses `python -m`) | update module/output paths inside |
| Ledger evidence paths (roots) | `v2/outputs` (14), `outputs/results_frozen` (4), `v2/outputs/distill` (4), `v2/qwen` (3), `v2/distill` (2), `outputs` (2), `VQA_V2/outputs/dynamic_150k_clsonly` (1) | update in lockstep with each move |

This is a ~150-import refactor that also rewrites the public `main` branch. It is genuinely large.
**Two profiles** (pick at execution time):
- **Profile A — full `src/` migration** (this document's target): clean, but the full reimport.
- **Profile B — lighter reframe** (fallback): keep code in place; add `results/INDEX.md` + a top-level
  method map; skip the import rewrite. Achieves "reads like one thesis project" at ~10% of the risk.

---

## 1. Current-state audit

| Folder | Contains | Thesis role | Method category | Class | Safe to move? | What breaks if moved |
|---|---|---|---|---|---|---|
| `GQA/dense/` | dense K=576 eval + `GQATestdevDataset` | diagnostic foundation | dense baseline + evaluation/data | main | yes (with reimport) | imports in static/dynamic runners that reuse the dataset |
| `GQA/shared/static.py` | `StaticPrunedLlava` (physical token removal) | static frontier | static pruning (model) | main | yes | all static/probe runners import it |
| `GQA/static/` | static runners + VisionZip + CLIP/LM probes | foundation | static pruning + question-cond. selection (frozen, negative) | main | yes | self-imports static.py + shared scorers |
| `GQA/dynamic/` | confidence cascade + sweep | foundation | dynamic budget | main | yes | reads `outputs/*` run dirs (globs) |
| `GQA/eval_runners/` | TextVQA / POPE / SQA eval | foundation | evaluation | main/appendix | yes | imports static.py + scorers |
| `GQA/shared/{official_score,m4c_evaluator,textvqa_score,pope_score,eval_pope_official,metrics}.py` | scorers | all | metrics | main | yes | imported widely |
| `GQA/shared/flops.py` | FastV Eq.5 FLOPs | efficiency | analysis | main | yes | imported by cascade_sweep etc. |
| `GQA/shared/dataset.py` | GQA loaders | foundation | data | main | yes | — |
| `GQA/shared/utils/` | config/seed/logger/io/checkpoint/device | infra | utils | main | yes | imported everywhere (one of 3 identical copies) |
| `VQA_V2/dense/`, `VQA_V2/static/` | trained dense/static answer-head wrappers | foundation | dense / static | appendix | yes | configs reference them |
| `VQA_V2/dynamic/` | `BudgetController` + scorer/selector + `train_dynamic.py` | **L2 (budget ties static)** | dynamic budget + training | main | yes | configs + generate_and_score |
| `VQA_V2/shared/evaluation/` | generation eval, instance_headroom, cascade_*, per_type, flops, make_figures | L2 analysis | evaluation + analysis | main/appendix | yes | reads `VQA_V2/outputs/static_k*` (globs) |
| `VQA_V2/shared/datasets/` | VQAv2 loader/answers | foundation | data | main | yes | — |
| `VQA_V2/shared/scripts/` | cache_features, vocab builders, gates | infra | scripts/data | appendix | yes | `.sh` callers |
| `VQA_V2/shared/training/cached/` | cached MLP trainer | exploratory | training | appendix | yes | — |
| `VQA_V2/shared/experiments/*.sh` | HPC/run scripts | infra | scripts | appendix | yes | hardcode `VQA_V2/outputs/...` |
| `VQA_V2/shared/utils/` | (identical to GQA utils) | infra | utils | — | yes | dedup target |
| `v2/qwen/qwen_pruner.py` | Qwen prune-before-LLM harness + mid-layer selector | **main (L5–L9)** | question-conditioned selection (model) | main | yes | 10 importers |
| `v2/qwen/qwen_{kcurve,layer_sweep,control,oracle*,budget_*,flops,…}.py` | the selection/budget experiments | **main (L5–L10,E)** | question-cond. selection + dynamic budget + analysis | main + paper | yes | read `v2/outputs/qwen_*` |
| `v2/eval/` | LLaVA-1.6 high-res evals + latency + layer sweep | main/appendix | evaluation + analysis | main/appendix | yes | read `v2/outputs/eval_*` |
| `v2/distill/` | `CheapQCSelector` + train/cache/eval | **main (L11,L12)** | distillation + training + evaluation | main | yes | `.pt`/`.json` paths |
| `v2/analysis/` | flops_frontier, oracle_decomposition, figures, verify_dense | efficiency/method | analysis | main | yes | — |
| `v2/data_setup/` | LLaVA-mix loader + downloaders | infra | data + scripts | appendix | yes | `data/llava_mix` (stays) |
| `v2/model/elastic_wrapper.py` | elastic LoRA backbone | exploratory | models/elastic (auxiliary) | appendix | yes | Stage-1 training only |
| `v2/shared/` | scorers (+docvqa/chartqa) + flops + utils | all | metrics/analysis/utils | main | yes | 3rd identical utils copy |
| `v2/training/` | `train_stage1.py` + tests | exploratory | training | appendix | yes | reads mix_dataset |
| `outputs/` | GQA-track results + `results_frozen/` | **L1,L3,L4** | results | main + appendix | **only with code** | `cascade_sweep.py`/analysis globs |
| `VQA_V2/outputs/` | VQAv2 results | **L2** | results | main + appendix | **only with code** | `instance_headroom.py`/`cascade_analyze.py` |
| `v2/outputs/` | v2 results + kept ckpts + paper raw | **L5–L12,E** | results + paper_candidates | main + paper | **only with code** | `qwen_flops.py` default + eval scripts |

---

## 2. Import & path dependency audit (how each is updated)

| Dependency type | Current form | Where | Why it exists | Update during migration |
|---|---|---|---|---|
| Package imports | `from GQA.shared.static import …`, `from VQA_V2.dynamic.…`, `from v2.qwen.qwen_pruner import …` (150) | all `.py` | track-named packages | rewrite to `from src.<method>.<module>` per §4 mapping (mechanical sed per batch + import smoke) |
| `sys.path.insert` | `sys.path.insert(0, …"..","..")` (61 files) | top of runners | makes `python file.py` work without install | **remove**; install the package once (`pip install -e .` with a `pyproject.toml`, or rely on `python -m src.…`) |
| Scorers/utils duplication | 3× identical `shared/utils`, near-identical `shared/flops.py` + scorers | GQA/VQA_V2/v2 | copied per track | **collapse to one** `src/utils`, `src/metrics`, `src/analysis`; update all importers |
| `data/` paths | `data/gqa/...`, `data/vqav2/...`, `data/llava_mix/...` (270) | py/yaml/sh | dataset locations | **NO CHANGE** — `data/` stays flat (documented decision) |
| Output paths | `outputs/...`, `VQA_V2/outputs/...`, `v2/outputs/...` (77) + `make_output_dir("outputs",…)` (11) | py | where results are written/read | repoint to `results/...` **in the same batch as the results move**; update `make_output_dir` base |
| YAML config paths | module names + `data/` + output dirs (27 files) | configs | drive trainers/evals | move to `configs/<method>/`; update output-dir keys; keep `data/` keys |
| Shell scripts | hardcoded module + `VQA_V2/outputs/...` (38 files) | experiments | HPC/run helpers | move to `experiments/<method>/`; update module + output paths; `bash -n` check |
| JSON result-path assumptions | analysis scripts glob `outputs/testdev_*`, `VQA_V2/outputs/static_k*`, `v2/outputs/qwen_*` | analysis py | read saved results | update globs to `results/...` with the results move |
| Ledger evidence paths | exact paths in `THESIS_EVIDENCE_LEDGER.md` (roots in §0) | docs | source of truth | edit ledger `File:` paths to new `results/...` **in lockstep**; re-run the path-existence check |
| Docs run-commands | `python -m …` | active docs | reproduction | **0 in active docs** (already clean); only the archived docs hold old commands (leave) |

---

## 3. Proposed final clean structure (adapted)

```text
vlm-thesis/
  README.md  requirements.txt  pyproject.toml        # NEW: makes `src` an installable package (kills sys.path hacks)
  docs/            (the 7 consolidated docs, unchanged)
  src/
    data/           # GQA + VQAv2 loaders, LLaVA-mix loader
    metrics/        # official_score, m4c_evaluator, textvqa/pope/docvqa/chartqa scorers
    models/
      backbone/     # frozen LLaVA-1.5/1.6 + Qwen wrappers (StaticPrunedLlava, QwenPruner, HighResPruner)
      dense/  static/  dynamic_budget/  distillation/  elastic/
    pruning/
      static/  dynamic_budget/  question_conditioned_selection/
    evaluation/
      gqa/  vqa/  textvqa/  docvqa/  chartqa/  pope/  scienceqa/
    analysis/       # flops, latency, oracle_decomposition, figures
    training/       # train_dynamic, train_student, train_stage1, cached trainer
    utils/          # config/seed/logger/io/checkpoint/device (ONE copy)
  configs/{dense,static,dynamic_budget,question_conditioned_selection,distillation,evaluation}/
  scripts/{data,evaluation,analysis,training,migration}/
  experiments/{diagnostic_foundation,dense_baselines,static_pruning,dynamic_budget,
               question_conditioned_selection,distillation,ablations}/
  results/{thesis_main,appendix,paper_candidates,archived}/
  literature/  data/  archive/
```

Notes: `question_conditioned_selection` holds both the **frozen negatives** (GQA CLIP/LM probes) and the
**mid-layer positive** (Qwen `textattn` + the high-res selector) — the v1→v2 story lives in `docs/`, the
code lives by method. `models/elastic/` is auxiliary (exploratory Stage-1).

---

## 4. File → destination mapping (per method)

| Destination | Source files (current) |
|---|---|
| `src/utils/` | `GQA/shared/utils/*` (canonical); delete the VQA_V2 & v2 duplicates after redirecting imports |
| `src/metrics/` | `GQA/shared/{official_score,m4c_evaluator,textvqa_score,pope_score,eval_pope_official,metrics}.py`; `v2/shared/{docvqa_score,chartqa_score}.py` (+ dedup v2/VQA_V2 scorer copies) |
| `src/analysis/` | `GQA/shared/flops.py` (canonical); `v2/qwen/qwen_flops.py`, `v2/analysis/{flops_frontier,oracle_decomposition,make_figures,verify_dense}.py`, `v2/qwen/make_qwen_figures.py`, `VQA_V2/shared/evaluation/{flops,make_figures}.py` |
| `src/data/` | `GQA/shared/dataset.py`, `VQA_V2/shared/datasets/*`, `v2/data_setup/mix_dataset.py` |
| `src/models/backbone/` + `src/models/static/` + `src/pruning/static/` | `GQA/shared/static.py`, `GQA/static/visionzip.py`, `VQA_V2/static/*` |
| `src/pruning/question_conditioned_selection/` | `GQA/static/{clip_select,question_cond}.py`; `v2/qwen/qwen_pruner.py` (the mid-layer selector); high-res selector inside `v2/eval/evaluate_textvqa_highres_kcurve.py` |
| `src/models/dynamic_budget/` + `src/pruning/dynamic_budget/` | `VQA_V2/dynamic/{budget_controller,token_scorer,token_selector,llava_wrapper}.py`; `GQA/dynamic/{cascade_sweep,run_speculative_testdev,run_pope_speculative}.py`; `v2/qwen/qwen_budget_*.py`, `v2/qwen/qwen_oracle*.py` |
| `src/models/distillation/` | `v2/distill/{student_selector,docvqa_data}.py` |
| `src/models/elastic/` | `v2/model/elastic_wrapper.py` (+ its tests) |
| `src/training/` | `VQA_V2/dynamic/train_dynamic.py`, `VQA_V2/shared/training/cached/train_cached.py`, `v2/distill/{train_student,cache_teacher}.py`, `v2/training/train_stage1.py` |
| `src/evaluation/gqa/` | `GQA/dense/run_dense_testdev.py`, `GQA/static/run_static*.py`, `GQA/static/run_visionzip_testdev.py` |
| `src/evaluation/{textvqa,pope,scienceqa}/` | `GQA/eval_runners/run_{textvqa,pope,sqa}.py` |
| `src/evaluation/vqa/` | `VQA_V2/shared/evaluation/{generate_and_score,per_type_accuracy,instance_headroom,cascade_pass,cascade_analyze}.py` |
| `src/evaluation/{docvqa,chartqa}/` | `v2/qwen/{qwen_kcurve,qwen_control,qwen_layer_sweep,qwen25_dense_eval}.py`, `v2/eval/evaluate_*highres*`, `v2/distill/{eval_gate,eval_control}.py` |
| `configs/<method>/` | `VQA_V2/{dense,dynamic}/*.yaml`, `VQA_V2/static/*.yaml` (21), `v2/configs/stage1_elastic.yaml` |
| `scripts/data/` | `VQA_V2/shared/scripts/*`, `v2/data_setup/download_*.py` |
| `experiments/<method>/` | all `*.sh` (`VQA_V2/shared/experiments/*`, `v2/training/*.sh`, `v2/distill/*.sh`) |
| `results/thesis_main/` + `appendix/` + `paper_candidates/` | `outputs/*`, `VQA_V2/outputs/*`, `v2/outputs/*` per `THESIS_EVIDENCE_LEDGER.md` class (move **with** the code that reads them) |

---

## 5. Batch execution order (dependency-safe; one commit per batch)

0. **Prep:** add `pyproject.toml` (package `src`), create the empty `src/ configs/ scripts/ experiments/
   results/` skeleton. No moves yet. Verify `pip install -e .` works.
1. **`src/utils/`** (leaf; everything imports it) → redirect all `*.shared.utils` imports → remove the 2
   duplicate utils trees.
2. **`src/metrics/`** (scorers; stdlib-only) → redirect importers.
3. **`src/analysis/`** (flops/figures) → redirect; reproduce `qwen_flops_summary.json` as the check.
4. **`src/data/`** (loaders) → redirect; load 2 samples as the check.
5. **`src/models/` + `src/pruning/`** per method (backbone/static → dynamic_budget → question_conditioned
   → distillation → elastic) → redirect; `keep-all==stock` + `static K=576==dense` smokes.
6. **`src/training/` + `src/evaluation/`** → redirect; tiny `--max-samples 2` eval smokes.
7. **`configs/` + `experiments/`** → update output-dir keys + module paths; `bash -n` on scripts.
8. **Results move** (`outputs/* → results/…`) **+ ledger path edits**, atomic per result group; then run
   the analysis scripts that glob results (e.g. `cascade_sweep`, `instance_headroom`) to confirm they
   still find their inputs; re-run the ledger path-existence check (22/22).
9. **Docs + README** final path refresh; `sys.path.insert` removal sweep (should be 0 left).
10. **Full smoke suite** (below) + update `MIGRATION_REPORT`/`REPOSITORY_CLEANUP_LOG`.

After each batch: `git mv` (tracked code), `python -c "import …"` across the moved modules, commit with a
clear message. Stop on any failure and fix before the next batch.

## 6. Smoke tests (reuse what already exists)
- **Import-all:** a script that imports every `src.*` module.
- **`python -m src.analysis.qwen_flops`** reproduces `qwen_flops_summary.json`.
- **`src.pruning.dynamic_budget.cascade_sweep`** reproduces `cascade_sweep.json` byte-identically.
- **static K=576 == dense** (`run_static … --method none --keep_k 576 --max-samples 2`).
- **keep-all == stock** (Qwen pruner self-test; high-res pruner `--self-test`).
- **distill gate smoke** (n=2).
- **ledger path-existence check** (every `File:` resolves).

## 7. Results + ledger update procedure (the correctness-critical part)
For each result group moved in Batch 8: (a) `git`-or-`mv` the dir to its `results/<class>/` home; (b) edit
the matching script glob/default; (c) edit the `File:` path in `THESIS_EVIDENCE_LEDGER.md`; (d) run the
reader script to confirm it loads; (e) commit. **Never move a result without updating its ledger row in
the same commit.** Keep a `results_move_manifest.md` (old→new) for rollback.

## 8. Rollback & safety
- **Code (tracked):** every batch is a commit → `git revert`/`git reset --hard <prev>` restores.
- **Results (git-ignored):** rely on the verified backup `vlm-thesis-evidence-20260624-111205/` + the
  per-batch `results_move_manifest.md`; reverse with `mv`.
- **Pre-flight:** refresh the backup first; confirm clean `git status`; do it on a branch
  (`git checkout -b method-migration`).
- **Never delete** during migration — only move; deletion stays a separate, post-defense step.

## 9. Risk & recommendation
- **Risk: high** (150 imports, 61 sys.path files, 77 output paths, public branch). Effort: 1–2 focused
  days with testing.
- **Thesis value: low** (writing doesn't need it). **Paper value: medium** (a clean public repo helps).
- **Recommendation: do NOT run Phase 3E before the thesis draft.** When ready, prefer **Profile A** if a
  clean public/paper repo is the goal; otherwise **Profile B** (results/INDEX + method map, no import
  rewrite) gets 80% of the readability for ~10% of the risk.

## 10. Preconditions before executing
1. Thesis draft complete (or explicit approval to migrate first).
2. Fresh backup taken.
3. Clean `git status` (commit the current `.gitignore`/README/early-proxy changes first).
4. Decision: Profile A (full `src/`) vs Profile B (lighter reframe).
5. Decision: keep `data/` flat (recommended) — confirmed.
6. **`.gitignore` GOTCHA (must fix before moving `src/data`):** the current rule `data/` is **unanchored**,
   so it ignores `data/` at *any* depth — including `src/data/` and `scripts/data/` (confirmed during the
   3E-1 scaffold: `src/data/.gitkeep` and `scripts/data/.gitkeep` were silently ignored). Before Phase 3E
   moves the data loaders into `src/data/`, change the rule to anchored `/data/` (top-level only). Same
   care for any other unanchored dir names. Otherwise migrated `src/data/*` files would be untracked.

---

## Phase status
- Phases 1, 2A, 3A, 3B, 3C, 3D: done. · **Phase 3E: STARTED** — Batch 1 (utils → `src/utils/`) done on `method-migration`; Batches 2–9 pending (see `PHASE3E_REAL_MIGRATION_MAP.md`).
