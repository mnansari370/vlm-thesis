# `v2/distill/` — Cheap Question-Conditioned Selector (thesis Part B / the method)

**Goal.** Turn the project's strongest *measurement* into a deployable *method*. Parts I–III showed:
question-conditioned **selection dominates** (mid-layer L16 attention: +60pp over blind on Qwen2.5-VL
DocVQA, 95% of dense at 88% FLOP-cut) but that selector is a **teacher** — it needs a full LLM forward,
so it is not deployable. This module **distills** that teacher into a **cheap front-end student** that
scores visual tokens from pre-LLM features only (ViT embeds + question embeds), adding negligible FLOPs
and **no decoder forward**.

This is a **no-lose pilot**:
- **PASS** (student recovers > 50% of the teacher's gain over blind at K=128) → build the full method;
  a deployable, question-conditioned selector competitive with VisionSelector / FlashVLM / FEATHER.
- **FAIL** → a publishable *characterization*: "the question-conditioned selection signal is irreducibly
  mid-LLM" — directly in the project's "are we solving the right problem?" lineage. (Then try the shallow
  proxy: first 2–3 LLM layers, ~10× cheaper than 16, before declaring the negative.)

## Isolation
Adds files only; imports `v2.qwen.qwen_pruner` and `v2.shared.docvqa_score` **read-only**. Edits nothing
in `GQA/`, `VQA_V2/`, `VQA_V2_early_proxy/`, or existing `v2/` files. Runs in `qwen_env`.

## Leakage guard (pilot)
Local DocVQA has only `validation`+`test` (the hub `train` split is not downloaded). The pilot therefore
trains on a deterministic slice `validation[0:N]` and gates on a **disjoint** tail `validation[N:N+n]`;
`eval_gate.py` **hard-asserts** the windows don't overlap. For the paper, download the real DocVQA
`train` split and pass `--split train` — the scripts are otherwise unchanged.

## Pipeline (run in `qwen_env`, from repo root)

```bash
# 1) cache the L16 teacher importance map for the train slice (~36 min, 3000 samples, GPU)
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen_env/bin/python -m v2.distill.cache_teacher \
    --split validation --start 0 --end 3000 --out v2/outputs/distill/teacher_docvqa_train.pt

# 2) distill the cheap student (CPU-cheap module; recomputes ViT features on the fly)
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen_env/bin/python -m v2.distill.train_student \
    --teacher v2/outputs/distill/teacher_docvqa_train.pt \
    --out v2/outputs/distill/student.pt --epochs 8

# 3) the GO/NO-GO gate on a held-out disjoint window (val[3000:3400])
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen_env/bin/python -m v2.distill.eval_gate \
    --student v2/outputs/distill/student.pt --n 400 --K 128
```

## Files
- `cache_teacher.py`  — Step 1: cache L16 teacher `qscores` per sample (the expensive signal).
- `student_selector.py` — `CheapQCSelector`: small cross-attention head (~8M params), pre-LLM inputs only.
- `train_student.py`  — Step 2: distill (listwise KL + top-K BCE); logs top-128 teacher-overlap.
- `eval_gate.py`      — Step 3: dense / blind / teacher / student ANLS at K=128 → recovery %, PASS/FAIL.

## Status
Pipeline smoke-verified end-to-end (model load, dataset, teacher extraction, training gradient path,
gate generation + leakage assert all run). Real pilot results pending the full cache+train+gate run.
