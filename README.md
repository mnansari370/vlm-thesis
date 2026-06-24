# Dynamic Question-Conditioned Visual Token Pruning for Efficient Vision-Language Models

*Master's thesis — finding: **Selection over Budget**.*

## Summary

This project studies how to reduce visual-token computation in Vision-Language Models (VLMs) while
preserving performance. A pruner makes two decisions: *which* tokens to keep (selection) and *how many*
per sample (a dynamic budget). My finding is that **question-conditioned token selection is the important
lever, not dynamic per-sample budget prediction** — *which* tokens you keep matters far more than *how
many*. I establish this with honest measurement: matched-FLOPs comparisons, reproduced dense baselines,
and an oracle-headroom estimate corrected for noise.

I do not claim to beat the state of the art; the deployable selector is not yet at SOTA quality; and the
gains come from the choice of signal and regime, not from training the backbone.

## Repository structure (method-based)

```
vlm-thesis/
  README.md  requirements.txt
  src/          all source code, organized by method:
    data/         dataset loaders (gqa, vqav2, docvqa, llava-mix)
    metrics/      scorers (official/GQA, M4C/TextVQA, POPE, DocVQA ANLS, ChartQA, …)
    models/       dense/ static/ dynamic_budget/ distillation/ elastic/
    pruning/      static/ dynamic_budget/ question_conditioned_selection/
    evaluation/   per task: gqa/ vqa/ textvqa/ docvqa/ pope/ scienceqa/
    analysis/     FLOPs, latency, oracle decomposition, figures
    utils/        config / seed / logger / io / checkpoint / device
  configs/      YAML configs by method (dense/ static/ dynamic_budget/ …)
  scripts/      runnable launchers (data/ training/ …)
  experiments/  experiment entry points (by method)
  results/      experiment results (git-ignored; see results/README.md + results/INDEX.md)
  docs/         thesis & paper planning documents (start at docs/README.md)
  data/         datasets (git-ignored, flat)
```

> The historical track folders (`GQA/`, `VQA_V2/`, `v2/`) and `outputs/` no longer exist at the top
> level — their code is now under `src/` by method and their results under `results/`. "v1/v2" survive
> only as historical labels inside the docs, not in the folder structure.

## Where things live

- **Code:** `src/` (by method) · **Configs:** `configs/` · **Launchers:** `scripts/` · **Experiments:** `experiments/`
- **Results:** `results/{thesis_main,appendix,paper_candidates,archived}/` (git-ignored; index in `results/INDEX.md`)
- **Thesis docs:** `docs/` — read `docs/THESIS_MASTER_PLAN.md` (story + chapter plan) and
  `docs/THESIS_EVIDENCE_LEDGER.md` (exact numbers, the source of truth).

## How to read the project

1. `docs/THESIS_MASTER_PLAN.md` — the thesis story, claim, datasets, and chapter plan.
2. `docs/THESIS_EVIDENCE_LEDGER.md` — every number, with its dataset/model/sample-size and evidence path.
3. `docs/PAPER_PUBLICATION_PLAN.md` — what a paper would need (it is possible, not yet ready).
4. `docs/REPOSITORY_CLEANUP_LOG.md` + `docs/FINAL_REPOSITORY_CLEANUP_REPORT.md` — how the repo was organized.

## The thesis narrative (one paragraph)

I write the thesis around a **selection-vs-budget decomposition**. The frozen low-resolution experiments
are the diagnostic foundation: the dense pipeline reproduces published accuracy, per-sample budget
headroom is small, and naive question-conditioned selection fails. The high-resolution / Qwen experiments
are the main result: a **mid-layer** question-conditioned selection signal is genuinely useful and
question-driven, while adaptive per-sample budget prediction stays weak. Conclusion: the dynamic,
question-dependent lever that pays off is **selection**, not the per-sample budget.

## Lightweight checks (no GPU/training)

```bash
python -m compileall src scripts          # syntax
python -c "import sys; sys.path.insert(0,'.'); import src.metrics, src.utils, src.analysis"  # imports
```

(Full evaluation/training is run via `python -m src.<...>` from the repo root; see `scripts/`.)

## Environment

```bash
conda create -n vlm_env python=3.11 -y && conda activate vlm_env
pip install -r requirements.txt
```

Backbones used across the project: LLaVA-1.5-7B, LLaVA-1.6-7B, Qwen-2.5-VL. The pinned
`torch` / `transformers` versions in `requirements.txt` are result-critical.
