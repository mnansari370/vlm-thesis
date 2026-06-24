# Dynamic Question-Conditioned Visual Token Pruning for Efficient Vision-Language Models

*Master's thesis — finding: **Selection over Budget**.*

## Summary

This project studies how to reduce visual-token computation in Vision-Language Models (VLMs) while
preserving performance. A VLM turns an image into many visual tokens and feeds them to a language model,
where self-attention cost grows with the number of tokens, so pruning visual tokens is the main
efficiency lever. A pruner makes two decisions: *which* tokens to keep (selection) and *how many* to keep
per sample (a dynamic budget).

My finding is that **question-conditioned token selection is the important lever, not dynamic per-sample
budget prediction.** In other words, *which* visual tokens are selected matters far more than *how many*
are selected per sample. I reach this through an honest measurement framework: I compare against tuned
static baselines at matched compute, reproduce published dense accuracy before making any pruning claim,
and correct the oracle-headroom estimates for noise.

I do not claim to beat the state of the art, the deployable selector is not yet at SOTA quality, and the
gains come from the choice of signal and regime — not from training the backbone.

## Current thesis status

- The thesis direction is **locked**: *Selection over Budget*.
- The evidence ledger is **locked** and is the single source of truth for every number.
- The planning documents have been **consolidated** into a small set under `docs/`.
- Thesis writing can start from **`docs/THESIS_MASTER_PLAN.md`**.
- Exact numbers must come only from **`docs/THESIS_EVIDENCE_LEDGER.md`**.
- A paper is **possible later, but not yet ready** — it needs full-validation reruns and stronger
  baselines (see `docs/PAPER_PUBLICATION_PLAN.md`).

## Folder map

> **Note on names:** `GQA/`, `VQA_V2/`, and `v2/` are **historical implementation folder names** from how
> the research developed. They do **not** reflect the final thesis organization. After the thesis draft I
> intend to migrate them into method-based folders (dense, static, dynamic budget, question-conditioned
> selection, distillation, evaluation, analysis). For now they are left in place so nothing breaks.

| Folder | What it holds |
|---|---|
| `docs/` | Clean thesis/paper planning documents (start here). |
| `GQA/` | *Historical:* the frozen low-resolution **diagnostic foundation** — dense reproduction, static pruning, oracle-band analysis, and the frozen question-conditioned selection probes. |
| `VQA_V2/` | *Historical:* the VQAv2 **dynamic-budget vs static-budget** experiments. |
| `v2/` | *Historical:* the high-resolution / Qwen-2.5-VL / LLaVA-1.6 work — selection, budget, FLOPs, latency, and the distilled student selector (the **main selection result**). |
| `outputs/` | Result files for the frozen diagnostic experiments (git-ignored). |
| `data/` | Local datasets (git-ignored; not part of the public repo). |
| `archive/` | Preserved old / failed / retired material (git-ignored). |
| `literature/` | Related-work notes. |
| `requirements.txt` | Python dependencies (result-critical pinned versions). |

## Active documents

Read the document that matches what you need:

- **`docs/THESIS_MASTER_PLAN.md`** — the thesis story, title, claim, contribution, dataset decisions, and
  the chapter-by-chapter writing plan. *Write the thesis from this.*
- **`docs/THESIS_EVIDENCE_LEDGER.md`** — the source of truth for every number, metric, dataset, model,
  sample size, evidence path, and safety label. *Take all numbers only from here.*
- **`docs/PAPER_PUBLICATION_PLAN.md`** — whether a paper is possible, what is already useful, and the
  exact extra work required.
- **`docs/REPOSITORY_CLEANUP_LOG.md`** — what cleanup has been done and what remains.

(`docs/README.md` indexes these; `docs/DOCS_CLEANUP_REPORT.md` records the docs consolidation.)

## The thesis narrative (one paragraph)

I write the thesis around a **selection-vs-budget decomposition**. The frozen low-resolution experiments
provide the diagnostic foundation: the dense pipeline reproduces published accuracy, the per-sample budget
headroom is small, and naive question-conditioned selection (frozen CLIP-space similarity and early-layer
attention) fails. The high-resolution / Qwen experiments provide the main result: a **mid-layer**
question-conditioned selection signal is genuinely useful and is driven by the question, while adaptive
per-sample budget prediction stays weak. This keeps the work aligned with *dynamic question-conditioned
visual-token pruning* while making the conclusion more precise: the dynamic, question-dependent lever that
pays off is **selection**, not the per-sample budget.

## What should NOT be used as a headline result

- Anything under `archive/failed_experiments/` (exploratory / low-value runs).
- The retired early-proxy classification-head code (now in `archive/retired_code/`).
- The elastic Stage-1 training (exploratory; the main results are on frozen models).
- The cross-model "generality" claim — narrative-only until it is recomputed and saved.
- Any paper claim that lacks full-validation numbers and strong baselines.

## Cleanup status

- Old planning docs consolidated; superseded docs archived under `archive/docs_consolidated_backup/`.
- Retired early-proxy code archived under `archive/retired_code/`.
- Generated Python caches (`__pycache__`) removed (regenerable).
- All results, evidence, and datasets **preserved**; the evidence ledger paths are intact.
- **No code migration yet** — the method-based folder restructuring is deliberately deferred until after
  the thesis draft.

## Next steps

- **A.** Start thesis writing from `docs/THESIS_MASTER_PLAN.md`.
- **B.** *(Optional, cheap)* recompute the cross-model generality evaluation so that claim becomes safe to
  write.
- **C.** *(Later)* create the method-based folder structure and migrate `GQA/`, `VQA_V2/`, `v2/`, and the
  result folders — only after the thesis draft.
- **D.** *(Later)* run the paper experiments and stronger baselines.

## Environment

```bash
conda create -n vlm_env python=3.11 -y
conda activate vlm_env
pip install -r requirements.txt
```

Backbones used across the project: LLaVA-1.5-7B, LLaVA-1.6-7B, and Qwen-2.5-VL. The pinned
`torch` / `transformers` versions in `requirements.txt` are result-critical.
