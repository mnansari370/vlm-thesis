# Dynamic Question-Conditioned Visual Token Pruning for Efficient Vision-Language Models

**Author:** Mo Nafees
**Supervisor:** Decebal Constantin Mocanu
**Advisor:** Boqian Wu

> Master's thesis repository. All experiments are complete and validated; every number in this
> README comes from a committed table under [`results/tables/`](results/tables/), produced by runs
> that passed an automatic fairness gate.

## Overview

Vision-language models such as LLaVA-1.5 and Qwen2.5-VL spend most of their language-model compute
on visual tokens. A single image occupies hundreds to over a thousand positions in the sequence,
while the answer to a question usually depends on a small part of the image. Pruning visual tokens
is therefore a natural way to make inference cheaper.

This thesis studies visual token pruning for efficient vision-language models. The goal is to reduce
visual-token computation while preserving answer quality across visual question answering tasks.
Just as important, the trade is measured honestly, with frozen backbones, locked evaluation samples,
identical prompts and decoding for every method, and compute accounting that charges every executed
forward pass.

## Research Questions

This thesis is organized around three research questions:

- **RQ1.** Can vision language models reduce visual token computation without losing too much
  task accuracy?
- **RQ2.** Can dynamic, question conditioned pruning improve the accuracy and efficiency trade
  off compared with dense inference and a strong static pruning baseline?
- **RQ3.** How did AI assisted development tools support the implementation, experimentation,
  and verification process of this thesis?

Any pruning method quietly makes two separate decisions, and this thesis evaluates them as two
separate axes:

- **Dynamic-WHICH** asks *which* visual tokens should be kept at a fixed budget. Does letting the
  question choose the tokens beat choosing them from the image alone?
- **Dynamic-COUNT** asks *how many* visual tokens should be kept for each sample. Does an adaptive
  per-sample budget beat the best single fixed budget at the same average compute?

## Thesis Scope

This repository focuses on two models and four datasets only:

| | |
|---|---|
| **Models** (both frozen, no backbone training) | LLaVA-1.5-7B (fixed 576 visual tokens at 336 px) · Qwen2.5-VL-7B (native per-image token count) |
| **Datasets** | GQA (testdev balanced, 12,578, exact match) · VQAv2 (25,000 validation questions, stratified by official answer type, seed 42, consensus) · TextVQA (val, 5,000, M4C soft accuracy, OCR in the prompt) · DocVQA (val, 5,349, ANLS) |
| **Budgets** | 15 / 25 / 35 / 50 / 75 % of each sample's dense visual-token count, plus dense (100 %) |
| **Methods** | Dense baseline · Static pruning · Dynamic-WHICH · Dynamic-COUNT (DC-D and DC-C) |

Fairness is enforced mechanically, not by convention: every method and both models score the exact
same sha256-locked sample lists (`configs/sample_ids/`), decode greedily at batch size 1 with
identical prompts, save per-sample predictions plus an aggregate record, and compute analytical
prefill FLOPs per sample before averaging. Token reduction and FLOP reduction are always reported
separately.

## Models and Datasets

LLaVA-1.5-7B is the fixed-grid case: every image costs exactly 576 tokens, so budgets map to exact
counts (86 / 144 / 202 / 288 / 432 / 576). Qwen2.5-VL-7B is the native-resolution case: its token
count varies per image (about 359 on GQA and VQAv2, about 964 on TextVQA, about 1,229 on DocVQA),
so budgets are applied per sample as a fraction of that image's own dense count. The four datasets
span scene reasoning (GQA), broad natural-image questions (VQAv2), scene text (TextVQA), and dense
document pages (DocVQA). Together they cover regimes where pruning is easy and regimes where it is
punishing.

## Method Summary

### Dense Baseline

Keeps all visual tokens. It is the full-compute reference: the accuracy ceiling that every pruning
result trades against, the reproduction anchor that shows the harness is trustworthy (the LLaVA
numbers track the published references closely), and the source of each sample's token counts.

| Model | GQA | VQAv2 | TextVQA | DocVQA |
|---|--:|--:|--:|--:|
| LLaVA-1.5-7B | 61.42 | 77.33 | 57.65 | 21.53 |
| Qwen2.5-VL-7B | 60.96 | 84.27 | 81.06 | 94.76 |

The low LLaVA DocVQA score is expected and informative: 576 low-resolution tokens cannot read a
dense document page.

### Static Pruning

Applies fixed-budget, image-based pruning; the question is never consulted. LLaVA ranks patches by
CLS attention in the vision encoder, and Qwen ranks merged visual embeddings by activation norm.
Tokens are physically removed before the language model, so the savings are real. Static is the
strong baseline every dynamic method must beat: at the 75 % budget it sits within 0.02 to 2.63
points of dense on every model-dataset cell while removing roughly a quarter of the FLOPs, and on
GQA and VQAv2 even the 15 % budget loses only about 4 to 8 points. It collapses only where
information is dense and spatial: Qwen TextVQA falls from 81.06 to 53.06 and Qwen DocVQA from 94.76
to 29.90 at the tightest budget.

### Dynamic-WHICH

Question-conditioned token selection at the same fixed budgets as static. The `textsim` selector
scores every visual token by its maximum cosine similarity to the question-token embeddings, keeps
the top K, and restores the original spatial order. The selector is training-free and needs no
extra language-model forward pass for scoring. Because the budget is identical to static per sample, the comparison
isolates the selection decision. The main success is Qwen2.5-VL-7B on TextVQA, confirmed by an
independent clean-room reimplementation that reproduces the predictions exactly.

### Dynamic-COUNT

Adaptive per-sample token budget, with the selector held fixed. A cheap probe pass records the
answer and 27 confidence, saliency, and question signals; a controller fitted on the first 20 % of
each sample list then decides, per sample, whether the probe budget sufficed or one second pass at a
larger budget is needed. Two variants: **DC-D**, a discrete cascade between fixed budget anchors,
and **DC-C**, the main method, which predicts a continuous per-sample integer token count. The
accounting is honest: an escalated sample pays for both passes. The comparison target is the
static accuracy-versus-FLOPs curve at matched average compute. The result is mostly negative or
near-tie, and that outcome is central to the thesis rather than a footnote.

## Repository Structure

```
dense/              method-facing entry point for the dense baseline
static/             method-facing entry point for static pruning
dynamic_which/      method-facing entry point for Dynamic-WHICH
dynamic_count/      method-facing entry point for Dynamic-COUNT
src/                implementation code grouped by role and method
scripts/            runnable scripts grouped by method, validation, and tables
results/            saved runs, result tables, and Dynamic-COUNT configs
configs/            locked sample manifests
data/               active datasets used by the final thesis experiments
logs/               run logs kept for provenance and debugging
archive/            retired/out-of-scope material preserved for traceability
```

Each method folder contains a `README.md` (what and why), `CODE_MAP.md` (exact implementation
files), `COMMANDS.md` (safe CPU validation, table generation, and GPU rerun commands), `RESULTS.md`
(the real numbers), and a safe wrapper script. The shared evaluation core (sample manifests,
output schema, fairness gate, FLOP accounting) lives in `src/common/`, and the per-method runner
cores live in `src/dense/`, `src/static/`, `src/dynamic_which/`, and `src/dynamic_count/`.

## Results Summary

The complete matrix: 2 models × 4 datasets, all budgets, every run on the full locked sample list
with its fairness gate passed. WHICH deltas are against static at the same budget; DC deltas are
against the static accuracy-compute curve at matched average FLOPs.

| Model | Dataset | Dense | Best static | Best WHICH Δstatic | Best DC-D Δcurve | Best DC-C Δcurve | Verdict |
|---|---|--:|---|--:|--:|--:|---|
| LLaVA-1.5 | GQA | 61.42 | 61.53 (p75) | −0.73 | −0.10 | −1.26 | static stands |
| LLaVA-1.5 | VQAv2 | 77.33 | 77.31 (p75) | −1.59 | +0.06 | −0.25 | static stands |
| LLaVA-1.5 | TextVQA | 57.65 | 57.39 (p75) | −3.91 | **+0.75** | +0.48 | DC-D beats static |
| LLaVA-1.5 | DocVQA | 21.53 | 21.36 (p75) | −3.97 | −0.89 | −0.54 | static stands |
| Qwen2.5-VL | GQA | 60.96 | 60.85 (p75) | −0.68 | −0.03 | −0.19 | static stands |
| Qwen2.5-VL | VQAv2 | 84.27 | 83.66 (p75) | −0.06 | +0.21 | −0.84 | static stands |
| Qwen2.5-VL | TextVQA | 81.06 | 78.43 (p75) | **+8.36** | +0.02 | −7.27 | **WHICH beats static** |
| Qwen2.5-VL | DocVQA | 94.76 | 93.98 (p75) | +3.19 | −4.61 | −11.07 | see caveat below |

The headline positive result, Qwen2.5-VL-7B on TextVQA over the full 5,000 validation questions:

| Budget | Dynamic-WHICH | Static | Dense | Gain vs static | FLOP reduction |
|--:|--:|--:|--:|--:|--:|
| 15 % | 60.56 | 53.06 | 81.06 | **+7.50** | 78.7 % |
| 25 % | 67.53 | 59.17 | 81.06 | **+8.36** | 69.6 % |
| 35 % | 71.36 | 63.57 | 81.06 | **+7.79** | 60.6 % |
| 50 % | 76.08 | 70.14 | 81.06 | **+5.94** | 46.8 % |
| 75 % | 79.80 | 78.43 | 81.06 | **+1.37** | 23.6 % |

The gain is largest exactly where the budget is tightest, and the clean-room reimplementation
reproduces the selector prediction-for-prediction (200/200 exact matches, zero score difference, at
every budget).

**Qwen DocVQA caveat.** Dynamic-WHICH shows a localized low-budget improvement there (+3.19 at the
15 % budget: 33.09 vs static 29.90), but it is not the overall best method for that dataset. The
win occurs where both methods have already collapsed, while higher static budgets and dense remain
far stronger in absolute accuracy (static p75 = 93.98, dense = 94.76).

## Key Findings

1. **Static pruning is a strong and reliable baseline.** Near-dense at the 75 % budget on every
   cell, graceful on scene-centric tasks even at 15 %. Much of the apparent benefit of dynamic
   pruning in the literature disappears against this floor.
2. **Dynamic-WHICH is not universal.** Across the 40-cell matrix it records 6 wins, 2 near-ties, and
   32 losses against static at the same budget; all 20 LLaVA cells are negative.
3. **Where it works, it works well.** On Qwen2.5-VL × TextVQA, question-conditioned selection gains
   +7.50 to +8.36 points at tight budgets, a validated and reproducible improvement. The win
   requires localized, question-addressable evidence and language-aligned visual features.
4. **Dynamic-COUNT does not consistently beat the static curve.** DC-D achieves one small win
   (LLaVA TextVQA, +0.75) and otherwise near-ties or loses; DC-C gives no clear win overall (best
   +0.48, a near-tie; the steep-curve cells lose by up to −11.07). Stacking COUNT on top of the
   successful WHICH selector adds no further increment.
5. **The oracle headroom is real but not recovered.** A perfect per-sample budget router would beat
   the best fixed budget by +2.09 to +6.25 points while removing 61 to 83 % of visual tokens, yet
   the real confidence-based controllers do not reliably recover it under honest two-pass accounting.
6. **Negative results are included by design.** All eight model-dataset cells are reported for every
   method, on identical samples, with the gate verdict stored in every result file. The negatives
   are as load-bearing for the thesis as the single strong positive.

## Reproducibility

The completed results can be re-verified on CPU in minutes, with no model and no GPU:

```bash
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
python -m scripts.validation.validate_dynamic_count
python -m scripts.validation.validate_dynamic_which
python -m scripts.validation.audit_dynamic_count
python -m scripts.validation.audit_dynamic_which
```

Expected output: all self-checks pass, both validators report `ALL_*_VALID=True`, the WHICH audit
reports 40/40 cells complete, and the COUNT audit reports zero probe-reproduction failures.

Full GPU reruns are expensive (two pinned conda environments, two GPUs, many hours) and are not
needed unless reproducing the experiments from scratch. Each method folder (`dense/`, `static/`,
`dynamic_which/`, `dynamic_count/`) documents its rerun commands in `COMMANDS.md`, and all launchers
skip existing results rather than overwrite them.

## Limitations

- Two models and four VQA-style datasets only; conclusions beyond this grid are not claimed.
- No training-based learned selector in the final thesis scope; all methods are training-free on
  frozen backbones.
- Dynamic-WHICH does not generalize across all tasks; it is a regime-specific tool.
- Dynamic-COUNT did not reliably recover the oracle headroom; the simple, explainable controllers
  used here (a rule controller and a small ridge model) are a deliberate design choice, not an
  exhaustive search.
- Compute is measured as analytical prefill FLOPs (identically for every method); wall-clock latency
  and memory were not re-measured in the final scope.
- Result paths and sample manifests are locked for reproducibility: result file names encode the
  references other methods resolve, so they must not be renamed.

## Future Work

- Learned question-conditioned selectors, especially for models whose visual features are not
  natively language-aligned.
- Stronger adaptive-budget controllers, and calibration methods for the confidence and risk signals
  the budget decision depends on.
- Broader coverage: more models, more datasets, and tasks beyond short-answer VQA.
- Training-time pruning or distillation, which the frozen-backbone scope deliberately excluded.
- Latency and memory analysis beyond analytical FLOPs.
- Task-aware or OCR-aware pruning for text-heavy datasets, where the static floor collapses and the
  potential gain from smarter selection is largest.

## Notes on Archived Material

Retired and out-of-scope material is preserved locally, outside the public repository, for
traceability. Every move is recorded in local migration manifests, and nothing was deleted. The
archive is history, not part of the thesis story: the four method folders and the tables under
`results/tables/` contain everything the thesis claims.
