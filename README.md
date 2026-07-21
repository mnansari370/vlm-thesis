# Dynamic Question-Conditioned Visual Token Pruning for Efficient Vision-Language Models

**Author:** Mo Nafees
**Supervisor:** Prof. Decebal Constantin Mocanu
**Advisor:** Boqian Wu
**Master in Information and Computer Science, University of Luxembourg, 2026**

> Master's thesis repository. All experiments are complete and validated; every number in this
> README comes from a committed table under [`results/tables/`](results/tables/), produced by runs
> that passed an automatic fairness gate.

## Overview

Vision-language models such as LLaVA-1.5 and Qwen2.5-VL spend most of their language-model compute
on visual tokens. A single image occupies hundreds to over a thousand positions in the sequence,
while the evidence for an answer is often local to a small part of the image. Pruning visual tokens
is therefore a natural way to make inference cheaper.

This thesis studies visual token pruning at inference time, on frozen backbones. The goal is to
reduce visual-token computation while preserving answer quality across visual question answering
tasks, and, just as important, to measure that trade honestly: locked evaluation samples, identical
prompts and decoding for every method, and computation accounting that charges every executed
language-model pass, including any pass used only to make a pruning decision.

## Research Questions

This thesis is organized around three research questions:

- **RQ1.** Can vision-language models reduce visual token computation without losing too much
  task accuracy?
- **RQ2.** Can dynamic, question-conditioned pruning improve the accuracy-efficiency trade-off
  compared with dense inference and a strong static pruning baseline?
- **RQ3.** How did AI-assisted development tools support the implementation, experimentation,
  and verification process of this thesis?

Any pruning method quietly makes two separate decisions, and this thesis evaluates them as two
separate axes:

- **Dynamic-WHICH** asks *which* visual tokens should be kept at a fixed budget. Does letting the
  question choose the tokens beat choosing them from the image alone?
- **Dynamic-COUNT** asks *how many* visual tokens should be kept for each sample. Does an adaptive
  per-sample budget beat the fixed-budget curve of the same selector at the same average compute?

## Thesis Scope

This repository focuses on two models and four datasets only:

| | |
|---|---|
| **Models** (both frozen, no backbone training) | LLaVA-1.5-7B (fixed 576 visual tokens at 336 px) · Qwen2.5-VL-7B-Instruct (native per-image token count) |
| **Datasets** | GQA (testdev balanced, 12,578, exact match) · VQAv2 (25,000 validation questions, stratified by official answer type, seed 42, adapted consensus) · TextVQA (val, 5,000, soft accuracy, OCR in the prompt) · DocVQA (val, 5,349, ANLS) |
| **Budgets** | 15 / 25 / 35 / 50 / 75 % of each sample's dense visual-token count, plus dense (100 %) |
| **Methods** | Dense baseline · Static pruning · Dynamic-WHICH · Dynamic-COUNT (DC-D and DC-C) |

The VQAv2 scorer is an adapted consensus score, min(matches/3, 1) over the ten reference answers
without the official leave-one-out averaging. It is applied identically to every method, so VQAv2
comparisons are internally consistent, but the absolute values are not official leaderboard scores.

Fairness is enforced mechanically, not by convention: every method and both models score the exact
same sha256-locked sample lists ([`configs/sample_ids/`](configs/sample_ids/)), decode greedily at
batch size 1 with identical prompts, save per-sample predictions plus an aggregate record, and
compute analytical prefill FLOPs per sample before averaging. The FLOP estimate covers language-model
input processing only; the vision encoder, selector operations, and decoding sit outside it and are
excluded identically for every method. Token reduction and FLOP reduction are always reported
separately. A score difference is called a win above +0.50 points, a near-tie within ±0.50, and a
loss below −0.50; that reporting convention (not a statistical test) is used everywhere below.

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

Keeps all visual tokens. It is the full-compute reference: it fixes the unpruned score and cost that
every pruning result is measured against, and it supplies each sample's dense token count from which
the budgets are derived. It is a reference, not an accuracy ceiling: a pruned run can land slightly
above it (LLaVA GQA at the 75 % static budget is +0.11 over dense), and the oracle analysis below
exceeds it in every cell.

| Model | GQA | VQAv2 | TextVQA | DocVQA |
|---|--:|--:|--:|--:|
| LLaVA-1.5-7B | 61.42 | 77.33 | 57.65 | 21.53 |
| Qwen2.5-VL-7B | 60.96 | 84.27 | 81.06 | 94.76 |

The low LLaVA DocVQA score is expected and informative: 576 low-resolution tokens cannot read a
dense document page.

### Static Pruning

Applies fixed-budget, image-based pruning; the question is never consulted. LLaVA ranks patches by
CLS attention in the vision encoder, and Qwen ranks merged visual embeddings by activation norm; the
two selectors are architecture-specific because the models expose different visual signals at the
pruning point. Tokens are physically removed before the language model, so the savings are real.
Static is the strong baseline every dynamic method must beat: at the 75 % budget it sits within 0.02
to 2.63 points of dense on every model-dataset cell while removing roughly a quarter of the FLOPs,
and on GQA and VQAv2 even the 15 % budget loses only about 4 to 8 points. It collapses on the
text-heavy Qwen settings: Qwen TextVQA falls from 81.06 to 53.06 and Qwen DocVQA from 94.76 to 29.90
at the tightest budget, plausibly because the required evidence there is fine-grained and spread
across many positions, though the experiments do not isolate the cause.

### Dynamic-WHICH

Question-conditioned token selection at the same fixed budgets as static. The `textsim` selector
scores every visual token by its maximum cosine similarity to the question-token embeddings, keeps
the top K, and restores the original spatial order. The selector is training-free and needs no extra
language-model pass for scoring. Because the per-sample budget is identical to static, the
comparison isolates the selection decision. The main success is Qwen2.5-VL-7B on TextVQA, confirmed
by an independent clean-room reimplementation that reproduces the predictions exactly.

### Dynamic-COUNT

Adaptive per-sample token budget, with the selector held fixed. A reduced-budget probe pass (15 % or
25 %) produces a provisional answer and 27 decision features: seven decoding-confidence signals, ten
selector-score statistics, and ten question and input features. A controller fitted on the first
20 % of each sample list then decides, per sample, whether the probe answer stands or a second pass
at a larger budget is needed. Two variants: **DC-D**, a discrete cascade that either keeps the probe
answer or escalates once to a calibrated fixed budget (50 % or 75 %), and **DC-C**, a continuous
controller that predicts a per-sample integer token count and runs a second pass only when the
prediction exceeds the probe count. The accounting is honest: an escalated sample pays for both
passes. Each controller is compared with the fixed-budget accuracy-versus-FLOPs curve of its own
selector, rebuilt on the held-out samples, at matched average compute. The result is mostly negative
or near-tie, and that outcome is central to the thesis rather than a footnote.

## Results Summary

The complete matrix: 2 models × 4 datasets, all budgets, with the fairness gate passed on every run.
Dense, static, and WHICH are measured on the full locked sample lists. DC-D and DC-C are calibrated
on the first 20 % of each list and evaluated on the held-out 80 %, so their deltas are against the
static accuracy-compute curve rebuilt on those same held-out ids at matched average FLOPs. WHICH
deltas are against static at the same budget.

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

The gain peaks at the 25 % budget (+8.36) rather than at the tightest one, and it stays above the
near-tie band even at 75 %. The clean-room reimplementation reproduces the selector
prediction-for-prediction (200/200 exact matches, zero score difference, at every budget).

**Qwen DocVQA caveat.** Dynamic-WHICH shows a localized low-budget improvement there (+3.19 at the
15 % budget: 33.09 vs static 29.90), but it is not the overall best method for that dataset. The
win occurs where both methods have already collapsed, while higher static budgets and dense remain
far stronger in absolute accuracy (static p75 = 93.98, dense = 94.76).

**Stacking COUNT on WHICH** (Qwen2.5-VL × TextVQA, held-out): DC-D lands +6.99 above the static
curve but −0.41 against the fixed-budget curve of its own question-conditioned selector, and DC-C is
−2.03 to −2.30 against that curve. The whole gain belongs to the selector; adaptive budgeting adds
no further increment on top of it.

## Key Findings

1. **Static pruning is a strong and reliable baseline.** Near-dense at the 75 % budget on every
   cell, graceful on scene-centric tasks even at 15 %. A dynamic method has to be judged against
   this floor, not only against dense inference; beating dense alone says little about the quality
   of a pruning rule.
2. **Dynamic-WHICH is not universal.** Across the 40-cell matrix it records 6 wins, 2 near-ties, and
   32 losses against static at the same budget; all 20 LLaVA cells are negative.
3. **Where it works, it works well.** On Qwen2.5-VL × TextVQA, question-conditioned selection gains
   +5.94 to +8.36 points at the four tighter budgets, a validated and reproducible improvement. The
   pattern is consistent with questions pointing at localized, readable evidence in
   language-aligned visual features, though the token-level mechanism was not verified directly.
4. **Dynamic-COUNT does not consistently beat the fixed-budget curve.** DC-D achieves one small win
   (LLaVA TextVQA, +0.75) and otherwise near-ties or loses; DC-C gives no win at all (best +0.48, a
   near-tie; the steep-curve cells lose by up to −11.07). Stacking COUNT on top of the successful
   WHICH selector adds no further increment.
5. **The oracle headroom is real but not recovered.** A perfect per-sample budget router over the
   five static budgets would beat the best fixed budget by +2.09 to +6.25 points, exceed even dense
   in every cell, and retain 61 to 83 % fewer visual tokens; yet the practical controllers do not
   reliably recover this headroom under honest two-pass accounting.
6. **Negative results are included by design.** All eight model-dataset cells are reported for every
   method, on identical samples, with the gate verdict stored in every result file. The negatives
   are as load-bearing for the thesis as the single strong positive.

## Repository Structure

```
dense/                  method-facing entry point for the dense baseline
static/                 method-facing entry point for static pruning
dynamic_which/          method-facing entry point for Dynamic-WHICH
dynamic_count/          method-facing entry point for Dynamic-COUNT
src/                    implementation code grouped by role and method
scripts/                runnable scripts grouped by method, validation, and tables
results/                committed result tables and fitted Dynamic-COUNT controller configs
configs/                sha256-locked sample manifests (sample_ids/*.json)
requirements.txt        pinned environment for the LLaVA-1.5-7B result cells
requirements-qwen.txt   pinned environment for the Qwen2.5-VL-7B result cells
```

Each method folder contains a `README.md` (what and why), `CODE_MAP.md` (exact implementation
files), `COMMANDS.md` (safe CPU validation, table generation, and GPU rerun commands), `RESULTS.md`
(the real numbers), and a safe wrapper script. The shared evaluation core (sample manifests,
output schema, fairness gate, FLOP accounting) lives in `src/common/`, and the per-method runner
cores live in `src/dense/`, `src/static/`, `src/dynamic_which/`, and `src/dynamic_count/`.

Datasets, model weights, per-sample run outputs, and run logs are local-only and are not part of
the public repository; the committed evidence is the summary tables, the fitted controller
configs, the locked sample manifests, and the two pinned environment specifications.

## Reproducibility

The committed results can be re-verified on CPU in minutes, with no model and no GPU:

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

Full GPU reruns are expensive (two pinned conda environments, two GPUs, many hours) and are only
needed to reproduce the experiments from scratch; because the per-sample records stay local,
regenerating the source tables externally requires such a rerun. Each method folder (`dense/`,
`static/`, `dynamic_which/`, `dynamic_count/`) documents its rerun commands in `COMMANDS.md`, and
all launchers skip existing results rather than overwrite them. Decoding is greedy, so there is no
sampling randomness, but bitwise-identical model reruns across different hardware or library
versions are not guaranteed.

## Limitations

- Two models and four VQA-style datasets only; conclusions beyond this grid are not claimed.
- The two static baselines use different architecture-specific ranking signals (CLS attention for
  LLaVA, activation norm for Qwen). The study does not establish that activation norm is the
  strongest possible question-independent selector for Qwen, and a stronger static baseline could
  shrink the TextVQA gains.
- No training-based learned selector in the final thesis scope; all selection methods are
  training-free on frozen backbones, and the Dynamic-COUNT controllers fit only small components
  outside the model.
- Dynamic-WHICH does not generalize across all tasks; it is a regime-specific tool.
- Dynamic-COUNT did not reliably recover the oracle headroom; the simple, explainable controllers
  used here (a rule controller and a small ridge model) are a deliberate design choice, not an
  exhaustive search.
- Compute is measured as analytical prefill FLOPs of language-model input processing, applied
  identically to every method; the vision encoder, selector operations, and decoding are outside
  the estimate, and wall-clock latency and memory were not measured in the final scope.
- The VQAv2 scorer is an adapted consensus metric, so VQAv2 values are internally comparable but
  not official leaderboard scores.
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

## Notes on Local-Only Material

Retired and out-of-scope material, internal planning documents, and the thesis manuscript are
preserved locally, outside the public repository. Every move into the archive is recorded in local
migration manifests, and nothing was deleted. The archive is history, not part of the thesis story:
the four method folders and the tables under `results/tables/` contain everything the thesis claims.
