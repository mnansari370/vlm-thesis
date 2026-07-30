# Dynamic Question-Conditioned Visual Token Pruning for Efficient Vision-Language Models

Master's thesis · **Mo Nafees** · University of Luxembourg, 2026

| | |
|---|---|
| **Degree** | Master in Information and Computer Sciences |
| **Faculty** | Faculty of Science, Technology and Medicine |
| **Supervisor** | Prof. Decebal Constantin Mocanu |
| **Reviewer** | Prof. Thomas Engel |
| **Advisor** | Boqian Wu |

> All experiments are complete. Every number in this README comes from a committed table under
> [`results/tables/`](results/tables/), produced by runs that passed an automatic schema and
> comparison check.

---

## Contents

- [In one paragraph](#in-one-paragraph)
- [Research questions](#research-questions)
- [The two pruning decisions](#the-two-pruning-decisions)
- [Evaluation scope](#evaluation-scope)
- [How comparisons are kept fair](#how-comparisons-are-kept-fair)
- [Methods](#methods)
- [Results](#results)
- [Findings](#findings)
- [Repository structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Thesis manuscript](#thesis-manuscript)
- [Citation](#citation)

---

## In one paragraph

A vision-language model turns an image into hundreds of visual tokens, and the language model
processes all of them before producing the first word of an answer. Removing some of those tokens is
a direct way to cut that cost. This thesis measures how far that can be pushed on **frozen**
backbones, and it separates something the literature usually changes all at once: pruning makes
**two** decisions, *which* tokens to keep and *how many*, and a method that changes both together
cannot tell you which decision earned the result. Each decision is therefore measured with the other
held fixed, under one shared evaluation, with every executed language-model pass charged, including
any pass used only to make a pruning decision.

The short answer: a fixed, question-independent rule is already a strong baseline. Question
conditioning helps consistently in exactly one of eight settings. Adaptive per-sample budgeting does
not consistently help at all, even though a retrospective oracle shows the headroom is real.

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Can vision-language models reduce visual token computation without losing too much task accuracy? |
| **RQ2** | Can dynamic, question-conditioned pruning improve the accuracy-efficiency trade-off compared with dense inference and a strong static pruning baseline? |
| **RQ3** | How did AI-assisted development tools support the implementation, experimentation, and verification process of this thesis? |

---

## The two pruning decisions

| Axis | Question | Held fixed | Compared against |
|---|---|---|---|
| **Dynamic-WHICH** | Which visual tokens to keep? | Retained-token count, identical per sample | Static pruning at the same budget |
| **Dynamic-COUNT** | How many visual tokens to keep? | The selection rule | Fixed-budget curve of the *same* selector, at matched mean computation |

Holding the count equal removes "it kept more tokens" as an explanation for a selection result.
Holding the selector fixed removes "it picked better tokens" as an explanation for a budget result.

---

## Evaluation scope

**Models** — both frozen, no backbone weights trained.

| Model | Visual tokens per image | Static selector |
|---|---|---|
| LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) | Fixed 576 | CLS attention in the vision encoder |
| Qwen2.5-VL-7B-Instruct (`Qwen/Qwen2.5-VL-7B-Instruct`) | Native: ≈359 GQA/VQAv2, ≈964 TextVQA, ≈1229 DocVQA | Activation norm on post-merger embeddings |

**Datasets**

| Dataset | Split | Samples | Metric |
|---|---|--:|---|
| GQA | testdev balanced | 12,578 | Exact match |
| VQAv2 | validation, stratified by official answer type, seed 42 | 25,000 | Adapted consensus, min(m/3, 1) |
| TextVQA | val, OCR text supplied in the prompt | 5,000 | Leave-one-out soft accuracy |
| DocVQA | validation | 5,349 | ANLS, threshold 0.5 |

**Budgets** — 15 / 25 / 35 / 50 / 75 % of each sample's own dense visual-token count, plus dense.
For LLaVA-1.5 that is 86 / 144 / 202 / 288 / 432 of 576. For Qwen2.5-VL the count is computed per
sample, rounded half to even, then clamped to at least 1 and at most the dense count.

> **On VQAv2 scores.** The scorer is an adapted direct-count consensus metric, not the official
> leave-one-out averaging. It is applied identically to every method, so VQAv2 comparisons here are
> internally consistent, but the absolute values are **not** official leaderboard scores.

---

## How comparisons are kept fair

Fairness is enforced mechanically rather than by convention.

- **Locked samples.** Every method and both models score the exact same ordered sample lists, stored
  with SHA-256 values in [`configs/sample_ids/`](configs/sample_ids/). A run whose identifiers,
  ordering or hash diverge is rejected.
- **Identical generation.** Greedy decoding, batch size 1, `max_new_tokens=64`, natural
  end-of-sequence stop, no repetition penalty, and one shared short-answer instruction across all
  four datasets.
- **One computation scope.** An analytical estimate of **language-model input-processing**
  computation, one multiply-accumulate counted as one operation. **Every executed pass is charged**,
  including a probe pass whose answer is later kept. Excluded identically for every method: vision
  encoding, projection or merging, token scoring and selection, controller inference, generated-token
  decoding, latency and memory. **This is not an end-to-end cost.**
- **Token reduction ≠ computation reduction.** The two are reported separately, never interchanged.
- **A descriptive reporting band.** A difference is called an *improvement* above +0.50 points, a
  *near-tie* within ±0.50, and a *decrease* below −0.50. This is a reporting convention, **not** a
  statistical test. No confidence intervals or hypothesis tests are reported.

---

## Methods

### Dense inference — the full-token reference

Keeps every visual token. It fixes the unpruned score and cost that pruned results are measured
against, and supplies each sample's dense token count from which budgets are derived.

It is a **reference, not an accuracy ceiling**: a pruned run can land slightly above it (LLaVA-1.5 on
GQA at 75 % retention is +0.11), and the retrospective oracle exceeds it in all eight settings.

| Model | GQA | VQAv2 | TextVQA | DocVQA |
|---|--:|--:|--:|--:|
| LLaVA-1.5 | 61.42 | 77.33 | 57.65 | 21.53 |
| Qwen2.5-VL | 60.96 | 84.27 | 81.06 | 94.76 |

The low LLaVA-1.5 DocVQA score is itself informative: 576 tokens at 336 px cannot resolve a dense
document page, so small changes there say little about document understanding.

### Static pruning — the question-independent baseline

Fixed budget, image-based ranking, question never consulted. Tokens are physically removed before the
language model, so the saving is real. The two selectors are architecture-specific because the models
expose different visual signals at the pruning point.

This is the baseline every dynamic method has to beat, and it is demanding: at 75 % retention it stays
within **0.02 to 2.63 points** of dense across all eight settings while cutting reported computation
by **22.30 to 25.03 %**.

### Dynamic-WHICH — question-conditioned selection

Training-free. Each visual token is scored by its maximum cosine similarity to the question-token
embeddings, the top *K* are kept, and the original spatial order is restored. No extra language-model
pass is needed for scoring, and the retained count is identical to static at every budget, so the
comparison isolates the selection decision alone.

### Dynamic-COUNT — adaptive per-sample budget

A reduced-budget probe pass (15 % or 25 %) produces a provisional answer plus decision features. A
controller fitted on the **first 20 %** of each sample list decides, per sample, whether the probe
answer stands or a second pass at a larger budget is needed. Evaluation uses the **held-out 80 %**.

- **DC-D** — discrete cascade: keep the probe answer, or escalate once to a calibrated budget.
- **DC-C** — continuous: predict a per-sample integer token count, run a second pass only if it
  exceeds the probe count.

An escalated sample pays for **both** passes. Each controller is compared with the fixed-budget curve
of its own selector, rebuilt on the same held-out samples, at matched mean computation.

---

## Results

### Complete matrix

Dense, static and WHICH use the full locked sample lists. DC-D and DC-C are calibrated on the first
20 % and evaluated on the held-out 80 %, so **their absolute scores are not comparable with the
full-set columns**. Only their paired differences against matched references are.

| Model | Dataset | Dense | Best static | WHICH Δ static | DC-D Δ curve | DC-C Δ curve | Outcome |
|---|---|--:|---|--:|--:|--:|---|
| LLaVA-1.5 | GQA | 61.42 | 61.53 (75 %) | −0.73 | −0.10 | −1.26 | static holds |
| LLaVA-1.5 | VQAv2 | 77.33 | 77.31 (75 %) | −1.59 | +0.06 | −0.25 | static holds |
| LLaVA-1.5 | TextVQA | 57.65 | 57.39 (75 %) | −3.91 | **+0.75** | +0.48 | DC-D improves |
| LLaVA-1.5 | DocVQA | 21.53 | 21.36 (75 %) | −3.97 | −0.89 | −0.54 | static holds |
| Qwen2.5-VL | GQA | 60.96 | 60.85 (75 %) | −0.68 | −0.03 | −0.19 | static holds |
| Qwen2.5-VL | VQAv2 | 84.27 | 83.66 (75 %) | −0.06 | +0.21 | −0.84 | static holds |
| Qwen2.5-VL | TextVQA | 81.06 | 78.43 (75 %) | **+8.36** | +0.02 | −7.27 | **WHICH improves** |
| Qwen2.5-VL | DocVQA | 94.76 | 93.98 (75 %) | +3.19 | −4.61 | −11.07 | see caveat |

### The one consistent selection improvement

Qwen2.5-VL on TextVQA, full 5,000 validation questions, question-conditioned versus static at
identical retained-token counts:

| Budget | Question-conditioned | Static | Dense | Δ static | Computation reduction |
|--:|--:|--:|--:|--:|--:|
| 15 % | 60.56 | 53.06 | 81.06 | **+7.50** | 78.65 % |
| 25 % | 67.53 | 59.17 | 81.06 | **+8.36** | 69.64 % |
| 35 % | 71.36 | 63.57 | 81.06 | **+7.79** | 60.56 % |
| 50 % | 76.08 | 70.14 | 81.06 | **+5.94** | 46.84 % |
| 75 % | 79.80 | 78.43 | 81.06 | **+1.37** | 23.63 % |

The gain peaks at 25 %, not at the tightest budget, and stays above the near-tie band even at 75 %.
An **independent reimplementation** of the selector reproduced all 200 checked predictions at every
budget with zero score difference. That supports implementation consistency in this setting. It does
not explain *why* the selector helps, and it does not establish correctness elsewhere.

> **Qwen2.5-VL DocVQA caveat.** There is a +3.19 improvement at 15 % retention (33.09 vs 29.90), but
> both conditions sit far below the dense reference of 94.76, and the improvement does not persist at
> any larger budget. It is not evidence that question conditioning helps on DocVQA.

### Adaptive budgeting

Across the **24** comparisons with static selection: **1 improvement, 10 near-ties, 13 decreases**.
The single improvement is DC-D on LLaVA-1.5 TextVQA at +0.75. **No continuous controller** exceeds
the +0.50 boundary anywhere.

Stacking budgeting on top of the successful selector (Qwen2.5-VL × TextVQA, held out):

| Controller | Score | Δ question-conditioned curve | Δ static curve |
|---|--:|--:|--:|
| Discrete cascade | 62.07 | −0.41 | **+6.99** |
| Continuous (ridge) | 77.09 | −2.30 | −0.35 |
| Continuous (rule) | 77.92 | −2.03 | −0.79 |

The +6.99 belongs to the **selector**, not the budget decision: the same result sits *below* the
fixed-budget curve of its own selector. Adaptive budgeting adds nothing on top.

### Retrospective oracle

Routing each sample to its best-scoring evaluated budget beats the best single fixed budget by
**+2.09 to +6.25 points** in all eight settings, exceeds dense in all eight, and retains
**61.50 to 82.99 %** fewer visual tokens.

This is an **upper bound, not a method.** It selects *after* seeing the outcomes, so it is not
deployable, and it says nothing about token counts outside the five evaluated budgets. It shows the
headroom exists. It does not show a controller can reach it.

---

## Findings

1. **Static pruning is a strong baseline.** Within 0.02–2.63 points of dense at 75 % retention on
   every setting. A dynamic method must be judged against this, not against dense alone, since
   beating dense says little about the quality of a pruning rule.
2. **Question conditioning is not general.** Across 40 matched comparisons: **6 improvements, 2
   near-ties, 32 decreases**. All 20 LLaVA-1.5 comparisons fall below static.
3. **Where it works, it works well.** Qwen2.5-VL × TextVQA gains +5.94 to +8.36 points at the four
   tighter budgets, reproducible and independently checked. This is consistent with questions
   pointing at localized readable evidence in language-aligned features, though the token-level
   mechanism was never verified directly.
4. **Adaptive budgeting does not consistently pay.** One improvement in 24 comparisons, none from a
   continuous controller, and no increment on top of question-conditioned selection.
5. **Oracle headroom is real but unrecovered.** The gap between retrospective potential and practical
   controllers is the central negative result, not a footnote.
6. **Negative results are reported by design.** All eight settings appear for every method, on
   identical samples, with the validation verdict stored in every result file.

---

## Repository structure

```
dense/                  entry point + docs for the dense reference
static/                 entry point + docs for static pruning
dynamic_which/          entry point + docs for question-conditioned selection
dynamic_count/          entry point + docs for adaptive budgeting
src/                    implementation, grouped by role and method
  common/                 sample manifests, output schema, validation, FLOP accounting
scripts/                runnable scripts: per method, validation, table generation
results/tables/         committed summary tables, the evidence behind this README
results/configs/        fitted Dynamic-COUNT controllers
configs/sample_ids/     SHA-256 locked sample manifests
requirements.txt        pinned environment for LLaVA-1.5 runs
requirements-qwen.txt   pinned environment for Qwen2.5-VL runs
```

Each method folder carries `README.md`, `CODE_MAP.md`, `COMMANDS.md` and `RESULTS.md`.

**Not in this repository:** datasets, model weights, per-sample prediction records, run logs and the
thesis manuscript. The committed evidence is the summary tables, the fitted controllers, the locked
manifests and the two environment specifications.

---

## Reproducibility

The committed results can be re-verified on CPU in minutes, with no model and no GPU:

```bash
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
python -m scripts.validation.validate_dynamic_which
python -m scripts.validation.audit_dynamic_which
python -m scripts.validation.validate_dynamic_count
python -m scripts.validation.audit_dynamic_count
```

Expected: all self-checks pass, both validators report `ALL_*_VALID=True`, the WHICH audit reports
40 of 40 settings complete, and the COUNT audit reports zero probe-reproduction failures.

Regenerate the summary tables:

```bash
python -m scripts.tables.make_dynamic_which_summary
python -m scripts.tables.make_dynamic_count_tables
python -m scripts.tables.make_final_thesis_tables
```

**Two pinned environments are required** for model runs, because the two models are not compatible
with a single `transformers` version:

```bash
conda create -n vlm_env python=3.11 -y && conda activate vlm_env
pip install -r requirements.txt          # torch 2.3.0+cu121, transformers 4.46.3

conda create -n qwen_env python=3.11 -y && conda activate qwen_env
pip install -r requirements-qwen.txt     # torch 2.5.1+cu121, transformers 4.51.3
```

Full GPU reruns are expensive and are only needed to reproduce from scratch. Because the per-sample
records stay local, regenerating the source tables externally requires such a rerun. Each method
folder documents its commands in `COMMANDS.md`, and launchers skip existing results rather than
overwrite them. Decoding is greedy, so there is no sampling randomness, but **bitwise-identical
reruns across different hardware or library versions are not guaranteed**.

---

## Limitations

- **Two models, four short-answer VQA datasets.** No claim is made beyond this grid. Captioning,
  long-form generation and multi-turn interaction were not evaluated.
- **The static baselines are architecture-specific** (CLS attention vs activation norm). The study
  does not establish that activation norm is the strongest question-independent selector for
  Qwen2.5-VL, and a stronger baseline could shrink the TextVQA gains.
- **One question-conditioned rule**, held fixed across all settings. Results describe that rule
  rather than question-conditioned selection in general.
- **Frozen backbones by design.** No trained selector, so the results say nothing about what training
  could recover.
- **Simple controllers by design.** A rule controller and a small ridge model, chosen for
  transparency rather than as an exhaustive search.
- **Computation is analytical, not end-to-end.** Vision encoding, projection or merging, token
  scoring and selection, controller inference, decoding, latency and memory are all outside the
  estimate.
- **No uncertainty estimates.** The ±0.50 band is descriptive, not a statistical test.
- **VQAv2 values are not official leaderboard scores** (adapted consensus scoring).
- **Result file names and manifests are load-bearing** and must not be renamed or regenerated.
- **Reproducibility gaps.** Manuscript table values were transferred by hand with no automated check,
  per-sample records remain local, and the DocVQA snapshot has a content fingerprint but no recorded
  upstream revision.

---

## Future work

- Compare the Qwen2.5-VL static selector against other question-independent selectors, to test
  whether the TextVQA advantage survives a different comparator.
- Alternative similarity functions, representations from other layers, and trained selectors.
- Additional frozen models, or one family at several scales, to separate architecture, scale and
  selector effects.
- Token-level coverage analysis on TextVQA and DocVQA, to test the proposed explanations directly.
- Controllers with a larger calibration set, alternative pre-answer signals, or a budget decision
  that avoids a full probe pass.
- End-to-end measurement including latency, memory and the currently excluded components.
- Uncertainty estimates such as bootstrap intervals and repeated controller fits, plus stronger
  external reproducibility.

---

## Thesis manuscript

The written thesis is a nine-chapter LaTeX document with two appendices covering the full result
tables and reproducibility notes. It is kept **local-only** and is not part of this repository.

---

## Citation

```bibtex
@mastersthesis{nafees2026pruning,
  title  = {Dynamic Question-Conditioned Visual Token Pruning for
            Efficient Vision-Language Models},
  author = {Nafees, Mo},
  school = {University of Luxembourg},
  year   = {2026},
  type   = {Master's thesis}
}
```
