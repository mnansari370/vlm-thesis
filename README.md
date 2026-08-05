# An Empirical Evaluation of Dynamic Question-Conditioned Visual Token Pruning in Vision-Language Models

**Master's Thesis** · **Nafees Mo** · University of Luxembourg, 2026

| | |
|---|---|
| **Degree** | Master in Information and Computer Sciences |
| **Faculty** | Faculty of Science, Technology and Medicine |
| **Supervisor** | Prof. Decebal Constantin Mocanu |
| **Reviewer** | Prof. Thomas Engel |
| **Advisor** | Boqian Wu |

> **Note:** All experiments are complete. Every number in this README comes from a committed table under [`results/tables/`](results/tables/), produced by runs that passed an automatic schema and comparison check.

---

## 📖 Contents

- [Overview](#-overview)
- [Research Questions](#-research-questions)
- [The Two Pruning Decisions](#-the-two-pruning-decisions)
- [Evaluation Scope](#-evaluation-scope)
- [Methodology & Strict Fairness](#-methodology--strict-fairness)
- [Empirical Results](#-empirical-results)
- [Core Findings](#-core-findings)
- [Repository Structure](#-repository-structure)
- [Reproducibility](#-reproducibility)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)

---

## 🔬 Overview

A vision-language model (VLM) converts an image into hundreds or thousands of visual tokens, and the core language model must process all of them before producing the first word of an answer. Removing some of these tokens is a direct way to reduce computational cost. This thesis measures how far that can be pushed on **frozen** backbones. 

Crucially, it separates a process the literature usually conflates: token pruning inherently makes **two distinct decisions**:
1. ***Which* tokens to keep** (Selection)
2. ***How many* tokens to keep** (Budgeting)

When a method changes both simultaneously, it is impossible to determine which decision drove the result. In this thesis, each decision is measured with the other held strictly fixed under a shared evaluation protocol. Every executed language-model pass is charged, including any exploratory pass used solely to make a pruning decision.

**The Short Answer:** A fixed, question-independent pruning rule is already an incredibly strong baseline. Question conditioning improves performance consistently in exactly one of eight settings. Adaptive per-sample budgeting does not consistently help at all, even though a retrospective oracle proves that theoretical headroom exists.

---

## ❓ Research Questions

| ID | Question |
|:---:|---|
| **RQ1** | What trade-offs between task accuracy and computation can visual token pruning achieve relative to dense inference? |
| **RQ2** | Can dynamic, question-conditioned pruning improve the trade-off between task accuracy and computation compared with dense inference and a strong fixed-budget image-based baseline? |
| **RQ3** | How did AI-assisted development tools support the implementation, experimentation, and verification process of this thesis? |

---

## ⚖️ The Two Pruning Decisions

| Axis | Question | Held Fixed | Compared Against |
|---|---|---|---|
| **Dynamic-WHICH** | Which visual tokens to keep? | Retained-token count, identical per sample | Static pruning at the identical budget |
| **Dynamic-COUNT** | How many visual tokens to keep? | The selection rule | Fixed-budget curve of the *same* selector, at matched mean computation |

Holding the count equal removes *"it kept more tokens"* as an explanation for a selection result.
Holding the selector fixed removes *"it picked better tokens"* as an explanation for a budget result.

---

## 🎯 Evaluation Scope

**Models** — Both backbones remain entirely frozen.
| Model | Visual tokens per image | Static selector |
|---|---|---|
| **LLaVA-1.5-7B** | Fixed exactly 576 | CLS attention in the vision encoder |
| **Qwen2.5-VL-7B-Instruct** | Native variable resolution: ≈359 (GQA), ≈964 (TextVQA), ≈1229 (DocVQA) | Activation norm on post-merger embeddings |

**Datasets** — Four distinct VQA tasks.
| Dataset | Split | Samples | Metric |
|---|---|--:|---|
| **GQA** | testdev-balanced | 12,578 | Exact match |
| **VQAv2** | validation (stratified subset) | 25,000 | Adapted consensus, `min(m/3, 1)` |
| **TextVQA** | val (OCR text supplied in prompt) | 5,000 | Leave-one-out soft accuracy |
| **DocVQA** | validation | 5,349 | ANLS, threshold 0.5 |

**Budgets** — `15%`, `25%`, `35%`, `50%`, and `75%` of each sample's own dense visual-token count, plus `Dense` (100%).

---

## 🛡️ Methodology & Strict Fairness

Fairness is enforced mechanically rather than by convention:

1. **Locked Samples:** Every method scores the exact same ordered sample lists, verified by SHA-256 hashes in [`configs/sample_ids/`](configs/sample_ids/).
2. **Identical Generation:** Greedy decoding, `batch_size=1`, `max_new_tokens=64`, and a shared short-answer instruction across all datasets.
3. **One Computation Scope:** An analytical estimate of **language-model input-processing** computation (FLOPs). **Every executed pass is charged**. Vision encoding, projection, decoding, latency, and memory are uniformly excluded.
4. **Descriptive Reporting Band:** A difference is described as an *improvement* above `+0.50` points, a *near-tie* within `±0.50`, and a *decrease* below `-0.50`. This is a reporting convention, not a statistical hypothesis test.

---

## 📊 Empirical Results

### Dense Inference & The Static Baseline
Dense inference keeps every token. **Static fixed-budget pruning** ranks tokens by image features (no text) and removes them. 
This is the baseline every dynamic method must beat. It is extremely demanding: at 75% retention, static pruning stays within **0.02 to 2.63 points** of dense inference across all settings while cutting reported computation by **22% to 25%**.

### Dynamic-WHICH (Question-Conditioned Selection)
Tokens are ranked by maximum cosine similarity to the question-token embeddings. 
- Evaluated across 40 matched configurations: **6 improvements, 2 near-ties, 32 decreases**.
- **The one consistent improvement:** Qwen2.5-VL on TextVQA gains `+5.94` to `+8.36` points at tighter budgets. An independent reimplementation reproduced all 200 checked predictions identically, confirming implementation consistency.

### Dynamic-COUNT (Adaptive Per-Sample Budget)
A reduced-budget probe pass yields an answer and decision features. A controller fitted on a 20% calibration split decides whether to accept the probe or run a second, larger pass. 
- Evaluated across 24 comparisons with static selection on the 80% held-out set: **1 improvement, 10 near-ties, 13 decreases**. 
- No continuous controller improved performance. The computational penalty of the probe pass outweighs the routing gains.
- **Oracle Headroom:** A retrospective oracle proves that routing each sample to its empirically best budget yields `+2.09` to `+6.25` points above the best fixed budget. The headroom is real, but deployable heuristic controllers fail to capture it.

---

## 💡 Core Findings

1. **Static pruning is a formidable baseline.** A dynamic method must be judged against this, not against dense inference alone.
2. **Question conditioning is not general.** Text-to-image similarity ranking frequently degrades performance compared to architecture-specific image-based signals.
3. **Where it works, it works well.** Qwen2.5-VL on TextVQA is the sole exception, proving highly effective for resolving localized, text-centric visual queries.
4. **Adaptive budgeting does not consistently pay off.** The overhead of executing a probe pass ruins the mathematical economics of adaptive routing.
5. **Oracle headroom is real but unrecovered.** Theoretical token-budget optimization is vast, but practical pre-generation signals fail to unlock it.

---

## 📁 Repository Structure

```text
dense/                  # Entry point + docs for the dense reference
static/                 # Entry point + docs for static pruning
dynamic_which/          # Entry point + docs for question-conditioned selection
dynamic_count/          # Entry point + docs for adaptive budgeting
src/                    # Core implementation
  └─ common/            # Sample manifests, validation logic, FLOP accounting
scripts/                # Runnable scripts (runs, validation, table generation)
results/tables/         # Committed summary tables (the evidence for this README)
results/configs/        # Fitted Dynamic-COUNT controllers
configs/sample_ids/     # SHA-256 locked sample manifests
requirements.txt        # Pinned environment for LLaVA-1.5 runs
requirements-qwen.txt   # Pinned environment for Qwen2.5-VL runs
```

> **Note:** Datasets, model weights, per-sample predictions, and the thesis manuscript itself are kept local and are not distributed in this repository. 

---

## ⚙️ Reproducibility

The committed results can be verified on CPU in minutes, with no model and no GPU:

```bash
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
python -m scripts.validation.validate_dynamic_which
python -m scripts.validation.audit_dynamic_which
python -m scripts.validation.validate_dynamic_count
python -m scripts.validation.audit_dynamic_count
```
*(Expected: all self-checks pass, both validators report `ALL_*_VALID=True`)*

To regenerate the summary tables:
```bash
python -m scripts.tables.make_dynamic_which_summary
python -m scripts.tables.make_dynamic_count_tables
python -m scripts.tables.make_final_thesis_tables
```

To run inference from scratch, **two pinned environments** are required due to `transformers` version incompatibilities:
```bash
# Environment 1: LLaVA-1.5
conda create -n vlm_env python=3.11 -y && conda activate vlm_env
pip install -r requirements.txt

# Environment 2: Qwen2.5-VL
conda create -n qwen_env python=3.11 -y && conda activate qwen_env
pip install -r requirements-qwen.txt
```

---

## 🚧 Limitations & Future Work

* **Architectural Boundaries:** The study evaluates exactly two frozen 7B-parameter architectures and four VQA datasets. Extrapolation to video, unconstrained generation, or parameter fine-tuning is unproven.
* **Controller Simplicity:** Controllers were deliberately kept simple (ridge regression and rule-based) to prioritize transparency over an exhaustive architectural search.
* **Computation Estimates:** FLOP estimations intentionally isolate the pruning mechanism but ignore external hardware bottlenecks like memory bandwidth and latency.
* **Future Directions:** Future work must develop pre-generation signals that bypass the need for a full exploratory probe pass. Eliminating the probe cost is the most critical hurdle for practical adaptive budgeting.

---

## 📄 Citation

```bibtex
@mastersthesis{mo2026pruning,
  title  = {An Empirical Evaluation of Dynamic Question-Conditioned Visual Token Pruning in Vision-Language Models},
  author = {Mo, Nafees},
  school = {University of Luxembourg},
  year   = {2026},
  type   = {Master's thesis}
}
```
