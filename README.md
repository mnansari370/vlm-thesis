# Dynamic, Question-Conditioned Visual Token Pruning for Efficient Vision–Language Models

**When can a per-sample visual token budget beat a well-tuned fixed budget — and how would you know
before building one?**

This repository contains the complete experimental framework of my Master's thesis on visual token
pruning in LLaVA-style Vision–Language Models. I implement a dynamic, question-conditioned pruning
framework on a **fully frozen LLaVA-1.5-7B**, evaluate it honestly against tuned static baselines at
matched compute on **five benchmark settings**, and contribute an **oracle-headroom diagnostic** that
measures — before any method is built — how much room per-sample token budgeting actually has.

---

## TL;DR — what this repository shows

1. **A working framework.** Question-conditioned token scoring (a CLS-attention prior plus a learned
   scorer), a learned per-sample budget controller, and a training-free confidence cascade — all on a
   frozen backbone, with only lightweight modules trained.
2. **An honest negative result, twice.** At *matched* average compute, dynamic budgeting **ties** tuned
   static pruning — on VQAv2 (75.76% vs 75.71%, Δ = +0.05 pp) and on GQA/TextVQA/POPE/ScienceQA, where
   the confidence cascade traces *along* the static frontier and never rises above it.
3. **The reason, quantified.** The **oracle budget-sensitive band** — the fraction of samples whose
   correctness actually depends on the token budget — is structurally thin on every benchmark
   (4.9–9.1%, always below 20%). Even a perfect, label-using budget allocator has almost nothing to win.
4. **A trustworthy pipeline.** The dense baseline reproduces published LLaVA-1.5-7B numbers within
   ≤ 1.65 pp on all four standard benchmarks, so the negative results are real, not evaluation artifacts.

> The actionable takeaway: *"give harder questions more tokens" sounds right but does not pay off on
> current benchmarks* — difficulty is not the same as token-need. The diagnostic tells you where it
> ever could.

---

## Method overview

```
                 ┌────────────────────────────────────────────────────────────┐
 image ──► CLIP ─┤ 576 patch tokens                                           │
                 │   1) SELECTION  rank by CLS→patch attention (training-free)│
 question ──────►│      (+ optional learned question-conditioned scorer)      │
                 │   2) BUDGET     how many tokens K for THIS sample?         │
                 │      a) learned controller  K ∈ [64, 576]                  │
                 │      b) confidence cascade  K=144 → escalate to 288        │
                 │         if first-token confidence < τ                      │
                 └───────────────┬────────────────────────────────────────────┘
                                 ▼
              top-K tokens (spatial order restored) ──► projector ──► frozen LLM ──► answer
```

**The oracle-headroom diagnostic** (the core contribution): answer every evaluation sample at a ladder
of static budgets K ∈ {144, 192, 288, 432, 576} and record its *first-correct budget*. This partitions
any benchmark into:

| Partition | Meaning |
|---|---|
| **easy@144** | already correct at the smallest budget — budgeting can't help |
| **budget-sensitive band** | wrong at 144, correct at some larger K — the *only* samples any budget policy can win |
| **never-correct** | wrong at every budget — no budgeting policy can recover them |

The band upper-bounds the gain of *any* per-sample budgeting method. A thin band means "don't bother";
a wide band means a dynamic budget is worth building. A method only counts as a win if it sits
**above the static frontier at equal FLOPs**.

---

## Key results

### 1. Pipeline validation (dense, official protocols)

| Benchmark | This work | Published | Offset |
|---|---|---|---|
| GQA (testdev-balanced) | 61.42 | 62.0 | −0.58 |
| TextVQA (val, OCR) | 57.65 | 58.2 | −0.55 |
| POPE (F1, mean of 3 subsets) | 85.78 | 85.9 | −0.12 |
| ScienceQA-IMG (test) | 65.15 | 66.8 | −1.65 |

### 2. The oracle-headroom band is thin everywhere (the headline)

| Benchmark | easy@K=144 | **budget-sensitive band** | never-correct |
|---|---|---|---|
| POPE | 84.36% | **4.93%** | 10.71% |
| TextVQA (OCR) | 56.24% | **6.08%** | 37.68% |
| TextVQA (no-OCR) | 44.82% | **6.24%** | 48.94% |
| ScienceQA-IMG | 64.30% | **7.73%** | 27.96% |
| GQA | 58.15% | **9.13%** | 32.72% |

The diagnostic *discriminates* — it orders benchmarks by their available headroom — yet no benchmark
comes anywhere near the ~20% band at which per-sample budgeting could clearly pay off.

### 3. Dynamic ties static at matched compute

**VQAv2** (val 10K, open-ended generation, VQA consensus scoring):

| Model | avg K | LM FLOPs | Accuracy |
|---|---|---|---|
| Dense (576 tokens) | 576 | 3.173 T | 76.44% |
| **Dynamic (type-adaptive budget)** | 264.3 | **1.530 T** | **75.76%** |
| Static, matched K=265 | 265 | 1.534 T | 75.71% |

Δ = +0.05 pp at identical cost — a wash. The dynamic operating point lands *on* the static curve.

**GQA** (testdev-balanced) — the confidence cascade versus the static frontier:

| Setting | Accuracy | LM FLOPs |
|---|---|---|
| Static K=144 | 58.15% | 0.904 T |
| Cascade τ=0.55 (144→288) | 59.41% | 1.487 T |
| Static K=288 | 60.53% | 1.648 T |
| Static K=432 | 61.53% | 2.402 T |
| Dense (576) | 61.42% | 3.168 T |

Sweeping τ moves smoothly between the two static endpoints — along the frontier, never above it.
For external calibration: published FastV is dominated by the static frontier (K=192: 52.7 vs 59.2),
and VisionZip's token merging lands on or below it at useful budgets.

### 4. Question-conditioned *selection* does not beat image saliency either

At K=64 (where selection headroom is largest), relative to the CLS-attention selector:

| Benchmark | CLS − Random | CLIP-space Qcond − CLS | LM-attention Qcond − CLS |
|---|---|---|---|
| TextVQA (no-OCR) | +20.7 | **−32.0** (below random) | −5.6 |
| TextVQA (OCR) | +8.1 | −12.5 | ≈ 0 |
| GQA | −1.5 | −9.2 | — |

Raw CLIP patch–text similarity is a poor localizer (consistent with CLIP-Surgery/MaskCLIP); CLS
attention already captures the exploitable selection signal.

---

## Repository structure

Three self-contained tracks, each with `dense/`, `static/`, `dynamic/`, and `shared/`:

```
GQA/                    Training-free multi-benchmark track (GQA · TextVQA · POPE · ScienceQA-IMG)
├── dense/              Dense K=576 baseline (honest testdev protocol; shared eval dataset)
├── static/             Static pruning (CLS-attn / random / spatial / L2) + VisionZip
│                       + the two question-conditioned selection probes (CLIP-space, LM-attention)
├── dynamic/            Confidence cascade (K1→K2 on low confidence) + τ-sweep analysis
├── eval_runners/       TextVQA / POPE / ScienceQA entry points
└── shared/             StaticPrunedLlava model, official scorers, FLOPs, dataset, utils

VQA_V2/                 Canonical VQAv2 track (trained modules, generation-protocol evaluation)
├── dense/  static/  dynamic/    model + configs per variant (dynamic also holds its trainer)
└── shared/             datasets, generation evaluator, oracle/cascade analyses, caching, utils

VQA_V2_early_proxy/     Retired first iteration (classification-head proxy) — kept only for the
                        ablation history; its numbers back no claim.
```

All ~120 modules run as `python -m <Track>.<path.to.module>` from the repository root.
Datasets (`data/`) and experiment outputs (`outputs/`, `*/outputs/`) are local-only and git-ignored.

---

## Installation

```bash
conda create -n vlm_env python=3.11 -y
conda activate vlm_env
pip install -r requirements.txt
```

Backbone: [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
(CLIP ViT-L/14@336 → 576 visual tokens + Vicuna-7B), downloaded automatically on first run.
The torch 2.3.0 / transformers 4.46.3 pins are **result-critical** — all frozen numbers were produced
with exactly these versions.

## Data setup

Place the datasets under `data/` (none are shipped with this repository):

| Dataset | Used for | Source |
|---|---|---|
| `data/vqav2/` (COCO train2014/val2014 + Q/A JSONs) | VQAv2 track, POPE images | [visualqa.org](https://visualqa.org/download.html) |
| `data/gqa/` (images + balanced question JSONs) | GQA track | [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html) |
| `data/textvqa/` (images + val JSON + OCR tokens) | TextVQA ±OCR | [textvqa.org](https://textvqa.org/dataset/) |
| `data/pope/coco/` (3 subset JSONs) | POPE | [POPE repo](https://github.com/RUCAIBox/POPE) |
| `data/scienceqa/sqa_img_test.parquet` | ScienceQA-IMG | [lmms-lab/ScienceQA](https://huggingface.co/datasets/lmms-lab/ScienceQA) |

## Reproducing the results

**GQA track** (training-free; every run uses the locked protocol — image padding, official prompts and
scorers, greedy decoding, `max_new_tokens=64`, batch size 1):

```bash
# dense baselines (validate against the published numbers first)
python -m GQA.dense.run_dense_testdev --image_pad
python -m GQA.eval_runners.run_textvqa --method none --keep_k 576
python -m GQA.eval_runners.run_pope    --method none --keep_k 576
python -m GQA.eval_runners.run_sqa     --method none --keep_k 576

# static frontier (repeat for K ∈ {144, 192, 288, 432})
python -m GQA.static.run_static_testdev --method cls_attn --keep_k 288

# confidence cascade + the τ-sweep (the sweep is CPU-only and reuses saved per-sample outputs)
python -m GQA.dynamic.run_speculative_testdev --tau 0.55
python -m GQA.dynamic.cascade_sweep

# question-conditioned selection probes (the second negative result)
python -m GQA.static.run_clip_probe
python -m GQA.static.run_qcond_probe
```

**VQAv2 track:**

```bash
# train the dynamic model (budget controller + answer head; backbone frozen)
python -m VQA_V2.dynamic.train_dynamic \
    --config VQA_V2/dynamic/llava_dynamic_150k_10k_fullvocab.yaml --output-dir VQA_V2/outputs/dynamic_run

# canonical generation-protocol evaluation (works with any checkpoint type)
python -m VQA_V2.shared.evaluation.generate_and_score \
    --config VQA_V2/static/llava_static_clsattn_150k_10k_fullvocab_k288.yaml \
    --model-type static --output-path VQA_V2/outputs/static_k288_eval.json

# analyses (no GPU): oracle headroom, per-type curves, cascade realizability, figures
python -m VQA_V2.shared.evaluation.instance_headroom
python -m VQA_V2.shared.evaluation.per_type_accuracy <eval.json>
python -m VQA_V2.shared.evaluation.make_figures
```

## FLOPs convention (unified across all experiments)

All reported FLOPs use the **FastV Eq. 5 full-LM prefill** convention from the published literature:

```
FLOPs = T · (4·n·d² + 2·n²·d + 2·n·d·m)        T=32 layers, d=4096, m=11008
n = K_visual + n_text   (measured full non-visual prompt tokens per benchmark:
                         VQAv2 35 · GQA 34 · TextVQA 86/32 · POPE 21 · ScienceQA 108)
```

Pruning is *physical* (prune-before-LLM): every transformer layer processes the shortened sequence.
Cascade runs are charged honestly — an escalated sample pays for **both** passes. The attention-only
proxy `2·T·n²·d` is available as a secondary diagnostic
(`python -m VQA_V2.shared.evaluation.flops` prints the full conversion table).

---

## Honest-evaluation commitments

This project adopts the evaluation safeguards motivated by the recent token-pruning critique
literature (Wen et al., *"Token Pruning in MLLMs: Are We Solving the Right Problem?"*, Findings of
ACL 2025):

- every dynamic-vs-static comparison is made at **matched average compute** against a **tuned** static
  baseline that uses the *same* selection signal;
- headline accuracy always comes from **official open-ended generation protocols** with official
  scorers — never from classification-head proxies;
- the dense pipeline must reproduce the published numbers before any pruning claim is interpreted;
- random / spatial / feature-norm selection floors are reported alongside every result;
- negative results are reported exactly as found.

## Author

**Mo Nafees** — MSc thesis, 2026.

This thesis is solo work: I designed, implemented, and ran all of the code, experiments, and analyses
in this repository myself. The thesis, *Dynamic, Question-Conditioned Visual Token Pruning for
Efficient Vision–Language Models*, has an accompanying paper in preparation. If you use the
oracle-headroom diagnostic in your own work, please cite the thesis/paper.
