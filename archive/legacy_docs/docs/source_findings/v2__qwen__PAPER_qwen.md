# Selection, Not Budget: A Rigorous Study of Question-Conditioned Dynamic Visual-Token Pruning on Modern VLMs

*Mo Nafees — MSc thesis Part III. Draft. Frozen Qwen2.5-VL-7B (current SOTA open VLM).*

---

## Abstract

Dynamic, question-conditioned visual-token pruning is usually framed as two coupled choices: *which*
tokens to keep (selection) and *how many* per sample (an adaptive budget). A growing line of work
(ATP-LLaVA, Dynamic-LLaVA, PACT) makes the per-sample **adaptive budget** a central contribution. We
ask, on the current SOTA backbone (Qwen2.5-VL-7B, native dynamic resolution), how much each choice
actually buys — measured **honestly**: strong baselines, matched FLOPs, oracle-noise corrected, and
training-aware. We find a sharp asymmetry. **Selection dominates:** keeping the tokens a mid-layer,
genuinely question-conditioned attention map points to retains **94.8% of dense accuracy at an 88% FLOP
reduction** on DocVQA (92.2 vs blind 32.2 at 128 tokens), and the advantage is +60pp. **The per-sample
budget is a mirage:** its oracle headroom is small (+0–7pp) and, crucially, **not realizable** — a
trained budget predictor captures **0%** of it on DocVQA and **hurts (−8.4pp)** on InfoVQA, because good
selection already fits most samples in the minimum budget and the residual hard cases are either
unpredictable (DocVQA, hard-tail AUC 0.59) or unfixable by more tokens (InfoVQA). The practical message:
**invest the token budget in question-conditioned selection; per-sample budgeting is not worth it.** Our
contribution is the rigorous joint decomposition — including a monotone oracle-noise correction and a
training-aware predictor test — on a modern high-numbers backbone.

---

## 1. Introduction

Visual tokens dominate VLM inference cost, and token pruning promises large savings. Methods split into
selecting *which* tokens to keep (FastV, VisionZip, FEATHER, FlashVLM) and adapting *how many* per sample
(ATP-LLaVA, Dynamic-LLaVA, PACT). The field is crowded and often evaluated optimistically — weak
baselines, FLOPs≠latency, and (we show) an oracle-noise illusion in the adaptive-budget claims.

This paper does not propose a new pruning *method*. It answers, rigorously, **where the gains in
question-conditioned dynamic pruning actually come from**, on the current SOTA open VLM. We decompose the
two axes and find selection is everything and per-sample budget is essentially worthless once selection
is good — a result that directly questions the premise of adaptive-budget pruning.

**Contributions.**
1. A clean **selection-vs-budget decomposition** of question-conditioned pruning on Qwen2.5-VL, with
   high, credible numbers (dense 85–97%).
2. A **monotone oracle-noise correction** for per-sample budget headroom (most prior oracle/upper-bound
   analyses ignore it), plus a **training-aware** budget-predictor test.
3. The finding that **per-sample budgeting is a mirage** (predictor captures 0% / hurts) across DocVQA,
   ChartQA, InfoVQA, with a mechanistic explanation (selection collapses the budget spread; the residual
   hard tail is unpredictable or unfixable).

---

## 2. Setup

- **Model (frozen):** Qwen2.5-VL-7B-Instruct (28-layer LLM, d=3584; native dynamic resolution → hundreds–
  thousands of visual tokens). Pruning = prune-before-LLM; we keep a subset of visual tokens **and their
  original M-RoPE 3D positions** (RoPE is position-value based), then decode. The harness is validated:
  keep-all reproduces stock generation **exactly** (6/6 on DocVQA, 5/5 on InfoVQA).
- **Benchmarks/metrics:** DocVQA (ANLS), ChartQA (relaxed-acc, human+augmented balanced), InfoVQA (ANLS).
  Generation protocol, official scorers, n=200–400. Dense fidelity reproduces published: DocVQA 94.85
  (pub 95.7), ChartQA 85.6 (pub 87.3). (InfoVQA dense 74.7 vs 82.6 — a resolution-cap gap; affects
  absolute, not the relative decomposition.)
- **Selectors:** blind **uniform** and **norm** (the hard-to-beat baselines per the analysis literature);
  **FastV-style** early-layer attention; **QC** = mid-layer (L16–20) question→visual attention. FLOPs:
  FastV Eq. 5 convention with Qwen2.5-7B constants.

---

## 3. Selection dominates (and is genuinely question-conditioned)

**K-curve (DocVQA, ANLS, dense 97.2).**

| K | uniform | norm | **QC** |
|---|---|---|---|
| 64  | 23.9 | 13.8 | **84.7** |
| 128 | 32.2 | 21.0 | **92.2** |
| 256 | 61.1 | 37.1 | **94.7** |
| 512 | 87.7 | 75.7 | **97.2** |

QC retains **94.8% of dense at 88% FLOP reduction** (K=128); the selection advantage is **+60pp**
(task-dependent: +60 on DocVQA, +22 on ChartQA — biggest where images are densest). Because we prune
**before** the LLM, this is a real **3.3× wall-clock decode speedup** (LLaVA-1.6, 355→107ms at K=128;
3.9× at K=64) — FLOPs → latency, unlike FastV's inside-layer pruning.

**The signal is mid-layer, not early (FastV).** Selector quality by LLM layer (K=128):

| layer | 2 (FastV) | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| ANLS | 44.6 | 45.7 | 47.7 | 73.7 | 93.2 | **94.9** | 91.2 |

Early-layer attention (FastV) is ~50pp below the mid-layer signal — confirming FEATHER's localization
finding on the modern backbone.

**Genuinely question-conditioned (control).** Selecting by a *mismatched* question gives 37.0 ANLS
(≈ blind 32.2); the *real* question gives 93.8 → **92% of the gain is due to the question**, not a better
blind saliency.

*(Honesty: the QC selector is the mid-layer **teacher** — it needs a full forward. The deployable cheap
version is occupied by FlashVLM/VisionSelector and is cited, not claimed. Our point is the **decomposition**,
not a new selector.)*

---

## 4. Per-sample budget is a mirage

We fix the selector and vary only K, measuring the per-sample budget headroom with a **monotone** oracle
(a sample counts at budget K only if correct at K *and all larger* — removing the noise that inflates
naive oracle/upper-bound estimates), then test a **trained predictor**.

**4.1 The oracle headroom is small, and shrinks with good selection.**

| selector | DocVQA | ChartQA | InfoVQA |
|---|---|---|---|
| uniform (blind) | +20.7pp | — | — |
| **QC (good)** | **+7.5 / +4.2pp** | **+2.7pp** | **−0.2pp** |

The large apparent budget benefit under *blind* selection (+20.7pp) is a symptom of weak selection;
a good selector collapses it (and cuts avg tokens 337→91).

**4.2 The headroom is not realizable (training-aware).** We train a per-sample budget predictor (the
adaptive-budget premise) on cheap QC-attention features, 5-fold CV, n=400:

| | DocVQA | InfoVQA |
|---|---|---|
| oracle dynamic | +4.2pp | −0.2pp |
| **trained predictor** | **+0.0pp** | **−8.4pp** |
| hard-tail predictability (AUC) | 0.59 | 0.72 |
| % correct at minimum (32 tok) | 81% | 55% |

- **DocVQA:** the oracle headroom exists but the hard tail (18% needing >32 tokens) is ~unpredictable
  (AUC 0.59) → the predictor collapses to "minimum-for-all" and captures **0%**.
- **InfoVQA:** there is **no** oracle headroom (−0.2pp); the hard tail *is* predictable (AUC 0.72) but
  more tokens don't fix it (those samples fail at full resolution) → the predictor only mis-allocates and
  **hurts (−8.4pp)**.

Either way — unrealizable or non-existent — **per-sample dynamic budgeting provides no realizable benefit.**

---

## 4.3 Generality across scale and architecture

The budget mirage holds across **2 families × 2 scales** (DocVQA, n=400; each pruner re-validated
keep-all==stock; each model's own best QC layer):

| model (family/scale) | dense | oracle budget | trained predictor | hard-tail AUC |
|---|---|---|---|---|
| Qwen2.5-VL-3B | 91.7 | +3.1pp | **−1.4pp** | 0.55 |
| Qwen2.5-VL-7B | 97.2 | +4.2pp | **0%** | 0.59 |
| Qwen2.5-VL-32B (4-bit) | 90.5 | +5.0pp | **+0.1pp (2%)** | 0.64 |
| LLaVA-1.6-7B (diff. architecture) | 67.8 | +3.1pp | **−10.7pp** | 0.60 |

On every model — **4 models, 2 families, 3 scales (3B→32B)** — selection dominates (+56–60pp), oracle
budget headroom is small (+3–5pp), the **trained predictor never helps** (0–2% or hurts), and the hard tail
is ~unpredictable (AUC 0.55–0.64). Confirmed at full-val scale (Qwen-7B n=1000: predictor 0%, dense 95.0).
A **general phenomenon across families, scales, and benchmarks** — not a single-model artifact.

## 5. Discussion

In question-conditioned dynamic pruning on a modern VLM, **which tokens you keep is everything; how many
per sample is not worth it.**

**Implication for adaptive-budget methods (a re-evaluation by construction).** Methods such as ATP-LLaVA,
Dynamic-LLaVA, and PACT add a *learned per-instance token budget* as a central contribution. Our trained
budget predictor (§4.2) **is exactly that mechanism** — a small learned head that maps per-sample features
to a token budget. We show it captures **0% of even the (small, noise-corrected) oracle headroom, and
often hurts**, across two families and two scales, because (i) good selection already fits most samples in
the minimum budget, and (ii) the residual hard tail is unpredictable (AUC ≈ 0.55–0.60) or unfixable by
more tokens. So the adaptive-budget component these methods headline is, under honest accounting, an
**oracle-noise illusion or an unfixable-tail artifact**; their real gains owe to *selection*. We recommend
that any dynamic-budget claim be reported with (a) a **monotone oracle correction** (to remove the noise
that inflates per-sample upper bounds) and (b) a **training-aware predictor test at matched FLOPs** (to
show the headroom is realizable, not just present).

## 6. Limitations
- Frozen model; the QC selector is a teacher (deployable cheap selection is prior work, cited).
- n=200–400 subsets (effect sizes large and consistent across 3 models; full-val tables in progress).
- Three models / two families (Qwen2.5-VL 3B+7B, LLaVA-1.6); broader families (InternVL) would strengthen
  further. InfoVQA dense below published due to a resolution cap (relative decomposition unaffected).
- Latency: we report both analytical FLOPs and measured LLM-generate latency (3.3× at K=128); the QC
  *selector* (mid-layer teacher) is not itself FLOPs-cheap — the deployable speedup assumes a cheap
  selector (prior work, cited), and our contribution is the decomposition, not the selector.

## 7. Conclusion
Across DocVQA, ChartQA, and InfoVQA on Qwen2.5-VL, question-conditioned **selection** retains ~95% of
dense accuracy at ~88% FLOP reduction, while per-sample **budgeting** — the premise of adaptive-pruning
methods — yields no realizable benefit (trained predictor: 0% of oracle, or worse). *Invest in selection;
per-sample budgeting is a mirage.*
