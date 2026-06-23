# Where Does Visual-Token Pruning Actually Help? A Rigorous Study of Resolution, Selection, and Dynamic Budgets

*MSc thesis — Part II. Draft. Mo Nafees, 2026.*
*(Companion to Part I, "Are We Solving the Right Problem?", which showed that on a frozen
low-resolution VLM, dynamic visual-token pruning has no headroom over a static baseline.)*

---

## Abstract

Visual-token pruning promises large efficiency gains for vision–language models (VLMs), and a
crowded literature reports impressive FLOPs reductions. We ask a more basic question: *where, and
by how much, does pruning genuinely help — once it is measured honestly?* Studying frozen LLaVA-1.5
(336px, 576 tokens) and LLaVA-1.6/AnyRes (≈2302 tokens) with a generation protocol and official
scorers, we find: **(1)** pruning has essentially no headroom at low resolution and only on
reasoning tasks; the room is concentrated in **high-resolution reading/document** tasks, jointly set
by resolution and task token-demand. **(2)** Where room exists, almost all of it comes from *which*
tokens are kept; the discriminative signal is a **mid-layer, genuinely question-conditioned**
attention map (early-layer attention, as used by FastV, is *worse than* CLS saliency). **(3)** The
*dynamic per-sample budget* — the headline of many adaptive-pruning papers — has real but **modest**
headroom once we remove **oracle noise**: a naive per-sample oracle overstates it by 2–13×. We
contribute a simple **monotone decomposition** that reports this honestly, and an explicit
**accuracy-vs-FLOPs frontier** that places our (frozen) numbers as *consistent with*, not beyond,
published SOTA. The work is a methodological cautionary study: weak baselines, FLOPs-dishonesty,
oracle-noise, and position bias each inflate pruning claims, and we quantify all four.

---

## 1. Introduction

Part I of this thesis built a dynamic, question-conditioned token-pruning method on a *frozen*
LLaVA-1.5 and found, via an oracle-headroom diagnostic, that it could not beat a static baseline at
matched compute — there was simply no room. That negative result raised the question this part
answers: **was the problem the method, or the regime?**

We investigate by moving to a higher-resolution backbone (LLaVA-1.6 / AnyRes, ≈2302 visual tokens vs
576) and asking, with deliberate rigor, three questions:
1. *Does pruning even have room here, and on which tasks?* (resolution × task)
2. *If so, where does the room come from — which tokens, or how many?* (selection vs budget)
3. *How large is each effect once measured honestly* (strong baselines, matched FLOPs, no oracle noise)?

**Contributions.**
- A resolution × task-demand characterization of *where* pruning headroom exists (§4.1).
- An analysis of the *selection* signal: a mid-layer, provably question-conditioned attention map,
  contrasted with the early-layer (FastV) and CLS (VisionZip-style) signals (§4.2).
- A **monotone oracle-noise decomposition** showing the dynamic-budget headroom is modest, and that
  the commonly-reported naive oracle overstates it by 2–13× (§4.3).
- An honest accuracy-vs-FLOPs frontier and an explicit accounting of four ways pruning results get
  inflated (§4.4, §5).

We make no SOTA claim. Every number is from a *frozen* model (no fine-tuning); the value is the
honest characterization, in the lineage of Part I.

---

## 2. Related Work (positioning)

High-resolution VLM token pruning is mature. **FastV** (ECCV'24) prunes by early-layer (layer-2)
attention; **VisionZip** and CLS-attention methods score by the vision encoder's CLS token;
**SparseVLM**, **CDPruner**, **HiRED**, **FEATHER** add text/question conditioning, multi-criteria
scoring, or per-region budgets, and report ~90%+ retention at aggressive budgets on LLaVA-NeXT.
Notably, **FEATHER already prunes using layer-8/16 attention** and documents that raw early-layer
attention has RoPE position bias (FastV "worse than random" under aggressive pruning).

Our findings are consistent with this literature rather than beyond it — and that is the point. We
do not propose a new SOTA method; we provide the **honest measurement framework** (matched-FLOPs
frontier, monotone oracle decomposition) that much of this literature omits, and we use it to locate
exactly where the real headroom is and how large it is.

---

## 3. Experimental setup

- **Models (frozen):** LLaVA-1.5-Vicuna-7B (CLIP-336, 576 tokens) and LLaVA-1.6-Vicuna-7B (same
  vision tower + LLM, AnyRes tiling, ≈2302 tokens). Same Vicuna-7B (T=32, d=4096, m=11008).
- **Benchmarks / metrics:** TextVQA (no-OCR; VQA soft-acc), GQA testdev-balanced (strict exact-match),
  DocVQA (ANLS), ChartQA (relaxed accuracy). n=300 subsets (deterministic first-N), generation
  protocol, greedy, official scorers.
- **Pruning:** prune-before-LLM (all 32 layers see K visual tokens). Selectors — *blind* (CLIP
  CLS→patch attention, packed across AnyRes tiles), *FastV* (early-layer LLM attention), *QC* (mid-
  layer LLM question→visual attention). AnyRes scores are aligned to the packed token sequence by
  passing them through the model's own `pack_image_features` (no geometry re-implementation); the
  splice is verified to reproduce the stock model exactly when no token is dropped.
- **FLOPs:** FastV Eq. 5 convention (Part I's `flops.py`): prune-before-LLM = T·f(K+n_text);
  faithful FastV = 3 full-token layers + 29 at K.

---

## 4. Results

### 4.1 Room exists only at high resolution, and chiefly on reading/document tasks

Retention at K=64 (% of each model's own dense; blind selector):

| benchmark (type) | LLaVA-1.5 → K=64 | LLaVA-1.6 → K=64 |
|---|---|---|
| TextVQA-noOCR (reading) | 102% (flat) | 57% |
| GQA (reasoning) | 85% | 66% |
| DocVQA (document) | — (cannot read at 336px) | 19% |

Low resolution is a "postage stamp": pruning to 64 tokens is near-lossless on reading and only
graceful on reasoning — nothing to remove. High resolution binds, but **task-dependently**: the room
is largest on reading/document tasks and only moderate on reasoning. *Pruning headroom = resolution ×
task token-demand.* Implication: FLOPs cuts reported on LLaVA-1.5-576, or on reasoning benchmarks,
measure little.

### 4.2 The room is in *which* tokens — a mid-layer, question-conditioned signal

Selector accuracy at K=128 (TextVQA-noOCR; dense 61.4):

| selector | acc | note |
|---|---|---|
| FastV early-layer (L2) | 39.1 | **below blind** — RoPE/position bias |
| blind CLS-attn | 44.2 | question-blind baseline |
| mid-layer attn (L12/16/20) | 51.7 / 54.7 / 54.0 | **+10.5pp** over blind |

The discriminative signal lives in the *middle* of the network, not early (FastV) or in the vision
encoder (CLS). It is **genuinely question-conditioned**: selecting tokens with a *mismatched* question
collapses accuracy to 43.8 (≈ blind 44.2), while the *real* question gives 54.7 — **104% of the gain
is attributable to the question.** The advantage is largest at aggressive compression and vanishes
when the budget is ample (TextVQA +18pp@K64 → +0.6pp@K256; DocVQA +36pp@K64 → +0.9pp@K1152).

*Caveat (FLOPs honesty):* the mid-layer signal requires running 16 full-token layers, so it is a
**teacher/upper-bound**, not a deployable method. Realising it cheaply (a learned front-end selector)
is deferred to Part B.

### 4.3 The dynamic per-sample budget headroom is modest — once oracle noise is removed

Fixing the selector and varying only K, we measure the per-sample headroom of a *perfect* budget
oracle. Reporting it naively (per-sample best K) is misleading: the oracle exploits noise (crediting
a sample at a budget where it is *accidentally* correct). The **monotone** oracle — a sample counts at
budget K only if it is correct at K *and every larger budget* — removes this. Binary correct@0.5,
consistent metric:

| benchmark | naive band | **honest (monotone, matched-FLOPs)** |
|---|---|---|
| TextVQA-noOCR | +7.0pp | **+2.6pp** |
| DocVQA | +9.7pp | **+5.0pp** |
| ChartQA | +6.7pp | **+0.5pp** |

The naive per-sample oracle **overstates the dynamic-budget headroom by ~2–13×.** The genuine,
noise-free win is mostly *efficiency* (full accuracy at ~2× fewer tokens), not accuracy — and much of
even that is reachable by a fixed budget. This recasts the central premise of adaptive-budget pruning
papers: the per-sample-budget prize is real but small.

### 4.4 An honest accuracy-vs-FLOPs frontier

Blind and QC share identical generation FLOPs at a given K (both prune-before-LLM), so the gap is a
clean matched-FLOPs gap; faithful FastV costs *more* (3 full-token layers). Retention vs our own dense
(DocVQA, ANLS, dense 67.2):

| K | FLOP-red | blind | QC | QC-retention |
|---|---|---|---|---|
| 128 | 94% | 27.8 | 55.1 | 82% |
| 256 | 89% | 41.4 | 57.4 | 86% |
| 512 | 79% | 50.7 | 61.4 | 91% |
| 768 | 68% | 57.9 | 63.2 | **94%** |
| 1152 | 52% | 63.0 | 63.9 | 95% |

QC reaches ~94% retention at 68% FLOP-reduction; the advantage over blind concentrates at aggressive
compression and vanishes by K=1152. These retention numbers are *consistent with* published SOTA
(HiRED DocVQA 68.7 @ 40% budget; CDPruner ~92% @160 tokens) and we do **not** claim to exceed them.
Our dense (67.2, n=300) is robust to generation length (identical at 20 vs 50 new tokens, 0% truncated)
and sits slightly below published LLaVA-NeXT dense; we therefore report **retention vs our own dense**.

### 4.5 Four ways pruning results get inflated (demonstrated)

1. **Weak baselines** — "+36pp over blind CLS" is real but vs SOTA the margin is ~par.
2. **FLOPs dishonesty** — the mid-layer teacher is not FLOPs-cheap; report matched FLOPs.
3. **Oracle noise** — naive per-sample oracle inflates dynamic-budget headroom 2–13×.
4. **Position/layer bias** — early-layer attention (FastV) is below blind; raw attention has RoPE bias.

---

## 5. Discussion

The headline is not a method but a map. Pruning research should target the regime where it matters
(high-res reading/document), report against strong baselines at matched FLOPs, and decompose
dynamic-budget claims to separate genuine headroom from oracle noise. Our results align with the
strongest existing methods (FEATHER's mid-layer signal; CDPruner/HiRED retention), which is itself a
useful confirmation obtained under honest measurement.

## 6. Limitations

- Frozen model only (no fine-tuning); the deployable cheap selector is future work (Part B).
- n=300 subsets. Stability check (DocVQA n=1000) confirms the conclusions: dense and QC are stable
  (QC K=128 55.1→54.0, K=256 57.4→57.4; dense 67.2→67.9), while the blind baseline drifts *down* ~5pp,
  so the reported QC-over-blind gaps are if anything conservative (e.g. K=128 +27→+31pp at n=1000).
  Full-val tables remain to be run for the final submission.
- One backbone family (LLaVA-1.5/1.6-Vicuna-7B); no Qwen2-VL / larger scales.
- The mid-layer teacher's FLOPs are not deployable as-is.
- Faithful FastV is approximated by an early-layer-attention proxy; a layer-split implementation is pending.

## 7. Conclusion

Across resolution, layer, selection, and budget, we give an honest account of where visual-token
pruning helps and by how much. The room is real but localized (high-res reading/document); the gains
come from a mid-layer, question-conditioned *selection* signal; and the much-touted dynamic per-sample
*budget* is, after removing oracle noise, modest. Part I showed there was no room on a frozen low-res
model; Part II shows where the room actually is — and how easily it is overstated.
