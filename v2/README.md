# `v2/` — Trained, Question-Conditioned Visual Token Pruning (thesis Part II)

> **Status: scaffold.** This folder holds the *plan* and the reusable foundations. The model and
> training code are built incrementally, each piece tested before the next. Nothing here imports from
> or modifies `GQA/`, `VQA_V2/`, `VQA_V2_early_proxy/`, or existing `data/` subfolders — v1 stays frozen.

## The idea in one line

A **lightly fine-tuned** LLaVA-1.5-7B (LoRA on the LLM + trained projector) that runs at **any token
budget**, a **question-conditioned selector** that picks *which* patches to keep, and a **learned budget
head** that picks *how many* — evaluated honestly (generation protocol, official scorers) against an
**equally-trained static baseline at equal FLOPs**.

## Why v2 (what v1 proved, and what training changes)

v1 (the frozen-backbone thesis, now finished) showed: on a frozen LLaVA-1.5, per-sample budgeting has
no room (oracle band 4.9–9.1%) and question-conditioned selection cannot beat image saliency. Both
walls are properties of the *frozen* model:

- it already extracts everything from ~144 saliency-ranked tokens (the "easy" wall), and
- ~30–49% of samples fail even at 576 tokens (the capability ceiling),
- and frozen CLIP / RoPE-biased LM attention give no usable question-conditioned signal.

**Training removes all three.** Operating at **low budgets (K = 32–128)** on a model adapted to run
there makes the accuracy-vs-K curve steep and *heterogeneous per sample* — the condition under which a
per-sample budget can finally win. And with training in the loop the selector learns from the answering
model itself, instead of probing broken frozen signals.

The honest rival is therefore **static CLS pruning with the *identical* LoRA training**, at equal FLOPs.
v2 is designed so the dynamic mechanism has advantages an equally-trained static method *cannot* copy:
per-sample variation in how many tokens, and per-question variation in which tokens.

## Architecture (and the FLOPs cost of each piece)

```
 image ─► CLIP ViT-L/14 (FROZEN) ─► 576 patch features        [cost: CONSTANT for all methods]
 question ─► token embeddings ──────────────┐
                                            ▼
   (B) SELECTOR  [trained, NEGLIGIBLE FLOPs]
       relevance[i] = MLP( cross_attn(question → patch_i), [optional saliency_i] )
       → always a function of the question; saliency is at most one input feature
   (C) BUDGET HEAD  [trained, NEGLIGIBLE FLOPs]
       K = g( question, image summary (CLIP CLS), shape/entropy of relevance map )
       K ∈ {32, 64, 128, 256, 576}   (ladder tunable from the pilot)
                                            ▼
   keep top-K patches (original spatial order) ─► 448+ patches PHYSICALLY DROPPED
                                            ▼
   (D) PROJECTOR [trained, tiny] ─► (E) LLM + LoRA, budget-aware [trained]
                                            ▼
                                      generated answer   [DOMINANT cost — what pruning shrinks]
```

**Question-conditioning, kept safe.** The selector output always depends on the question. Saliency
enters only as an optional input feature the question-conditioned network may use or ignore — which
guarantees it can never regress below the proven saliency baseline (v1's selector had no such floor and
fell below random). A scalar-gate variant `α(q)·relevance + (1−α(q))·saliency` is kept only as an
*interpretability* exhibit (plot α(q) by question type).

## Training (staged curriculum — order matters)

- **Stage 0 — Teacher attribution (offline, one-time):** run the dense 576-token model, extract per-sample
  visual-token importance via attention×gradient attribution (gradient term corrects the RoPE position
  bias FEATHER documents). Subsample ~50–80K. → targets for warm-starting the selector.
- **Stage 1 — Elastic backbone:** train projector + LLM-LoRA with selection fixed (saliency/random) at a
  **randomly sampled K** each step → the LLM learns to function at every budget. Optional learned budget
  embedding (ablated).
- **Stage 2 — Selector:** warm-start on Stage-0 targets, then refine end-to-end (soft→hard top-K; reuse
  the soft-selection idea from v1's `VQA_V2/dynamic/token_selector.py`) so the answer loss trains it.
- **Stage 3 — Budget head:** train on **token-sufficiency labels** (smallest K whose gold-answer NLL is
  within ε of the K=576 NLL — cheap, 5 short forwards/sample, no generation). Learns token-need directly.
- **Stage 4 — Light joint polish (optional):** brief co-adaptation of (B)+(C)+LoRA at a tiny LR.

## FLOPs accounting (central to the thesis)

Convention unchanged from v1 — FastV Eq. 5 full-LM prefill, `f(K)=T·(4nd²+2n²d+2ndm)`, `n=K+n_text`.
Additions for v2, reported explicitly:
- `f_selector`, `f_budget_head` — counted; shown to be ~4 orders of magnitude below the LLM savings.
- **LoRA = 0 extra at inference** (merged into base weights).
- **Decode savings** reported as a bonus (pruned tokens never enter the KV cache).
- **One LLM pass** (predict-then-run); the run-then-escalate cascade is one ablation, charged for both passes.
- **Win criterion:** a point counts only if it lies **above the equally-trained static frontier at equal
  total FLOPs** (selector + budget head included).

## Datasets

- **Training:** the LLaVA-1.5 instruction mix `llava_v1_5_mix665k.json` (one file; image-grounded subset,
  ShareGPT text-only rows dropped). Stratified ~300K subsample for the pilot, full mix later. Same data
  PruMerge+/ATP train on → fair comparison + bias diversity. Setup: see `data_setup/README.md`.
- **Evaluation (headline):** GQA, TextVQA (±OCR), POPE; **contrast:** ScienceQA-IMG; **sanity:** MME/MMBench;
  **comparability appendix:** VQAv2; **deployment win:** mixed-workload pool. **Stretch:** DocVQA/ChartQA
  (needs their train splits). All eval reuses the existing `data/` (read-only) and v1's official scorers
  (copied into `v2/shared/`).
- **Baselines:** dense+LoRA (ceiling); **static-CLS at each K with identical LoRA** (the real rival);
  elastic-uniform (Stage-1 at fixed K); published methods via the existing harness.

## Folder map

```
v2/
├── data_setup/   LLaVA-mix download checklist + loader (conversations format → existing collator)
├── model/        selector, budget_head, elastic LoRA wrapper, projector   [to build]
├── training/     stage0_attribution, stage1_elastic, stage2_selector, stage3_budget, stage4_polish  [to build]
├── eval/         generation eval + matched-compute FLOPs                   [to build]
├── shared/       COPIES of v1 helpers — flops.py, official_score.py, textvqa_score.py, m4c_evaluator.py,
│                 pope_score.py, eval_pope_official.py, metrics.py, utils/  (byte-identical to GQA/shared)
├── configs/      one yaml per stage                                        [to build]
└── README.md     (this file)
```

## First milestone — the pilot (the go/no-go gate)

Train the Stage-1 elastic model (CLS selection only), then measure the per-sample first-correct-K spread
at low K on val. If wide (literature predicts 20–40% on text tasks, vs v1's 6–9%) → build the full
method. If not → Components B+C are still independently publishable, and the mixed-workload result is a
second net. The pilot tests the single riskiest assumption before any method is built.

## Isolation rules (do not break v1)

1. v2 only **adds** files; it never edits/deletes anything in `GQA/`, `VQA_V2/`, `VQA_V2_early_proxy/`,
   or existing `data/` subfolders.
2. Reusable v1 logic is **copied** into `v2/shared/`, never imported.
3. New training data goes under `data/llava_mix/` (new) and via **symlinks** to existing GQA/TextVQA images.
4. Work stays on branch `v2-trained`; nothing is pushed to public `main` until ready. `data/`, `*.pt`,
   and `v2/outputs|feature_cache|logs` are git-ignored.
