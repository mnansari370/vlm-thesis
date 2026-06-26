# THESIS EVIDENCE LEDGER — single source of truth for evidence

> **Purpose.** Every number written in the thesis/paper must trace to a row here. Narrative anchor:
> [`THESIS_MASTER_PLAN.md`](THESIS_MASTER_PLAN.md). Created 2026-06-24; all values re-verified against
> saved artifacts on that date. **If a number is not in this ledger, do not write it as a result.**
> *(This file was `THESIS_LEDGER.md`; renamed and kept byte-for-byte during the Phase-3B docs
> consolidation — no claim, number, or path was changed.)*
>
> Status legend — **Thesis:** `safe` (write as-is) · `careful` (write with stated qualifier) · `risky`
> (do not write unless upgraded). **Paper:** `safe` · `careful` · `not-enough` (needs more work).
> **Placement:** `main` / `appendix` / `exclude`. **Backing:** `artifact` (in saved JSON/log) ·
> `computable` (script exists, output not saved) · `narrative-only` (only in a `.md`, not reproducible
> from saved results).

---

## A. Master summary (at-a-glance)

| ID | Claim (short) | Canonical value | v1/v2 | Backing | Placement | Thesis | Paper |
|----|---|---|---|---|---|---|---|
| L1 | Dense reproduces published | ≤1.65pp on 4 benchmarks | v1 | artifact | main | safe | safe |
| L2 | Per-sample budget ties tuned static | 75.76 vs 75.71 (+0.05pp) | v1 | artifact | main | safe | safe |
| L3 | Oracle budget band is thin | 4.93–9.13% | v1 | artifact | main | safe | safe |
| L4 | Naive frozen Q-cond selection fails | −32.04 / −5.58pp | v1 | artifact | main | safe | safe |
| L5 | Q-cond selection dominates | +59.99pp (K128) | v2 | artifact | main | careful | careful |
| L6 | Mid-layer > early-layer signal | L2 44.6 → L16 93.2 | v2 | artifact | main | safe | careful |
| L7 | Selection is genuinely Q-conditioned | 92.2% question-driven | v2 | artifact | main | careful | careful |
| L8 | Honest oracle budget headroom small | +7.50pp | v2 | artifact | main | safe | safe |
| L9 | Trained budget predictor ≈ captures none | +0.09pp (1.8%) | v2 | artifact | main | safe | careful |
| L10 | Budget mirage generalizes (families/scales) | 3B/32B/InfoVQA/LLaVA-1.6 | v2 | **narrative-only** | appendix→main if recomputed | risky | not-enough |
| L11 | Student recovers majority, below SOTA | 57.9% recovery; ~69% retention | v2 | artifact | main | careful | careful |
| L12 | Student Q-conditioned; data > epochs | 99.2%; 46→58% (3K→12K) | v2 | artifact | main | careful | careful |
| E1 | Token reduction | 86.8% (K128) | v2 | artifact | main | safe | safe |
| E2 | FLOP reduction | 87.9% (K128) | v2 | computable→artifact | main | careful | careful |
| E3 | Measured latency | 3.31× (K128) | v2 | artifact | main | careful | careful |
| E4 | Teacher selector cost | full LLM forward | v2 | artifact(code) | main | safe | safe |
| E5 | Student selector cost | negligible, no decoder fwd | v2 | artifact(code) | main | safe | careful |

---

## B. Full rows (14 fields each)

### L1 — Dense pipeline reproduces published accuracy
- **Claim:** Our frozen dense baselines match published LLaVA-1.5-7B within ≤1.65pp, so downstream negatives are trustworthy.
- **Number:** GQA 61.42/62.0 (−0.58); TextVQA-OCR 57.65/58.2 (−0.55); POPE-F1 85.78/85.9 (−0.12); SQA 65.15/66.8 (−1.65).
- **Metric:** accuracy / F1. **Task:** GQA, TextVQA, POPE, ScienceQA-IMG. **Model:** LLaVA-1.5-7B. **Track:** v1.
- **Setting:** locked honest protocol (image_pad, official prompt/scorer, greedy, bs=1). **n:** full splits.
- **File:** `results/thesis_main/gqa/results_frozen/all.json` ("validation"). **Placement:** main. **Thesis:** safe. **Paper:** safe.
- **Notes:** v2 backbones' dense is *below* published on some tasks (InfoVQA 74.7 vs 82.6) → for v2 report **retention**, not absolute.

### L2 — Per-sample dynamic budget ties tuned static (the budget negative)
- **Claim:** A trained per-sample budget controller equals a tuned static budget at matched average cost.
- **Number:** dynamic 75.76 vs static K265 75.71 = **+0.05pp**; dynamic avg K=264.3.
- **Metric:** VQA gen accuracy. **Task:** VQAv2 val 10K. **Model:** LLaVA-1.5-7B (frozen; trained head+controller). **Track:** v1.
- **Setting:** generation protocol, CLS-attn ranking, matched-K static comparison. **n:** 10,000.
- **File:** `results/thesis_main/vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json`. **Placement:** main. **Thesis:** safe. **Paper:** safe.
- **Notes:** also corroborated by the GQA confidence cascade tracing the static frontier (`results/thesis_main/gqa/results_frozen/all.json` "cascade"). Oracle ceiling 81.13% uses labels+noise (upper bound only).

### L3 — Oracle budget-sensitive band is thin (frozen low-res)
- **Claim:** The fraction of samples whose correctness depends on the token budget is small everywhere (<20%).
- **Number:** POPE 4.93 < TextVQA-OCR 6.08 < TextVQA-noOCR 6.24 < SQA 7.73 < GQA 9.13 (%).
- **Metric:** % of samples in the band (first-correct at K>min). **Task:** 4 benchmarks. **Model:** LLaVA-1.5-7B. **Track:** v1.
- **Setting:** per-sample first-correct-K over K∈{144,192,288,432,576}. **n:** full splits.
- **File:** `results/thesis_main/gqa/results_frozen/all.json` ("bands"); `results/thesis_main/gqa/testdev_frontier_analysis.json`. **Placement:** main. **Thesis:** safe. **Paper:** safe.

### L4 — Naive frozen question-conditioned selection fails
- **Claim:** On frozen low-res, the *naive* question-conditioned selectors (CLIP-space cosine; early-layer LM attention) lose to image-saliency CLS-attn, sometimes below random.
- **Number:** CLIP-Qcond − CLS = **−32.04pp**; LM-Qcond − CLS = **−5.58pp** (K=64).
- **Metric:** VQA soft-acc. **Task:** TextVQA-noOCR. **Model:** LLaVA-1.5-7B. **Track:** v1.
- **Setting:** K=64 selection probe. **n:** 5,000.
- **File:** `results/thesis_main/gqa/week1_all_numbers.json`; `results/thesis_main/gqa/results_frozen/tables.md` (T_C). **Placement:** main (as motivating negative). **Thesis:** safe. **Paper:** safe.
- **Notes:** frame as regime+signal-specific (it motivates the v2 mid-layer signal); not a claim that "selection never works."

### L5 — Question-conditioned selection dominates (v2)
- **Claim:** With a mid-layer question-conditioned signal at high resolution, selection vastly outperforms blind selection.
- **Number:** textattn(QC) **92.18** vs uniform **32.19** = **+59.99pp**; vs norm 20.98 = +71.2pp.
- **Metric:** ANLS. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B (frozen). **Track:** v2.
- **Setting:** K=128, prune-before-LLM, keep-all==stock validated. **n:** 200.
- **File:** `results/thesis_main/highres/qwen_kcurve_docvqa.json`. **Placement:** main. **Thesis:** careful. **Paper:** careful.
- **Notes:** ALWAYS qualify "Qwen-7B DocVQA, K=128, vs uniform, n=200." Gain ranges +17 to +71pp (see Reconciliation R2). Baselines are blind floors (uniform/norm), NOT strong published methods.

### L6 — The selection signal is mid-layer, not early (FastV)
- **Claim:** Question→visual selection quality peaks at a mid LLM layer; early-layer attention (FastV) is far below.
- **Number:** L2=44.6, L8=47.7, L12=73.7, **L16=93.2**, L20=94.9, L24=91.2.
- **Metric:** ANLS. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B. **Track:** v2.
- **Setting:** K=128 layer sweep. **n:** 120.
- **File:** `results/thesis_main/highres/qwen_layer_sweep.json`. **Placement:** main. **Thesis:** safe. **Paper:** careful.
- **Notes:** this is the *proxy* for FastV (early-layer attention), not a faithful layer-split FastV implementation.

### L7 — Selection is genuinely question-conditioned
- **Claim:** The selection gain is driven by the question, not disguised saliency.
- **Number:** real 93.84 / mismatched 36.98 / blind 32.19 → **92.2% question-driven**. (Student control: 99.2%.)
- **Metric:** ANLS. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B. **Track:** v2.
- **Setting:** mismatched-question control, K=128. **n:** 100 (teacher); 200 (student).
- **File:** `results/thesis_main/highres/qwen_control_docvqa.json`; `results/thesis_main/highres/distill/control_12k.json`. **Placement:** main. **Thesis:** careful. **Paper:** careful.
- **Notes:** n=100 for teacher control is small — rerun larger for the paper.

### L8 — Honest oracle budget headroom is small
- **Claim:** Under a monotone, FLOPs-matched oracle (noise removed), the per-sample budget headroom is small even with a good selector.
- **Number (CANONICAL):** **+7.50pp** (naive 99.01, band 1.82 → monotone 98.0). DocVQA. ChartQA: +2.66pp.
- **Metric:** ANLS (DocVQA) / relaxed-acc (ChartQA). **Task:** DocVQA, ChartQA. **Model:** Qwen2.5-VL-7B. **Track:** v2.
- **Setting:** monotone oracle, FLOPs-matched, QC selector. **n:** 200.
- **File:** `results/thesis_main/highres/qwen_oracle_docvqa_qc.json`; `results/thesis_main/highres/qwen_oracle_chartqa_qc.json`. **Placement:** main. **Thesis:** safe. **Paper:** safe.
- **Notes:** see Reconciliation R1 for why this is 7.50 and not 4.99/4.2/3.9.

### L9 — Trained budget predictor captures almost none of the oracle gain
- **Claim:** A realistic per-sample budget predictor captures essentially none of the (already small) oracle headroom — the "mirage."
- **Number:** oracle 4.99 → trained predictor **+0.09pp = 1.8% of oracle**; rule +0.72; best fixed 93.0; hard-tail AUC 0.643.
- **Metric:** ANLS. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B. **Track:** v2.
- **Setting:** MLP predictor on QC-attention features, 5-fold CV. **n:** 200.
- **File:** `results/thesis_main/highres/qwen_budget_eval.json`; `results/thesis_main/highres/qwen_budget_robust.json`. **Placement:** main. **Thesis:** safe. **Paper:** careful.
- **Notes:** "captures ≈0%" is shorthand for 1.8% / +0.09pp. Only DocVQA-7B is artifact-backed (cross-model = L10).

### L10 — Budget mirage generalizes across families/scales ⚠️ NARRATIVE-ONLY
- **Claim (NOT YET WRITABLE):** the mirage holds across 3B/7B/32B and LLaVA-1.6, and on InfoVQA.
- **Number (unverified):** 3B +3.1/−1.4, 32B +5.0/+0.1, LLaVA-1.6 +3.1/−10.7, InfoVQA −0.2/−8.4 (oracle/predictor).
- **Metric:** ANLS. **Task:** DocVQA/InfoVQA. **Model:** Qwen 3B/7B/32B, LLaVA-1.6. **Track:** v2.
- **Setting:** budget oracle + predictor. **n:** ~200–400 each.
- **File:** narrative in `docs/source_findings/v2__qwen__FINDINGS_qwen.md`; **raw data** exists (`results/paper_candidates/qwen_budget_data_{docvqa_3b,docvqa_32b,infovqa,docvqa_7b_n1000}.json`) but **the eval gains are not in any saved JSON/log**. **Placement:** appendix (→ main only if recomputed+saved). **Thesis:** risky. **Paper:** not-enough.
- **Notes:** **DO NOT WRITE the "general phenomenon" claim until the eval is rerun and per-model result files are saved.** Raw data is present, so this is recomputable cheaply.

### L11 — Cheap student recovers a meaningful part of the teacher, below SOTA
- **Claim:** A no-decoder-forward student recovers the majority of the teacher's selection gain but remains below dedicated SOTA selectors.
- **Number:** dense 95.0 / teacher 91.1 / **student 65.9** / blind 31.2 → **recovery 57.9%** (PASS); ~69% retention.
- **Metric:** ANLS. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B + CheapQCSelector. **Track:** v2.
- **Setting:** distill L16 teacher; train[0:12000] → held-out val[0:400], K=128. **n:** 400 (gate).
- **File:** `results/thesis_main/highres/distill/gate_12k.json`; `docs/source_findings/v2__distill__FINDINGS_distill.md`. **Placement:** main. **Thesis:** careful. **Paper:** careful.
- **Notes:** DocVQA only; SOTA cheap selectors ~90% retention. Recovery at other K: K64 47.9%, K256 45.0% (`gate_K64.json`,`gate_K256.json`).

### L12 — Student is genuinely question-conditioned; data > epochs
- **Claim:** The student's gain is question-driven, and more data (not more epochs) is the lever.
- **Number:** 99.2% question-driven; 3K→12K data lifts recovery 46→58%; 10→25 epochs overfits 58→52%.
- **Metric:** ANLS / recovery%. **Task:** DocVQA. **Model:** Qwen2.5-VL-7B + student. **Track:** v2.
- **Setting:** control + data/epoch ablation. **n:** 200/400.
- **File:** `results/thesis_main/highres/distill/control_12k.json`; `results/thesis_main/highres/distill/gate_12k_25ep.json`. **Placement:** main. **Thesis:** careful. **Paper:** careful.

### E1 — Token reduction (efficiency, kept separate)
- **Claim:** Selection removes most visual tokens. **Number:** **86.8%** (K128); 91.8% (K64); 76.8% (K256); 56.9% (K512).
- **Metric:** 1 − avg_tokens/dense_tokens (measured). **Task/Model:** DocVQA / Qwen-7B. **Track:** v2. **n:** 200.
- **File:** `results/thesis_main/highres/qwen_kcurve_docvqa.json` (avg_tokens) → `results/thesis_main/highres/qwen_flops_summary.json`. **Placement:** main. **Thesis:** safe. **Paper:** safe.

### E2 — FLOP reduction (efficiency, kept separate)
- **Claim:** Prefill FLOPs drop super-linearly with tokens. **Number:** **87.9%** (K128); 92.5% (K64); 78.5% (K256); 59.5% (K512). Dense 7.30 TFLOPs.
- **Metric:** FastV Eq.5 analytical prefill (T=28,d=3584,m=18944,n=K+40). **Task/Model:** DocVQA / Qwen-7B. **Track:** v2. **n:** 200.
- **File:** `results/thesis_main/highres/qwen_flops_summary.json` (script `src/analysis/qwen_flops.py`). **Placement:** main. **Thesis:** careful. **Paper:** careful.
- **Notes:** analytical, EXCLUDES selector cost (E4). State convention + exclusions every time.

### E3 — Measured wall-clock latency (efficiency, kept separate)
- **Claim:** Because pruning is before the LLM, FLOP cuts convert to real speedup. **Number:** **3.31×** (K128); 3.86× (K64).
- **Metric:** LLM generate() ms (warmup+sync). **Task/Model:** DocVQA / **LLaVA-1.6** (not Qwen). **Track:** v2. **n:** 40.
- **File:** `results/thesis_main/highres/llava_latency.json`. **Placement:** main. **Thesis:** careful. **Paper:** careful.
- **Notes:** different model from the FLOPs/accuracy headline; excludes selector cost.

### E4 — Teacher selector cost (the central caveat)
- **Claim:** The QC teacher selector is NOT FLOPs-cheap — it needs a full LLM forward. The reduction numbers exclude it.
- **Evidence:** `src/pruning/question_conditioned_selection/qwen_pruner.py` `encode_qc` (full `model.model(...)` forward to read L16 attention).
- **Track:** v2. **Placement:** main. **Thesis:** safe. **Paper:** safe.
- **Notes:** teacher = analysis/upper-bound selector; the deployable path is the student (E5 / L11).

### E5 — Student selector cost
- **Claim:** The student adds negligible FLOPs and needs no decoder forward (deployable), at the cost of lower retention.
- **Evidence:** `src/models/distillation/student_selector.py` (~20M params, pre-LLM features only). Recovery/retention in L11.
- **Track:** v2. **Placement:** main. **Thesis:** safe. **Paper:** careful.

---

## C. Reconciliations (resolve the confusing numbers — cite these in the thesis)

### R1 — Oracle headroom: 7.50 vs 4.99 vs 4.2 vs 3.9
Different measurements, not contradictions. **Canonical = +7.50pp.**

| Number | What it is | n | Backing | Use |
|---|---|---|---|---|
| **+7.50pp** | monotone, FLOPs-matched oracle gain (QC selector) — the upper bound | 200 | `qwen_oracle_docvqa_qc.json` | **canonical (cite this)** |
| +4.99pp | oracle vs *best single fixed budget* in the predictor eval (different baseline) | 200 | `qwen_budget_eval.json` | appendix only |
| +4.2pp | quoted in FINDINGS generality table | — | narrative-only | **drop** |
| +3.9pp | quoted as "full-val n=1000" | 1000 | narrative-only (log saved data only) | **drop unless recomputed+saved** |

Realizable predictor gain (paired with the oracle): **+0.09pp = 1.8% of oracle** (`qwen_budget_eval.json`).

### R2 — "+60pp" selection gain vs smaller gains
Gain = (model × task × K × baseline)-dependent; all true. Canonical headline = **Qwen-7B DocVQA, K=128, vs uniform, n=200 = +59.99pp**.

| Setting | QC − baseline | File |
|---|---|---|
| Qwen-7B DocVQA K128 vs uniform | +59.99pp | `qwen_kcurve_docvqa.json` |
| Qwen-7B DocVQA K128 vs norm | +71.2pp | `qwen_kcurve_docvqa.json` |
| Qwen-7B DocVQA K256 vs uniform | +33.6pp | `qwen_kcurve_docvqa.json` |
| LLaVA-1.6 DocVQA K128 vs blind | +27.4pp (n=300) / +30.9pp (n=1000) | `eval_highres_docvqa.json` / `eval_highres_docvqa_n1000.json` |
| LLaVA-1.6 ChartQA K128 vs blind | +17.3pp | `eval_highres_chartqa.json` |

### R3 — Token reduction vs FLOP reduction vs latency (never conflate)
- Token reduction (E1): **86.8%** @K128 — measured, from avg_tokens.
- FLOP reduction (E2): **87.9%** @K128 — analytical prefill (slightly higher; quadratic term).
- Measured latency (E3): **3.31×** @K128 — on LLaVA-1.6, n=40, decode only.
- None include the selector's own cost (E4).

### R4 — Teacher selector vs student selector
- Teacher (QC/textattn): the headline accuracy/selection numbers (L5–L8); **needs a full forward** (E4); analysis/upper-bound.
- Student (CheapQCSelector): the deployable path; negligible cost; recovers 57.9% / ~69% retention (L11–L12, E5).
- **Rule:** efficiency claims pairing "95% retention at 88% FLOP-cut" use the TEACHER selector; state that the deployable student is lower.

### R5 — v1 negatives vs v2 positives (not a contradiction)
Same instrument, different regime/signal — **all frozen.** v1 (576-token LLaVA-1.5, naive CLIP/early-LM signals) → selection fails + budget capped. v2 (high-res LLaVA-1.6/Qwen, mid-layer signal) → selection dominates + budget still a mirage. Axis = resolution × task token-demand × signal layer. **Not training.** (See `THESIS_MASTER_PLAN.md` §2 → v1→v2 transition.)

### R6 — Thesis-ready vs paper-ready
- Thesis-ready now: L1–L4, L8 (v1 + methodology + DocVQA-7B oracle/mirage). L5–L7, L9, L11–L12 are thesis-ready *with qualifiers (n, model, baselines)*.
- Paper-ready needs: full-val (L5,L7,L9), strong baselines (L5), recomputed+saved generality (L10), saved FLOPs artifact (E2 — now done), pinned numbers (R1).

---

## D. Things explicitly marked NOT reproducible from saved artifacts (do not cite as results)
- L10 cross-model generality gains (3B/32B/InfoVQA/LLaVA-1.6) — narrative-only; raw data present, eval not saved.
- Oracle numbers +4.2pp and +3.9pp — narrative-only.
- Any "matches/beats SOTA" — no strong baseline implemented (verified: no FEATHER/CDPruner/HiRED/VisionSelector code in repo).
- Full-validation headline numbers — current headline cells are n=100–300 (only DocVQA highres has n=1000).

## E. Pointers
- Narrative / writing plan: [`THESIS_MASTER_PLAN.md`](THESIS_MASTER_PLAN.md)
- Paper plan: [`PAPER_PUBLICATION_PLAN.md`](PAPER_PUBLICATION_PLAN.md)
- FLOPs artifact: [`../results/thesis_main/highres/qwen_flops_summary.json`](../results/thesis_main/highres/qwen_flops_summary.json)
- Prior maps (archived, history): `archive/docs_consolidated_backup/archive_docs/report_evidence_map.md`
