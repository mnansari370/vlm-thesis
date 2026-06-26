# Thesis Master Plan

*The single document I write the thesis from. It fixes the title, the claim, the story, what goes where,
the datasets, and the chapter plan. Every number lives in [`THESIS_EVIDENCE_LEDGER.md`](THESIS_EVIDENCE_LEDGER.md)
(claims L1–L12, E1–E5, reconciliations R1–R6) — I cite ledger IDs here rather than repeat numbers, so the
two files never drift. Consolidated 2026-06-24 from the earlier story-lock, outline, narrative,
experiments, appendix, and dataset-audit notes (all preserved in `archive/docs_consolidated_backup/`).*

---

## 1. Title, claim, contribution

**Title (recommended):** *Dynamic, Question-Conditioned Visual Token Pruning for Efficient
Vision–Language Models: Selection over Budget.* (Keeps my registered topic verbatim and adds the finding.)

**Main claim.** On vision–language models, *which* visual tokens you keep — chosen by a question-conditioned,
mid-layer signal — is the dominant efficiency lever, while *how many* per sample is not. Question-conditioned
selection retains ~95% of dense accuracy at ~88% fewer prefill FLOPs (Qwen-2.5-VL-7B, DocVQA, K=128; L5,
E1–E2); per-sample budgeting only ties a tuned static budget (L2) and its honest oracle headroom is small
and unrealizable (L8, L9).

**Main contribution.** A rigorous **selection-vs-budget decomposition** of question-conditioned visual-token
pruning, with an honest measurement methodology — matched-FLOPs comparison, strong/floor baselines, a dense
pipeline that reproduces published numbers, and a **monotone oracle-noise correction** — showing across a
frozen low-resolution model and high-resolution / SOTA backbones that **selection dominates and per-sample
budgeting is a mirage**, plus a **cheap distilled question-conditioned selector** that realizes most (not
all) of the expensive teacher signal.

---

## 2. Final story: "Selection over Budget"

A pruner makes two decisions per sample: *which* tokens to keep (**selection**) and *how many*
(**per-sample budget**). I measure both honestly and find an asymmetry: **selection is the lever; budgeting
is not.** "Dynamic" survives honestly as *input-/question-dependent selection* (genuinely dynamic
computation), not as a per-sample token count.

The work reads as **one diagnostic applied across regimes**, not as a pile of separate experiments:

1. **Diagnostic foundation (frozen LLaVA-1.5, 576 tokens).** I build the measurement framework (oracle-
   headroom band + monotone correction) and find two negatives: per-sample budgeting has almost no room
   (L3), and the *naive* question-conditioned selection signals available on this model — frozen CLIP-space
   similarity and early-layer LLM attention — fail, even below random (L4). The pipeline reproduces
   published dense accuracy (L1), so the negatives are trustworthy.
2. **Main result (frozen high-res LLaVA-1.6 / Qwen-2.5-VL).** I ask whether those negatives are properties
   of *pruning* or of the *regime/signal*. The budget negative **survives and sharpens** into a mirage (L8,
   L9); the selection negative **inverts** once the signal is **mid-layer** and the regime is high-resolution
   / token-dense: selection dominates (L5) and is genuinely question-driven (L6, L7).
3. **From measurement to method (distillation).** I distil the expensive mid-layer teacher into a cheap,
   no-decoder-forward student that recovers ~58% of the gain and is genuinely question-conditioned (L11,
   L12), with an honest below-SOTA gap.

### How v1/v2 map onto the thesis (without exposing "v1/v2")
- **"v1" = the diagnostic foundation** (the frozen GQA pipeline + the VQAv2 budget track). It keeps its
  *function* — baselines, oracle band, budget-ties-static, frozen-selection-fails — but I never call it
  "v1" in the thesis; it is "the frozen low-resolution diagnostic."
- **"v2" = the main selection result** (high-res LLaVA-1.6 / Qwen + distillation). Again, no "v2" label;
  it is "the high-resolution / SOTA study."
- The folder names `GQA/`, `VQA_V2/`, `v2/` were *historical* and are **now method-based** under `src/`
  (data/metrics/models/pruning/evaluation/analysis); the thesis is organized by **method**, and "v1/v2"
  appears only as this one explanatory sentence.

**The axis separating the two regimes is resolution × task token-demand × which-layer signal — NOT
training.** The v2 headline wins are on frozen models; the only trained pieces are the (failing) budget
predictor and the (sub-SOTA) student. I must never write "training fixed it."

---

## 3. What goes in the main thesis / appendix / excluded

**Main thesis (the spine):** L1 (dense reproduction) · L2 (budget ties static) · L3 (thin oracle band) ·
L4 (frozen Q-cond selection fails) · L5 (selection dominates) · L6 (mid-layer signal) · L7 (genuinely
question-conditioned) · L8 (small honest oracle) · L9 (predictor mirage) · L11–L12 (cheap student) ·
E1–E3 (token/FLOP reduction, latency) · E4 (teacher-cost caveat) · the monotone-correction methodology.

**Appendix:** the static CLS-attention frontier; the confidence cascade detail; the VQAv2 BudgetController
detail (per-type curves, oracle ceiling); the frozen selection-probe internals (CLIP-space / LM-attention
+ the three fairness fixes); LLaVA-1.6 high-res resolution×task detail; the 7.50/4.99/4.2/3.9 oracle
reconciliation (R1); FLOPs conventions; the bug log / reproducibility protocol; the original proposal.

**Excluded (do not write as results):** the retired classification-head VQAv2 proxy (now in
`archive/retired_code/`); the failed budget-predictor diagnostics (`data/budget_oracle/`); the elastic
Stage-1 training; everything in `archive/failed_experiments/`.

---

## 4. Dataset decisions

The thesis needs **four main datasets** to tell "Selection over Budget" cleanly:

| Dataset | Role in the story | Ledger |
|---|---|---|
| **DocVQA** (Qwen-2.5-VL) | the selection win (95% retention @ 88% FLOP-cut; budget mirage) | L5–L12, E1–E3 |
| **VQAv2** (frozen LLaVA-1.5) | budget ties static (+0.05pp) | L2 |
| **GQA** (frozen LLaVA-1.5) | dense reproduction + oracle band + cascade | L1, L3, L4 |
| **TextVQA** (frozen LLaVA-1.5) | naive question-conditioned selection fails | L1, L3, L4 |

**Appendix datasets:** POPE, ScienceQA-IMG (4-benchmark dense reproduction + band *ordering*); ChartQA
(selection generalizes, smaller gain); LLaVA-1.6 high-res (resolution × task).
**Paper-candidate (not written until recomputed):** InfoVQA + cross-model DocVQA (3B/32B) for the L10
generality.
**Excluded:** the early-proxy classification head, `data/budget_oracle/` diagnostics, the elastic Stage-1
training (`data/llava_mix/` is its data — kept on disk, not featured).

DocVQA/ChartQA/InfoVQA load from the HuggingFace hub; the rest live in the (git-ignored) `data/`, kept flat.

---

## 5. Chapter-by-chapter writing plan

1. **Introduction** — visual tokens dominate VLM cost; the two-lever framing (selection vs budget); my
   pivot to "Selection over Budget"; contributions; the four research questions. *(From this plan + the
   original proposal in the archive.)*
2. **Background & related work** — token reduction in ViTs; VLM pruning (FastV, VisionZip, FasterVLM,
   FEATHER, CDPruner, HiRED, SparseVLM, ATP-LLaVA, Dynamic-LLaVA); adaptive computation; the
   "are-we-solving-the-right-problem" critique (Wen et al., ACL-2025). *(From `docs/literature/related_work.md`.)*
3. **Measurement framework** — the oracle-headroom band + the monotone oracle-noise correction + the
   matched-FLOPs win criterion. *(The methodological contribution.)*
4. **The frozen low-resolution diagnostic** — L1 (dense reproduction), L2 (budget ties static), L3 (thin
   band), L4 (naive selection fails). Establishes the negatives that motivate the rest.
5. **Where pruning helps: selection** — L5 (selection dominates), L6 (mid-layer signal), L7 (genuinely
   question-conditioned), plus the resolution×task context.
6. **Per-sample budget is a mirage** — L8 (small honest oracle), L9 (trained predictor captures ≈0), the
   selection–budget coupling.
7. **From teacher to student** — L11–L12 (the cheap distilled selector and the recovery law).
8. **Efficiency** — E1 (token reduction), E2 (FLOP reduction), E3 (measured latency), E4 (teacher-cost
   caveat). Kept as separate, clearly-labelled quantities (R3).
9. **Discussion** — the one-story-across-regimes argument; honest-measurement methodology; practical
   guidance ("invest in selection").
10. **Limitations & future work** (see §7 below).
11. **Conclusion** — selection over budget; a validated diagnostic; a deployable-but-partial student.
12. *(Optional) RQ4 — reflection on AI-assisted development* (needs `git`-cadence data; decide in or out).

---

## 6. Figures and tables I will need

- **From the frozen diagnostic (v1):** the accuracy–FLOPs static frontier; the oracle-band bar chart
  (band ordering across benchmarks); the confidence-vs-features chart; the CLS-vs-CLIP qualitative panel;
  the per-type cascade adaptivity chart. *(The originals are archived in
  `archive/docs_consolidated_backup/figs_old/` — F1, F2, F3, F5, and the CLS-vs-CLIP panel; regenerable.)*
- **From the main result (v2):** the DocVQA selection K-curve (QC vs blind); the layer sweep (mid > early);
  the mismatched-question control; the accuracy-vs-FLOPs frontier; the student recovery chart. *(Sources in
  `results/thesis_main/highres/figures/` and the `results/thesis_main/highres/qwen_*` / `distill/*` JSONs.)*
- **Tables:** dense-reproduction (L1); oracle-band ordering (L3); selection K-curve (L5); layer sweep (L6);
  budget oracle-vs-predictor (L8, L9); efficiency (E1–E3); student recovery (L11).

I will regenerate final figures at writing time from the saved result JSONs; nothing figure-related is
needed before then.

---

## 7. Limitations to state honestly

- The headline selection signal is a **mid-layer teacher that needs a full LLM forward** — it is an
  analysis/upper-bound selector, not a deployable method (E4). The deployable path is the student, which
  is below dedicated SOTA selectors (~69% retention; L11).
- Several v2 headline numbers are on **n=200–400 subsets** (full-val is paper work).
- The cross-model **generality of the budget mirage (L10) is narrative-only** until recomputed — I do not
  write it as a result in the thesis without the saved per-model evaluation.
- The frozen low-resolution negatives are **regime/signal-specific**, not a claim that selection or
  budgeting can never help — the whole point of the v1→v2 bridge.
- Distillation is **DocVQA-only** so far; the ~42% irreducible gap is itself a finding.

---

## 8. Claims to avoid (do not write unless the ledger row is upgraded)

- "training fixed it" — the v2 wins are frozen (R5).
- "we beat / match SOTA" — no strong published baseline was run.
- "the budget mirage generalizes across families/scales" as a proven result — L10 is narrative-only.
- unqualified "+60pp" (state model/task/K/baseline/n; range +17→+71pp, R2) or unqualified
  "88% FLOP reduction" (analytical prefill, excludes selector cost, K=128 Qwen-7B n=200, R3).
- the oracle numbers +4.2 / +3.9 (canonical = +7.50pp, R1).
- "paper is ready" — it is possible but not ready (see `PAPER_PUBLICATION_PLAN.md`).

---

## 9. Status & next step

Research is ~85% done and honest; the deliverable is mostly **writing** plus one focused round of full-scale
numbers and baselines (paper-only). The repository is clean enough to write from. The recommended next move
is to **start writing chapters 3–7** (which are fully supported today), optionally preceded by the cheap
**L10 recompute** so the generality claim becomes writable. Code/folder reorganization is deferred until a
complete draft exists (see `REPOSITORY_CLEANUP_LOG.md`).
