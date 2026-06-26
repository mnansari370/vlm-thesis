# FINAL EVIDENCE LEDGER (locked scope)

*Created 2026-06-27 by extracting only the **final-scope-relevant** evidence from the now-archived
`THESIS_EVIDENCE_LEDGER.md`, `THESIS_AUDIT.md`, `THESIS_VERIFICATION.md`, `THESIS_MASTER_PLAN.md`,
`PAPER_PUBLICATION_PLAN.md`, the two understanding/clarity reports, and `docs/source_findings/`. Old docs
live in `archive/legacy_docs/docs/`. **Every number here was already verified against a saved result JSON in
prior audits** (paths in §9). If a number is not in this ledger, do not write it as a final result.*

> **Scope filter applied:** kept only {LLaVA-1.5-7B, Qwen-2.5-VL-7B} × {GQA, TextVQA, DocVQA, VQAv2}.
> Dropped from the active ledger (archived, not deleted): POPE / ScienceQA / ChartQA / InfoVQA results, the
> LLaVA-1.6 high-res bridge, the Qwen-3B/32B generality (old **L10**), and the elastic Stage-1 track.

---

## 1. Final thesis claim
> **This thesis decomposes VLM visual-token pruning into token selection and token budgeting, showing that
> question-conditioned selection is the useful lever while per-sample budgeting must be tested carefully
> against matched FLOPs.**

Long form: *which* visual tokens you keep — chosen by a question-conditioned, mid-layer signal — is the
dominant efficiency lever; *how many* per sample (a dynamic budget) is not, once measured honestly
(matched-FLOPs, noise-corrected oracle). All headline results are on **frozen** backbones.

## 2. Final models (both frozen)
| Model | ID | Visual tokens | Notes |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | 576 fixed @336px | low-res diagnostic; runs in `vlm_env` (torch 2.3 / transformers 4.46.3 — result-critical pins) |
| Qwen-2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | native dynamic (~1286 avg DocVQA) | modern backbone; runs in `qwen_env` (transformers 4.51, `qwen_vl_utils`) |

## 3. Final datasets
GQA (testdev_balanced, exact-match) · TextVQA (OCR + no-OCR, M4C soft-acc) · DocVQA (hub, ANLS) ·
VQAv2 (val 10k stratified, VQA consensus).

## 4. Final methods
1. **Dense** — keep all tokens (ceiling + reproduction anchor).
2. **Static pruning** — question-independent fixed-budget selection (LLaVA: CLS-attention saliency; Qwen:
   blind `uniform`/`norm` — **a fair CLS-equivalent is still missing on Qwen**).
3. **Dynamic WHICH** — question-conditioned token selection at a **fixed %** (the question chooses which
   tokens; count fixed). Qwen signal = mid-layer (L16) question→visual attention.
4. **Dynamic COUNT** — adaptive per-sample budget. **Being redesigned fresh**; old attempts kept as evidence.

Budget levels = {15, 25, 35, 50, 75, 100}% of dense. LLaVA-1.5 K-map = 86 / 144 / 202 / 288 / 432 / 576.

---

## 5–6. Evidence with status (DONE / PARTIAL / TODO)

### A. Dense + Static (the baselines/frontier)

| Claim | Number (final scope) | Status | Notes |
|---|---|---|---|
| **L1** Dense reproduces published | GQA **61.42**/62.0; TextVQA-OCR **57.65**/58.2 | DONE | trust anchor; VQAv2 dense **76.44**; Qwen-DocVQA dense **97.19** (n=200) |
| **Static frontier — LLaVA-1.5 GQA** | K144/192/288/432 = 58.15/59.19/60.53/61.53 (ret. 94.7–100.2%) | PARTIAL | 25/50/75% done; **15% (K86), 35% (K202) TODO** |
| **Static frontier — LLaVA-1.5 TextVQA** | OCR 55.97/56.55/56.40/57.39; noOCR 44.80/45.36/45.69/46.59 | PARTIAL | 15/35% TODO |
| **Static frontier — LLaVA-1.5 VQAv2** | K64/128/144/192/288/432 = 71.02/74.27/74.44/75.31/75.82/76.27 | PARTIAL | 15/35% TODO |
| **Static — LLaVA-1.5 DocVQA** | — | TODO | DocVQA never run on LLaVA-1.5 (576 tok can't read dense docs — informative) |
| **Static (blind) — Qwen DocVQA** | uniform K64/128/256 = 23.93/32.19/61.12; norm lower | PARTIAL | fixed-K not %-of-dense; **no CLS-equiv baseline** |
| **Static/Dense — Qwen × {GQA,TextVQA,VQAv2}** | — | TODO | no Qwen harness for these 3 datasets yet |

### B. Dynamic WHICH (the selection win)

| Claim | Number | Status | Notes |
|---|---|---|---|
| **L4** Naive frozen Q-cond selection **fails** (LLaVA-1.5) | TextVQA-noOCR K64: CLIP-space **−32.04pp**, LM-attn(L8) **−5.58pp** vs CLS; GQA probe mixed (CLS itself < random) | DONE | the motivating low-res negative; n=4000–5000, K=64 only |
| **L5** Q-cond selection **dominates** (Qwen DocVQA) | K128 textattn **92.18** vs uniform 32.19 = **+59.99pp**; K64 +60.8; K256 +33.6 | PARTIAL | n=200; baseline is the **weakest floor** (see caveats) |
| **L6** Signal is **mid-layer** | L2=44.6 → **L16=93.2** → L20=94.9 (K128) | DONE | n=120; early-layer (FastV-style) far below |
| **L7** Genuinely **question-conditioned** | real 93.84 / mismatched 36.98 / blind 32.19 = **92.2% question-driven** | PARTIAL | n=100 (teacher) — rerun larger |
| **L11** Cheap distilled **student** recovers majority | dense 95.0 / teacher 91.1 / **student 65.9** / blind 31.2 = **57.9% recovery** (PASS) | DONE | DocVQA only; ~69% retention vs SOTA ~90%; K64 47.9%, K256 45.0% |
| **L12** Student Q-conditioned; data>epochs | **99.2%** question-driven; 3K→12K data 46→58%; 10→25 ep overfits 58→52% | DONE | n=200/400 |
| Dynamic-WHICH at the 15–75% grid, all cells | — | TODO | the headline must be run at the budget grid on 4 datasets × 2 models, with a fair baseline |

### C. Dynamic COUNT (the budget mirage — redesign pending)

| Claim | Number | Status | Notes |
|---|---|---|---|
| **L2** Per-sample budget **ties** tuned static (LLaVA-1.5 VQAv2) | dynamic **75.76** vs static-K265 75.71 = **+0.05pp**; avg K 264.3, std 44.9 | DONE | mechanism works; Jensen caps gain (per-type curves ≈ same concave shape) |
| **L3** Oracle budget **band is thin** (LLaVA-1.5) | GQA **9.13%**, TextVQA-OCR 6.08% / noOCR 6.24% (first-correct-K band) | DONE | <20% everywhere in scope |
| **L8** Honest oracle headroom **small** (Qwen DocVQA) | **+7.50pp** FLOPs-matched (monotone 98.0 @ ~90 tok); **+0.81pp** pure-accuracy over best fixed K | DONE | with blind selector the same oracle is **+20.66pp @ 337 tok** → budget benefit is a symptom of weak selection |
| **L9** Trained predictor captures **≈ none** | oracle 4.99 → MLP **+0.09pp = 1.8%**; hard-tail AUC 0.643 (base rate 0.195) | DONE | 81% already correct at min-32; tail unfixable |
| **New Dynamic-COUNT mechanism** | — | TODO | must beat best fixed % at same avg FLOPs; reuse the monotone-oracle bound before training |

### D. Efficiency (always report separately — never conflate)

| Claim | Number | Status | Notes |
|---|---|---|---|
| **E1** Token reduction | **86.8%** @K128 (Qwen DocVQA) | DONE | measured from avg_tokens |
| **E2** FLOP reduction | **87.9%** @K128 (analytical FastV Eq.5 prefill) | DONE | EXCLUDES selector cost |
| **E3** Measured latency | 3.31× @K128 | **CAVEAT/TODO** | measured on **LLaVA-1.6** (now out of scope), decode-only, n=40, `arange` tokens — **re-measure on Qwen/LLaVA-1.5 or drop** |
| **E4** Teacher selector cost | full LLM forward | DONE | the QC teacher is analysis/upper-bound, not deployable |
| **E5** Student selector cost | negligible, no decoder forward | DONE | the deployable path (L11) |

**Status tally (final-scope method cells):** ~4 DONE baselines, ~9 PARTIAL, ~15 TODO. Full matrix in
[`CLEAN_EXPERIMENT_MATRIX.md`](CLEAN_EXPERIMENT_MATRIX.md); run order in [`TODO_NEXT_RUNS.md`](TODO_NEXT_RUNS.md).

---

## 7. Caveats & claims to AVOID (carried from THESIS_VERIFICATION / AUDIT)
- ❌ **"training fixed it."** Every headline win is **frozen**. The axis is resolution × task token-demand ×
  signal-layer, not training.
- ❌ **unqualified "+60pp" selection win.** It is vs the **weakest** baseline (`uniform`). Qwen has **no
  CLS-attention baseline implemented**; a fair one likely shrinks the gap to **+20–30pp** (cf. the archived
  LLaVA-1.6 number was +27pp vs CLS). Always state model/task/K/baseline/n.
- ❌ **"+7.50pp budget headroom"** without the matched-budget qualifier. Pure-accuracy ceiling over best
  fixed K = **+0.81pp**; +7.50pp is a FLOPs-matched efficiency number.
- ❌ **"the hard tail is unpredictable."** Saved AUC = **0.643** (not 0.59). Say **"weakly identifiable but
  unfixable"** — realized gain ≈0 *despite* AUC 0.64.
- ❌ **"beats / matches SOTA."** No strong published baseline (FEATHER/CDPruner/HiRED/VisionSelector) is
  implemented.
- ❌ **"the budget mirage generalizes across families/scales"** as proven — that was old **L10**
  (3B/32B/InfoVQA/LLaVA-1.6), now **out of scope**; raw data preserved in `results/paper_candidates/` but the
  per-model eval was never saved.
- ⚠️ Headline cells **L5/L7** are **n=100–200** → need full-validation before quoting to 0.1pp (Qwen DocVQA
  dense 97.19 > published 95.7 is small-n optimism).

## 8. Known risks that still matter
1. **No Qwen harness for GQA/TextVQA/VQAv2** — every Qwen cell on those three is TODO (needs a dataset adapter
   + scorer dispatch; the Qwen pruner only loads DocVQA/ChartQA today). Biggest build item.
2. **Fair Qwen blind baseline missing** — gates the credibility of every cross-model selection claim.
3. **Qwen-on-low-res-reasoning may show QC barely helps** (GQA/VQAv2 have little to prune) — the +60pp DocVQA
   result may not transfer; could echo the LLaVA-1.5 negative.
4. **LLaVA-1.5 × DocVQA is TODO** and expected weak (576 tokens can't read dense docs) — run it anyway as a
   resolution×task data point.
5. **Dynamic-COUNT redesign** must clear the +0.05/+0.09pp bar honestly or be reported as a confirmed negative.
6. **`answer_vocab_full.json` missing** (referenced by LLaVA-1.5 dense/static/dynamic YAMLs) — only the
   retired classification head needs it; regenerate via `scripts/data/build_answer_vocab_full.py` if revived.
7. **FLOPs conventions** — two calculators (`flops.py`/`flops_vqav2.py` vs `qwen_flops.py`); `qwen_flops.py`
   hardcodes `N_TEXT=40` (DocVQA) → set per-dataset for GQA/TextVQA/VQAv2.

## 9. Result files supporting each claim (in-scope, on disk under `results/`)
| Claim | File(s) |
|---|---|
| L1 dense | `thesis_main/gqa/results_frozen/all.json`; `testdev_dense_honest_bs1_*/metrics.json`; `textvqa_analysis_ocr.json`; `vqav2/dense_pad/generation_eval_10k.json`; `highres/qwen25_dense_docvqa.json` |
| Static frontiers | GQA `thesis_main/gqa/testdev_static_cls_attn_k{144,192,288,432}_*/metrics.json` + `testdev_frontier_analysis.json`; TextVQA `textvqa_analysis_{ocr,noocr}.json`; VQAv2 `vqav2/static_k*_pad/generation_eval_10k.json` + `static_baseline_locked_expand2square.json` |
| L2 budget ties static | `vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json` |
| L3 thin band | `gqa/testdev_frontier_analysis.json`; `textvqa_analysis_{ocr,noocr}.json`; `results_frozen/all.json` (bands) |
| L4 naive selection fails | `gqa/week1_all_numbers.json`; `qcond_textvqa_noocr_*/results.json`; `qcond_gqa_*/results.json` |
| L5 selection dominates | `highres/qwen_kcurve_docvqa.json` |
| L6 mid-layer | `highres/qwen_layer_sweep.json` |
| L7 question-driven | `highres/qwen_control_docvqa.json` |
| L8 oracle (small) | `highres/qwen_oracle_docvqa_qc.json` (+ `qwen_oracle_docvqa_uniform.json` for the coupling) |
| L9 predictor ≈0 | `highres/qwen_budget_eval.json`; `qwen_budget_robust.json` |
| L11/L12 student | `highres/distill/{gate_12k,control_12k,gate_12k_25ep,gate_K64,gate_K256}.json` |
| E1/E2 efficiency | `highres/qwen_flops_summary.json`; `qwen_kcurve_docvqa.json` |
| E3 latency (caveat) | `highres/llava_latency.json` (LLaVA-1.6 — out of scope; number only) |

All result trees are git-ignored and stay in `results/thesis_main/`; the mapping of in-scope vs out-of-scope
files inside `results/thesis_main/highres/` is in [`../results/legacy_index.md`](../results/legacy_index.md).

## 10. Source docs this ledger was extracted from (now archived)
`archive/legacy_docs/docs/`: `THESIS_EVIDENCE_LEDGER.md` (claim IDs + canonical numbers + reconciliations
R1–R6), `THESIS_VERIFICATION.md` (the +7.50→+0.81pp reframe, the +60pp baseline-asymmetry, the AUC-0.643
correction), `THESIS_AUDIT.md` (the LLaVA-1.5 result inventory + FLOPs conventions), `THESIS_MASTER_PLAN.md`
(story + scope), `PAPER_PUBLICATION_PLAN.md` (what's not paper-ready), `FULL_REPOSITORY_UNDERSTANDING_REPORT.md`
+ `METHOD_DATASET_MODEL_CLARITY_REPORT.md` (the per-method/dataset/model breakdown), and
`docs/source_findings/` (the original Qwen/distill FINDINGS drafts).
