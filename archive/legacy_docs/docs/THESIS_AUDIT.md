# Thesis Scope Audit — LLaVA-1.5 "Idea 1" (dynamic *which*, fixed *count*)

> **Read-only audit.** Produced without moving, editing, deleting, renaming, or running anything.
> Scope under audit: **one model (LLaVA-1.5-7B frozen)**, datasets **VQAv2 / GQA / TextVQA / POPE**,
> three approaches on one accuracy-vs-FLOPs frontier: **dense**, **static-K** (not question-aware),
> **question-conditioned at fixed K** (Idea 1: content varies per question, count K fixed).
> Every cell below is verified against an actual file; numbers and paths are quoted, not inferred.
> Where I could not verify, I write **uncertain**.

---

## Headline conclusions (read these first)

1. **Static pruning is in good shape.** The requested grid is `{72, 144, 288, 432}`. For **static CLS-attention**,
   **K=144, 288, 432 already exist on all four datasets**, on the locked honest protocol. **Only K=72 is missing
   everywhere** (the closest existing point is K=64 on VQAv2). Dense exists everywhere.
2. **Question-conditioned-at-fixed-K (Idea 1) barely exists.** There is **no** clean question-conditioned-selection
   cell at K∈{72,144,288,432} on **any** dataset. The only QC-selection runs are **probe** runs at **K=64 (and K=96
   for TextVQA)** on **GQA and TextVQA only**, on **n=4000–5000 subsets**. VQAv2 and POPE have **zero** QC-selection
   results. The whole `QC × {72,144,288,432}` matrix must be generated.
3. **The existing LLaVA-1.5 QC-selection evidence is mixed-to-negative, and the one selector that beats CLS is not
   FLOPs-cheap.** On TextVQA-noOCR, question-conditioned (LM-attention L8) selection at K=64 *loses* to CLS by
   −5.58pp. On GQA at K=64 it *wins* by +3.5pp, but CLS there is itself below random. The selector that does best
   (mid/early-LLM attention) **requires a full LLM forward to score** — a cost not counted in the FLOP frontier.
   The cheap one (CLIP-space) **fails badly** (GQA K=64: 45.73 vs CLS 54.93). **This is the central scope risk: the
   "selection wins" result in the wider thesis is a high-res / Qwen finding; on frozen low-res LLaVA-1.5 it is a
   negative/ties result** (ledger L4). See §6.
4. **FLOP accounting is internally consistent per dataset but uses two different conventions across datasets**
   (VQAv2 reports *attention-only* GFLOPs; GQA/TextVQA/POPE report *FastV-full* TFLOPs). Both come from unified
   formulas, but the reported headline numbers are not directly comparable across datasets as-saved. See §3.

---

## Section 1 — What LLaVA-1.5 results actually exist

**Protocol legend.** "Locked honest protocol" = `image_pad`/expand2square, official prompt+scorer per benchmark,
greedy, `bs=1`, no min-new-tokens / no repetition penalty. Verified in
[run_static_testdev.py:1-23](../src/evaluation/gqa/run_static_testdev.py#L1-L23),
[static_baseline_locked_expand2square.json](../results/thesis_main/vqav2/static_baseline_locked_expand2square.json) `eval_protocol`.

Selector note: every "static" cell below uses **CLS-attention** (VisionZip/FasterVLM "dominant" scoring) unless
stated. The selector is set in [static.py:121](../src/models/static/static.py#L121) (`StaticPrunedLlava`, method `cls_attn`).

### 1A. VQAv2 (mandatory)

Canonical reference = [static_baseline_locked_expand2square.json](../results/thesis_main/vqav2/static_baseline_locked_expand2square.json)
(val2014, **10k stratified** subset, seed 42, generation, `max_new_tokens=10`, VQA-consensus scorer, expand2square).

| approach | K | exists? | result file path | accuracy | config | selector | protocol matches dense/static? | notes |
|---|---|---|---|---|---|---|---|---|
| Dense | 576 | **Y** | `results/thesis_main/vqav2/dense_pad/generation_eval_10k.json` | **76.44** | `configs/dense/llava_dense_150k_10k_fullvocab.yaml` | none (all 576) | Y (reference) | locked, "do not re-run" |
| Static | 72 | **N** | — | — | — | — | — | **missing; closest is K=64=71.02** |
| Static | 144 | **Y** | `results/thesis_main/vqav2/static_k144_pad/generation_eval_10k.json` | **74.44** | static k144 | CLS-attn | Y | −2.00pp vs dense |
| Static | 288 | **Y** | `results/thesis_main/vqav2/static_k288_pad/generation_eval_10k.json` | **75.82** | static k288 | CLS-attn | Y | −0.62pp |
| Static | 432 | **Y** | `results/thesis_main/vqav2/static_k432_pad/generation_eval_10k.json` | **76.27** | static k432 | CLS-attn | Y | −0.17pp |
| *(extra static present)* | 64/128/192 | Y | `static_k{64,128,192}_pad/` | 71.02 / 74.27 / 75.31 | — | CLS-attn | Y | in locked curve |
| QC @ fixed K | 72/144/288/432 | **N** | — | — | — | — | — | **none exist** |

**VQAv2 question-conditioned reality:** there is **no** question-conditioned *selection-at-fixed-K* run on VQAv2.
The only "dynamic" VQAv2 artifacts are:
- `dynamic_150k_clsonly/` — uses `scoring_mode=cls_only` (CLS-attn ranking, **not** question-conditioned selection) **plus a budget controller** (Idea 2). Headline 75.76% @ avg-K 264.3 ([RESULT_SUMMARY.json](../results/thesis_main/vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json)). **Cannot be reinterpreted as Idea 1** — its token *ranking* is plain CLS, and its variation is in *count*, not *content*.
- `gate_real/` — budget-controller smoke test (Idea 2 mechanism), std(K)=44.3 ([gate_results.json](../results/thesis_main/vqav2/gate_real/gate_results.json)). Not a selection result.

So: **new clean QC-selection runs are required on VQAv2 for the entire grid** (the learned `QuestionConditionedTokenScorer` in [token_scorer.py](../src/models/dynamic_budget/token_scorer.py) exists in code but was never run at fixed K with the budget controller disabled; all saved VQAv2 runs used `cls_only`).

**Preprocessing-variant caveat:** VQAv2 has parallel `static_k*_pad` (expand2square, **canonical/locked**), `static_k*_fixed`, `static_k*_pertype`, `static_k*_matched`, `static_k*_v1` dirs. The locked curve references the **`_pad`** files. The `_fixed`/`_v1` preprocessing is **uncertain** — do not mix them with `_pad` numbers without checking.

### 1B. GQA (mandatory)

Canonical reference = [testdev_frontier_analysis.json](../results/thesis_main/gqa/testdev_frontier_analysis.json)
(testdev_balanced, **full n=12,578**, locked honest protocol, official GQA scorer
[official_score.py](../src/metrics/official_score.py): `strip().rstrip('.').lower()` strict).

| approach | K | exists? | result file path | accuracy | config/driver | selector | protocol matches? | notes |
|---|---|---|---|---|---|---|---|---|
| Dense | 576 | **Y** | `results/thesis_main/gqa/testdev_dense_honest_bs1_20260605_010314/` | **61.42** | `run_dense_testdev.py` | none | Y (reference) | ledger L1 (pub 62.0) |
| Static | 72 | **N** | — | — | — | — | — | **missing**; `run_static_testdev.py` argparse `--keep_k` **rejects 72** (choices = {576,432,288,192,144,96,64}) |
| Static | 144 | **Y** | `results/thesis_main/gqa/testdev_static_cls_attn_k144_20260605_015942/` | **58.15** | `run_static_testdev.py --method cls_attn` | CLS-attn | Y | retention 94.7% |
| Static | 288 | **Y** | `results/thesis_main/gqa/testdev_static_cls_attn_k288_20260605_021446/` | **60.53** | same | CLS-attn | Y | retention 98.5% |
| Static | 432 | **Y** | `results/thesis_main/gqa/testdev_static_cls_attn_k432_20260605_021439/` | **61.53** | same | CLS-attn | Y | retention 100.2% |
| *(extra static present)* | 192 | Y | `testdev_static_cls_attn_k192_*/` | 59.19 | same | CLS-attn | Y | not in requested grid |
| *(VisionZip baseline)* | 144/192/288/432 | Y | `testdev_visionzip_k*/` | (≈ CLS-attn) | `run_visionzip_testdev.py` | VisionZip | Y | second static baseline |
| *(val_balanced static set)* | 144/192/288/432 | Y | `oracle_cls_attn_k*_20260602_*/` | — | `run_static.py` | CLS-attn | **different split** (val_balanced, not testdev) | do not mix with testdev |
| QC @ fixed K | 72/144/288/432 | **N** | — | — | — | — | — | **none at requested K** |
| QC @ K=64 (probe) | 64 | **Y (subset)** | `results/thesis_main/gqa/qcond_gqa_20260605_164346/results.json` | qcond-L8 **58.43**, CLS 54.90, random 56.90 | `run_qcond_probe.py` | LM-attn (L2/5/8), fusion | **Y on protocol** (same harness/suffix/scorer) but **n=4000, K=64 only** | qcond-L8 **beats** CLS here |
| CLIP-space QC @ K=64 | 64 | **Y (subset)** | `results/thesis_main/gqa/clip_gqa_20260606_034827/results.json` | clip **45.73** vs CLS 54.93 | `run_clip_probe.py` | CLIP-space | Y on protocol, n=4000 | **fails** (ledger L4) |

GQA also has the **oracle-headroom** and **confidence-cascade** artifacts (Idea 2 — see §2):
[results_frozen/all.json](../results/thesis_main/gqa/results_frozen/all.json) "cascade", and the per-sample first-correct-K
band (9.13%) inside `testdev_frontier_analysis.json`.

### 1C. TextVQA

Reference = [textvqa_analysis_ocr.json](../results/thesis_main/gqa/textvqa_analysis_ocr.json) and
[week1_all_numbers.json](../results/thesis_main/gqa/week1_all_numbers.json) (`textvqa_band`). **Both OCR and no-OCR
variants exist** (n=5000 each). M4C VQA soft-acc scorer ([textvqa_score.py](../src/metrics/textvqa_score.py)).

| approach | K | exists? | result file | accuracy (OCR / noOCR) | selector | protocol matches? | notes |
|---|---|---|---|---|---|---|---|
| Dense | 576 | **Y** | `textvqa_dense_full_20260605_125126/` (+ band json) | **57.65 / 46.73** | none | Y | OCR dense in ledger L1 |
| Static | 72 | **N** | — | — | — | — | **missing** |
| Static | 144 | **Y** | `textvqa_cls_attn_k144_*` (+ `_noocr`) | **55.97 / 44.80** | CLS-attn | Y | |
| Static | 288 | **Y** | `textvqa_cls_attn_k288_*` (+ `_noocr`) | **56.40 / 45.69** | CLS-attn | Y | |
| Static | 432 | **Y** | `textvqa_cls_attn_k432_*` (+ `_noocr`) | **57.39 / 46.59** | CLS-attn | Y | |
| *(extra static)* | 192 | Y | `textvqa_cls_attn_k192_*` | 56.55 / 45.36 | CLS-attn | Y | not requested |
| QC @ fixed K | 72/144/288/432 | **N** | — | — | — | — | **none at requested K** |
| QC @ K=64/96 (probe) | 64, 96 | **Y (subset)** | `qcond_textvqa_{ocr,noocr}_20260605_153609/results.json` | noOCR: qcond-L8 **37.65** vs CLS 43.23 (**−5.58**) | LM-attn, fusion | Y on protocol, n=5000 | **QC loses to CLS** (ledger L4) |
| CLIP-space QC @ K=64/96 | 64, 96 | **Y (subset)** | `clip_textvqa_{ocr,noocr}` (in week1) | noOCR clip **11.21** vs CLS 43.25 | CLIP-space | Y on protocol | **fails badly** |

### 1D. POPE (optional)

Reference = [pope_analysis.json](../results/thesis_main/gqa/pope_analysis.json) and `week1_all_numbers.json`
(`pope_*`). **All three subsets exist** (random / popular / adversarial), n=3000 each. Scorer
[pope_score.py](../src/metrics/pope_score.py).

| approach | K | exists? | result file | accuracy/F1 (mean over 3 subsets) | selector | protocol matches? | notes |
|---|---|---|---|---|---|---|---|
| Dense | 576 | **Y** | `pope_dense_20260606_001005/` | acc **84.64** / F1 **85.78** | none | Y | ledger L1 (F1 pub 85.9) |
| Static | 72 | **N** | — | — | — | — | **missing** |
| Static | 144 | **Y** | `pope_cls_attn_k144_20260606_012929/` | acc 84.36 / F1 **85.71** | CLS-attn | Y | |
| Static | 288 | **Y** | `pope_cls_attn_k288_20260606_023833/` | acc 83.35 / F1 **85.06** | CLS-attn | Y | |
| Static | 432 | **Y** | `pope_cls_attn_k432_20260606_023935/` | acc 84.46 / F1 **85.76** | CLS-attn | Y | |
| *(extra static)* | 192 | Y | `pope_cls_attn_k192_*` | 83.90 / 85.41 | CLS-attn | Y | not requested |
| QC @ any K | — | **N** | — | — | — | — | **no question-conditioned POPE runs at all** |

POPE has **only** dense + static-CLS + a speculative/confidence cascade (`pope_speculative_tau055`, Idea 2). No QC selection.

### Cross-cutting comparability flags

- **F1: K=72 exists nowhere.** Static `{144,288,432}` are present on all four datasets; QC-at-the-grid is absent everywhere.
- **F2: Subset/n mismatch.** Static frontiers are full-split (GQA 12,578; TextVQA 5,000; POPE 9,000) **except VQAv2
  (10k subset)**. QC probes are **K=64/96 only, n=4000–5000**. A QC-vs-static comparison at the same K and same n
  **does not exist** for any dataset.
- **F3: Protocol — GOOD news.** The QC probes ([run_qcond_probe.py:86](../src/evaluation/gqa/run_qcond_probe.py#L86))
  subclass `StaticPrunedLlava` and use `image_pad=True, honest=True`, the **same** instruction suffix and **same**
  official scorers as the static runs (`append_suffix=True` for GQA; the TextVQA prompt already carries the
  instruction). So QC and static are **protocol-comparable where they overlap** — the gap is K-grid and n, not protocol.
- **F4: FLOP convention differs by dataset** (see §3) — VQAv2 headline FLOPs are *attention-only*; GQA/TextVQA/POPE
  are *FastV-full*. Within a dataset it is consistent across dense/static.
- **F5: GQA has two static sets** — `testdev_*` (canonical, full 12,578) and `oracle_cls_attn_*` (val_balanced split).
  Use testdev for the frontier; do not mix.

---

## Section 2 — Reality check on Idea 2 (variable per-sample budget)

### 2.1 — Where oracle headroom was measured (every file)

| file | dataset / model | K-set the oracle picks from | oracle definition | headroom reported |
|---|---|---|---|---|
| [qwen_oracle.py](../src/pruning/dynamic_budget/qwen_oracle.py) (selector=uniform) | **DocVQA/ChartQA, Qwen-2.5-VL** | {32,64,128,256,512,768,1024,full} | naive (max over K) **and** monotone (correct at K *and all larger*); honest = monotone − static@avg-budget | DocVQA blind +20.7pp; (Qwen) |
| [qwen_oracle_qc.py](../src/pruning/dynamic_budget/qwen_oracle_qc.py) (selector=QC) | **DocVQA/ChartQA, Qwen-2.5-VL** | same ladder | same monotone correction, but with the **good** QC selector | DocVQA **+7.50pp**, ChartQA +2.66pp (ledger L8) |
| [qwen_budget_eval.py](../src/pruning/dynamic_budget/qwen_budget_eval.py) | **DocVQA, Qwen-2.5-VL** | reads `qwen_budget_data_docvqa.json` ladder | oracle = sufficiency-K per sample (smallest robustly-correct K), vs **best fixed** | oracle +4.99pp; predictor **+0.09pp = 1.8%** (ledger L9) |
| [oracle_decomposition.py](../src/analysis/oracle_decomposition.py) | **LLaVA-1.6** TextVQA/DocVQA/ChartQA | {…,full=2302} | monotone vs naive, FLOPs-matched | TextVQA +2.6, DocVQA +5.0, ChartQA +0.5pp |
| `testdev_frontier_analysis.json` "oracle_headroom" | **GQA, LLaVA-1.5** | {144,192,288,432,576} | per-sample first-correct-K band | **band 9.13%**, never 32.72% |
| `pope_analysis.json` "pope_oracle" | **POPE, LLaVA-1.5** | {144,192,288,432,576} | first-correct-K band | **band 4.93%**, never 10.71% |
| `textvqa_analysis_{ocr,noocr}.json` | **TextVQA, LLaVA-1.5** | {144,192,288,432,576} | first-correct-K band | **band 6.08% (OCR) / 6.24% (noOCR)** |
| [val_oracle_static_dense_summary.json](../data/budget_oracle/val_oracle_static_dense_summary.json) | **VQAv2, LLaVA-1.5** | {144,288,432,576}, threshold 1/3 | per-sample smallest-correct-budget (naive, label-based) | oracle resolved **77.04%** @ 279 tok vs best-fixed (dense) 66.73% → **≈ +10.3pp naive** |
| `dynamic_150k_clsonly/RESULT_SUMMARY.json` "instance_headroom_oracle" | **VQAv2, LLaVA-1.5** | {64,96,128,160,219,265,276,334,357} | oracle ceiling (labels + noise) | ceiling **81.13%**; oracle−uniform **+5.42 to +8.21pp**; **token-need band 10.4%**, never 19.1% |

### 2.2 — Was oracle headroom measured on VQAv2 or GQA with LLaVA-1.5?

**Yes, on both — extensively.**
- **VQAv2:** [val_oracle_static_dense_summary.json](../data/budget_oracle/val_oracle_static_dense_summary.json)
  (naive label oracle, +≈10.3pp over dense, but exploits noise) **and** the **monotone-style** instance-headroom in
  [RESULT_SUMMARY.json](../results/thesis_main/vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json): oracle ceiling 81.13%,
  realizable token-sensitive band **only 10.4%**, oracle-vs-uniform **+5.4pp** at K=265.
- **GQA:** `testdev_frontier_analysis.json` → first-correct-K **band 9.13%**, never-correct 32.72%.

So the claim "Idea-2 headroom is small" **is backed by LLaVA-1.5 VQAv2/GQA artifacts**, not only by the Qwen track.
The **honest** (noise-corrected / realizable) numbers are small (band ~9–10%); the **naive** numbers look big
(+10pp) but credit accidental correctness.

### 2.3 — Predictor inputs tried (and the critical image-feature question)

| predictor | file | inputs | output | captured headroom | image features? |
|---|---|---|---|---|---|
| `BudgetController` (MLP) | [budget_controller.py](../src/models/dynamic_budget/budget_controller.py) | `question_projected` (LLM text embed) + **7 score-stats** (mean/std/max/min/top5/top10/entropy of token scores) + optional qtype embed | per-sample keep-ratio→K | ties static (**+0.05pp**, L2 / RESULT_SUMMARY) | **Indirect only** — score-stats derive from CLS-attn/learned scores, but **no raw image features** |
| Qwen budget MLP (5-fold CV) | [qwen_budget_eval.py](../src/pruning/dynamic_budget/qwen_budget_eval.py) | 5 **attention-distribution** features (`norm_entropy, eff_support, top64/128/256_mass`) | sufficiency-K level | **+0.09pp (1.8% of oracle)** | **No** — attention stats only |
| Qwen spread rule | same | `eff_support` quantile bins | K level | +0.72pp | No |
| **Binary route predictor** | [binary_routing_eval_summary.json](../data/budget_oracle/binary_routing_eval_summary.json) + [val_budget_visual_features_binary.jsonl](../data/budget_oracle/val_budget_visual_features_binary.jsonl) | **CLS-attn distribution stats + patch-norm stats** (image-derived) and/or tf-idf-visual + qtype | binary route (K=144 vs dense-576) | **never closes the gap** — best route leaves **+2.8 to +7.0pp** accuracy gap to dense at 30–50% token saving | **YES — image-derived summary stats** |
| Per-qtype budget map | [qtype_oracle_summary.json](../data/budget_oracle/qtype_oracle_summary.json), `dynamic_stage6c_*`, `exp2–5_*` | question **type** only | per-type K | per-type allocation ≈ 62.3% @ ~211–220 tok (classification-era metric) | No (type label only) |

**Critical answer (Q3).** A predictor **was** given **image-derived features** — but only once, on VQAv2, in the
early phase: the `val_budget_visual_features_binary.jsonl` records per-sample **CLS-attention distribution statistics**
(`cls_attn_mean/std/top1/top2/margin/top5/top10/top25/entropy/effective_tokens`) **and patch-norm statistics**
(`patch_norm_mean/std/top1.../entropy`). These fed the **binary route** experiment
([binary_routing_eval_summary.json](../data/budget_oracle/binary_routing_eval_summary.json)). Two honest caveats:
(1) they are **hand-crafted summary stats**, not a learned image encoder or raw patch embeddings; (2) the predictor
was a **binary route (k144 vs dense)**, not a fine-grained per-sample K head; and (3) it still **did not beat the
static frontier**. The two *learned* per-sample predictors (`BudgetController`, Qwen MLP) used **only question +
score/attention statistics — no image features.** These `data/budget_oracle/` files are dated **May 1–19 2026**,
the **early VQAv2 classification-head phase** (accuracies ~60–66% on a 1/3 threshold, *not* the June generation
protocol at 75.76%); provenance of exact scoring is **uncertain** and the master plan marks this phase **excluded**.

### 2.4 — Per-question-type budgeting

`qtype_oracle_summary.json` (VQAv2, 10k): the **oracle token-need correlates with question type** — yes/no needs
~194 tok (89.6% resolvable), what/which ~368 (50%), color ~269, where ~401 (43%), how ~450 (32%), why ~458 (31%).
But **the hard types have low "resolved_rate"**: for where/how/why, even the oracle's extra tokens fix <45% of them —
i.e., the budget-sensitive questions are mostly *unfixable*, not *token-starved*.

`dynamic_stage6c_6type` / `exp2–exp5_7type` apply a type→budget map and report ~62.3% @ ~211–220 avg tokens
([6type txt](../data/budget_oracle/dynamic_stage6c_6type_diagnostic.txt),
[exp5 txt](../data/budget_oracle/exp5_7type_diagnostic.txt)). Per-type std is tiny for easy types (yes/no std 2.4)
and only grows for spatial/complex — confirming type carries little within-type signal.

**Was per-type oracle larger than per-sample?** No — per-type allocation (~62%) is **well below** the per-sample
oracle (77% naive / 81% ceiling). Type is a **coarser, weaker** signal than per-sample. **Did any per-type predictor
capture meaningful gain?** No: the matched-cost comparison (`RESULT_SUMMARY.json`) is **+0.05pp** vs uniform K, and
the binary type-route leaves a 2.8–7pp gap.

### 2.5 — Honest verdict on Idea 2

**Idea 2 is genuinely capped, and on VQAv2/GQA specifically — not merely undertested.** The small-headroom result
is reproduced *natively on LLaVA-1.5 VQAv2 and GQA* (band 9–10%; oracle-vs-uniform +5.4pp that is itself
noise-inflated), not imported from Qwen. The mechanism is the same everywhere: the budget-sensitive band is small,
and the samples in it are dominated by *unfixable* questions (low resolved-rate / high never-correct), so a predictor
that escalates them mostly wastes tokens — confirmed by the confidence cascade only **tracing**, never beating, the
static frontier (`results_frozen/all.json` "cascade"). **One honest qualifier:** the *image-conditioned* predictor
was tried only as a **binary route on hand-crafted CLS/patch summary stats in the retired classification phase**, and
the two *learned per-sample* predictors saw **no image features at all**. So "a strong learned image-conditioned
per-sample-K predictor on the final generation protocol" is **technically untried** — but given (a) the oracle ceiling
itself is small after noise correction and (b) the headroom is concentrated in unfixable questions, the *expected*
upside is low. **Bottom line: Idea 2 looks genuinely dead for the accuracy goal; the only residual is a modest,
honestly-reported *efficiency* (token-saving at fixed accuracy) story, which static pruning already captures.**

---

## Section 3 — Comparability and FLOP accounting

### 3.1 — Canonical FLOP calculator and `n_text` constants

**There are two calculators, not one** (same formulas, "unified", but separate files):
- [flops.py](../src/analysis/flops.py) — canonical for **GQA / TextVQA / POPE** (LLaVA-1.5: `T=32, D=4096, M=11008,
  N_VISUAL_DENSE=576`). Per-benchmark `n_text` constants present and verified:
  `N_TEXT_TESTDEV=34` (GQA), `N_TEXT_TEXTVQA_OCR=86`, `N_TEXT_TEXTVQA_NOOCR=32`, `N_TEXT_POPE=21`, `N_TEXT_SQA=108`,
  `N_QUESTION=11` (legacy raw-question supplement).
  **It has NO VQAv2 constant.**
- [flops_vqav2.py](../src/analysis/flops_vqav2.py) — the VQAv2 calculator (`N_TEXT_VQAV2=35`), explicitly "unified with
  GQA/shared/flops.py", same per-layer Eq. 5.

So **`flops.py` is canonical for 3 of the 4 datasets**; VQAv2's `n_text=35` lives in the separate (consistent)
`flops_vqav2.py`. For one cross-dataset frontier, the thesis must call both or consolidate. **Flag:** the README/ledger
sometimes imply a single `flops.py` — that is **not** literally true for VQAv2.

### 3.2 — What "FLOPs" includes / excludes

The per-layer formula (both files) is **FastV Eq. 5: `4·n·D² + 2·n²·D + 2·n·D·M` summed over all 32 layers**, with
`n = K + n_text`. This is **LLM prefill only**:
- **Included:** the 32 decoder layers' prefill at the pruned sequence length (QKVO + attention + FFN).
- **Excluded:** the **CLIP vision encoder**, the **multimodal projector**, and the **decode/generation** steps
  (a few short tokens). These are excluded for *all* of dense/static/QC alike, so the *comparison* is fair — but the
  reported "TFLOPs" is **not** total model FLOPs.
- **Consistency across approaches:** within a dataset, dense/static/QC would all use `n=K+n_text` → the convention is
  consistent and the dense-vs-K reduction is honest. **However**, the *saved* VQAv2 baseline reports the
  **attention-only proxy** (`2·T·n²·D`, the `attention_flops_gflops` field, S=K+35), while GQA/TextVQA/POPE saved
  analyses report **FastV-full** (`lm_full_TFLOPs` / `fastv_full_TFLOPs`). Both are computable from the same files, but
  **the as-saved headline numbers use different conventions across datasets** → pick ONE (FastV-full Eq.5 is the
  thesis-wide convention per `flops_vqav2.py` docstring) and re-extract VQAv2 under it.

### 3.3 — Question-conditioned scoring cost (the critical caveat)

The LLaVA-1.5 question-conditioned **selection cost is NOT free, and for the best selector it is large**:
- **LM-attention QC** ([question_cond.py:73](../src/pruning/question_conditioned_selection/question_cond.py#L73)):
  `score_sample` runs **a full `self.backbone(...)` forward over the dense 576-visual-token sequence with
  `output_attentions=True`** to read layer-2/5/8 attention. That is **one full dense prefill** (≈ the K=576 cost,
  GQA ≈ **3.17 TFLOPs**) *before* any pruning. At K=144 the prefill *saving* is only 3.17→0.90 TFLOPs (≈2.27 TFLOPs).
  **So the scoring forward costs more than the saving** — the LM-attention QC selector is a **teacher / upper-bound,
  not a deployable method**, exactly like the Qwen mid-layer teacher (ledger E4).
- **CLIP-space QC** ([clip_select.py](../src/pruning/question_conditioned_selection/clip_select.py)): cheap (reuses
  CLIP features + question embeds, no extra LLM forward) — **but it fails** (GQA K=64: 45.73 vs CLS 54.93).
- **Is this cost in the FLOP accounting?** **No.** The frontier JSONs report only generation/prefill FLOPs at the
  pruned K; **the QC scoring forward is excluded.** For the thesis this must be stated as a caveat, and the honest
  framing is "matched-FLOPs *content* comparison at fixed K with a teacher selector," not "free question-conditioning."

---

## Section 4 — What needs to be re-run

### 4.1 — Exist and clean (directly usable)

- **Dense**: VQAv2 76.44, GQA 61.42, TextVQA-OCR 57.65 / noOCR 46.73, POPE F1 85.78. ✅
- **Static CLS-attn K∈{144,288,432}** on **all four** datasets, locked protocol. ✅ (plus K=64/128/192 extras on VQAv2/GQA/TextVQA/POPE)
- **Static frontiers + oracle bands** (GQA/TextVQA/POPE full split; VQAv2 10k). ✅
- **Idea-2 evidence** (VQAv2 instance headroom, GQA band, cascade, qtype oracle). ✅ (for the "budget is a mirage" section)

### 4.2 — Exist but need re-running / re-extraction

| cell | issue | effort |
|---|---|---|
| VQAv2 FLOPs column | saved as attention-only; re-extract under FastV-full Eq.5 for a single cross-dataset convention | **small** (re-run `flops_vqav2.py`, no GPU) |
| GQA/TextVQA QC @ K=64/96 probes | only n=4000–5000, K=64/96; not the grid; need full-split + grid if used at all | **medium** (re-run eval) |
| VQAv2 "dynamic" 75.76 | this is CLS-ranking + budget controller (Idea 2), **not** Idea 1 — cannot be reinterpreted | n/a (different idea) |

### 4.3 — Don't exist — must be generated (the `dense + static + QC × {72,144,288,432} × {VQAv2,GQA,TextVQA}` matrix)

Priority-ordered:

1. **QC-selection × {72,144,288,432} on VQAv2, GQA, TextVQA** — **the core missing deliverable** (no clean cell exists).
   - **Decision needed first (see §6):** which QC selector? LM-attention teacher (works-ish but full-forward cost) vs a
     cheap learned scorer (`QuestionConditionedTokenScorer`, never run at fixed K). **large** if a learned scorer must be
     trained; **medium** if using the LM-attention teacher probe at the 4 K values (re-run eval, full split).
2. **Static K=72** on VQAv2, GQA, TextVQA (and POPE if kept) — **medium** each.
   - GQA/TextVQA/POPE: relax `--keep_k` choices + `SUPPORTED_K` in [static.py:116](../src/models/static/static.py#L116)
     to allow 72 (cls_attn uses `topk`, so no precomputed spatial index needed) → 1 eval run per dataset.
   - VQAv2: a K=72 generation eval; **uncertain** whether the VQAv2 path needs a new per-K feature cache
     (`cache_static_k*.sh`) or runs CLS-attn topk at runtime — **verify before estimating** (small→medium).
3. **VQAv2 full-split (not 10k)** — optional; current VQAv2 is a 10k subset while GQA/TextVQA are full. For a clean
   cross-dataset story, decide whether to keep 10k (cheaper) or go full-val — **medium/large**.
4. **K=72 on POPE** — only if POPE is promoted from optional — **medium**.

**Note on grid choice:** the existing static grid is `{64,128,144,192,288,432}`. If the thesis can tolerate
**K=64 instead of 72** (11% vs 12.5%), then **static is already complete** for {64/144/288/432} and only the **QC track**
needs generation — a much smaller job. This single substitution removes ~4 static re-runs.

---

## Section 5 — Archive-candidate inventory (nothing moved)

> Dependency rule used below: a file is "safe to archive" only if nothing **in scope** imports it. In-scope =
> dense + static + LLaVA-1.5 question-conditioned selection + the four datasets' eval + shared utils/metrics/data/flops.

### `archive/qwen/` — Qwen code/results
- **Code:** [src/pruning/question_conditioned_selection/qwen_pruner.py](../src/pruning/question_conditioned_selection/qwen_pruner.py),
  `src/pruning/dynamic_budget/qwen_{oracle,oracle_qc,budget_data,budget_eval,budget_robust}.py`,
  `src/evaluation/docvqa/{qwen_kcurve,qwen_control,qwen_layer_sweep,qwen25_dense_eval}.py`,
  [src/analysis/qwen_flops.py](../src/analysis/qwen_flops.py), [src/analysis/make_qwen_figures.py](../src/analysis/make_qwen_figures.py),
  `scripts/data/dl_qwen{3b,32b}.py`, `scripts/data/download_qwen25vl.py`.
- **Results:** `results/thesis_main/highres/qwen_*` (~40 files), `results/paper_candidates/qwen_budget_data_*`.
- **Dependency:** Qwen cluster is self-contained **except** it is the backbone of the distillation track (§ below).
  No in-scope file imports it. **Safe to archive.**

### `archive/llava16/` — high-res LLaVA-1.6
- **Code:** `src/evaluation/textvqa/evaluate_textvqa_highres*.py`, `src/evaluation/gqa/evaluate_gqa_highres.py`,
  `src/evaluation/docvqa/evaluate_highres_*`, [src/analysis/llava_latency.py](../src/analysis/llava_latency.py),
  [src/analysis/oracle_decomposition.py](../src/analysis/oracle_decomposition.py) (reads LLaVA-1.6 spread files),
  [src/analysis/make_figures_highres.py](../src/analysis/make_figures_highres.py).
- **Results:** `results/thesis_main/highres/eval_*highres*`, `eval_spread_*`, `llava_latency.json`, `figures/`.
- **Dependency:** none in-scope. **Safe to archive.**

### `archive/budget/` — budget controller + predictors + oracle diagnostics (Idea 2)
- **Code:** [src/models/dynamic_budget/](../src/models/dynamic_budget/) (`budget_controller.py`, `token_selector.py`,
  `llava_wrapper.py`, `answer_head.py`, **and `token_scorer.py`** — see ⚠), [src/training/train_dynamic.py](../src/training/train_dynamic.py),
  [scripts/data/budget_variance_gate.py](../scripts/data/budget_variance_gate.py), the Qwen budget files (overlap with `archive/qwen/`).
- **Data/results:** **all of [data/budget_oracle/](../data/budget_oracle/)** (binary routing, qtype oracle, stage6c/exp2–5
  diagnostics, val_oracle_static_dense, val_budget_* labels/features), `results/thesis_main/vqav2/{dynamic_150k_clsonly,gate_real}/`.
- ⚠ **Dependency conflicts:**
  - [generate_and_score.py:68](../src/evaluation/vqa/generate_and_score.py#L68) (in scope — VQAv2 eval) **lazily imports
    `LlavaDynamicVQAModel`** for `model_type=="dynamic"`. Archiving `models/dynamic_budget` **breaks the dynamic branch**
    (static branch fine). Needs a guard/refactor (wrap the import in try/except or drop the branch) before archiving.
  - `token_scorer.py` (`QuestionConditionedTokenScorer`) is the **learned question-conditioned scorer** — conceptually
    **Idea 1**, not Idea 2. It lives inside `models/dynamic_budget/` and is only ever instantiated by the budget model.
    **If you keep the option of a learned QC scorer, do not blanket-archive this file** — extract it first.

### `archive/distillation/` — teacher/student
- **Code:** [src/models/distillation/student_selector.py](../src/models/distillation/student_selector.py),
  [src/training/{cache_teacher,train_student}.py](../src/training/train_student.py),
  `src/evaluation/docvqa/{eval_gate,eval_control,control_question_conditioning}.py`.
- **Results:** `results/thesis_main/highres/distill/*` (`*.pt`, gate/control JSONs, logs).
- **Dependency:** imports `qwen_pruner` (archive/qwen). No in-scope importer. **Safe to archive** (move with/after qwen).

### `archive/cascade/` — speculative / confidence cascade
- **Code:** `src/evaluation/gqa/run_speculative_testdev.py`, `src/evaluation/pope/run_pope_speculative.py`,
  `src/evaluation/vqa/{cascade_pass,cascade_analyze,instance_headroom}.py`, [src/analysis/cascade_sweep.py](../src/analysis/cascade_sweep.py).
- **Results:** `results/thesis_main/gqa/{cascade_sweep.json,testdev_speculative_*,pope_speculative_*}`, `vqav2/cascade/`.
- ⚠ **Dependency:** `cascade_pass.py` imports `generate_and_score.load_model` and `run_speculative_testdev.py` imports
  `StaticPrunedLlava` — both **in-scope** (one-way; archiving cascade does not break scope). **Safe to archive.**
  **Caveat:** the cascade is currently cited as *evidence for the budget-is-a-mirage* claim (`results_frozen/all.json`
  "cascade"); if Idea 2 is discussed in the thesis, keep the cascade *numbers* even if the code is archived.

### `archive/elastic/` — elastic Stage-1 / LoRA
- **Code:** [src/models/elastic/](../src/models/elastic/) (`elastic_wrapper.py`, tests),
  [src/training/train_stage1.py](../src/training/train_stage1.py), `configs/stage1_elastic.yaml`,
  `scripts/training/launch_stage1_full.sh`.
- ⚠ **Dependency:** **`src/evaluation/gqa/evaluate_gqa.py`, `src/evaluation/textvqa/evaluate_textvqa.py`, and
  `src/evaluation/test_generate.py` import `ElasticPrunedLlava`.** These three eval files would break if `models/elastic`
  is archived. Check whether any are your intended in-scope GQA/TextVQA driver (the canonical drivers are
  `run_static_testdev.py` / `run_textvqa.py`, which do **not** use elastic) — if not, archive the three `evaluate_*`
  files **together with** elastic.
- **Results:** `results/archived/stage1_*` (already in an archived/ area).

### `archive/superseded_v1/` — VQAv2 classification-head phase + deprecated drivers
- **Code:** the classification answer-head path inside [src/models/static/](../src/models/static/) (`answer_head.py`,
  `llava_wrapper.py`'s `LlavaStaticVQAModel`, `token_selector.py`'s `StaticCLSAttentionTokenSelector`) — **⚠ keep if the
  VQAv2 generation eval uses `LlavaStaticVQAModel`** (it is imported by `generate_and_score.py:65`, in scope).
  `src/training/train_cached.py`, `scripts/data/cache_static_*.sh` (per-K caching) — used only by the cached
  classification pipeline; archive if VQAv2 K=72 runs at runtime instead.
- **Data/results:** VQAv2 `classification_curve` block (inside the locked baseline json), `static_k*_v1`, `static_k*_pertype`,
  `static_k*_fixed` duplicate-preprocessing dirs (keep `_pad` canonical).
- ⚠ Highly entangled with in-scope static — **do not bulk-move `src/models/static/`**; it holds the canonical
  `StaticPrunedLlava` (in scope) **and** the retired classification head in the same package.

### `archive/extra_benchmarks/` — DocVQA / ChartQA / InfoVQA / ScienceQA
- **Code:** [src/data/docvqa.py](../src/data/docvqa.py), `src/evaluation/docvqa/*` (all), `src/evaluation/scienceqa/run_sqa.py`,
  `src/metrics/{docvqa_score,chartqa_score}.py`, `scripts/data/download_docchart.py`.
- **Results:** `results/thesis_main/gqa/sqa_*`, `results/thesis_main/highres/*docvqa*`/`*chartqa*` (overlap with qwen/llava16),
  `results/paper_candidates/*`.
- ⚠ **Dependency:** `run_sqa.py` imports `StaticPrunedLlava` (in scope, one-way — safe). `m4c_evaluator.py`/`textvqa_score.py`
  are shared with TextVQA (**in scope — do NOT archive**). **Safe to archive** the DocVQA/ChartQA/SQA-specific files.

**Shared infrastructure that must STAY (imported across scope):** `src/utils/*`, `src/metrics/{official_score,
textvqa_score,m4c_evaluator,pope_score,eval_pope_official,metrics}.py`, `src/data/{gqa.py,vqav2/}`,
`src/analysis/{flops.py,flops_vqav2.py}`, `src/models/static/static.py` (`StaticPrunedLlava`),
`src/pruning/static/visionzip.py`, and the LLaVA-1.5 QC probes
`src/pruning/question_conditioned_selection/{question_cond.py,clip_select.py}` + `run_{qcond,clip}_probe.py`
(**these ARE the in-scope question-conditioned approach**).

---

## Section 6 — Open questions (prioritized; these block the cleanup decision)

1. **[BLOCKER] Which question-conditioned selector is "the method" on LLaVA-1.5?** The existing evidence says the only
   QC selector that beats CLS is **LM-attention, which needs a full LLM forward** (not FLOPs-cheap), and the cheap
   CLIP-space one **fails**. So under the proposed LLaVA-1.5-only scope, the "question-conditioned pruning" arm has
   **no cheap winner and a mixed-to-negative record** (GQA +3.5pp at K=64, TextVQA −5.58pp; ledger L4). **Do you want
   the QC arm to be (a) the LM-attention teacher reported honestly as an upper-bound at matched-FLOPs-content, (b) a
   newly-trained cheap learned scorer (`QuestionConditionedTokenScorer`) that has never been run at fixed K, or (c) a
   re-scoped negative result ("on frozen low-res, question-conditioned selection does not beat CLS")?** This determines
   whether §4.3-item-1 is medium (probe) or large (train), and whether the thesis story is a win or a characterized
   negative.
2. **[BLOCKER] K=72 vs K=64.** Will you accept **K=64** (already run everywhere) as the 12.5% point, or must it be
   exactly **72**? Accepting 64 makes the static track essentially complete and removes ~4 re-runs.
3. **VQAv2 split.** Keep the **10k stratified** subset (cheaper, already done) or move to **full val** for parity with
   GQA/TextVQA full-split? Affects every VQAv2 re-run estimate.
4. **FLOP convention.** Confirm **FastV-full Eq.5** (`flops_vqav2.py`/`flops.py`) as the single cross-dataset convention
   (VQAv2 saved numbers are currently attention-only) — and whether the vision-encoder + projector should be **added**
   to make the number a true total (currently prefill-only).
5. **Does the VQAv2 K=72 eval need a new feature cache?** I could not verify whether the VQAv2 generation eval runs
   CLS-attn `topk` at runtime or reads per-K caches. This changes the K=72/K=64-substitution effort from small to medium.
6. **POPE in or out?** It is "optional"; it has dense+static {144,288,432} but **no QC runs** — if kept, a QC track must
   be generated for it too.
7. **Idea-2 in the thesis: discuss or drop?** The strongest *native LLaVA-1.5* Idea-2 evidence (VQAv2 instance headroom,
   GQA band, cascade) is genuinely useful for a "why fixed-K" justification. If you keep it as a section, the cascade
   and oracle **numbers** must be retained even though their **code** is an archive candidate (§5).

---

## Section 7 — What I did not read or could not verify (honest gaps)

- **Read in full / verified:** the VQAv2 locked baseline json (all K + protocol), `dense_pad` / `gate_real` jsons,
  GQA `testdev_frontier_analysis.json`, the `qcond_gqa` / `clip_gqa` / `qcond_textvqa_noocr` probe results,
  `textvqa_analysis_ocr.json`, `pope_analysis.json`, `week1_all_numbers.json`, `qtype_oracle_summary.json`,
  `val_oracle_static_dense_summary.json`, `binary_routing_eval_summary.json`, the `stage6c_6type` / `exp5_7type` txt
  diagnostics, the head of `val_budget_visual_features_binary.jsonl` (feature keys), and the source files
  `run_qcond_probe.py`, `run_static_testdev.py` (header), `question_cond.py`, `budget_variance_gate.py`,
  `flops.py`, `flops_vqav2.py`, `qwen_oracle.py`/`qwen_oracle_qc.py`/`qwen_budget_eval.py`, plus the full
  cross-module import grep for the dependency analysis.
- **Skimmed / inferred, not line-by-line:**
  - Per-cell **metrics.json** for most GQA/TextVQA/POPE static dirs — I trusted the aggregated `*_analysis.json` /
    `testdev_frontier_analysis.json` rather than re-reading each run dir's `metrics.json`. Individual run-dir accuracies
    could differ slightly from the aggregates if an aggregate was hand-edited (**uncertain**, low risk).
  - The VQAv2 `static_k*_fixed` / `_pertype` / `_v1` / `_matched` directories — I did **not** open each; I only
    established `_pad` is canonical. The exact preprocessing of `_fixed`/`_v1` is **uncertain**.
  - `clip_select.py` body (I read its role and the GQA/TextVQA result numbers, not the full scoring code), and
    `run_clip_probe.py`.
  - The VQAv2 caching/generation path (`train_cached.py`, `generate_and_score.py` full body, `cache_static_*.sh`) —
    I did not fully trace whether K=72 needs a cache rebuild (Open Q5).
  - The big `val_budget_labels_3class.json` / `val_oracle_static_dense.json` (6.5 MB) — I read only their summaries,
    not the per-record bodies.
  - I did not open the `_1k` VQAv2 eval variants, per-sample prediction dumps, `.tex`/`master_table.md`, or the
    `dynamic_stage6c_7type` / `exp2–4` json bodies (read the 6type/exp5 `.txt` representatives only).
- **Could not verify (need you):** whether `data/budget_oracle/*` accuracies are on the classification-head proxy or a
  generation protocol (dated May, likely classification — **uncertain**); whether any in-scope driver you actually use
  depends on the `evaluate_gqa.py`/`evaluate_textvqa.py` (elastic) path rather than `run_static_testdev.py`/`run_textvqa.py`.

*No files were moved, edited, deleted, or run. Awaiting your decisions on the Section 6 blockers before proposing actions.*
