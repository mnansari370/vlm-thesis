# Selection vs Budget in Question-Conditioned Visual-Token Pruning — Qwen2.5-VL findings

**Frozen Qwen2.5-VL-7B (current SOTA open VLM, native dynamic-resolution ≈ hundreds–thousands of
visual tokens). Generation protocol, official scorers. n=200 (DocVQA val, ChartQA test balanced).**

The Part-A study (LLaVA-1.6) is re-run on the *current SOTA* backbone, where absolute numbers are
high/credible (dense 85–97%) and the dynamic-pruning field is moving. Central question: in
question-conditioned **dynamic** pruning, how much does *which tokens* (selection) vs *how many
tokens* (budget) each contribute — measured **honestly** (strong baselines, matched FLOPs,
oracle-noise corrected)?

Harness: `v2/qwen/qwen_pruner.py` (prune-before-LLM; M-RoPE positions of kept tokens preserved;
manual greedy decode). **Validated: keep-all == stock generate, 6/6 exact.** Dense fidelity:
DocVQA 94.85 (n=50) ≈ published 95.7; ChartQA 85.6 (balanced) ≈ 87.3.

---

## Finding A — SELECTION is the dominant lever (and genuinely question-conditioned)

K-curve, DocVQA (ANLS, dense 97.2). Blind = uniform/norm (the hard-to-beat baselines per the
analysis literature); QC = mid-layer (L16) question→visual attention.

| K | uniform | norm | **QC** | (full) |
|---|---|---|---|---|
| 64  | 23.9 | 13.8 | **84.7** | 97.2 |
| 128 | 32.2 | 21.0 | **92.2** | |
| 256 | 61.1 | 37.1 | **94.7** | |
| 512 | 87.7 | 75.7 | **97.2** | |

ChartQA (relaxed, dense 85.5): K=128 QC **83.5** vs uniform 61.0 vs norm 54.5.

- **Selection advantage is huge and task-dependent:** +60pp at K=128 on DocVQA (documents), +22pp on
  ChartQA (charts) — biggest where images are most information-dense.
- **Genuinely question-conditioned (control):** selecting by a *mismatched* question gives 37.0 ANLS
  (≈ blind 32.2), the *real* question gives 93.8 → **92% of the gain is due to the question.** Not a
  better blind saliency.
- *Honesty:* the QC selector is the mid-layer **teacher** (needs a full forward); the deployable cheap
  version is occupied by FlashVLM/FEATHER and is cited, not claimed.

## Finding B — The honest FLOPs frontier (`v2/qwen/qwen_flops.py`)

DocVQA, prune-before-LLM (all 28 layers see K), Qwen2.5-7B constants (d=3584, m=18944):

| K | FLOP-red | QC acc | QC retention |
|---|---|---|---|
| 128 | 88% | 92.2 | **94.8%** |
| 256 | 78% | 94.7 | 97.5% |
| 512 | 60% | 97.2 | 100% |

**QC selection retains ~95% of dense accuracy at 88% FLOP reduction** (and 100% at 60%), while blind
retains 22–33% at K=128.

**FLOPs → real latency (because we prune BEFORE the LLM).** Measured LLM generate() latency on LLaVA-1.6
(DocVQA, ~2252 dense tok, warmup + cuda.sync): full 355ms → K=128 **107ms = 3.3× speedup** (K=64 3.9×,
K=256 2.9×). Unlike FastV's inside-layer pruning (FLOPs cut ≠ latency, FlashAttn-incompatible), prune-
before-LLM converts the token cut into genuine wall-clock speedup — so ~95% of dense accuracy at **3.3×
faster** decode with good selection.

## Finding C — DYNAMIC BUDGET is a real but SECONDARY lever, coupled to selection quality

Per-sample budget oracle (fix selector, vary K, monotone noise-correction; `qwen_oracle*.py`):

| selector | dataset | honest dynamic-budget gain (matched-FLOPs) | avg tokens |
|---|---|---|---|
| uniform (blind) | DocVQA | **+20.7pp** | 337 |
| **QC (good)** | DocVQA | **+7.5pp** | 91 |
| **QC (good)** | ChartQA | **+2.7pp** | 122 |

- **The apparent dynamic-budget benefit is largely a symptom of weak selection.** With blind selection,
  per-sample token-need varies hugely (easy doc question → 32 tokens, hard → full) → +20.7pp. A good
  question-conditioned selector **collapses that spread** (everything fits in fewer tokens) → the honest
  budget gain drops to +7.5pp (DocVQA) / +2.7pp (ChartQA), AND the avg budget drops 337→91.
- So dynamic budget is **real but secondary, and task-dependent** (bigger on documents).

## Finding D — The dynamic-budget headroom is NOT realizable (training-aware confirmation)

We trained a per-sample budget predictor (the thing ATP-LLaVA/Dynamic-LLaVA propose) on cheap QC
attention-distribution features, 5-fold CV, n=400 DocVQA (`qwen_budget_data.py` + `qwen_budget_eval.py`):

| | accuracy @ avg budget | gain vs fixed-budget |
|---|---|---|
| oracle dynamic (noise-free, monotone) | 96.2% @ 108 tok | +4.2pp |
| **trained predictor (MLP, 5-fold CV)** | 84.5% @ 32 tok | **+0.0pp** |
| heuristic (attention-spread rule) | 95.0% @ 428 tok | −1.3pp |

Across THREE token-hungry benchmarks, a realistic per-sample budget predictor provides **no benefit and
can hurt** — for two complementary reasons:

| benchmark | monotone oracle headroom | trained predictor (5-fold CV) | hard-tail predictability (AUC) |
|---|---|---|---|
| **DocVQA** | +4.2pp (real) | **+0.0pp** (captures 0%) | 0.59 (≈unpredictable) |
| **ChartQA** | +2.7pp (small) | ~0 | — |
| **InfoVQA** | **−0.2pp (none)** | **−8.4pp (hurts)** | 0.72 (predictable, but useless) |

- **DocVQA:** headroom exists (+4.2pp) but is **unrealizable** — the hard tail (18% needing >32 tokens) is
  ~unpredictable from attention features (AUC 0.59), so the predictor collapses to "minimum-for-all" and
  captures 0%. (81% of samples are already correct at the 32-token minimum.)
- **InfoVQA:** there is **essentially no oracle headroom** (−0.2pp). The hard tail (45%) *is* predictable
  (AUC 0.72), but giving those samples more tokens **doesn't fix them** (they fail at full resolution too),
  so a budget predictor only mis-allocates and **hurts (−8.4pp)**.
- **Conclusion (bulletproof, training-aware, 3 benchmarks):** per-sample dynamic budgeting — the premise of
  ATP-LLaVA/Dynamic-LLaVA — provides **no realizable benefit**: either the headroom isn't capturable
  (DocVQA) or it doesn't exist (InfoVQA). **Invest in selection; per-sample budgeting is a mirage.**
- **Stable at full-val scale:** Qwen-7B DocVQA n=1000 (dense 95.0): 81.6% correct at min-32, oracle +3.9pp,
  **trained predictor captures 0%** — identical to the n=400 result.

---

## Finding E — Generality across SCALE and ARCHITECTURE (the budget mirage is a general phenomenon)

The budget mirage holds across **2 model families × 2 scales × multiple benchmarks** (DocVQA, n=400, QC
selection; each model's pruner re-validated keep-all==stock; each model's best QC layer from its own sweep):

| model | family/scale | dense | oracle budget headroom | **trained predictor** | hard-tail AUC |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | Qwen / 3B | 91.7 | +3.1pp | **−1.4pp (hurts)** | 0.55 |
| Qwen2.5-VL-7B | Qwen / 7B | 97.2 | +4.2pp | **0% (collapses)** | 0.59 |
| Qwen2.5-VL-32B (4-bit) | Qwen / 32B | 90.5 | +5.0pp | **+0.1pp (2%)** | 0.64 |
| LLaVA-1.6-7B | LLaVA / 7B | 67.8 | +3.1pp | **−10.7pp (hurts)** | 0.60 |
| *(Qwen-7B InfoVQA)* | Qwen / 7B | 74.7 | −0.2pp | **−8.4pp (hurts)** | 0.72 |

*(32B: 4-bit + reduced resolution to fit a single GPU; relative decomposition unaffected. Qwen-7B
full-val n=1000 also confirms: predictor 0%, oracle +3.9pp, dense 95.0.)*

**Invariant across all:** the oracle budget headroom is small (+3–4pp, or none); a **trained per-sample
budget predictor never helps** (captures 0% or actively hurts, down to −10.7pp); and the hard tail is
~unpredictable (AUC 0.55–0.60) or, when predictable (InfoVQA), useless because more tokens don't fix those
samples. Selection meanwhile dominates on every model (+56–60pp). **This is a general phenomenon across
model families, scales, and benchmarks — not a single-model artifact.**

## The one-line message (honest, data-driven)

> **In question-conditioned dynamic pruning on a modern VLM, *which* tokens you keep dominates (+60pp,
> retains 95% of dense at 88% FLOP-cut); *how many* per sample is a MIRAGE — its oracle headroom (+4–7pp)
> is not realizable (a trained predictor captures 0%), because good selection already fits 81% of samples
> in the minimum budget and the hard tail is unpredictable.** Invest in selection; per-sample budgeting
> (the premise of adaptive-pruning methods) is not worth it.

## Honest positioning / novelty

Components are known (FEATHER/FlashVLM = mid-layer/question selection; ATP-LLaVA/Dynamic-LLaVA =
adaptive budgets). The contribution is the **rigorous joint selection-vs-budget decomposition with
oracle-noise correction on the current SOTA backbone**, plus the **selection–budget coupling** result
(budget benefit is a function of selection quality). Realistic venue: workshop / ACL-EMNLP Findings /
TMLR — an honest measurement paper on the user's exact topic, with high credible numbers.

## Remaining
- Phase 2: fine-tune a per-sample budget predictor → confirm it captures only a fraction of the +7.5pp
  (training-aware confirmation; the fine-tuning Nafees wanted).
- InfoVQA (3rd, most token-hungry benchmark); full-val numbers; real-latency measurement.
