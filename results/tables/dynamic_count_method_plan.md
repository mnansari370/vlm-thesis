# Dynamic-COUNT method plan v2 — continuous per-sample token budgets

*Redesigned 2026-07-04 after the correction: the main Dynamic-COUNT method must predict a
sample-specific INTEGER token count K_i (e.g. 64, 97, 188, 240, 391), not merely choose among the
five hand-set percentages. The five budgets p15/p25/p35/p50/p75 are kept as (1) fixed static
baselines, (2) evaluation anchors, (3) the static accuracy-vs-FLOP curve, and (4) a discrete
baseline version of Dynamic-COUNT — but they do NOT constrain the real controller.*

---

## 1. Two method variants

### DC-D — Discrete Dynamic-COUNT (baseline variant)
Controller chooses from the anchor set {p15, p25, p35, p50, p75}. This is the confidence-gated
cascade of plan v1: probe cheap, escalate unconfident samples to an anchor. Kept because it is
(a) directly comparable to every anchor baseline, (b) composable OFFLINE from the frozen finals
(escalation outcomes already on disk → nearly free), and (c) the clean ablation that isolates what
continuity adds.

### DC-C — Continuous / integer Dynamic-COUNT (**the main method — real dynamic pruning**)
Per sample, the controller predicts an integer token budget:

  K_i = clamp(round(f(signals_i)), min_tokens, max_tokens)      e.g. 64, 97, 188, 240, 391

with `min_tokens` = 32 (default) and `max_tokens` = the sample's full visual count (576 for LLaVA;
the native n_visual for Qwen). Execution policy (two passes maximum):

1. **Probe pass** at K_probe (chosen from {p15, p25} on the calibration split) with the locked
   static selector; record the answer + all controller signals (free, same forward).
2. **Controller** predicts K_i from the signals.
3. If **K_i ≤ K_probe** → the sample "needed few tokens": keep the probe answer (cost = probe only).
   If **K_i > K_probe** → re-run at exactly K_i tokens (cost = probe + K_i; honest double-pay).

The per-sample K_i distribution is a headline deliverable (histogram per cell) — the visual proof
that the method allocates a *continuum* of budgets, not five bins.

## 2. Controller design (DC-C)

**Signals** (all recorded during the probe pass at zero extra cost):
- *Answer confidence*: first-token max probability; top1−top2 logit margin; first-token entropy;
  length-normalized answer log-probability.
- *Saliency-distribution statistics* (from the selector scores already computed): score entropy,
  participation ratio (effective support), top-64/128/256 mass.
- *Question features*: token/word length, wh-type flags.
- *Visual token count* (Qwen: native n_visual; LLaVA: constant 576, dropped).

**Two calibrated mappings** (both fit ONLY on the calibration split — first 20% of manifest order —
and evaluated on the remaining 80%):

- **Rule-based monotone mapping (primary, fully transparent).** Bin calibration samples by
  predicted risk (a single confidence signal, or the regression score below). For each risk bin,
  estimate the accuracy-vs-K curve from the sample's outcomes at the anchors and take the smallest K
  (LINEARLY INTERPOLATED between anchors) at which the bin reaches within ε of its saturation
  accuracy → that bin's target K. Continuous risk → continuous K by interpolating between bin
  targets. A compute knob λ scales all targets, sweeping out the accuracy-vs-FLOPs curve.
- **Small calibrated regression (secondary).** Ridge/logistic regression on the full signal vector
  → calibrated risk → the same risk→K map. Tiny (≤10 weights), fit on the calibration split with
  CV; disclosed in the method section. No neural budget head, no backbone training.

**Why a probe pass is mandatory (not a design choice):** input-only budget prediction was tested
twice and is dead — the June `dynamic_budget` phase (trained MLP captured 1.8% of oracle headroom;
hard-tail AUC 0.643) and the 2026-07-03 full-scale replication (question-length and token-count
routers: Δ vs static curve −0.32…+0.07pp on all 8 cells ≈ random control). All predictive power
about "how many tokens this sample needs" lives in the model's own reaction to the sample — which
requires one cheap forward. The probe is honestly charged to the method's FLOPs.

## 3. Evidence that continuity beats the discrete cascade (frozen data, perfect-signal ceilings)

Graded escalation (probe → cheapest sufficient anchor) vs binary escalation (probe → p75), exact
per-sample computation with double-pay; Δ = above the static curve at matched average compute:

| Cell | binary Δcurve | **graded Δcurve** | compute saved |
|---|--:|--:|--:|
| qwen × textvqa | +8.88 | **+11.64** | 9.0% |
| qwen × docvqa | +0.73 | **+6.01** | 20.4% |
| llava × gqa | +4.33 | **+5.45** | 3.6% |
| qwen × gqa | +3.36 | **+4.28** | 2.9% |
| llava × textvqa | +3.06 | **+4.05** | 2.1% |
| qwen × vqav2 | +4.06 | **+4.70** | 4.4% |
| llava × vqav2 | +2.87 | **+3.59** | 2.9% |
| llava × docvqa | +1.77 | **+2.98** | 2.2% |

Gradation improves the ceiling on **all 8 cells**, and rescues Qwen DocVQA (binary: impossible even
with a perfect signal; graded: +6.0 by escalating most samples only to mid budgets). DC-C, which
refines gradation from 3 anchor rungs to a continuum, can only sharpen this frontier further.
(These are perfect-signal ceilings; realized gains depend on the measured confidence ROC — §5.)

## 4. Fairness protocol

- Same locked manifests, prompts, scorers, greedy bs=1, max_new_tokens=64, TextVQA OCR-on, DocVQA
  instruction-on; same schema and fairness gate (`method="dynamic_count"`, `variant=dc_discrete |
  dc_continuous`); per-sample records carry `k_predicted`, `k_executed`, `escalated`, and every
  controller signal.
- **The five anchors stay in every table and plot** as the static baselines and the
  accuracy-vs-FLOP curve; DC-D is reported beside DC-C at matched compute. Dense and WHICH
  (textsim) remain on the same plots.
- **FLOPs** = per-sample sum of ALL executed prefills (probe + optional K_i pass; FastV Eq.5
  handles arbitrary K), then averaged — identical convention to dense/static/WHICH. Decode excluded
  consistently for all methods (noted: escalated samples decode twice).
- **Comparison is curve-vs-curve**: DC-C traced by the λ sweep; a win = above the interpolated
  static curve at the same average FLOPs. Calibration split (first 20%) is excluded from the
  evaluated 80%; thresholds/mappings are never chosen post-hoc on test data.
- Between-anchor accuracy for calibration is an interpolation ASSUMPTION — stated openly; the pilot
  measures real outcomes at the controller's actual K_i values, which is the ground truth that
  validates or corrects it.

## 5. Pre-registered go/no-go (unchanged gate, revised DocVQA expectation)

The controller's power still hinges on one measurable: the probe-confidence ROC for identifying
wrong answers. Required wrong-capture at 15% false-escalation for +1pp over the static curve
(binary p25→p75 basis): llava-textvqa 47%, llava-gqa 54%, qwen-vqav2 58%, qwen-gqa 63%,
llava-vqav2 64%, qwen-textvqa 51% (CGC-S basis; DC-C/W is primary there), llava-docvqa 87%
(expected negative). **Qwen DocVQA: binary variant remains impossible; the graded/continuous
variant is newly viable at ceiling (+6.0) but demands a strong signal — kept in the matrix as the
stretch cell, not excluded.** ROC AUC ≈ 0.70–0.78 satisfies most cells — a realistic range for
short-answer VLM confidence.

## 6. Implementation notes (frozen generators respected)

- **Qwen**: `QwenPruner.generate(selector="norm", K=K_i)` already accepts ANY integer K — DC-C
  works out of the box. Confidence: `QwenConfidencePruner(QwenPruner)` subclass overriding
  `_greedy` to record per-step log-probs; gated by exact-match reproduction of the frozen static
  finals (free equivalence check, same clean-room discipline as WHICH).
- **LLaVA**: arbitrary K_i via the established WHICH-wrapper pattern (compute cls_attn scores, take
  top-K_i sorted, set `keep_k`, call `_build_inputs`) — no `static.py` changes;
  `generate_answers(return_confidence=True)` already exists, extended at the wrapper level for
  margin/entropy/mean-logprob.
- **DC-D** escalation outcomes come free from the frozen anchor finals (offline composition).
  **DC-C** escalated samples need real generation at their K_i (new GPU, bs=1) — this is the true
  cost of continuity and is budgeted in the stages.
- New: `src/pruning/dynamic_count/` (confidence wrappers + controller), 
  `src/final_scope/dynamic_count_eval.py`, `scripts/final_scope/run_dynamic_count_probe.py`,
  `scripts/final_scope/run_dynamic_count_continuous.py`, offline composer + tables scripts, CPU
  tests in `test_final_scope.py`.

## 7. Staged execution (each GPU stage needs explicit approval)

- **S0 — CPU (done):** honest oracle decomposition; router null; binary + graded ceilings; go/no-go.
- **S1 — GPU pilot:** n=1000 probe passes (p15 AND p25) with full signal recording on Qwen TextVQA
  (static-norm probe AND textsim probe), Qwen GQA, LLaVA GQA. Predictions cross-checked against
  frozen finals (equivalence gate).
- **S2 — CPU:** measure real ROC per signal/cell vs the pre-registered table; fit both controllers
  on the calibration split; compose DC-D offline; select DC-C operating points (λ grid).
- **S3 — GPU pilot:** execute DC-C second passes for the pilot cells' escalated samples at their
  actual K_i (n=1000 scale) → first real continuous-K accuracy/FLOPs points; validate the
  interpolation assumption.
- **S4 — GPU full (only cells passing the gate):** full-n probes + DC-C escalations; DC-D composed
  offline for all 8 cells (including both DocVQA cells — reported even if negative).
- **S5 — CPU:** final tables/curves: `dynamic_count_final_summary.{csv,md}`, K_i histograms,
  dense/static/WHICH/DC-D/DC-C combined comparison, win/loss summary — same freeze discipline as
  WHICH.

## 8. Expected outcome and thesis framing

DC-C is the thesis's centerpiece dynamic-pruning method: *training-free, self-routed, per-sample
continuous token budgets*. Realistically: a strong result on Qwen TextVQA (probe=textsim stacks the
validated WHICH win; ceiling +11.6), moderate broad wins elsewhere if the measured ROC clears the
pre-registered bars, and DocVQA as the honest hard case. If the confidence signal underdelivers,
the axis closes with pre-registered rigor. Either way the final matrix reads:
**dense → static anchors → WHICH (fixed-budget selection) → DC-D (discrete routing) → DC-C
(continuous per-sample budgets)** — a complete, fair, honestly-accounted progression from static to
real dynamic pruning.
