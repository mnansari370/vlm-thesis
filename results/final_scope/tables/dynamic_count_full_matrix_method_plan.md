# Dynamic-COUNT full-matrix implementation plan (Stage 0 — infrastructure frozen)

*The main dynamic-pruning phase of the thesis. Method design: `dynamic_count_method_plan.md` (v2).
This document is the implementation-level plan for the full 8-cell matrix.*

## The method in one paragraph

The five percentages p15/p25/p35/p50/p75 are **static baselines and evaluation anchors only**.
The real Dynamic-COUNT predicts a **sample-specific integer token count**
`K_i = clamp(round(f(signals_i)), 32, N_i)` — e.g. 64, 97, 188, 240, 391. Execution: one cheap
**probe** pass at K_probe (p15 or p25, chosen on the calibration split) records the answer and all
controller signals for free; if the controller says the sample needed ≤ K_probe the probe answer
stands, otherwise exactly one second pass runs at K_i. Every executed prefill is charged
(double-pay). Two variants: **DC-D** (discrete cascade over the anchors — baseline/ablation,
composed offline from frozen finals) and **DC-C** (continuous integer K_i — THE main method, real
GPU second passes).

## Controllers (fit on the FIRST 20% of manifest order; evaluated on the held-out 80%)

1. **Rule controller (primary, transparent):** risk = one probe-confidence signal → quantile bins →
   per-bin sufficiency fraction read from the sample-level anchor outcomes (linear interpolation
   BETWEEN anchors → continuous targets; isotonic so higher risk never gets fewer tokens); compute
   knob λ sweeps the accuracy-vs-FLOPs curve.
2. **Ridge controller (secondary):** ≤30-weight closed-form ridge on
   confidence + saliency-statistics + question features → calibrated risk → the same rule mapping.
Signals recorded per probe sample: first-token max-prob / margin / entropy, length-normalized
answer log-prob, mean/min token prob, answer length; saliency entropy / participation ratio /
top-{32,64,128,256} mass / max / mean / std / concentration; question length + wh-flags +
number-flag; native visual token count.

## Fairness

Same locked manifests, prompts, scorers, greedy bs=1 mnt=64, TextVQA OCR-on, DocVQA
instruction-on, same fairness gates. FLOPs = per-sample sum of ALL executed prefills. The static
curve is rebuilt from frozen per-sample anchor JSONLs restricted to EXACTLY the evaluated ids
(plus dense), and Δcurve is reported at matched average FLOPs with labels
dynamic_win (> +0.50), near_tie (±0.50), dynamic_loss (< −0.50). All 8 cells reported, negatives
included. No frozen file is modified; all Dynamic-COUNT outputs use the `dynamic_count_` prefix.

## Reproduction gates (hard requirement before any DC evaluation)

The probe reruns the cheap pass through wrappers that mirror the frozen paths exactly (LLaVA:
cls_attn top-K → `_build_inputs` → honest greedy with `output_scores` recording; Qwen: frozen
selection formulas + a `_greedy` mirror whose argmax stays on the original logits tensor). Every
probe prediction is compared per-sample against the frozen reference final (static cls_attn/norm;
WHICH textsim for the Qwen×TextVQA textsim probe). **Any mismatch fails the gate, exits rc≠0, and
blocks the phase.**

## Stages

- **S0 (done):** infrastructure + CPU tests (this freeze).
- **S1:** probe launcher `run_dynamic_count_probe_full_matrix.sh` — 18 full-manifest probe jobs
  (8 cells × p15/p25 + Qwen×TextVQA textsim × 2), GPU0=LLaVA ∥ GPU1=Qwen, skip-safe, per-job logs,
  failure summary. Gate: all reproduction checks pass.
- **S2 (CPU):** `make_dynamic_count_controller_calibration` — DC-D sweep + rule/ridge fits +
  signal-quality report, calibration split only.
- **S3 (CPU):** `compose_dynamic_count_discrete` — DC-D for all 8 cells from frozen anchors.
- **S4:** `run_dynamic_count_continuous_full_matrix.sh` — DC-C second passes at real K_i, both
  controllers, both GPUs.
- **S5 (CPU):** audit + validate + final tables (probe/DC-D/DC-C summaries, five-way comparison,
  win/loss, K_i histograms).

## Cost estimate (S1+S4)

Probes: p15+p25 ≈ 40% of one dense pass per cell → roughly one dense-final-equivalent per model
across all four datasets. DC-C: only escalated samples (expected 20–50%) × one pass at K_i.
Both GPUs in parallel throughout.
