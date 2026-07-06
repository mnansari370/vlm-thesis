# Dynamic-COUNT — code map

The shared evaluation runtime currently lives under `src/final_scope/` for backward compatibility.
The method-facing entry point is this folder. Exact files implementing Dynamic-COUNT:

## Runner core (CPU)

| File | Role |
|---|---|
| `src/final_scope/dynamic_count_eval.py` | probe (reproduction-gated), DC-D composer, DC-C runner, same-ids static-curve builder, honest multi-pass FLOPs |

## Signals, controllers, and GPU wrappers

| File | Role |
|---|---|
| `src/pruning/dynamic_count/__init__.py` | pure helpers: budget bookkeeping, calibration split, saliency stats, question features |
| `src/pruning/dynamic_count/confidence.py` | confidence-signal math (probe-side) |
| `src/pruning/dynamic_count/controllers.py` | RuleController (risk-bin sufficiency fractions, isotonic, λ knob) + RidgeRiskModel (≤30 weights) |
| `src/pruning/dynamic_count/llava_dynamic_count.py` | LLaVA-1.5 probe + arbitrary-K second pass; mirrors the frozen engine |
| `src/pruning/dynamic_count/qwen_dynamic_count.py` | Qwen2.5-VL probe + arbitrary-K second pass; mirrors the frozen engine |

## Pipeline launchers (5 stages)

| Stage | File |
|---|---|
| 1. probes (GPU, reproduction-gated) | `scripts/dynamic_count/run_probe.py` · `run_dynamic_count_probe_full_matrix.sh` |
| 2. calibration (CPU, first-20% split) | `scripts/dynamic_count/make_controller_calibration.py` |
| 3. DC-D compose (CPU) | `scripts/dynamic_count/run_discrete.py` · `compose_dynamic_count_discrete.py` |
| 4. DC-C run (GPU) | `scripts/dynamic_count/run_continuous.py` · `run_dynamic_count_continuous_full_matrix.sh` |
| 5. oracle bound (CPU, not a method) | `scripts/dynamic_count/analyze_oracle.py` |

## Validation and audit

| File | Role |
|---|---|
| `scripts/validation/validate_dynamic_count.py` | every probe/DC-D/DC-C aggregate: gates, n, jsonl line counts, probe reproduction |
| `scripts/validation/audit_dynamic_count.py` | all 8 cells complete (probes + config + DC-D + DC-C) |

## Inputs and outputs

- Reads: the frozen **static finals** (and, for COUNT-on-WHICH, the frozen **WHICH finals**) whose
  predictions the probe reproduces; the dense final; `configs/sample_ids/{dataset}.json`.
- Fitted controllers (tracked): `results/configs/dynamic_count/{model}_{dataset}[_textsim].json`.
- Writes: `results/runs/{model}/{dataset}/dynamic_count_{probe,dcd,dcc}_*[...].{json,jsonl}`.

> **COUNT-on-WHICH is kept separate from pure COUNT** (its own `_textsim` config and result rows) so
> the WHICH-vs-COUNT attribution stays clean. Result basenames are load-bearing — do not rename.
