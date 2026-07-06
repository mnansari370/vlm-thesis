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
| 1. probes (GPU, reproduction-gated) | `scripts/final_scope/run_dynamic_count_probe.py` · `run_dynamic_count_probe_full_matrix.sh` |
| 2. calibration (CPU, first-20% split) | `scripts/final_scope/make_dynamic_count_controller_calibration.py` |
| 3. DC-D compose (CPU) | `scripts/final_scope/run_dynamic_count_discrete.py` · `compose_dynamic_count_discrete.py` |
| 4. DC-C run (GPU) | `scripts/final_scope/run_dynamic_count_continuous.py` · `run_dynamic_count_continuous_full_matrix.sh` |
| 5. oracle bound (CPU, not a method) | `scripts/final_scope/analyze_dynamic_count_oracle.py` |

## Validation and audit

| File | Role |
|---|---|
| `scripts/final_scope/validate_dynamic_count_final.py` | every probe/DC-D/DC-C aggregate: gates, n, jsonl line counts, probe reproduction |
| `scripts/final_scope/audit_dynamic_count_full_matrix.py` | all 8 cells complete (probes + config + DC-D + DC-C) |

## Inputs and outputs

- Reads: the frozen **static finals** (and, for COUNT-on-WHICH, the frozen **WHICH finals**) whose
  predictions the probe reproduces; the dense final; `configs/final_scope/sample_ids/{dataset}.json`.
- Fitted controllers (tracked): `results/final_scope/dynamic_count_configs/{model}_{dataset}[_textsim].json`.
- Writes: `results/final_scope/{model}/{dataset}/dynamic_count_{probe,dcd,dcc}_*[...].{json,jsonl}`.

> **COUNT-on-WHICH is kept separate from pure COUNT** (its own `_textsim` config and result rows) so
> the WHICH-vs-COUNT attribution stays clean. Result basenames are load-bearing — do not rename.
