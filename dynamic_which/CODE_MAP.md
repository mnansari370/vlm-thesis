# Dynamic-WHICH — code map

The shared evaluation core lives under `src/common/`; the method runner cores live under `src/dense/`, `src/static/`, `src/dynamic_which/`, and `src/dynamic_count/`.
The method-facing entry point is this folder. Exact files implementing Dynamic-WHICH:

## Runner core (CPU)

| File | Role |
|---|---|
| `src/dynamic_which/evaluate_dynamic_which.py` | WHICH runner core; same budget contract as static, loads the dense final AND the same-budget static final so it can report Dyn−Dense and Dyn−Static over the same samples |

## Selectors (GPU)

| File | Role |
|---|---|
| `src/pruning/dynamic_which/llava_textsim.py` | LLaVA-1.5 `textsim` (+ `textsim_cls_mix`) selector; composes the frozen LLaVA engine |
| `src/pruning/dynamic_which/qwen_textsim.py` | Qwen2.5-VL `textsim` (+ `textsim_norm_mix`) selector; composes the frozen Qwen engine |
| `src/pruning/dynamic_which_ref/` | independent clean-room re-implementation (`llava_ref.py`, `qwen_ref.py`, pure helpers) used only to validate the selectors above |

## Launchers and validation

| File | Role |
|---|---|
| `scripts/dynamic_which/run_dynamic_which.py` | one WHICH cell (`--selector textsim`, `--budget-pct`) |
| `scripts/dynamic_which/run_dynamic_which_ref.py` | clean-room reference pilots |
| `scripts/dynamic_which/run_dynamic_which_full_matrix.sh` | the 35 full finals (Qwen×TextVQA finals already existed) |
| `scripts/validation/validate_dynamic_which.py` | validates the five Qwen×TextVQA WHICH finals |
| `scripts/validation/audit_dynamic_which.py` | audits all 40 WHICH cells for completeness |
| `scripts/dynamic_which/compare_qwen_textvqa_current_vs_ref.py` | current textsim vs clean-room, per budget |
| `scripts/dynamic_which/check_equivalence{,_extended}.py` | keep-all == dense / α=0-mix == static (GPU, small n) |

## Inputs and outputs

- Reads: the matching **dense final** AND the matching **static final** (both required) +
  `configs/sample_ids/{dataset}.json`.
- Writes: `results/runs/{model}/{dataset}/dynamic_which_final_textsim_p{b}[...].{json,jsonl}`
  (and `dynamic_which_ref_pilot_*` for the clean-room validation).

> Result **basenames are load-bearing** — the Dynamic-COUNT COUNT-on-WHICH path resolves the frozen
> WHICH finals by name. Do not rename result files.
