# Dynamic-WHICH — code map

The shared evaluation runtime currently lives under `src/final_scope/` for backward compatibility.
The method-facing entry point is this folder. Exact files implementing Dynamic-WHICH:

## Runner core (CPU)

| File | Role |
|---|---|
| `src/final_scope/dynamic_which_eval.py` | WHICH runner core; same budget contract as static, loads the dense final AND the same-budget static final so it can report Dyn−Dense and Dyn−Static over the same samples |

## Selectors (GPU)

| File | Role |
|---|---|
| `src/pruning/dynamic_which/llava_textsim.py` | LLaVA-1.5 `textsim` (+ `textsim_cls_mix`) selector; composes the frozen LLaVA engine |
| `src/pruning/dynamic_which/qwen_textsim.py` | Qwen2.5-VL `textsim` (+ `textsim_norm_mix`) selector; composes the frozen Qwen engine |
| `src/pruning/dynamic_which_ref/` | independent clean-room re-implementation (`llava_ref.py`, `qwen_ref.py`, pure helpers) used only to validate the selectors above |

## Launchers and validation

| File | Role |
|---|---|
| `scripts/final_scope/run_dynamic_which_eval.py` | one WHICH cell (`--selector textsim`, `--budget-pct`) |
| `scripts/final_scope/run_dynamic_which_ref_eval.py` | clean-room reference pilots |
| `scripts/final_scope/run_dynamic_which_textsim_full_missing.sh` | the 35 full finals (Qwen×TextVQA finals already existed) |
| `scripts/final_scope/validate_dynamic_which_final.py` | validates the five Qwen×TextVQA WHICH finals |
| `scripts/final_scope/audit_dynamic_which_full_final_matrix.py` | audits all 40 WHICH cells for completeness |
| `scripts/final_scope/compare_qwen_textvqa_current_vs_ref.py` | current textsim vs clean-room, per budget |
| `scripts/final_scope/check_dynamic_which_equivalence{,_extended}.py` | keep-all == dense / α=0-mix == static (GPU, small n) |

## Inputs and outputs

- Reads: the matching **dense final** AND the matching **static final** (both required) +
  `configs/final_scope/sample_ids/{dataset}.json`.
- Writes: `results/final_scope/{model}/{dataset}/dynamic_which_final_textsim_p{b}[...].{json,jsonl}`
  (and `dynamic_which_ref_pilot_*` for the clean-room validation).

> Result **basenames are load-bearing** — the Dynamic-COUNT COUNT-on-WHICH path resolves the frozen
> WHICH finals by name. Do not rename result files.
