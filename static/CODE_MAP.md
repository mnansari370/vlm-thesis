# Static — code map

The shared evaluation runtime currently lives under `src/final_scope/` for backward compatibility.
The method-facing entry point is this folder. Exact files implementing static pruning:

## Runner core (CPU)

| File | Role |
|---|---|
| `src/final_scope/static_eval.py` | static runner core; budget→K bookkeeping, loads the dense final as the same-sample reference, computes token/FLOP reductions and Δdense |

## Selection + generation (GPU, frozen)

| File | Role |
|---|---|
| `src/models/static/static.py` | LLaVA-1.5 engine; the `cls_attn` selector (CLS→patch attention, top-K) lives here |
| `src/pruning/question_conditioned_selection/qwen_pruner.py` | Qwen2.5-VL engine; the `norm` and `uniform` selectors live here |
| `src/pruning/static/` | `visionzip.py` — the VisionZip (dominant + contextual merge) published baseline (legacy evidence) |

## Launchers

| File | Role |
|---|---|
| `scripts/static/run_static.py` | one static cell (`--model`, `--dataset`, `--budget-pct {15,25,35,50,75}`, `--selector`) |
| `scripts/static/run_llava_static.sh` | all LLaVA-1.5 `cls_attn` static finals (GPU 0) |
| `scripts/static/run_qwen_static.sh` | all Qwen2.5-VL `norm` static finals (GPU 1) |

## Budget → K

- LLaVA-1.5 (fixed 576): K ∈ {86, 144, 202, 288, 432}.
- Qwen2.5-VL (per image): `K_i = clamp(round(pct · dense_nvis_i), 1, dense_nvis_i)`, with
  `dense_nvis_i` read from the sample's dense final (no extra encode).

## Inputs and outputs

- Reads: the matching **dense final** (required) + `configs/sample_ids/{dataset}.json`.
- Writes: `results/runs/{model}/{dataset}/static_final_{cls_attn|norm}_p{b}[...].{json,jsonl}`.

> Result **basenames are load-bearing** — Dynamic-WHICH resolves the same-budget static reference by
> reconstructing the `static_final_{sel}_p{b}[...]` name. Do not rename result files.
