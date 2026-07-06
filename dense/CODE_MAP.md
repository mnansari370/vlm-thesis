# Dense — code map

The shared evaluation core lives under `src/common/`; the method runner cores live under
`src/dense/`, `src/static/`, `src/dynamic_which/`, and `src/dynamic_count/`. The method-facing entry
point is this folder. The exact files that implement the dense baseline:

## Runner core (CPU — data, scoring, schema, FLOPs)

| File | Role |
|---|---|
| `src/dense/evaluate_dense.py` | dense runner core; also defines the per-dataset adapters, prompts, and scorers that every other method reuses |

Supporting core modules (shared by all methods): `src/common/sample_ids.py` (locked manifests),
`output_writer.py` (unified schema), `schema_validator.py` (the fairness gate), `token_flops.py`
(per-sample-then-averaged prefill FLOPs).

## Generation engines (GPU, frozen)

| File | Role |
|---|---|
| `src/models/static/static.py` | LLaVA-1.5 engine; dense uses `method="none"`, K=576 |
| `src/pruning/question_conditioned_selection/qwen_pruner.py` | Qwen2.5-VL engine; dense uses `selector="full"` (validated equal to stock generation) |

## Launchers

| File | Role |
|---|---|
| `scripts/dense/run_dense.py` | one dense cell (`--model`, `--dataset`, `--n 200` pilot or `--full`) |
| `scripts/dense/run_llava_dense.sh` | all LLaVA-1.5 dense finals (GPU 0, vlm_env) |
| `scripts/dense/run_qwen_dense.sh` | all Qwen2.5-VL dense finals (GPU 1, qwen_env) |
| `scripts/data/build_sample_manifests.py` | builds the sha256-locked sample manifests read by every run (one-time setup) |

## Inputs and outputs

- Sample manifests (read): `configs/sample_ids/{dataset}.json`
- Results (written): `results/runs/{model}/{dataset}/dense_final[...].{json,jsonl}`

> The result **basenames are load-bearing** — static/WHICH/COUNT resolve the dense reference by
> reconstructing the `dense_final[...]` name. Do not rename result files.
