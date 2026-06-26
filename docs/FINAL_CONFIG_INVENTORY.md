# FINAL CONFIG INVENTORY

*Locked scope (2026-06-27). Inventory only — **no configs were rewritten**. Maps each final-scope
model×dataset×method×budget to an existing config (reusable?) or a needed new one. Budget levels =
{15,25,35,50,75,100}% of dense. LLaVA-1.5 K-map: 86/144/202/288/432/576.*

## A. LLaVA-1.5 configs (exist; YAML-driven)

| Method | Existing config(s) | Reusable? | Needs new? | Missing params | Output folder |
|---|---|---|---|---|---|
| Dense | `configs/dense/llava_dense_150k_10k_fullvocab.yaml` (+443k) | **Yes** (generation path) | no | `answer_vocab_full.json` referenced but **missing on disk** (only matters for the retired classification head, not generation) | `results/final_scope/llava15/<dataset>/dense_100.json` |
| Static 25/50/75% | `configs/static/llava_static_clsattn_150k_10k_fullvocab_k{144,288,432}.yaml` | **Yes** | no | — | `.../static_{25,50,75}.json` |
| Static 15% (K=86) | — (closest: `_k96.yaml` ≈17%) | partial | **Yes** (`..._k86.yaml`) | `token_selection.keep_tokens: 86` | `.../static_15.json` |
| Static 35% (K=202) | — (closest: `_k192.yaml` ≈33%, `_k219.yaml` ≈38%) | partial | **Yes** (`..._k202.yaml`) | `token_selection.keep_tokens: 202` | `.../static_35.json` |
| Dynamic WHICH (QC selection, fixed %) | none as a config (the QC probes are argparse scripts: `run_qcond_probe.py`, `run_clip_probe.py`) | partial | **Yes** (or keep argparse) | a QC-selector + `keep_tokens` per % level; decide LM-attn teacher vs CLIP-space vs learned scorer | `.../which_{15..75}.json` |
| Dynamic COUNT | `configs/dynamic_budget/llava_dynamic_150k_10k_fullvocab.yaml` (+gate_2k, gate_smoke) | **as reference** | **Yes (redesign)** | new mechanism TBD | `.../count.json` |

> Note: the LLaVA-1.5 static/dense configs are wired to **VQAv2** (`dataset.name: vqav2`). Running them on
> **GQA/TextVQA/DocVQA** uses the argparse harnesses (`run_static_testdev.py`, `run_textvqa.py`, etc.), which
> take `--keep_k` directly and do **not** read these YAMLs. So for GQA/TextVQA the "config" is the CLI flag,
> not a YAML — the YAMLs matter mainly for the VQAv2 cached/classification path and the dynamic trainer.

## B. Qwen-2.5-VL-7B configs (**none exist** — argparse-driven, new infra needed)

| Method | Existing | Reusable? | Needs new? | Missing | Output folder |
|---|---|---|---|---|---|
| Dense | `src/evaluation/docvqa/qwen25_dense_eval.py` (DocVQA/ChartQA only) | partial | **Yes** | GQA/TextVQA/VQAv2 `load_bench` branches + scorers | `results/final_scope/qwen25vl7b/<dataset>/dense_100.json` |
| Static (blind) | `qwen_pruner.py` selectors `uniform`/`norm` | partial | **Yes** | **fair CLS-equivalent baseline** (ViT/merger attention); **%-of-dense** budgeting (per-sample K) | `.../static_{15..75}.json` |
| Dynamic WHICH | `qwen_pruner.py` selector `textattn` (L16) | **Yes** (DocVQA) | extend | %-of-dense; GQA/TextVQA/VQAv2 adapters; full-val | `.../which_{15..75}.json` |
| Dynamic COUNT | `qwen_oracle*.py` + `qwen_budget_eval.py` (DocVQA) | **as methodology** | **Yes (redesign)** | new mechanism; GQA/TextVQA/VQAv2 adapters | `.../count.json` |

## C. New configs to create (place under `configs/final_scope/`)
1. `configs/final_scope/llava15/static_k86.yaml`, `static_k202.yaml` (15%, 35% — copy `_k144.yaml`, change `keep_tokens`).
2. `configs/final_scope/llava15/which_<sel>_k{86,144,202,288,432}.yaml` — once the QC selector is chosen.
3. `configs/final_scope/qwen25vl7b/*.yaml` — only if you convert the Qwen argparse scripts to YAML; otherwise
   record the exact CLI per cell in the run script under `scripts/final_scope/`.

## D. Cross-cutting config gaps
- **FLOPs `N_TEXT`:** `qwen_flops.py` hardcodes `N_TEXT=40` (DocVQA estimate). Measure and set per-dataset
  values for GQA/TextVQA/VQAv2 before quoting Qwen FLOPs there.
- **`answer_vocab_full.json` missing:** referenced by every LLaVA-1.5 dense/static/dynamic YAML; only needed
  for the retired classification head. Regenerate via `scripts/data/build_answer_vocab_full.py` only if you
  revive that path (the generation protocol does not need it).
- **Budget %→K rounding:** define one rule (`K = round(pct × dense_tokens)`, clamp ≥1) and put it in the new
  run scripts so LLaVA-1.5 (fixed 576) and Qwen (per-sample) stay consistent.
