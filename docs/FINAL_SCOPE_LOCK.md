# FINAL SCOPE LOCK

*Locked 2026-06-27. This document fixes the thesis scope. Anything outside it is **archived, not deleted**
(see `archive/migration_manifests/archive_manifest_20260627.md`). Supersedes the broader scope discussed in
`THESIS_MASTER_PLAN.md` for the purpose of the final experiment matrix.*

## Thesis sentence
> **This thesis decomposes VLM visual-token pruning into token selection and token budgeting, showing that
> question-conditioned selection is the useful lever while per-sample budgeting must be tested carefully
> against matched FLOPs.**

## 1. Final models (2)
| Model | ID | Visual tokens | Role |
|---|---|---|---|
| **LLaVA-1.5-7B** | `llava-hf/llava-1.5-7b-hf` | 576 (fixed, @336px) | frozen low-res model |
| **Qwen-2.5-VL-7B** | `Qwen/Qwen2.5-VL-7B-Instruct` | native dynamic (~hundreds–1286) | frozen modern model |

Both **frozen** (no backbone training in scope). All other backbones (LLaVA-1.6, Qwen-3B/32B, elastic LoRA)
are archived.

## 2. Final datasets (4)
GQA · TextVQA · DocVQA · VQAv2. (POPE, ScienceQA, ChartQA, InfoVQA, LLaVA-665K mix are archived.)

## 3. Final method families (4)
1. **Dense baseline** — keep all visual tokens (ceiling + reproduction anchor).
2. **Static pruning** — question-independent fixed-budget selection (CLS-attention saliency / blind floors).
3. **Dynamic WHICH** — question-conditioned token selection at a **fixed token percentage** (the question
   chooses *which* tokens; the count is fixed per budget level).
4. **Dynamic COUNT** — adaptive per-sample token budget. **To be redesigned fresh** (the old BudgetController
   and Qwen budget-oracle/predictor are kept as evidence/methodology, but the new mechanism is new code).

## 4. Static / fixed budget levels (% of dense tokens)
**15%, 25%, 35%, 50%, 75%, 100%** of dense.

Token mapping for LLaVA-1.5 (576 dense): 15%→**86**, 25%→**144**, 35%→**202**, 50%→**288**, 75%→**432**,
100%→**576**. (Existing static runs at K=144/288/432 already equal 25/50/75%; K=86 and K=202 are new.)
For Qwen-2.5-VL the dense token count varies per image, so the % budget is computed **per sample** from that
sample's dense token count (K_sample = round(pct × dense_tokens_sample)); existing Qwen runs used fixed K
(64/128/256/512) and must be re-expressed as %-of-dense.

## 5. What is archived (preserved, not deleted)
- LLaVA-1.6 / high-res bridge code + results
- Qwen-2.5-VL-3B and 32B scripts + results
- Elastic LoRA / Stage-1 (code, config, checkpoints, the 665K mix loader)
- POPE, ScienceQA evaluation code; ChartQA/InfoVQA are scorer-branch-only (no standalone code archived)
- Retired `data/budget_oracle/` diagnostics
Full list + restore paths: `archive/migration_manifests/archive_manifest_20260627.md`.

## 6. What is explicitly NOT deleted
**Nothing was deleted.** Archived items live on disk under `archive/` (git-ignored). The `results/thesis_main/*`
evidence trees and `data/{vqav2,gqa,textvqa}` stay in place; `data/llava_mix/` (34G) stays for now (delete
only after submission). DocVQA loads from the HF hub at run time.

## 7. Shared code held back from archiving (clean during the method rewrite, not now)
`src/models/{static,dynamic_budget}`, the Qwen budget files, `src/metrics/{chartqa,pope}_score.py`,
`src/analysis/flops.py`, `cascade_sweep.py` — each mixes in-scope and out-of-scope use; see manifest §C.
