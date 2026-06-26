# CLEAN EXPERIMENT MATRIX

*Locked scope (2026-06-27). Status legend:*
- **DONE** — result exists under the final protocol (or close enough to lift directly).
- **PARTIAL** — code and/or a result exists but at non-standard budgets/baselines/n; needs re-expression to the
  15/25/35/50/75% protocol or a fair baseline.
- **TODO** — missing; must be run.
- **ARCHIVED** — an old out-of-protocol result exists, moved to `archive/` (do not cite as final).

Budget levels = {15, 25, 35, 50, 75, 100}% of dense tokens. "Dynamic WHICH" = question-conditioned selection
at fixed %. "Dynamic COUNT" = adaptive per-sample budget (to be redesigned fresh).

## Master matrix

| Model | Dataset | Dense (100%) | Static 15/25/35/50/75% | Dynamic-WHICH 15/25/35/50/75% | Dynamic-COUNT | Metric | FLOPs util |
|---|---|---|---|---|---|---|---|
| **LLaVA-1.5** | GQA | **DONE** 61.42 | **PARTIAL** (25/50/75% = 58.15/60.53/61.53; 15%,35% TODO) | **PARTIAL** (probe K64 only, **negative** vs CLS; not at grid) | **PARTIAL** (confidence cascade, ties) | exact-match | `flops.py` ✓ |
| LLaVA-1.5 | TextVQA | **DONE** 57.65 OCR / 46.73 noOCR | **PARTIAL** (25/50/75% done; 15/35% TODO) | **PARTIAL** (probe K64, **−5.58pp** vs CLS; not at grid) | **PARTIAL** (cascade, ties) | M4C soft | `flops.py` ✓ |
| LLaVA-1.5 | DocVQA | **TODO** | **TODO** | **TODO** | **TODO** | ANLS | `flops.py` ✓ |
| LLaVA-1.5 | VQAv2 | **DONE** 76.44 | **PARTIAL** (25/50/75% = 74.44/75.82/76.27; +K64/128/192; 15/35% TODO) | **TODO** (no QC-selection run; scorer code exists) | **PARTIAL** (BudgetController ties +0.05pp; **redesign**) | VQA consensus | `flops_vqav2.py` ✓ |
| **Qwen-2.5-VL-7B** | GQA | **TODO** | **TODO** | **TODO** | **TODO** | exact-match | `qwen_flops.py` (needs N_TEXT) |
| Qwen-2.5-VL-7B | TextVQA | **TODO** | **TODO** | **TODO** | **TODO** | M4C soft | `qwen_flops.py` (needs N_TEXT) |
| Qwen-2.5-VL-7B | DocVQA | **DONE** 97.19 (n=200) | **PARTIAL** (uniform/norm at K64/128/256/512; **no CLS-equiv**; not %-of-dense) | **PARTIAL** (textattn 92.18 @K128, n=200; +student; re-express as %, full-val) | **PARTIAL** (oracle +7.50pp / predictor +0.09pp; **redesign**) | ANLS | `qwen_flops.py` ✓ |
| Qwen-2.5-VL-7B | VQAv2 | **TODO** | **TODO** | **TODO** | **TODO** | VQA consensus | `qwen_flops.py` (needs N_TEXT) |

## Reading the matrix (what's actually missing)

**The LLaVA-1.5 static row is nearly done** — only the new budget points (15% = K86, 35% = K202) are missing;
25/50/75% already exist at K=144/288/432.

**Everything Qwen-on-{GQA,TextVQA,VQAv2} is TODO** — there is no Qwen harness for these three datasets yet
(the Qwen pruner only has DocVQA/ChartQA loaders). This is the single biggest build item.

**Dynamic-WHICH is the headline but thinly covered at the grid:** strong only on Qwen-DocVQA (and as a
probe-negative on LLaVA-1.5). It must be run at the 15–75% grid on the four datasets for both models, with a
**fair blind baseline** (CLS-attention on Qwen, not just uniform/norm).

**Dynamic-COUNT is intentionally PARTIAL everywhere** — the old attempts (LLaVA-1.5 BudgetController; Qwen
oracle+predictor) are kept as evidence/methodology, but the family is being **redesigned fresh**, so no cell
is "DONE" for the new mechanism.

**LLaVA-1.5 × DocVQA is fully TODO** — DocVQA was only ever run on high-res models. Expect weak absolute
numbers (LLaVA-1.5 can't read dense documents at 576 tokens), which is itself an informative resolution×task
data point — but it must be run to fill the cell.

## Count of cells by status
| Status | Count (of 28 method-cells, excl. metric/FLOPs cols) |
|---|---|
| DONE | 4 (LLaVA-1.5 dense×{GQA,TextVQA,VQAv2}; Qwen dense×DocVQA) |
| PARTIAL | 9 |
| TODO | 15 |
| ARCHIVED-only | static/QC at out-of-protocol budgets, LLaVA-1.6/Qwen-3B/32B (all in `archive/`) |

See `docs/TODO_NEXT_RUNS.md` for the exact run order and `docs/FINAL_CONFIG_INVENTORY.md` for which configs
exist vs must be created.
