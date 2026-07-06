# Dynamic-WHICH (question-conditioned selection)

## What the method does

Dynamic-WHICH holds the token budget **identical to static** and changes only *which* tokens
survive, based on the question. The `textsim` selector scores every visual token by its maximum
cosine similarity to the question-token embeddings, keeps the top-K, then restores the original
spatial order. It is training-free, needs no answer head, and performs no extra language-model
forward pass for scoring — so any accuracy difference from static at the same budget is attributable
purely to the selection.

## Why it exists in the thesis

This is the *which tokens* axis of the decomposition. The headline metric is **Dyn−Static**:
question-conditioned selection minus the image-only floor at the same per-sample budget. A positive
value means the question information paid off.

## What code implements it

The method-facing entry point is this folder; exact files are in [CODE_MAP.md](CODE_MAP.md). The
selectors compose the two frozen engines (no engine edits), and an independent clean-room
re-implementation exists purely to validate them.

## What scripts run it

- Method-facing wrapper (safe, CPU): [`scripts/validate_dynamic_which.sh`](scripts/validate_dynamic_which.sh)
- Full commands: [COMMANDS.md](COMMANDS.md)

## Where results are stored

Per cell: `results/runs/{model}/{dataset}/dynamic_which_final_textsim_p{b}[...].{json,jsonl}`.
Committed reports: `results/tables/dynamic_which_final_report.md`,
`dynamic_which_textsim_full_final_summary.md`, `qwen_textvqa_current_vs_ref_validation.md`;
reproduced in [RESULTS.md](RESULTS.md).

## Final conclusion

Dynamic-WHICH is a **regime-specific** tool, not a universal improvement. It wins decisively on
**Qwen2.5-VL × TextVQA** (+7.50 to +8.36 pp over static at tight budgets, largest where the budget is
tightest), reproduced prediction-for-prediction by the clean-room implementation. Across the full
40-cell matrix it loses on 32 cells: all LLaVA cells (its projector features are not language-aligned
for the cosine signal), Qwen GQA and VQAv2 (broad-coverage tasks), and Qwen DocVQA at every budget
except a collapsed-floor corner at p15. The win requires both localized, question-addressable
evidence and natively language-aligned visual features.
