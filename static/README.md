# Static pruning

## What the method does

Static pruning selects visual tokens **from the image alone** — the question is never consulted — at
a fixed budget, then feeds the shortened sequence to the language model (prune-before-LLM, so the
FLOP savings are real). Budgets are 15/25/35/50/75 % of each sample's dense visual-token count.
Selectors: LLaVA-1.5 ranks patches by CLS-attention in the vision encoder (`cls_attn`); Qwen2.5-VL
ranks merged visual embeddings by activation norm (`norm`).

## Why it exists in the thesis

Static is the honest floor. A question-aware method has no claim unless it beats image-only
selection **at the same budget**. Because the static accuracy-vs-FLOPs curve is strong, it is the
comparison target for both dynamic methods — a much harder baseline than the weak uniform floors
common in the pruning literature.

## What code implements it

The method-facing entry point is this folder; the tested implementation is in the shared evaluation
core and the frozen engines — see [CODE_MAP.md](CODE_MAP.md). The runner core handles the budget→K
bookkeeping and the same-sample deltas against the dense reference; the selection happens inside the
two frozen engines.

## What scripts run it

- Method-facing wrapper (safe, CPU): [`scripts/validate_static.sh`](scripts/validate_static.sh)
- Full commands: [COMMANDS.md](COMMANDS.md)

## Where results are stored

Per cell: `results/runs/{model}/{dataset}/static_final_{cls_attn|norm}_p{b}[...].{json,jsonl}`.
Committed summaries: `results/tables/final_dense_static_dynamic_comparison.md` (section B)
and `static_final_summary.csv`; reproduced in [RESULTS.md](RESULTS.md).

## Final conclusion

Static pruning is a strong floor, often near-dense at the 75% budget (within 0.02–2.63 pp of dense on
every cell, ~23–25% FLOP reduction). It degrades gracefully on scene-centric tasks (GQA, VQAv2) and
catastrophically where information is dense and spatially distributed (Qwen TextVQA falls
81.06 → 53.06 and Qwen DocVQA 94.76 → 29.90 at the tightest budget). How steep each curve is turns
out to govern whether adaptive budgeting (Dynamic-COUNT) can help at all, and where question
conditioning (Dynamic-WHICH) has room to win.
