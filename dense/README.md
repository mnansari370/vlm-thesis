# Dense baseline

## What the method does

The dense baseline keeps **all** visual tokens — no pruning. Each image is encoded to its full
visual-token sequence (LLaVA-1.5: a fixed 576 tokens; Qwen2.5-VL: the native per-image count), and
the model answers under the locked evaluation protocol (greedy decoding, batch size 1,
`max_new_tokens=64`, one instruction string, one canonical scorer per dataset).

## Why it exists in the thesis

Dense is the reference point for everything else. It serves three roles:

1. **Accuracy ceiling** — every pruning result is a trade against the dense score.
2. **Reproduction anchor** — the harness is only trustworthy because dense reproduces the published
   references (LLaVA GQA 61.42 vs published 62.0; TextVQA-OCR 57.65 vs 58.2).
3. **Per-sample reference** — the dense output records each sample's native visual-token count
   (which defines the Qwen per-sample budgets), its text-token count, and the FLOPs denominator that
   every reduction number divides by. Static, Dynamic-WHICH, and Dynamic-COUNT all load the matching
   dense result before running.

## What code implements it

The method-facing entry point is this folder. The tested implementation lives in the shared
evaluation core — see [CODE_MAP.md](CODE_MAP.md) for exact paths. In short: a CPU runner core that
handles data, scoring, schema, and FLOPs, plus the two frozen generation engines (LLaVA and Qwen)
that produce the answers.

## What scripts run it

- Method-facing wrapper (safe, CPU): [`scripts/validate_dense.sh`](scripts/validate_dense.sh)
- Full commands (CPU validation, tables, GPU rerun): [COMMANDS.md](COMMANDS.md)

## Where results are stored

Per cell: `results/runs/{llava15,qwen25vl7b}/{gqa,vqav2,textvqa,docvqa}/dense_final[...].{json,jsonl}`
(git-ignored local evidence). The committed summary is
`results/tables/final_dense_static_dynamic_comparison.md` (section A), reproduced in
[RESULTS.md](RESULTS.md).

## Final conclusion

Dense is the measurement standard, not the practical operating point. On GQA and VQAv2 a 75% static
budget matches dense within about one accuracy point at roughly a quarter of the FLOPs, so the
interesting question is never "dense vs pruned" but which pruning decision — *which* tokens or *how
many* — buys accuracy back at a given cost. That question is answered by the other three method
folders.
