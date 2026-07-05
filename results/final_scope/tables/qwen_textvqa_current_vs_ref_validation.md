# Qwen2.5-VL × TextVQA — current textsim vs clean-room textsim_ref (n=200 validation)

Independent validation of the headline Dynamic-WHICH win. Comparison set = first 200 manifest IDs (the ref n=200 sample set). p25/p35/p50 compare against the current n=200 pilot; p15/p75 against the first 200 rows of the current full final. All scores recomputed over the same 200.

| p | current source | same IDs | same order | pred match | score diff | cur score | ref score | cur dyn−static | ref dyn−static | static | dense | cur vis | ref vis | meta |
|--:|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| p15 | final_first200 | True | True | 200/200 | 0 | 58.95 | 58.95 | +11.85 | +11.85 | 47.1 | 80.6 | 146.21 | 146.21 | none |
| p25 | n200_pilot | True | True | 200/200 | 0 | 66.25 | 66.25 | +6.90 | +6.90 | 59.35 | 80.6 | 243.64 | 243.64 | none |
| p35 | n200_pilot | True | True | 200/200 | 0 | 70.45 | 70.45 | +8.80 | +8.80 | 61.65 | 80.6 | 341.25 | 341.25 | none |
| p50 | n200_pilot | True | True | 200/200 | 0 | 73.25 | 73.25 | +0.50 | +0.50 | 72.75 | 80.6 | 487.31 | 487.31 | none |
| p75 | final_first200 | True | True | 200/200 | 0 | 77.75 | 77.75 | +1.65 | +1.65 | 76.1 | 80.6 | 730.7 | 730.7 | none |

## Interpretation

The headline claim is validated when, for every budget, the clean-room `textsim_ref` reproduces the current `textsim` predictions (pred match = n, 0 score diffs) and the ref dyn−static ≈ current dyn−static. **All compared budgets match exactly → the Qwen×TextVQA win is implementation-robust.**
