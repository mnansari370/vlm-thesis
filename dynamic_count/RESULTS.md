# Dynamic-COUNT — results

Sources: `results/final_scope/tables/dynamic_count_dc_d_summary.md`, `dynamic_count_dc_c_summary.md`,
`dynamic_count_win_loss_summary.md`, `dynamic_count_oracle_summary.md`, and
`final_thesis_results_summary.md`. Evaluated on the held-out 80% split; Δ is vs the static
accuracy-vs-FLOPs curve at matched average compute (labels at ±0.50 pp).

## Oracle bound (upper bound, not a method)

A perfect per-sample budget router (peeking at correctness) would beat the best fixed budget by
**+2.09 to +6.25 pp** on every cell while removing 61–83% of visual tokens. Dynamic-COUNT is the
honest test of whether a real, input-side controller can approach this.

## DC-D (discrete cascade) — pure: 1 win, 5 near-ties, 2 losses

| Cell | Δcurve | Label |
|---|--:|---|
| LLaVA × TextVQA | **+0.75** | win |
| Qwen × VQAv2 | +0.21 | near-tie |
| LLaVA × VQAv2 | +0.06 | near-tie |
| Qwen × TextVQA | +0.02 | near-tie |
| Qwen × GQA | −0.03 | near-tie |
| LLaVA × GQA | −0.10 | near-tie |
| LLaVA × DocVQA | −0.89 | loss |
| Qwen × DocVQA | −4.61 | loss |

## DC-C (continuous integer K_i, the main method) — pure: 0 wins, 3 near-ties, 5 losses (best per cell)

| Cell | Best Δcurve | Label |
|---|--:|---|
| LLaVA × TextVQA | +0.48 | near-tie |
| Qwen × GQA | −0.19 | near-tie |
| LLaVA × VQAv2 | −0.25 | near-tie |
| LLaVA × DocVQA | −0.54 | loss |
| Qwen × VQAv2 | −0.84 | loss |
| LLaVA × GQA | −1.26 | loss |
| Qwen × TextVQA | −7.27 | loss |
| Qwen × DocVQA | −11.07 | loss |

The continuity itself worked — hundreds of distinct integer K_i per run, real second passes at exact
K_i — but the allocation could not beat the curve. The steep-curve cells (Qwen TextVQA, DocVQA) lose
heavily because double-pay punishes every wasted probe.

## COUNT-on-WHICH (Qwen × TextVQA, reported separately)

DC-D shows +6.99 vs the *static* floor, but **−0.41 vs the textsim fixed-budget curve**; DC-C is
−0.35/−0.79 vs static and −2.30/−2.03 vs the textsim curve. All the gain is WHICH's — **COUNT adds
no increment on top of WHICH**.

## Reading

The confidence signals capture 47–73% of wrong answers at 15% false-escalation, yet the oracle
headroom is not harvestable under honest double-pay accounting: flat-curve cells (GQA, VQAv2) leave
too little room above the curve, and steep-curve cells punish over-allocation. The simpler binary
DC-D matches or beats the continuous DC-C in practice. Dynamic-COUNT is reported as a documented
negative result, on all eight cells, with no cherry-picking.
