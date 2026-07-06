# Dynamic-WHICH: current vs clean-room reference — failure attribution

Per (model, dataset, budget): dyn−static for the CURRENT textsim path (n=200) vs the fresh clean-room REFERENCE selectors (n=200), against the same static/dense references. **Strict n=200 on both sides** — n=20 smoke outputs are excluded (mixing them caused the earlier false IMPL-SUSPECT rows).

**Verdict key:** *IMPLEMENTATION-RELATED* = ref textsim_ref ≠ current textsim at n=200 (> 1.5pp) ⇒ a bug. *SELECTOR-DESIGN-RELATED* = implementations agree and a better selector materially lifts dyn−static. *TASK/REGIME-RELATED* = no tested selector beats static.

## A) In-ref-scope decision cells

| Model | Dataset | p | Static | Dense | cur textsim | cur best (sel) | ref textsim_ref | ref best (sel) | Impl check | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| llava15 | docvqa | 25 | 14.65 | 16.47 | -7.50 | -2.99 (textsim_cls_mix) | -7.50 | -1.33 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | docvqa | 35 | 16.20 | 16.47 | -8.19 | -3.83 (textsim_cls_mix) | -8.19 | -1.54 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | docvqa | 50 | 17.01 | 16.47 | -7.46 | -3.12 (textsim_cls_mix) | -7.46 | -1.28 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | textvqa | 25 | 36.00 | 41.50 | -8.05 | -5.50 (textsim_cls_mix) | -8.05 | -2.45 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | textvqa | 35 | 56.71 | 57.65 | -8.25 | -8.05 (textsim_cls_mix) | -8.25 | -2.95 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | textvqa | 50 | 56.40 | 57.65 | -7.55 | -4.05 (textsim_cls_mix) | -7.55 | -1.10 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | vqav2 | 25 | 74.67 | 77.33 | -3.17 | -1.34 (textsim_cls_mix) | -3.17 | -1.17 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| llava15 | vqav2 | 35 | 75.50 | 77.33 | -3.83 | -2.67 (textsim_cls_mix) | -3.83 | +1.00 (static_guarded_textsim_ref) | impl-consistent | SELECTOR-DESIGN (rescue 'static_guarded_textsim_ref' recovers, +1.00pp vs static) |
| qwen25vl7b | docvqa | 25 | 49.10 | 88.20 | -9.07 | -9.07 (textsim) | -9.07 | -7.77 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| qwen25vl7b | docvqa | 35 | 67.84 | 88.20 | -14.48 | -14.25 (textsim_norm_mix) | -14.48 | -11.38 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| qwen25vl7b | docvqa | 50 | 80.15 | 88.20 | -15.46 | -9.98 (textsim_norm_mix) | -15.46 | -8.55 (static_guarded_textsim_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| qwen25vl7b | gqa | 25 | 67.00 | 68.00 | -7.00 | +0.00 (textsim_norm_mix) | -7.00 | +0.00 (saliency_mix_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| qwen25vl7b | gqa | 35 | 68.00 | 68.00 | -7.00 | -0.50 (textsim_norm_mix) | -7.00 | -0.50 (saliency_mix_ref) | impl-consistent | TASK/REGIME (no selector beats static) |
| qwen25vl7b | vqav2 | 25 | 79.83 | 83.00 | -3.33 | +1.50 (textsim_norm_mix) | -3.33 | +1.50 (static_guarded_textsim_ref) | impl-consistent | SELECTOR-DESIGN (some selector beats static; margin small) |
| qwen25vl7b | vqav2 | 35 | 80.50 | 83.00 | -2.00 | +0.00 (textsim_norm_mix) | -2.00 | +0.00 (saliency_mix_ref) | impl-consistent | TASK/REGIME (no selector beats static) |

## B) Out-of-ref-scope cells (context only — not tested by the rescue by design)

Qwen×TextVQA is the completed FINAL success — **independent ref validation pending** (run `run_qwen_textvqa_ref_validation.sh` + `compare_qwen_textvqa_current_vs_ref.py`). Other rows here were intentionally not covered by the rescue launcher; their existing current evidence stands (NOT a pending failure).

| Model | Dataset | p | Static | Dense | cur textsim | cur best (sel) | ref textsim_ref | ref best (sel) | Impl check | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| llava15 | gqa | 25 | 66.00 | 63.50 | +1.00 | +1.00 (textsim) | — | — | n/a | not-in-ref-scope (current evidence stands; ref not run by design) |
| llava15 | gqa | 35 | 65.70 | 67.30 | +0.00 | +0.00 (textsim) | — | — | n/a | not-in-ref-scope (current evidence stands; ref not run by design) |
| qwen25vl7b | textvqa | 15 | 47.10 | 80.60 | +7.50 | — | +11.85 | +11.85 (textsim_ref) | n/a | FINAL SUCCESS; independent ref validation pending |
| qwen25vl7b | textvqa | 25 | 59.17 | 81.06 | +6.90 | +6.90 (textsim) | +6.90 | +6.90 (textsim_ref) | n/a | FINAL SUCCESS; independent ref validation pending |
| qwen25vl7b | textvqa | 35 | 63.57 | 81.06 | +8.80 | +8.80 (textsim) | +8.80 | +8.80 (textsim_ref) | n/a | FINAL SUCCESS; independent ref validation pending |
| qwen25vl7b | textvqa | 50 | 70.14 | 81.06 | +0.50 | +0.50 (textsim) | +0.50 | +0.50 (textsim_ref) | n/a | FINAL SUCCESS; independent ref validation pending |
| qwen25vl7b | textvqa | 75 | 78.43 | 81.06 | +1.37 | — | +1.65 | +1.65 (textsim_ref) | n/a | FINAL SUCCESS; independent ref validation pending |

## Rollup (in-ref-scope decision cells)

- TASK/REGIME: 13 cell-budget(s)
- SELECTOR-DESIGN: 2 cell-budget(s)

## Conclusion (ref pilots complete)

The clean-room ref rescue pilots are **complete** (45 runs, no failures). The code audit confirmed the current Dynamic-WHICH path is training-free, frozen-generation, no answer head, no old dynamic-budget code, and fairness-correct; keep-all==dense and static-criterion==static equivalence passed. With current and ref matched **strictly at n=200**, current textsim and ref textsim_ref agree. **No IMPLEMENTATION-RELATED cells remain** — the earlier IMPL-SUSPECT rows were a reporting bug (the ref branch did not filter n==200, so n=20 smoke outputs were mixed with n=200 ref outputs; fixed here). 
So the failures are **not implementation bugs**; the remaining split is SELECTOR-DESIGN vs TASK/REGIME per the table above. The one positive result (Qwen×TextVQA) still needs its own independent ref validation (pending).
