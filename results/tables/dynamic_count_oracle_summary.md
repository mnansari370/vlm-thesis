# Dynamic-COUNT ORACLE — adaptive-budget upper bound (NOT a deployable method)

> **ORACLE / UPPER BOUND.** For each sample this peeks at the per-sample correctness (`per_sample_score` vs gold) to pick a budget from {15,25,35,50,75}% while reusing the EXISTING static selectors (LLaVA→`cls_attn`, Qwen→`norm`). It is **not runnable at inference** and is **not** a claimed controller — it bounds what a perfect per-sample budget router could achieve. No training, no new labels, no GPU.

Two policies:
- **Best-score oracle** — per sample, max static score over the 5 budgets, charging the *smallest* budget that attains it. Upper bound on adaptive-budget accuracy + its token cost.
- **Match-dense oracle** — per sample, the *smallest* budget whose static score ≥ dense (else fall back to dense). A 'no worse than dense per sample' router and its token cost.

Reductions are **visual-token** reductions vs dense (mean kept visual tokens / dense mean).

## Best-score oracle

| Model | Dataset | Metric | Dense | Best fixed (budget) | **Oracle** | Oracle−Dense | Oracle−BestFixed | Oracle vis tok | Dense vis tok | Vis red.% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava15 | gqa | gqa_exact | 61.42 | 61.53 (p75) | **67.78** | +6.36 | +6.25 | 102 | 576 | 82.3 |
| llava15 | vqav2 | vqa_consensus | 77.33 | 77.31 (p75) | **81.46** | +4.13 | +4.15 | 100 | 576 | 82.6 |
| llava15 | textvqa | m4c_soft | 57.65 | 57.39 (p75) | **62.52** | +4.87 | +5.13 | 98 | 576 | 83.0 |
| llava15 | docvqa | anls | 21.53 | 21.36 (p75) | **26.14** | +4.61 | +4.79 | 109 | 576 | 81.1 |
| qwen25vl7b | gqa | gqa_exact | 60.96 | 60.85 (p75) | **66.41** | +5.45 | +5.56 | 62 | 359 | 82.7 |
| qwen25vl7b | vqav2 | vqa_consensus | 84.27 | 83.66 (p75) | **87.00** | +2.73 | +3.35 | 67 | 359 | 81.3 |
| qwen25vl7b | textvqa | m4c_soft | 81.06 | 78.43 (p75) | **83.36** | +2.29 | +4.93 | 251 | 964 | 74.0 |
| qwen25vl7b | docvqa | anls | 94.76 | 93.98 (p75) | **96.07** | +1.31 | +2.09 | 473 | 1229 | 61.5 |

### Per-budget static accuracy + where the oracle 'solves' each sample first

| Model | Dataset | s@p15 | s@p25 | s@p35 | s@p50 | s@p75 | first@p15 | first@p25 | first@p35 | first@p50 | first@p75 | unsolved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llava15 | gqa | 56.1 | 58.1 | 59.4 | 60.5 | 61.5 | 56.1 | 5.4 | 2.6 | 2.1 | 1.6 | 32.2 |
| llava15 | vqav2 | 73.4 | 75.5 | 76.1 | 76.9 | 77.3 | 77.0 | 5.3 | 2.0 | 1.8 | 1.5 | 12.5 |
| llava15 | textvqa | 56.0 | 56.0 | 56.7 | 56.4 | 57.4 | 59.9 | 3.5 | 1.8 | 1.2 | 1.6 | 32.0 |
| llava15 | docvqa | 17.7 | 19.3 | 20.1 | 20.8 | 21.4 | 19.5 | 5.7 | 3.4 | 3.1 | 2.7 | 65.5 |
| qwen25vl7b | gqa | 56.0 | 58.8 | 60.0 | 60.5 | 60.9 | 56.0 | 5.4 | 2.2 | 1.6 | 1.1 | 33.6 |
| qwen25vl7b | vqav2 | 76.2 | 78.9 | 80.5 | 82.1 | 83.7 | 78.6 | 5.6 | 3.0 | 2.7 | 2.6 | 7.4 |
| qwen25vl7b | textvqa | 53.1 | 59.2 | 63.6 | 70.1 | 78.4 | 53.3 | 10.3 | 6.8 | 8.1 | 9.7 | 11.9 |
| qwen25vl7b | docvqa | 29.9 | 49.5 | 68.9 | 84.8 | 94.0 | 17.0 | 19.7 | 22.8 | 23.1 | 14.8 | 2.6 |

*first@pX = % of samples whose cheapest max-scoring static budget is pX; unsolved = % where no budget scored > 0. Reading the distribution: mass at p15 means many samples are already solved at the tightest budget (adaptive routing can be cheap); mass at p75/unsolved means the task needs many tokens regardless.*

## Match-dense oracle (cheapest per-sample budget that is ≥ dense)

| Model | Dataset | Acc (≥dense) | Mean vis tok | Vis red.% vs dense | needs-dense% |
|---|---|---:|---:|---:|---:|
| llava15 | gqa | 66.12 | 103 | 82.2 | 0.9 |
| llava15 | vqav2 | 80.50 | 101 | 82.5 | 0.8 |
| llava15 | textvqa | 61.45 | 100 | 82.7 | 1.1 |
| llava15 | docvqa | 24.18 | 108 | 81.3 | 1.4 |
| qwen25vl7b | gqa | 65.11 | 62 | 82.7 | 0.5 |
| qwen25vl7b | vqav2 | 86.84 | 69 | 80.7 | 1.3 |
| qwen25vl7b | textvqa | 84.08 | 272 | 71.7 | 3.9 |
| qwen25vl7b | docvqa | 95.53 | 480 | 60.9 | 2.2 |

*needs-dense% = share of samples where NO static budget matches the dense score (must keep dense). A low needs-dense% with a high vis-reduction% means most samples are dense-lossless at a small budget — the strongest motivation for Dynamic-COUNT.*

## How to read this for the thesis

1. **Oracle−BestFixed** is the accuracy headroom a perfect budget router could add over the single best static budget. If it is large, adaptive *count* is worth pursuing; if small, one fixed budget is already near-optimal and the interesting axis stays *which* tokens.
2. **Match-dense vis-reduction% at low needs-dense%** quantifies free compute: how many visual tokens could be dropped, per sample, with no accuracy loss vs dense — the clean, honest motivation for Dynamic-COUNT.
3. These are **ceilings**. A real controller must predict the budget from the input alone (image/question), with NO access to correctness — so realized gains will be strictly lower. This script exists to decide whether that gap is worth closing, not to claim it.
