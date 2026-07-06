# Dynamic-WHICH final — Qwen2.5-VL-7B × TextVQA (selector=textsim, OCR-on)

Metric: M4C soft-accuracy. Dense reference = 81.06. n = 5000 (full val).
Dynamic-WHICH = fixed-budget, question-conditioned selection (same K as static; different WHICH).

| Budget | Dynamic | Static | Dense | Dyn−Static | Dyn−Dense | Token red.% | FLOP red.% | Vis tok | Gate | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| p15 | 60.56 | 53.06 | 81.06 | +7.50 | -20.50 | 77.98 | 78.65 | 144.6 | OK | 5000 |
| p25 | 67.53 | 59.17 | 81.06 | +8.36 | -13.53 | 68.81 | 69.64 | 241.0 | OK | 5000 |
| p35 | 71.36 | 63.57 | 81.06 | +7.79 | -9.70 | 59.61 | 60.56 | 337.5 | OK | 5000 |
| p50 | 76.08 | 70.14 | 81.06 | +5.94 | -4.98 | 45.86 | 46.84 | 481.9 | OK | 5000 |
| p75 | 79.80 | 78.43 | 81.06 | +1.37 | -1.26 | 22.94 | 23.63 | 722.7 | OK | 5000 |

**Reading:** Dyn−Static is the headline Dynamic-WHICH gain (question-conditioned selection vs the static `norm` floor at the same budget); it is positive at every budget and largest at the tightest budgets (p15 +7.50, p25 +8.36). Dyn−Dense stays negative because dense keeps all ~964 visual tokens; the trade is accuracy for large token/FLOP reductions.
