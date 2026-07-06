# Dynamic-WHICH — results

Sources: `results/final_scope/tables/dynamic_which_final_report.md`,
`dynamic_which_textsim_full_final_summary.md`, `qwen_textvqa_current_vs_ref_validation.md`. Full
manifests (GQA 12,578 · VQAv2 25,000 · TextVQA 5,000 · DocVQA 5,349), all fairness gates passed.
Win/loss vs static at the same budget: **6 wins, 2 near-ties, 32 losses**.

## The headline win: Qwen2.5-VL × TextVQA (OCR-on)

| Budget | Dynamic-WHICH | Static | Dense | **Dyn−Static** | FLOP red.% |
|--:|--:|--:|--:|--:|--:|
| p15 | 60.56 | 53.06 | 81.06 | **+7.50** | 78.65 |
| p25 | 67.53 | 59.17 | 81.06 | **+8.36** | 69.64 |
| p35 | 71.36 | 63.57 | 81.06 | **+7.79** | 60.56 |
| p50 | 76.08 | 70.14 | 81.06 | **+5.94** | 46.84 |
| p75 | 79.80 | 78.43 | 81.06 | **+1.37** | 23.63 |

The gain is largest where the budget is tightest. The clean-room re-implementation reproduces the
production textsim **200/200 predictions, score difference 0, at every budget** — the win is not an
implementation artifact.

## The rest of the matrix (best Dyn−Static per cell)

| Model | Dataset | Best Dyn−Static | Verdict |
|---|---|--:|---|
| LLaVA | GQA | −0.73 | loss (all 5 budgets) |
| LLaVA | VQAv2 | −1.59 | loss |
| LLaVA | TextVQA | −3.91 | loss |
| LLaVA | DocVQA | −3.97 | loss |
| Qwen | GQA | −0.68 | loss |
| Qwen | VQAv2 | −0.06 | near-tie at p75 (no win) |
| Qwen | TextVQA | **+8.36** | **win (all 5 budgets)** |
| Qwen | DocVQA | +3.19 | isolated p15 only — see caveat |

**Qwen DocVQA caveat:** the +3.19 at p15 is 33.09 vs static 29.90 — but dense is 94.76 and static
p75 is 93.98. The "win" lives where the static floor has already collapsed; at p25–p75 WHICH loses
by −2.98 to −13.43. It is not a usable DocVQA operating point and must not be presented as one.

## Interpretation

Question-conditioned selection wins only when the answer evidence is localized and
question-addressable (scene text in TextVQA) **and** the visual features are natively
language-aligned so the cosine signal is meaningful (Qwen's merged ViT features live in the language
embedding space; LLaVA's projector output does not). This regime-specific result motivated testing
the complementary axis — how many tokens per sample — in Dynamic-COUNT.
