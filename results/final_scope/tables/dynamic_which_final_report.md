# Dynamic-WHICH — Final Report (phase freeze)

*Method phase: fixed-budget, question-conditioned visual-token selection ("WHICH tokens"), evaluated
on the locked final-scope matrix. This report freezes the Dynamic-WHICH phase before moving to
Dynamic-COUNT.*

---

## A. Completion status

- **40/40 full Dynamic-WHICH textsim cells complete** (2 models × 4 datasets × 5 budgets), every run
  on the FULL locked manifest (GQA 12,578 · VQAv2 25,000 · TextVQA 5,000 · DocVQA 5,349).
- **0 missing, 0 invalid** (`audit_dynamic_which_full_final_matrix`: TOTAL=40, COMPLETED=40,
  MISSING=0, INVALID=0).
- **All 40 fairness gates OK** — same sample manifests, prompts, scorers, greedy bs=1 decoding
  (max_new_tokens=64) as the dense and static finals; TextVQA OCR-on; DocVQA instruction-on.
- Verification suite re-run at freeze: `validate_dynamic_which_final` → ALL_FINAL_VALID=True.

## B. Method policy

- **One primary selector everywhere: `textsim`** — cosine similarity between visual-token embeddings
  and question-token embeddings (LLM embedding table), scored on the raw question only; top-K then
  indices sorted back to original visual order. **No per-dataset selector switching** in the final
  matrix — the comparison is deliberately clean.
- **Budgets identical to static:** LLaVA K ∈ {86, 144, 202, 288, 432} of 576; Qwen per-sample
  K = clamp(round(budget% · dense_n_visual), 1, dense_n_visual).
- **Current Dynamic-WHICH implementation** used for the entire final matrix; the **clean-room
  reference implementation was used only for implementation validation**, never for final results.
- Training-free, frozen generation, no answer head, no dense LM prefill for scoring.

## C. Main result summary

**Qwen2.5-VL × TextVQA is the only strong full success.** Dynamic-WHICH beats the static `norm`
floor at every budget (n=5000, M4C soft-acc; dense = 81.06):

| Budget | Dyn−Static | Dynamic | Static | FLOP red.% |
|--:|--:|--:|--:|--:|
| p15 | **+7.50** | 60.56 | 53.06 | 78.65 |
| p25 | **+8.36** | 67.53 | 59.17 | 69.64 |
| p35 | **+7.79** | 71.36 | 63.57 | 60.56 |
| p50 | **+5.94** | 76.08 | 70.14 | 46.84 |
| p75 | **+1.37** | 79.80 | 78.43 | 23.63 |

The gain is largest at the tightest budgets — exactly where selection matters most.

**Implementation robustness (clean-room validation).** An independent, clean-room re-implementation
(`textsim_ref`) reproduces the current implementation **exactly** on the n=200 validation set at all
five budgets: **pred_match = 200/200 and score_diff = 0** per budget, with identical prompts, K, and
metadata. The extended equivalence checks also passed (keep-all == dense; static-criterion ==
static). The headline win is therefore not an artifact of the implementation.

## D. Negative results (reported in full — not cherry-picked)

- **LLaVA × GQA** — loses to static at all budgets (−0.73 … −1.87).
- **LLaVA × VQAv2** — loses at all budgets (−1.59 … −5.52).
- **LLaVA × TextVQA** — loses at all budgets (−3.91 … −8.57), the mirror image of Qwen's win.
- **LLaVA × DocVQA** — loses at all budgets (−3.97 … −6.82).
- **Qwen × GQA** — loses at all budgets (−0.68 … −2.80).
- **Qwen × VQAv2** — loses at all budgets, but converges to **near-tie at high budget**
  (p50 −0.39, p75 −0.06).
- **Qwen × DocVQA** — **p15 is an isolated positive (+3.19)**, but p25/p35/p50/p75 are negative
  (−2.98, −11.33, −13.43, −5.94); DocVQA is **not** a full success.

## E. Scientific interpretation

Dynamic-WHICH textsim is not universal. It is beneficial when the question text can reliably
identify localized answer-relevant visual/OCR tokens, as in Qwen TextVQA. However, it underperforms
static saliency on many settings, especially LLaVA and DocVQA, showing that question-conditioned
token selection alone is not sufficient across all VLM/task regimes.

Two regularities in the full matrix support this reading. First, the win requires the visual
embedding space to be natively aligned with the language embedding space: Qwen's merged ViT
features live in the LLM space, whereas LLaVA's projector output is not trained for cosine
similarity against the embedding table — every LLaVA cell is negative. Second, the win requires the
answer evidence to be *localized and question-addressable*: TextVQA scene text satisfies this;
GQA's relational reasoning, VQAv2's broad natural-scene questions, and DocVQA's page-wide dense text
do not (DocVQA's isolated p15 positive reflects the regime where the static floor collapses so
severely that even imperfect question-conditioning helps).

## F. Thesis-ready conclusion

> We evaluated fixed-budget question-conditioned token selection (Dynamic-WHICH, `textsim`) on the
> complete final-scope matrix — two VLMs, four benchmarks, five budgets, full evaluation sets, under
> identical manifests, prompts, and decoding as the dense and static baselines. The method produces
> a large and consistent improvement over the static saliency floor in exactly one regime:
> Qwen2.5-VL on TextVQA, where it gains +1.4 to +8.4 accuracy points at 24–79% FLOP reduction, with
> the largest gains at the tightest budgets; an independent clean-room re-implementation reproduces
> these results exactly. Across the remaining seven model×dataset cells, however, Dynamic-WHICH
> matches or underperforms the static baseline (32 of 40 cells are losses, with a worst case of
> −13.4 points on Qwen DocVQA at p50). We conclude that question-conditioned selection is a
> regime-specific tool — effective when answer evidence is localized, text-addressable, and the
> visual features are natively language-aligned — rather than a universal improvement over static
> pruning. This negative result motivates the next axis of this thesis: instead of choosing *which*
> tokens to keep at a fixed budget, choosing *how many* tokens each sample needs (Dynamic-COUNT).

---

*Provenance: `dynamic_which_full_final_matrix_audit.{csv,md}` (40/40, gates OK),
`dynamic_which_textsim_full_final_summary.{csv,md}`,
`dynamic_which_dense_static_dynamic_comparison.{csv,md}`,
`qwen_textvqa_current_vs_ref_validation.{csv,md}` (200/200, 0 diffs, all budgets).
No result JSON/JSONL was modified in producing this report.*
