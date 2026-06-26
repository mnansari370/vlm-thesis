# Where Does Visual-Token Pruning Actually Help? — Path-A Findings

**Frozen-model study (no fine-tuning). LLaVA-1.5 (576 tok) vs LLaVA-1.6 / AnyRes (~2302 tok),
Vicuna-7B. Diagnostics on n=300 subsets; generation protocol; official scorers.**

This is the rigorous, honest backbone for the analysis thesis/paper — a continuation of v1
("Are We Solving the Right Problem?"). The aim is not to win a SOTA bake-off (the high-res
question-conditioned pruning field is mature: FEATHER, CDPruner, HiRED, SparseVLM), but to
characterize *where, why, and how much* pruning truly helps — and to supply the methodological
tools (matched-FLOPs frontier, oracle-noise decomposition) the field often skips.

---

## Finding 1 — Pruning's room is set by resolution × task token-demand (not universal)

Pruning cost at K=64 tokens, blind CLS-attn, n=300, **retention = % of each model's own dense**
(same 300 samples / metric / selector across backbones; LLaVA-1.5 via ElasticPrunedLlava,
LLaVA-1.6 via HighResPruner; GQA = strict exact-match, TextVQA/DocVQA = soft/ANLS):

| benchmark (type) | LLaVA-1.5 (576 dense) → K=64 | LLaVA-1.6 (2302 dense) → K=64 |
|---|---|---|
| TextVQA-noOCR (**reading**) | 42.2 → 43.0 = **102%** (flat) | 61.4 → 34.8 = **57%** |
| GQA (**reasoning**) | 58.7 → 50.0 = **85%** | 64.3 → 42.7 = **66%** |
| DocVQA (**document**) | — (336px can't read docs) | 67.2 → 12.8 = **19%** |

- **Low-res (336px) is a "postage stamp":** pruning to 64 is ~lossless on reading (102%) and only
  graceful on reasoning (85%) — there is little to throw away. This is why v1 found no room and the
  LLaVA-1.5 fine-tuning attempt tied.
- **High-res binds — but task-dependently:** the room is **largest on reading/document tasks**
  (TextVQA 57%, DocVQA 19% retention at K=64) and only **moderate on reasoning** (GQA 66%). So pruning
  headroom is jointly determined by **resolution AND the task's token-demand**, not resolution alone.
- **Implication for the field:** "big FLOPs cuts" on LLaVA-1.5-576, or on reasoning benchmarks, measure
  little. Pruning research is meaningful chiefly on **high-res reading/document** settings.

*(Cross-verified: eval_base.json is the frozen base, not the trained checkpoint — its K=576 strict 58.67
≠ the verbose model's ~4%; GQA uses the SAME first-300 qids on both backbones; instruction appears exactly
once in the high-res GQA prompt.)*

## Finding 2 — The gains come from WHICH tokens; the signal is mid-layer & question-conditioned

Selection quality at fixed budget K=128 (TextVQA, no-OCR; dense 61.4):

| selector | acc | note |
|---|---|---|
| FastV-style early-layer attn (L2) | 39.1 | **worse than blind** (RoPE/position bias; cf. FEATHER) |
| blind CLS-attn (VisionZip-like) | 44.2 | question-blind baseline |
| mid-layer attn (L12/16/20) | 51.7 / 54.7 / 54.0 | **+10.5pp over blind** |

- Layer matters decisively: early-layer attention (what FastV uses) is *below* blind saliency;
  mid-layer attention is far above. (Matches FEATHER's layer-8/16 finding.)
- **Genuinely question-conditioned** (capstone control): selecting tokens with a *mismatched*
  question → 43.8 (≈ blind 44.2); with the *real* question → 54.7. **104% of the gain is due to
  the question.** Not disguised saliency.

## Finding 3 — The DYNAMIC per-sample budget headroom is real but MODEST once oracle-noise is removed

Per-sample first-correct-K decomposition (fixed blind selector, vary only K; binary correct@0.5,
consistent metric; `v2/analysis/oracle_decomposition.py`). **Generalizes across all 3 benchmarks:**

| benchmark | best static | naive oracle | **naive band (inflated)** | **honest (noise-free, matched-FLOPs)** |
|---|---|---|---|---|
| TextVQA-noOCR | 62.3 | 69.3 | **+7.0pp** | **+2.6pp** |
| DocVQA | 74.0 | 83.7 | **+9.7pp** | **+5.0pp** |
| ChartQA | 51.7 | 58.3 | **+6.7pp** | **+0.5pp** |

- The naive per-sample oracle (+6.7 to +9.7pp) is **mostly noise-exploitation** — crediting a sample at
  a budget where it is *accidentally* right. The **monotone** oracle (correct at K *and all larger K*)
  removes that: the honest, noise-free, FLOPs-matched gain is **+0.5 to +5.0pp** — i.e. the naive number
  overstates the dynamic-budget headroom by **~2–13×**. v1's thin band had the same illusion.
- The real, noise-free win is **efficiency** (full accuracy at ~2× fewer tokens), not accuracy — and much
  of even that is reachable by static pruning.
- **Methodological contribution (the paper's most distinctive result):** the monotone decomposition is a
  clean, reusable way to report per-sample/dynamic-budget headroom honestly. Most papers report the
  inflated naive number.

## Finding 4 — Honest accuracy-vs-FLOPs frontier (`v2/analysis/flops_frontier.py`)

Convention = v1's `flops.py` (prune-before-LLM: all 32 layers see K; faithful FastV: 3 full + 29 at K).
Blind and QC share identical generation FLOPs at a given K → the gap is a clean matched-FLOPs gap.
Retention = % of our own dense (which is itself below published LLaVA-NeXT dense due to n=300 /
20-token generation — so we compare *retention*, not absolute, and do NOT claim to beat SOTA).

**DocVQA (ANLS, dense 67.2):**

| K | FLOP-red | blind | QC | QC-retention |
|---|---|---|---|---|
| 64 | 96% | 12.8 | 49.2 | 73% |
| 128 | 94% | 27.8 | 55.1 | 82% |
| 256 | 89% | 41.4 | 57.4 | 86% |
| 512 | 79% | 50.7 | 61.4 | 91% |
| 768 | 68% | 57.9 | 63.2 | **94%** |
| 1152 | 52% | 63.0 | 63.9 | 95% |

- QC reaches ~94% retention at 68% FLOP-reduction; the QC-over-blind advantage is **largest at
  aggressive compression (+36pp @ K=64)** and **vanishes by K=1152 (+0.9pp)** — blind already
  suffices when the budget is ample. Same pattern on ChartQA (+22→+13pp) and TextVQA (+18→+0.6pp).
- **Honest positioning:** these retention numbers are *consistent with* published SOTA
  (HiRED DocVQA 68.7 @ 40% budget; CDPruner ~92% retention @ 160 tok), **not beyond** them. Our
  large "+pp" gaps are over the *naive* blind/FastV baselines, and shrink against real methods.

## Finding 5 — Methodological cautions (the "rigor" thesis, continuing v1)

Four ways pruning results get inflated, each demonstrated above:
1. **Weak baselines** — "+36pp over blind CLS" looks huge; vs SOTA it's ~par. Always report strong baselines.
2. **FLOPs dishonesty** — the mid-layer teacher needs 16 full-token layers; it is a *teacher*, not a
   deployable method. Matched-FLOPs accounting is mandatory.
3. **Oracle-noise** — naive per-sample oracle inflates dynamic-budget headroom ~3×; use the monotone bound.
4. **Position/layer bias** — early-layer attention (FastV) is below blind; raw attention has RoPE bias (FEATHER).

---

## What is solid vs. what still needs doing (for the write-up)

**Solid (measured):** Findings 1–4 above, on TextVQA-no-OCR / ChartQA / DocVQA (n=300), plus the
question-conditioning control and the monotone oracle decomposition.

**Stability:** DocVQA n=300→n=1000 confirms conclusions; QC & dense stable, blind drifts down ~5pp
(so QC-over-blind gaps are conservative: K=128 +27→+31pp). Effect sizes robust.

**To strengthen before submission:**
- Full-benchmark numbers (n=300/1000 → full val) for the headline tables.
- Faithful FastV (keep tokens through layer 2-3) as the literature baseline, not just the early-layer proxy.
- Generalize the oracle-noise decomposition (Finding 3) to DocVQA/ChartQA, not only TextVQA.
- ~~Establish our dense ≈ published LLaVA-NeXT (longer max-new-tokens)~~ — **DONE: not truncation.**
  Dense is identical at 20 vs 50 max-new-tokens (DocVQA 67.2, TextVQA 61.4, ChartQA 50.3; 0% hit the
  cap, mean answer 3–5 tok). The dense ceiling is robust; the gap to any "~74" figure is subset
  (n=300) + eval settings, not a bug. Report **retention vs our own dense** (standard for pruning).
  For the paper: re-run headline tables on full-val and cite the exact published baseline from source.
- Add GQA / POPE for the low-res "no room" side (have v1 numbers; re-confirm under this harness).

**Explicitly out of scope for Path A (deferred to Path B):** training a cheap selector to distill the
mid-layer teacher; beating SOTA head-to-head. Path A is the honest characterization; Path B is the method.
