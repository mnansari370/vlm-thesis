# Related Work — synthesis for the thesis/paper

Organized around the two results this work defends: **(1)** static CLS-attention pruning is a
near-oracle, training-free frontier that per-sample dynamic budgeting only ties; **(2)** question-
conditioned *selection* is saturated by image-saliency. Citations are grouped so each Related-Work
paragraph maps to one paragraph here. Specific numbers are from the per-paper notes / `SYNTHESIS_tier1.md`
and the project's frozen reports (`../docs/`); treat them as the values to cite, not re-derive.

---

## 1. Token reduction in vision transformers (the ancestors)
Patch tokens in ViTs are redundant, and reducing them cuts the quadratic self-attention cost
(Vaswani et al. 2017). Vision-only methods either **prune** (DynamicViT — learned per-token keep
decisions; A-ViT — adaptive halting; SPViT — latency-aware soft pruning) or **merge** (ToMe — combine
similar tokens). These motivate token sparsity but target image *classification*, with no notion of a
*question* conditioning which tokens matter — the gap this thesis started from.

## 2. Visual token pruning in VLMs — where to score
For LLaVA-style VLMs the question is *which signal* ranks the 576 visual tokens.

**(a) LLM-side attention (position-biased).** FastV (ECCV 2024) prunes after an early LLM layer using
the attention that visual tokens *receive* inside the decoder. It is plug-and-play but its scores are
corrupted by RoPE position bias — later (raster-scan bottom) tokens are systematically favored —
which FEATHER shows wrecks localization (FastV 6.7 vs FEATHER 44.1 on RefCOCO at ~64% FLOPs cut). In
our frontier FastV is **dominated** (GQA K=192: 52.70 vs static CLS-attn 59.19).

**(b) CLIP-side CLS→patch attention (training-free, strong).** FasterVLM/VisPruner and VisionZip's
"dominant" scoring rank patches by the CLIP CLS token's attention to each patch (CLIP uses *learned 2D*
position embeddings, not RoPE, so scores reflect content). FasterVLM reports 75.8% VQAv2 @128 tokens
training-free; VisionZip adds a contextual *merge*. LLaVA-PruMerge uses penultimate-layer CLS attention
with an IQR rule for a per-image adaptive count (72.0% @~32 tok base; 76.8% with a 1-epoch LoRA). This
family is the basis for our **selection** component: CLS-attention is a near-oracle, zero-cost signal.

**(c) Question-conditioned / cross-modal selection (our negative).** ATP-LLaVA (CVPR 2025) scores with
in-decoder self- and cross-modal attention and a learnable per-layer threshold (76.4% @144 tok), but it
**fine-tunes the LLM** — its own ablation shows the frozen-LLM variant loses 2–3 points, the penalty our
frozen architecture accepts. The DivPrune / SparseVLM family and CLIP-space question↔patch relevance are
the canonical *question-conditioned* selectors. **Our finding:** on a frozen backbone, neither beats
CLS-attention — LM-attention Qcond −5.58 pp and CLIP-space Qcond −32.04 pp (below random) at K=64
TextVQA-noOCR, fusion ±0.3 pp (`../docs/week1_results.md`). The CLIP-space failure is explained by raw
CLIP dense features being poor localizers (CLIP-Surgery; MaskCLIP) — verified visually (panel F4).

## 3. Adaptive computation & budgeting (the "how many" axis)
Input-dependent compute has a long line: Adaptive Computation Time (Graves), Universal Transformers,
and early-exit networks (BranchyNet, MSDNet) allocate *depth/steps* by difficulty. The proposal cast
visual tokens as an analogous **discrete budget** problem. Our result reframes it: the per-sample budget
is **not predictable** from frozen input features (CLIP-feature↔correctness r < 0.06) and is only weakly
**revealed** by decoding confidence (r ≈ 0.35–0.49), so our budget mechanism is a **confidence cascade**
(escalate low-confidence samples), conceptually an early-exit on token budget rather than a learned
feature-to-budget predictor.

## 4. The critique line (where our contribution sits)
A growing "are we solving the right problem?" line questions whether VLM token-pruning gains are real or
artifacts of evaluation. This thesis contributes a concrete instrument for that question — the
**oracle-headroom diagnostic** (per-sample first-correct-at-K): it quantifies the budget-sensitive band
(4.9–9.1% across GQA/TextVQA/POPE/ScienceQA, always < 20%), orders benchmarks by how much dynamic
budgeting *could* help, and predicts (correctly) that a confidence cascade only **ties** the static
frontier. To our knowledge this per-sample budget-headroom characterization, paired with a
selection-saturation result on two question-conditioned formulations, is novel.

---

## 5. Positioning (one paragraph for the paper)
We **adopt** the training-free CLS-attention selection that the FasterVLM/VisionZip family established;
we **add** a confidence-driven per-sample cascade for the budget; and we **contribute** the diagnostic
that explains why this — and, we argue, frozen-backbone dynamic pruning generally — ties rather than
beats a well-chosen static budget. Unlike ATP-LLaVA we keep the backbone frozen (and quantify that
penalty); unlike FastV we score CLIP-side (avoiding RoPE bias); unlike the DivPrune/CLIP-space line we
show question-conditioned selection does not beat image saliency on a frozen model.

## 6. Gaps / to verify before submission
**Verified 2026-06-11** (full entries in `../docs/thesis_proposal.md` references):
- Critique paper = Wen, Gao, Li, He, Zhang. *Token Pruning in Multimodal Large Language Models: Are We
  Solving the Right Problem?* Findings of ACL 2025 (arXiv:2502.11501).
- FEATHER = Endo, Wang, Yeung-Levy. *Feather the Throttle: Revisiting Visual Token Pruning for
  Vision-Language Model Acceleration.* ICCV 2025 (arXiv:2412.13180).
- FastV ECCV 2024 (2403.06764) · LLaVA-PruMerge CVPR 2025 (2403.15388) · FasterVLM/VisPruner
  arXiv:2412.01818 · VisionZip arXiv:2412.04467 · ATP-LLaVA CVPR 2025 (2412.00447).

**Still to verify:**
- Final venues for FasterVLM and VisionZip (cited as arXiv; check if since published).
- DivPrune, SparseVLM, CLIP-Surgery, MaskCLIP exact entries (broader set in project memory).
- ATP-LLaVA frozen-LLM ablation row (the fair-comparison target for the VQAv2 track).
