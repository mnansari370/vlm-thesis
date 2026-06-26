# `literature/` — related-work notes

Source notes for the thesis Background and the paper's Related Work. Backbone of interest throughout:
LLaVA-1.5-7B (CLIP ViT-L/14@336 → 576 visual tokens).

## Files
- `related_work.md` — **the synthesis to write from**: organizes the field around the thesis's two
  results (static CLS-attn is near-oracle; question-conditioned selection + per-sample budgeting don't
  beat it), with a claim → citation map.
- `SYNTHESIS_tier1.md` — earlier 4-paper engineering synthesis (FastV, LLaVA-PruMerge, FasterVLM,
  ATP-LLaVA): comparison table + code-change ideas + open questions. **Historical**: several of its
  "open questions" were resolved by the project's pivot (e.g. Q1 — the 61.44% was the *classification
  head*, not the generative model; generative static is ≈75.8%, see `../docs/vqav2_findings.md`).
- Per-paper notes: `fastv.md`, `fastervlm.md`, `llava-prumerge.md`, `atp-llava.md`. **Historical:** any
  "improvement target" or "our model = 62.35%/61.44%" in these notes predates the pivot and uses the
  retired classification proxy — use the generation numbers in `../docs/vqav2_findings.md` for all claims.

## Claim → citation map (for the paper)
| Thesis claim | Primary citations |
|---|---|
| Attention cost scales with token count; visual tokens are redundant | Vaswani 2017; DynamicViT, A-ViT, ToMe, SPViT (ViT ancestors, from proposal) |
| LLM-side attention scoring is position-biased (RoPE) | FastV; FEATHER |
| CLIP CLS→patch attention is a strong, training-free selection signal | FasterVLM/VisPruner; VisionZip (dominant scoring); LLaVA-PruMerge |
| Question-conditioned / cross-modal selection (our negative) | ATP-LLaVA; DivPrune / SparseVLM family; CLIP-space relevance |
| Raw CLIP dense patch features localize poorly (why CLIP-Qcond < random) | CLIP-Surgery; MaskCLIP |
| Adaptive/conditional computation & budgeting | ACT (Graves); Universal Transformers; early-exit (BranchyNet, MSDNet) |
| "Are we solving the right problem?" — our diagnostic's framing | Wen et al., Findings of ACL 2025 (arXiv:2502.11501) |

> Note: a broader 40-paper survey (FastV, SparseVLM, VisionZip, PyramidDrop, PruMerge, DivPrune, DART,
> PACT, TopV, HiRED, TRIM, VTW, ATP-LLaVA, Dynamic-LLaVA, …) is captured in the project memory, not in
> this folder. Pull the ones above into the paper; expand `related_work.md` if the committee wants more.
