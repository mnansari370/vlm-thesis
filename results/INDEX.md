# Results index

Maps the `results/` classes to the evidence they hold. Authoritative per-number paths are in
`docs/THESIS_EVIDENCE_LEDGER.md` (this index is a navigation aid). Contents are git-ignored.

| Class | Location | Holds | Ledger |
|---|---|---|---|
| thesis main (GQA) | `results/thesis_main/gqa/results_frozen/`, `results/thesis_main/gqa/{testdev_frontier_analysis,cascade_sweep,week1_all_numbers}.json` + run dirs | dense reproduction, oracle band, cascade, frozen-selection probe | L1, L3, L4 |
| thesis main (VQAv2) | `results/thesis_main/vqav2/dynamic_150k_clsonly/`, `results/thesis_main/vqav2/static_k*/` | budget ties static; static frontier (appendix) | L2 (+ L3 support) |
| thesis main (high-res) | `results/thesis_main/highres/qwen_*.json`, `eval_highres_*.json`, `llava_latency.json`, `qwen_flops_summary.json`, `distill/{gate,control}_*.json`, `distill/*.pt`, `figures/` | selection dominates, mid-layer signal, Q-conditioned, student | L5–L12, E1–E3 |
| paper candidates | `results/paper_candidates/qwen_budget_data_*.json` | raw inputs for the L10 generality recompute (not written as results yet) | L10 |
| archived | `results/archived/stage1_*/` | elastic Stage-1 checkpoints (excluded from thesis) | none (history) |

> **Main vs appendix is decided by the ledger's *Placement* column**, not by which folder a file sits in.
> The per-track physical split under `thesis_main/` exists only to keep the ledger paths and the `src/`
> reader scripts mapping by a simple prefix. The original external archive (failed experiments, retired
> code, redundant checkpoints, consolidated-docs backup) is kept outside the repo.
