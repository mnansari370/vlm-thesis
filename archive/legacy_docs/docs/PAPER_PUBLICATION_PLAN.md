# Paper Publication Plan

*What a paper from this work would be, what is ready, and the exact work still needed. Numbers live in
[`THESIS_EVIDENCE_LEDGER.md`](THESIS_EVIDENCE_LEDGER.md); story in [`THESIS_MASTER_PLAN.md`](THESIS_MASTER_PLAN.md).
Consolidated 2026-06-24 from the earlier paper roadmap (preserved in `archive/docs_consolidated_backup/`).*

> **Bottom line: a paper is possible, but NOT ready yet.** Order of work: finish the thesis and the repo
> cleanup first; the paper experiments come after. The realistic paper is an **honest measurement /
> analysis paper** ("Selection over Budget"), **not** a SOTA-method paper.

---

## 1. Is publication possible, and as what?
Yes — as an honest measurement/analysis paper. Realistic venues: a **workshop or short paper now**, or
**ACL/EMNLP Findings or TMLR** after the gaps below are closed. It is **not** a top-tier method paper:
there is no SOTA result and no strong-baseline comparison yet. Drafts to build from are preserved at
`docs/source_findings/v2__qwen__PAPER_qwen.md` and `docs/source_findings/v2__analysis__PAPER_DRAFT.md`.

## 2. What is already paper-useful (artifact-backed today)
| Ledger | Claim |
|---|---|
| L1 | dense reproduces published (trust anchor) |
| L2 | per-sample budget ties tuned static (frozen LLaVA-1.5) |
| L3 | thin oracle budget band |
| L4 | naive frozen question-conditioned selection fails |
| L8 | honest (monotone) oracle headroom is small |
| L9 | trained budget predictor captures ≈0 (DocVQA-7B) |
| method | the monotone oracle-noise correction; matched-FLOPs; FLOPs-vs-latency honesty (E1–E3) |

Together these already make a coherent, internally consistent "selection-vs-budget" story on DocVQA-7B
(+ ChartQA).

## 3. What is NOT paper-ready
| Ledger | Issue | Needed |
|---|---|---|
| L5, L7, L9 | headline cells are n=100–300 | **full-validation reruns** |
| L5 | baselines are blind floors (uniform/norm) | **strong published baselines** |
| L6 | "FastV is worse" uses an early-layer proxy | **faithful layer-split FastV** |
| L10 | generality (3B/32B/InfoVQA/LLaVA-1.6) is **narrative-only** | **recompute + SAVE per-model result JSONs** |
| L11 | student is DocVQA-only, ~69% retention | distil on ChartQA/InfoVQA; honest framing |

## 4. Exact extra work before submission
1. **Full-validation reruns** of L5, L7, L9 (replace n=100–300) — `src/evaluation/docvqa/{qwen_kcurve,qwen_control}.py`, `src/pruning/dynamic_budget/qwen_budget_eval.py`.
2. **Strong baselines at matched FLOPs** — FEATHER, CDPruner, HiRED, VisionSelector, **and faithful FastV
   (layer-split)**. None are implemented in the repo yet; add as new selectors in `src/pruning/question_conditioned_selection/qwen_pruner.py`.
3. **Recompute & SAVE the L10 generality** — raw data already exists
   (`results/paper_candidates/qwen_budget_data_{docvqa_3b,docvqa_32b,infovqa,docvqa_7b_n1000}.json`); re-run
   `qwen_budget_eval.py` per model and write per-model result JSONs. **Highest-leverage single fix** (turns
   a risky narrative claim into evidence).
4. **Pin numbers** to the ledger canon (oracle = +7.50pp; predictor = +0.09pp/1.8%; drop +4.2/+3.9).
5. **Careful claim framing** per §6.

## 5. What reviewers will criticize (anticipate)
- Small n on headline cells; weak baselines ("+60pp over uniform").
- The **teacher selector needs a full forward** (efficiency circularity) — answer with the student (L11)
  and by framing the teacher as analysis/upper-bound, not a deployed method.
- DocVQA-centric distillation; dense below published on some tasks (report retention, not absolute).
- "Components are known" (mid-layer selection ≈ FEATHER; adaptive budget ≈ ATP-LLaVA) — answer: the
  contribution is the **rigorous decomposition + the budget negative**, not a new mechanism.

## 6. Claims to avoid in a paper
- "we beat / match SOTA" (no strong baseline run).
- "training fixed it" (the wins are frozen).
- "budget mirage generalizes across families/scales" as proven (L10 narrative-only until recomputed).
- unqualified "+60pp" (state model/task/K/baseline/n) or "88% FLOP reduction" (analytical, excludes
  selector cost).
- the +4.2 / +3.9 oracle numbers (canonical = +7.50pp).

## 7. Minimum vs stronger paper
- **Minimum (workshop/short), supportable with today's artifacts:** the honest selection-vs-budget
  decomposition + the monotone oracle-noise correction — selection dominates, budget is a mirage — on
  DocVQA-7B (+ChartQA), framed as measurement (L2, L3, L8, L9 + methodology).
- **Stronger (Findings/TMLR), after §4:** add full-val headline tables (L5–L9), strong baselines at
  matched FLOPs (L5/L6), **saved** multi-model generality (L10), and multi-dataset student (L11) — a
  re-evaluation of the adaptive-budget pruning literature by construction, on the current SOTA backbone.

## 8. Files/data that must be preserved for paper work (never delete before submission)
- `results/paper_candidates/qwen_budget_data_*.json` (raw, for the L10 recompute).
- All ledger L/E evidence files; the distill teachers/students (`results/thesis_main/highres/distill/*.pt`); the
  analysis code (now `src/pruning/`, `src/evaluation/`, `src/models/distillation/`, `src/analysis/`).
- The paper drafts (`docs/source_findings/v2__qwen__PAPER_qwen.md`, `docs/source_findings/v2__analysis__PAPER_DRAFT.md`,
  `docs/literature/related_work.md`) and the FLOPs artifact (`results/thesis_main/highres/qwen_flops_summary.json`).

## 9. Practical roadmap (after thesis + cleanup)
1. Recompute & save L10 generality (cheap; raw data exists) → unblocks the central claim.
2. Implement + run ≥2 strong baselines + faithful FastV at matched FLOPs.
3. Full-val reruns of L5/L7/L9.
4. ChartQA/InfoVQA distillation (L11 generality).
5. Draft from `docs/source_findings/v2__qwen__PAPER_qwen.md` using ledger numbers only.

## 10. Thesis-first vs paper-first
**Thesis first.** Most of the thesis is writable today; the paper experiments (full-val, strong baselines)
improve the thesis too but are not blocking for a Master's. Run the cheap L10 recompute soon; defer the
heavy paper experiments until after the thesis draft.
