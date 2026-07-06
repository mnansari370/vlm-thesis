# Dynamic-COUNT — next-phase plan

*Written at the Dynamic-WHICH freeze. Planning document only — no Dynamic-COUNT GPU work has been
run or approved yet.*

## Goal

Dynamic-WHICH chooses **WHICH** tokens to keep at a fixed per-sample budget. Dynamic-COUNT changes
the axis: choose **HOW MANY** tokens each sample needs, selecting a per-sample budget from the
locked grid {15, 25, 35, 50, 75}% while keeping the **existing static selectors** (LLaVA `cls_attn`,
Qwen `norm`) to pick which tokens fill that budget. No new selection method is introduced — only the
budget becomes adaptive.

## Reason

The completed full-final matrix shows Dynamic-WHICH textsim is **not universal**: it wins only on
Qwen×TextVQA (and one isolated DocVQA p15 cell); 32 of 40 cells lose to the static floor. Meanwhile,
two pieces of existing evidence point at the *budget* axis instead:

1. **Static accuracy-vs-budget curves** rise steeply and saturate at different points per sample and
   per dataset — many samples are already correct at p15 while others need p75 or dense.
2. **The CPU oracle analysis** (`analyze_dynamic_count_oracle.py`, already run on the dense + static
   finals) shows a perfect per-sample budget router over the same static selectors would beat the
   best fixed budget by **+2 to +6 pp on every one of the 8 model×dataset cells** at ~74–83%
   visual-token reduction, and could match dense accuracy with only ~60–108 mean visual tokens
   (needs-dense on only 0.5–3.9% of samples). That headroom is an upper bound (the oracle peeks at
   correctness), but it is consistent and large — unlike the WHICH axis, it does not depend on a
   favorable task regime.

## Tasks for the Dynamic-COUNT phase

1. **Re-run / verify the Dynamic-COUNT oracle summary** (CPU only) against the frozen dense/static
   finals: `python -m scripts.final_scope.analyze_dynamic_count_oracle` → confirm
   `dynamic_count_oracle_summary.{csv,md}` reproduces (best-score and match-dense oracles, per-cell
   headroom, first-solved-budget distributions).
2. **Define deployable (correctness-blind) controller candidates** — each must decide the budget
   from the input alone, never from gold:
   - **Confidence/entropy routing** — run the cheapest budget first; escalate to a larger budget when
     the model's answer confidence (first-token max-prob / entropy) is low.
   - **Static score-distribution statistics** — route on the shape of the static saliency
     distribution (e.g. entropy/participation-ratio of cls_attn or norm scores: concentrated ⇒ few
     tokens suffice; flat ⇒ more needed).
   - **Question-type / prompt features** — route on cheap text features (question type, length,
     wh-word, counting/reading cues) fitted on a held-out split.
   - **Hybrid confidence + score-statistics router** — the escalation rule gated jointly on model
     confidence and saliency-distribution statistics.
3. **Start CPU-only**: oracle + any router that can be simulated from existing per-sample outputs
   (e.g. confidence values already recorded, saliency statistics recomputable offline) before any new
   GPU generation.
4. **Then n=200 controller pilots** (GPU, on approval) for the most promising router(s), under the
   same fairness schema (same manifests, prompts, scorers; method="dynamic_count").
5. **n=1000 confirmation** only for pilots that clearly beat the best fixed budget at matched
   average compute.
6. **Full finals only if confirmation is strong** — same staged-batch discipline as the WHICH phase.

## Honest-reporting rules carried over from the WHICH phase

- The oracle is labeled **ORACLE / upper bound, not a deployable method**, everywhere it appears.
- Controllers are evaluated at **matched average compute** vs fixed budgets (otherwise a router that
  simply spends more would trivially "win").
- Escalation-style routers must account for the cost of the extra pass(es) in the FLOP accounting.
- Negative controller results are reported in full, exactly as the WHICH negatives were.

## Explicitly out of scope for now

- No Dynamic-COUNT GPU runs yet (this document is the plan, not a launch).
- No training, no learned budget heads (training-free controllers first; learned routing is a
  separate decision if simple routers fail).
- No changes to the frozen dense / static / Dynamic-WHICH results.
