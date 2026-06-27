# DENSE DECISION LOCK (Phase 1)

*Locked 2026-06-27. This is the authoritative record of the dense-baseline protocol decisions. It finalizes the
Decision Table in `docs/DENSE_PROTOCOL.md` (which is the menu; this is the chosen set). No code changed, no
experiments run. "LOCKED" = decided now; "OPEN-pilot" = deliberately deferred to the n=200 pilot, with the
default and the rule for resolving it stated here.*

## Correction logged first: sample-ID manifest location
The canonical, version-controlled sample-ID manifests live under **`configs/`**, not `results/` (which is
git-ignored):
```
configs/final_scope/sample_ids/{gqa,textvqa,docvqa,vqav2}.json   # TRACKED in git (ordered IDs + sha256)
```
Run outputs under `results/final_scope/` only **reference** these by path and **record their sha256**; they do
not copy the IDs. `DENSE_PROTOCOL.md` §1, §7, §12, Option C, and Phase 2 were updated to this path.

---

## Locked decisions

| # | Decision | Status | Choice (locked / default) | Rationale | Revisit condition |
|---|---|---|---|---|---|
| 1 | **Implementation approach** | **LOCKED** | **Hybrid toolkit** — shared `sample_ids` builder/loader · shared `output_writer` · shared `token_flops` helper · shared `schema_validator`/fairness gate · **reuse existing validated generation paths** (`StaticPrunedLlava`/LLaVA stock; `QwenPruner.generate(selector=full)`) | keeps the validated numbers while guaranteeing one schema/token/FLOPs/validator; reused by every later method | only if the toolkit can't wrap a generation path without reimplementing decoding |
| 2 | **VQAv2 subset** | **LOCKED** | **25,000 stratified validation questions, seed 42** (i.e. **VQAv2 final subset is now LOCKED to 25,000 stratified validation questions sampled with seed 42**) | 10k was too small for reliable per-type precision; 25k gives per-type ±~1.1pp at acceptable runtime (~62 GPU-h across all VQAv2 method-runs). The old 10k result **76.44 is reference only**. 20k/30k are documented only as alternatives in `DENSE_RUNTIME_BATCHSIZE_ANALYSIS.md` | freeze `configs/final_scope/sample_ids/vqav2.json` recording `n=25000`, `seed=42`, `split=validation`, `selection_method`, `stratification_method`, per-`answer_type` counts, the **ordered `question_ids`**, and `sha256`; the **same 25k manifest is used for dense, static, dynamic-WHICH, and dynamic-COUNT**. **Sampling = proportional stratified sampling by the official VQAv2 annotation `answer_type`**, strata = **`yes/no`, `number`, `other`** (joined from `v2_mscoco_val2014_annotations.json`). **Do NOT use** the repo's old heuristic `_question_type_id` buckets (yes/no/attribute/counting/spatial) for the final subset. **Build note:** the existing val loader truncates (stratifies only `train`), so the builder applies the `answer_type` stratification explicitly to the val split |
| 3 | **DocVQA subset** | **LOCKED (staged)** | **n=200 pilot first, then FULL validation (~5,349) as the final** | pilot validates the new LLaVA-DocVQA + Qwen-DocVQA harness cheaply before the full run | pilot fails fairness checks → fix harness before full |
| 4 | **TextVQA OCR prompt wording** | **LOCKED default; plural = OPEN-pilot** | default **`"Reference OCR token:"` (singular)** = reproduction wording matching the cached jsonl + old 57.65 run. Plural **`"Reference OCR tokens:"`** is an **optional pilot only**, never the final default unless explicitly chosen later | singular reproduces the published anchor; a one-word change can move the number, so don't switch silently | a plural pilot on n=200 shows a clear, accepted improvement AND is explicitly chosen |
| 5 | **DocVQA instruction (on/off)** | **OPEN-pilot** | **OPEN** — pilot **instruction-on vs instruction-off on n=200** before finalizing | a single-word instruction can depress extractive ANLS; decide on evidence | resolve at pilot; pick the higher-ANLS variant and lock it before the full run |
| 6 | **max_new_tokens** | **LOCKED (provisional)** | **64 for all four datasets** during dense pilots and final | greedy stops at EOS, so 64 is harmless for short answers and sufficient for DocVQA; reuses LLaVA GQA/TextVQA old runs | a pilot reveals a concrete problem (e.g. truncation/runaway) → adjust that dataset only |
| 7 | **VQAv2 scorer** | **LOCKED; official = OPEN-pilot** | **inline consensus `min(matches/3,1)`** for continuity (produced 76.44). Optional **pilot vs official-style normalization**; **do NOT switch unless every method will use the same scorer** | keeps dense comparable to the existing static curve; no mixed scorers across budgets | only switch if the official-vs-inline delta is material AND all methods are re-scored identically |
| 8 | **Batch size** | **LOCKED** | **bs=1** for accepted final numbers | the locked honest protocol; avoids LLaVA left-padding generation drift | bs>1 allowed only after proving byte-identical predictions |
| 9 | **Old LLaVA dense results** | **LOCKED** | **reference only** — **re-run under the final unified schema** and **compare** to old (GQA 61.42 / TextVQA-OCR 57.65 / VQAv2 76.44) | old runs lack tokens/FLOPs/sample-ID sha; re-running makes them protocol-clean (expect ≈ identical scores) | re-run score diverges materially from old → investigate before accepting |
| 10 | **Qwen DocVQA old results** | **LOCKED** | **reference only** (old n=50 = 94.85, n=200 = 97.19) — **re-run with per-sample predictions, IDs, token counts, and FLOPs** at the agreed n | old runs save no per-sample artifacts and have inconsistent n; n=200 dense > published (small-n optimism) | — |

## Summary
- **Fully locked now:** #1 Hybrid, **#2 VQAv2 = 25k stratified seed 42**, #6 mnt=64 (provisional), #7 inline scorer, #8 bs=1, #9/#10 re-run-old-as-reference, #3 full-DocVQA-after-pilot.
- **Deliberately OPEN until the n=200 pilot:** #5 DocVQA instruction on/off; #4 TextVQA plural wording (singular is the default unless plural is later chosen).
- **Pilot-then-decide ordering:** #3 (DocVQA full after pilot), #4/#5 (resolve at pilot), #7 (optional official-scorer pilot).

These supersede the recommendations in `DENSE_PROTOCOL.md`'s Decision Table for execution purposes; the protocol
doc remains the detailed spec, this doc is the locked choice set. Next: **Phase 2 — shared infrastructure**
(build the Hybrid toolkit + freeze the `configs/final_scope/sample_ids/` manifests).
