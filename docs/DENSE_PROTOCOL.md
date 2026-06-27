# DENSE BASELINE PROTOCOL (final scope)

*Protocol design only — written 2026-06-27. No code was changed, no experiments run. Defines the single
dense-baseline protocol for {LLaVA-1.5-7B, Qwen-2.5-VL-7B} × {GQA, TextVQA, DocVQA, VQAv2}. Builds on
`DENSE_BASELINE_AUDIT.md` (what exists today) and `FINAL_EVIDENCE_LEDGER.md` (numbers). Genuinely open
choices are flagged **[OPEN]** and collected in the Decision Table at the end — I recommend, I do not decide.*

**Why dense first:** dense is the **ceiling** and the **FLOPs/token reference** that every static and dynamic
result is reported against, and it is the **reproduction anchor** (must match published numbers, or the
downstream negatives aren't trustworthy). So the dense protocol must be airtight and identical across methods.

---

## 1. Sample subset policy
**Principle: one canonical, version-controlled sample-ID list per dataset, generated once, reused by dense
AND every pruning method AND both models.** Dense on LLaVA and Qwen must run the *same* IDs per dataset so
the two models are comparable; every method at every budget must run the *same* IDs as dense so retention is
exact.

- The canonical sample-ID manifest is **version-controlled under `configs/`, not `results/`**:
  `configs/final_scope/sample_ids/{dataset}.json` = ordered list of sample IDs + a `sha256`. (`results/` is
  git-ignored, so a manifest there would not be tracked — hence `configs/`.) All runs **read** the manifest from
  `configs/`; each run's result JSON only **references the path and records the sha256** (it does not copy the
  IDs), so a subset mismatch is caught.
- Per dataset:
  - **GQA** — testdev_balanced, **full 12,578** (deterministic dict order; already used by `run_static_testdev`).
  - **TextVQA** — val, **full 5,000** (deterministic JSONL order). OCR and no-OCR share the same IDs.
  - **DocVQA** — `lmms-lab/DocVQA` validation; **[OPEN]** full (~5,349) vs a fixed 2,000.
  - **VQAv2** — **LOCKED:** the final subset is **25,000 stratified validation questions sampled with seed 42**
    (the old 10k/76.44 run is **reference only**). 20k/30k are documented as alternatives only in
    `DENSE_RUNTIME_BATCHSIZE_ANALYSIS.md`.
- IDs: GQA `questionId`; TextVQA `question_id` (= image_id); DocVQA `questionId`/`docId`+row-index; VQAv2 `question_id`.
- The **VQAv2 manifest** (`configs/final_scope/sample_ids/vqav2.json`) records: **`n=25000`**, **`seed=42`**, the
  **exact stratification method** (below), the **ordered `question_ids`**, and the **`sha256`** of that ordered
  list — the **same 25k manifest is reused by dense, static, dynamic-WHICH, and dynamic-COUNT**.
- **Stratification (defined explicitly).** Strata = the repo's **4 heuristic question-type buckets**
  (`0=yes/no, 1=attribute, 2=counting, 3=spatial`), inferred from the question text by `_question_type_id` and
  sampled **proportionally** per bucket by `_stratified_sample(seed=42)` in `src/data/vqav2/vqav2.py`. This is
  the method the repo already documents — it is **NOT** VQAv2's official `answer_type` field
  (yes/no / number / other). Selection is deterministic given seed 42.
- **Build note (correctness).** The *existing* val data path **truncates** (`build_vqav2_dataset` passes
  `stratify=False` for the val split; `_build_samples` stratifies only `train`). So the 25k **val** manifest must
  be built with the above stratification **explicitly applied to the val split** by the Phase-2 sample-ID builder,
  then frozen + sha'd. (This also means the old "10k stratified" val label was effectively plain truncation —
  one more reason the 10k/76.44 number is **reference only**.)

## 2. Exact prompt per dataset
Same user-visible instruction string across models; the chat-template wrapping differs by model
(LLaVA `apply_chat_template` vicuna_v1 vs Qwen `apply_chat_template` — inherent, documented, accepted).

| Dataset | Prompt (text the model sees after `{question}`) |
|---|---|
| GQA | `{question}\nAnswer the question using a single word or phrase.` |
| TextVQA (OCR, primary) | `{question}\nReference OCR token{S}: {ocr}\nAnswer the question using a single word or phrase.` — **[OPEN: `token` vs `tokens`]** |
| TextVQA (no-OCR, secondary) | `{question}\nAnswer the question using a single word or phrase.` |
| DocVQA | `{question}\nAnswer the question using a single word or phrase.` **[OPEN: instruction on/off — it can hurt extractive ANLS]** |
| VQAv2 | `{question}\nAnswer the question using a single word or phrase.` (standardized to `\n`+instr; old VQAv2 runs used a space — re-run needed for parity) |

- **[OPEN] TextVQA OCR phrasing — `"Reference OCR token:"` (singular) vs `"Reference OCR tokens:"` (plural).**
  The cached `data/textvqa/llava_textvqa_val_v051_ocr.jsonl` already embeds the OCR line, and the old runs used
  the **singular** form ("token"); the LLaVA-1.5 published convention is also commonly the singular. This is a
  one-word prompt difference that can shift the reproduction number, so do **not** change it silently — pick
  singular (matches the cached jsonl / old 57.65 run) or plural and record the choice. Default: keep the cached
  jsonl's wording verbatim and pilot the alternative on n=200 if curious.
- LLaVA: `image_pad=True` (expand2square, the LLaVA-1.5 default). Qwen: `max_pixels` fixed and recorded (see §8).

## 3. max_new_tokens per dataset
**Recommendation: `max_new_tokens = 64` for all four datasets.** Greedy stops at natural EOS, so a longer cap
is harmless for short-answer tasks (GQA/TextVQA/VQAv2) and necessary for DocVQA's longer extractive answers.
Consequence: LLaVA GQA/TextVQA old results (already 64) are directly reusable; VQAv2 old used 10 → a
confirmation re-run is needed (answers are short, so the value rarely changes under greedy). **[OPEN]** if you
prefer faster runs, use 20 for GQA/TextVQA/VQAv2 and 64 for DocVQA — but then GQA/TextVQA need re-runs.

## 4. Decoding settings (identical everywhere)
`do_sample=False`, `num_beams=1`, temperature 0 (implicit), **no** `repetition_penalty`, **no**
`min_new_tokens`, natural EOS stop, `pad_token_id=eos`. Dtype: LLaVA fp16, Qwen bf16; `attn_implementation="sdpa"`.
**`batch_size=1`** for the accepted final number (the locked honest protocol); bs>1 allowed only after verifying
it produces byte-identical predictions (LLaVA left-padding can perturb generation).

## 5. Metric / scorer (canonical, one per dataset)
| Dataset | Scorer | Notes |
|---|---|---|
| GQA | `src/metrics/official_score.py` | strict `strip().rstrip('.').lower()` equality; per-semantic-type breakdown |
| TextVQA | `src/metrics/textvqa_score.py` (+ `m4c_evaluator.py`) | official M4C soft-acc; report OCR (primary) and no-OCR |
| DocVQA | `src/metrics/docvqa_score.py` `anls` | ANLS, τ=0.5 |
| VQAv2 | consensus `min(matches/3, 1)` (currently inline in `generate_and_score.py`) | **[OPEN]** keep the light inline consensus vs the full official VQAv2 eval (article/number normalization) |

- **[OPEN] VQAv2 scorer.** The inline consensus (`normalize_answer` + `min(matches/3,1)`) is **acceptable for
  continuity** — it produced the existing 76.44 and keeps dense comparable to the old static curve. However, it
  uses a lighter normalization than the **official VQAv2 eval** (full article/number/punctuation processing).
  **Pilot both on the same n=200 and report the delta**; only switch to the official scorer if the delta is
  material AND you re-score every method the same way (don't mix scorers across budgets).
- All cells **save predictions + IDs** so any number is re-scorable offline.

## 6. Required saved fields per sample (unified schema, JSONL — one row per sample)
```json
{"sample_id": "...", "dataset": "gqa", "model": "llava15", "method": "dense", "budget_pct": 100,
 "image_id": "...", "question": "...", "prompt": "<full text after template>",
 "pred_raw": "<decoded, .strip() only>", "pred_norm": "<scorer-normalized>",
 "gold": ["..."], "per_sample_score": 1.0,
 "n_visual_tokens": 576, "n_text_tokens": 34, "seq_len": 610}
```
Every cell must populate **all** fields (today Qwen-DocVQA populates none of the per-sample fields).

## 7. Required aggregate fields (one JSON per cell)
```json
{"model": "...", "dataset": "...", "split": "...", "method": "dense", "budget_pct": 100,
 "n": 12578, "metric": "gqa_exact", "score_pct": 61.42,
 "per_type": {...},                              // GQA semantic types / VQAv2 qtype / TextVQA where applicable
 "visual_tokens": {"avg": 576, "max": 576},      // Qwen: real per-image avg/max
 "text_tokens_avg": 34, "seq_len_avg": 610,
 "flops_prefill_TFLOPs_avg": 3.17, "flops_per_sample": "computed per sample then averaged (Qwen varies)",
 "flops_convention": "FastV Eq.5 prefill, prune-before-LLM",
 "token_reduction_pct": 0.0, "flop_reduction_pct": 0.0,   // dense = reference (0%)
 "prompt_template": "...", "max_new_tokens": 64, "decoding": "greedy bs=1 no-rep no-minnew",
 "image_pad": true, "max_pixels": null,          // Qwen records max_pixels
 "sample_ids_path": "configs/final_scope/sample_ids/gqa.json", "sample_ids_sha256": "...",
 "published_ref": 62.0, "diff_from_ref_pp": -0.58,
 "git_sha": "...", "env": "vlm_env|qwen_env", "timestamp": "..."}
```

## 8. Token counting method
- **LLaVA-1.5:** visual tokens = **576** (24×24 @336px, `vision_feature_layer=-2`, `default` strategy — fixed).
  text tokens = tokenized non-visual prompt length = `attention_mask.sum() − 576` (measure, don't estimate).
  **Hard sanity assert (every sample):** `n_visual_tokens == 576` **and** `seq_len == n_visual_tokens +
  n_text_tokens`. Any sample violating this fails the run (not just the aggregate) — see §13 check 3.
- **Qwen-2.5-VL:** visual tokens = `Σ (t·h·w) // merge²` over `image_grid_thw` (exactly as `qwen25_dense_eval.py`
  computes it) — **varies per image**, controlled by `max_pixels`; record per-sample, plus avg + max. text tokens
  measured the same way (total minus visual block). **Fix and record `max_pixels`** so the dense token count is
  reproducible. *Correction (vs the audit's first read):* the two old Qwen dense runs used the **same** resolution
  cap — `1280·28·28 == 1,003,520` — so the discrepancy between them is **not** `max_pixels`. The real problems are
  the **missing per-sample predictions/IDs** and the **inconsistent n (50 vs 200)** (`DENSE_BASELINE_AUDIT.md` §2).
  Still pin `max_pixels` explicitly so the per-image token count stays reproducible.
- `seq_len = n_visual_tokens + n_text_tokens` (per sample; report avg). For LLaVA this is an exact identity to
  assert; for Qwen it is a per-sample measurement.

## 9. FLOPs calculation method
Analytical LLM **prefill** only, FastV Eq.5 per layer `4·n·d² + 2·n²·d + 2·n·d·m`, summed over T layers, where
for each sample `n_i = (that sample's visual tokens) + (that sample's text tokens)`.

**Compute FLOPs PER SAMPLE, then average over the run — do NOT plug an average token count into the formula.**
Because the formula has an `n²` term (convex), `mean_i FLOPs(n_i) ≠ FLOPs(mean_i n_i)`; using the mean token
count **undercounts** the true average FLOPs. This matters specifically for **Qwen**, where the visual-token
count varies per image. For **LLaVA-1.5** every sample has `n_visual = 576`, so per-sample and averaged agree —
but compute per-sample anyway so both models use one uniform code path.

- **LLaVA-1.5:** `src/analysis/flops.py` / `flops_vqav2.py` — T=32, d=4096, m=11008. Measure each sample's text
  tokens; reference `n_text` means for cross-check only: GQA 34, TextVQA-OCR 86 / no-OCR 32, VQAv2 35,
  **DocVQA = measure new**.
- **Qwen-2.5-VL:** `src/analysis/qwen_flops.py` — T=28, d=3584, m=18944. Measure each sample's visual **and**
  text tokens per sample. The hardcoded `N_TEXT=40` (a DocVQA estimate) must **not** be reused for
  GQA/TextVQA/VQAv2.
- **Report** `flops_prefill_TFLOPs_avg` = mean of the per-sample FLOPs, and (for Qwen) optionally the per-sample
  min/max so the spread is visible.
- **State exclusions every time:** excludes the vision encoder, the projector, and the decode steps. Dense is
  the **100% reference**; report **token-reduction and FLOP-reduction as separate numbers** (both 0% for dense),
  never conflated. Fold the dense FLOPs into the aggregate JSON (§7) so every method can divide by it.

## 10. Pilot run size (validate the harness before any full run)
**n = 200** per cell (first 200 of the canonical sample list). Acceptance before scaling up:
- Reproduces published within tolerance (§13).
- LLaVA dense == full-576 sanity; Qwen `selector=full` == stock `generate` (already validated in `qwen_pruner`).
- Per-sample schema fully populated; tokens recorded and sane; <1% empty predictions.
- sample-IDs sha matches the manifest.
- **Prompt-parity pilot (required for the 3 existing LLaVA cells — GQA, TextVQA, VQAv2).** Because the new
  standardized prompt may differ from what the old saved results used (VQAv2 ` `→`\n`+instr; TextVQA OCR
  `token`/`tokens` wording; any template drift), the pilot MUST run **OLD prompt vs STANDARDIZED prompt on the
  SAME n=200** and report the score delta. Finalize the standardized prompt only if the delta is within
  tolerance; if it shifts the number materially, surface it and decide (keep old vs adopt new) **before** the
  full run — never swap prompts silently. (New cells — LLaVA-DocVQA, Qwen-GQA/TextVQA/VQAv2 — have no old prompt
  to compare, so they only need the reproduction + schema checks.)

## 11. Final run size options
| Tier | GQA | TextVQA | DocVQA | VQAv2 | Use |
|---|---|---|---|---|---|
| **Pilot** | 200 | 200 | 200 | 200 | harness validation only |
| **Standard (recommended)** | full 12,578 | full 5,000 | full ~5,349 | **25k (seed 42, LOCKED)** | the citable final dense baseline |
| **Extended** | 12,578 | 5,000 | full | >25k toward full VQAv2 | only if a reviewer later demands a larger VQAv2 than the locked 25k |
Qwen runs the **same IDs** as LLaVA per dataset at the chosen tier. (Full-val VQAv2 214k is out — too costly.)

## 12. Output file paths
```
configs/final_scope/sample_ids/{gqa,textvqa,docvqa,vqav2}.json        # shared ID manifest (+ sha) — TRACKED in git
results/final_scope/{llava15,qwen25vl7b}/{gqa,textvqa,docvqa,vqav2}/  # run outputs — git-ignored
    dense_100.json      # aggregate (§7) — references the manifest path + records its sha256
    dense_100.jsonl     # per-sample (§6)
```
(TextVQA emits `dense_100_ocr.*` and `dense_100_noocr.*`.) The **sample-ID manifests are version-controlled
under `configs/`** (so the exact subset is reproducible from git); only the **run outputs** live under the
git-ignored `results/final_scope/` scaffold, and they reference the manifest by path + sha rather than copying it.

## 13. Fairness checks before accepting a dense result
A dense cell is **accepted** only if ALL hold:
1. **Reproduction:** GQA within ±0.5pp of 62.0; TextVQA-OCR within ±1pp of 58.2; DocVQA within ~3pp of 95.7
   (looser at small n); VQAv2 within ±1pp of 76.4. (Tolerances tighten at full n.)
2. **Same samples:** `sample_ids_sha256` equals the canonical manifest that every method will use.
3. **Dense / token sanity (hard asserts):** **LLaVA — every sample `n_visual_tokens == 576` AND
   `seq_len == n_visual_tokens + n_text_tokens`** (fail the run on any violation); Qwen `full` == stock
   `generate`; Qwen visual-token avg consistent with the recorded `max_pixels`.
4. **Decoding:** bs=1 (or proven bs-invariant), greedy, no rep-penalty, no min-new-tokens.
5. **Re-scorable:** predictions + IDs + gold saved; re-scoring the JSONL reproduces the aggregate score.
6. **Logging:** prompt template, `max_new_tokens`, `image_pad`/`max_pixels`, scorer, git sha, env recorded.
7. **Health:** empty-prediction rate <1%; per-type breakdown present (GQA/VQAv2); FLOPs + token avg written.
8. **Cross-model parity:** LLaVA and Qwen dense ran the identical sample-ID manifest for that dataset.

---

## Implementation options (do not implement yet)

### Option A — one unified dense evaluator
A single new module (e.g. `src/final_scope/dense_eval.py`) with: model adapters (LLaVA-1.5, Qwen-2.5-VL),
dataset adapters (GQA/TextVQA/DocVQA/VQAv2), scorer dispatch, the unified per-sample + aggregate schema, the
token/FLOPs recorder, and the fairness-check gate — all in one place. **It must reuse the already-validated
generation paths internally** (`StaticPrunedLlava`/the LLaVA stock path; `QwenPruner.generate(selector=full)`),
not reimplement decoding.

| Pros | Cons | Risk |
|---|---|---|
| One protocol, one schema → guaranteed cross-model/dataset comparability | New code to write + validate against old numbers | **Medium** |
| Single place for fairness checks, tokens, FLOPs, sample-ID manifest | Bigger upfront effort before the first number | Mitigate by wrapping validated `generate` paths, not rewriting them |
| Trivial to extend to static / dynamic-WHICH later (add a selector arg) | A subtle divergence here propagates to *all* cells | Validate each cell against its pilot tolerance before trusting |
| Naturally fills the 4 missing cells with the same code | Two envs (vlm_env/qwen_env) → the module must run under both | Keep model/dataset adapters import-light; lazy-import heavy deps |

### Option B — patch the existing scripts
Extend `run_dense_testdev.py`, `run_textvqa.py`, `generate_and_score.py` (LLaVA) and `qwen25_dense_eval.py`
(Qwen) to emit the unified schema + tokens + FLOPs + read the shared sample-ID manifest; add the 4 missing
cells by extending Qwen `load_bench` (GQA/TextVQA/VQAv2) and adding a LLaVA-1.5 DocVQA path.

| Pros | Cons | Risk |
|---|---|---|
| Reuses already-validated generation/scoring → numbers stay trustworthy | 5+ scripts to keep in lockstep → schema drift | **Medium** |
| Smaller diffs; LLaVA GQA/TextVQA/VQAv2 cells done fastest | Four code styles, four output writers to align | Easy for one script to diverge silently |
| Lowest risk of changing the already-good LLaVA numbers | The 4 missing cells still need substantial new adapters anyway | Same new-code risk as A for those cells, minus the unification benefit |
| No new module/import surface | Fairness checks duplicated per script (or skipped) | Harder to guarantee identical protocol across files |

### Option C — Hybrid (**recommended first implementation path**)
Build a **small shared toolkit** that both the existing scripts and any new cell import, instead of either a
single monolith (A) or N independently-patched scripts (B). The toolkit is method-agnostic, so it serves dense
now and static / dynamic-WHICH / dynamic-COUNT later. Five shared pieces:

1. **`sample_ids`** — one builder/loader that writes/reads `configs/final_scope/sample_ids/{dataset}.json` (+ sha256);
   every run reads the same manifest (kills subset drift, §1).
2. **`output_writer`** — one writer for the unified per-sample JSONL (§6) + aggregate JSON (§7), so every cell
   emits byte-identical schema.
3. **`token_flops` helper** — per-sample token counting (§8) + per-sample-then-averaged FLOPs (§9) for both
   models, in one place (no convexity mistakes, no `N_TEXT` reuse bugs).
4. **`schema_validator` + fairness gate** — asserts the §6/§7 fields and runs the §13 checks; a cell is only
   "accepted" if it passes.
5. **Reuse existing generation paths** — the toolkit does **not** reimplement decoding; it wraps the already-
   validated `StaticPrunedLlava`/LLaVA stock path and `QwenPruner.generate(selector=full)`.

The existing LLaVA scripts (`run_dense_testdev`, `run_textvqa`, `generate_and_score`) and the Qwen path
(`qwen25_dense_eval` / `qwen_kcurve full`) call into this toolkit for IDs/output/tokens/FLOPs/validation; the
4 missing cells get a thin new adapter that uses the same toolkit.

| Pros | Cons | Risk |
|---|---|---|
| Reuses validated generation → trustworthy numbers (B's strength) | A bit more design up front than B (define the toolkit API) | **Low–Medium** |
| One schema/token/FLOPs/validator path → real comparability (A's strength) | Two call-sites styles (old scripts + new adapters) to keep calling the toolkit | Mitigated: the toolkit *is* the single source of truth |
| Incremental: wire the 3 ready LLaVA cells first, add missing cells later | Slightly more files than a single monolith | Lowest blast radius — a toolkit bug is caught by its own validator |
| Toolkit is reused by every later method (static/dynamic), not just dense | — | — |

---

## Decision Table (recommendations — **FINALIZED in `docs/DENSE_DECISION_LOCK.md`**)

> Phase-1 lock: these decisions are now recorded authoritatively in **`docs/DENSE_DECISION_LOCK.md`**
> (D1=Hybrid; D2=VQAv2 **LOCKED 25k seed 42**; D3=DocVQA full after n=200 pilot; D4=TextVQA singular default; D5=DocVQA
> instruction OPEN-pilot; D6=mnt 64; D7=inline scorer; D8=bs=1; D9/D10=re-run old as reference). The table
> below is the menu they were chosen from.

| # | Decision | Options | Recommended | Why |
|---|---|---|---|---|
| D1 | **Implementation** | A unified / B patch / **C Hybrid** | **C — Hybrid (shared toolkit + reuse existing generate paths)** | keeps the validated LLaVA/Qwen numbers (B) while guaranteeing one schema/token/FLOPs/validator (A); incremental and reused by every later method |
| D2 | **VQAv2 subset** | 20k / 25k / 30k seed 42 | **LOCKED — 25k seed 42** | per-type precision ±~1.1pp at acceptable runtime; 10k reference only; 20k/30k kept as alternatives in `DENSE_RUNTIME_BATCHSIZE_ANALYSIS.md` |
| D3 | **DocVQA subset** | full ~5,349 / fixed 2,000 | **full ~5,349** | cheap; avoids a subset-choice question |
| D4 | **DocVQA instruction** | on / off | **on, but pilot both** | consistency; verify it doesn't depress ANLS at pilot |
| D5 | **max_new_tokens** | 64 all / 20 short+64 DocVQA | **64 all** | reuses LLaVA GQA/TextVQA old runs; harmless under greedy |
| D6 | **TextVQA primary setting** | OCR / no-OCR | **OCR primary** (report both) | matches the published 58.2 reproduction anchor |
| D7 | **VQAv2 scorer** | inline consensus / full official VQA | **inline consensus** (pilot vs official) | matches existing 76.44; check delta to official at pilot |
| D8 | **Final tier** | Pilot / Standard / Extended | **Standard** | full GQA/TextVQA/DocVQA + **25k VQAv2 (D2)** = citable, affordable |
| D9 | **Batch size** | bs=1 / bs>1 verified | **bs=1** | locked honest protocol; avoid padding-induced drift |
| D10 | **Reuse old LLaVA dense numbers?** | reuse as-is / re-run under new schema | **re-run under the unified schema** (compare to old) | old runs lack tokens/FLOPs/sample-ID sha; re-run makes them protocol-clean (expect ≈ identical scores) |

---

## Dense Phase Plan

A staged plan from "protocol locked" to "verified dense baseline". Each phase has a clear exit gate; no GPU
work happens before Phase 3.

### Phase 1 — Protocol finalization (no GPU)
Resolve the Decision Table (D1–D10): implementation = Hybrid (D1), and the open prompt/subset/scorer/tokens
choices (D2–D7). Lock the per-dataset prompt strings (incl. TextVQA `token`/`tokens`, DocVQA instruction
on/off), `max_new_tokens=64`, the scorers, and the unified schema (§6/§7). **Exit gate:** every `[OPEN]` flag
in this doc is decided and written down.

### Phase 2 — Shared infrastructure (no GPU; code = the Hybrid toolkit)
Build the 5 shared pieces (Option C): `sample_ids` builder, `output_writer`, `token_flops` helper,
`schema_validator` + fairness gate, and the thin wrappers over the existing validated generation paths. Also
generate and freeze `configs/final_scope/sample_ids/{dataset}.json` (+ sha) for all four datasets, and set the
per-dataset Qwen `N_TEXT` / measure DocVQA `n_text`. **Exit gate:** `compileall` clean; the toolkit unit-validates
on a tiny synthetic record; sample-ID manifests exist with stable shas.

### Phase 3 — Pilot n=200 (GPU, small)
Run all 8 cells (the 4 ready + 4 new) at n=200 on the first 200 canonical IDs. For the 3 existing LLaVA cells,
also run the **prompt-parity pilot** (old vs standardized prompt, §10). **Exit gate:** each cell passes the §13
fairness checks at pilot tolerance; prompt-parity deltas are within tolerance or explicitly accepted; the new
cells reproduce sane numbers (DocVQA-LLaVA expected low; flag, don't fail).

### Phase 4 — Final dense runs (GPU)
Run the **Standard tier** (§11): GQA full 12,578 · TextVQA full 5,000 (OCR + no-OCR) · DocVQA full ~5,349 · VQAv2
**25k (seed 42, D2)**, on **both** models, same sample-ID manifests. Emit `dense_100.{json,jsonl}` per cell with tokens
+ per-sample-averaged FLOPs. **Exit gate:** all §13 checks pass at full-n tolerances; cross-model parity holds.

### Phase 5 — Dense verification report (no GPU)
Produce `docs/DENSE_VERIFICATION.md`: the 8-cell dense table (score, n, tokens, FLOPs, Δ-from-published), the
prompt-parity deltas, the reproduction pass/fail per cell, and a side-by-side of new-protocol vs old legacy
numbers (GQA 61.42 / TextVQA 57.65 / VQAv2 76.44 / Qwen-DocVQA 97.19). **Exit gate:** every dense cell is
"accepted" and becomes the locked reference denominator for the static + dynamic phases.

---

*Tracking note: as of this revision, `.gitignore` now negates `docs/DENSE_PROTOCOL.md` and
`docs/DENSE_BASELINE_AUDIT.md`, so both are tracked alongside the other active final-scope docs.*
