# DENSE BASELINE AUDIT (final scope)

*Audit-only, read-only — produced 2026-06-27. No code changed, no experiments run. Every fact below was
read directly from the source file or the saved result JSON cited. Scope = {LLaVA-1.5-7B, Qwen-2.5-VL-7B} ×
{GQA, TextVQA, DocVQA, VQAv2}. "Dense" = keep all visual tokens, generation protocol, official scorer.*

> **Headline:** 4 of 8 dense cells have working code (LLaVA-1.5 × GQA/TextVQA/VQAv2; Qwen × DocVQA); the
> other 4 are **TODO** (LLaVA-1.5 × DocVQA has a loader but no harness; Qwen has **no** GQA/TextVQA/VQAv2
> loader at all). Even the 4 that exist are **not standardized**: prompts, `max_new_tokens`, and saved
> artifacts differ; tokens are recorded only for Qwen; **FLOPs are recorded nowhere** (computed analytically,
> separately); and the **Qwen DocVQA dense run saves no per-sample predictions or IDs**.

---

## 1. Dense-support matrix

| Model | Dataset | Dense code? | Loader | Prompt | Metric | Saves preds? | Saves IDs? | Tokens? | FLOPs? | Status | Risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | **GQA** | **Yes** — `src/evaluation/gqa/run_dense_testdev.py` (inline `LlavaTestdevEval`, stock HF `generate`) | inline `GQATestdevDataset` (testdev_balanced, full 12,578) | `{q}\nAnswer the question using a single word or phrase.` (vicuna_v1 template); greedy, `max_new_tokens=64`, image_pad opt | `official_score.py` (strict `rstrip('.').lower()`) | **Yes** (`testdev_balanced_predictions.json` + `results.jsonl`) | **Yes** (`questionId`) | No (implicit 576) | No | **READY** | Low |
| LLaVA-1.5 | **TextVQA** | **Yes** — `src/evaluation/textvqa/run_textvqa.py` (`StaticPrunedLlava` method=`none`,K=576) | inline `TextVQADataset` (val, full 5,000; OCR + no-OCR) | OCR: `{q}\nReference OCR token: {ocr}\n{INSTR}`; noOCR: `{q}\n{INSTR}`; greedy, `max_new_tokens=64`, image_pad=True | `textvqa_score.py` (M4C soft-acc) | **Yes** (`predictions.json` + `per_sample_scores.json`) | **Yes** (`question_id`=image_id) | No | No | **READY** | Low |
| LLaVA-1.5 | **DocVQA** | **No harness** (loader `src/data/docvqa.py` exists but is wired only to Qwen/distill) | `src/data/docvqa.py` (`load_docvqa`, hub val/test + train parquet) — unused by any LLaVA eval | — | (would be `docvqa_score.anls`) | — | — | — | — | **TODO** | Med — must build LLaVA-1.5 DocVQA harness; expect very low score (576 tok can't read dense docs) |
| LLaVA-1.5 | **VQAv2** | **Yes** — `src/evaluation/vqa/generate_and_score.py --model-type dense` (`LlavaDenseVQAModel`, stock `generate`) | `src/data/vqav2/` (`build_vqav2_dataset`, val **10k first-N truncation** — the loader stratifies only `train`, `stratify=False` for val) | `{q} Answer the question using a single word or phrase.` (note: space, not newline); greedy, `max_new_tokens=10` (default) | inline `vqa_consensus_score` = `min(matches/3,1)` | **Yes** (`generation.predictions`) | **Yes** (`question_id` + `image_id` + `raw_answers`) | Computed internally (`_get_token_stats`) but **not written to JSON** | No | **READY** | Low–Med — different prompt/`max_new_tokens` from GQA/TextVQA; also runs a (retired) classification head unless `--skip-classification` |
| Qwen-2.5-VL-7B | **GQA** | **No** | — (Qwen scripts only `load_dataset("lmms-lab/DocVQA"/"ChartQA")`) | — | — | — | — | — | — | **TODO** | High — no Qwen GQA loader/scorer adapter exists |
| Qwen-2.5-VL-7B | **TextVQA** | **No** | — | — | — | — | — | — | — | **TODO** | High — no Qwen TextVQA loader/scorer adapter |
| Qwen-2.5-VL-7B | **DocVQA** | **Yes (2 harnesses)** — `src/evaluation/docvqa/qwen25_dense_eval.py` (stock Qwen `generate`) **and** `qwen_kcurve.py --selectors full` (`QwenPruner`, manual greedy; keep-all==stock validated) | `datasets.load_dataset("lmms-lab/DocVQA","DocVQA","validation")` | `{q}` + optional ` {INSTR}` (`--instr` / `QwenPruner` INSTR=on); greedy, `max_new_tokens=32`; `max_pixels` configurable | `docvqa_score.anls` (τ=0.5) | **No** (aggregate only) | **No** | **Yes** (`avg_vis_tokens`) | No (separate `qwen_flops_summary.json`) | **PARTIAL** | Med — **no per-sample preds/IDs** (can't re-score/audit); `n=50` vs `n=200` between the two; `n=200` dense 97.19 > published 95.7 (small-n optimism) |
| Qwen-2.5-VL-7B | **VQAv2** | **No** | — | — | — | — | — | — | — | **TODO** | High — no Qwen VQAv2 loader/scorer adapter |

### Per-question answers (1–12), summarized
1. **Code exists?** Yes for 4 cells (LLaVA GQA/TextVQA/VQAv2; Qwen DocVQA); No for 4.
2. **Script path:** see column 3 above.
3. **Model loading:** LLaVA — `LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=fp16, attn_implementation="sdpa")` (in `run_dense_testdev.py`, `static.py`, `models/dense/llava_wrapper.py`). Qwen — `Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", bf16, sdpa)` (in `qwen25_dense_eval.py` / `qwen_pruner.py`).
4. **Dataset loader:** GQA inline `GQATestdevDataset`; TextVQA inline `TextVQADataset`; VQAv2 `src/data/vqav2/build_vqav2_dataset`; DocVQA `datasets.load_dataset` (Qwen) / `src/data/docvqa.py` (unused for LLaVA dense).
5. **Prompt:** see column 5 — **three different instruction styles** across the 4 working cells.
6. **Metric:** GQA exact (`official_score`), TextVQA M4C soft, VQAv2 consensus, DocVQA ANLS.
7. **Saves predictions:** GQA ✓, TextVQA ✓, VQAv2 ✓, Qwen-DocVQA ✗.
8. **Saves sample IDs:** GQA ✓, TextVQA ✓, VQAv2 ✓, Qwen-DocVQA ✗.
9. **Records tokens/seq-len:** only Qwen-DocVQA (avg visual tokens). LLaVA cells: no (576 implicit). No cell records text-token count or full sequence length in its output.
10. **Records FLOPs:** **none** — FLOPs are analytical, produced separately by `src/analysis/{flops,flops_vqav2,qwen_flops}.py`.
11. **Same subset as other methods?** GQA full testdev (shared with `run_static_testdev`) ✓; TextVQA full 5,000, deterministic file order (shared with static) ✓; VQAv2 10k deterministic first-N (the val loader truncates, not stratified; shared with `static_k*_pad`) ✓; Qwen-DocVQA first-N of validation — `qwen_kcurve` full+selectors share n=200 ✓, but `qwen25_dense_eval` uses n=50 (inconsistent).
12. **Final-scope readiness:** READY = LLaVA GQA/TextVQA/VQAv2; PARTIAL = Qwen DocVQA; TODO = LLaVA DocVQA + Qwen GQA/TextVQA/VQAv2.

---

## 2. Old dense result files (reference only — "do not trust blindly")

| Cell | Path | n | Metric | Score | Prompt (recoverable?) | Split/subset | Preds? | IDs? | Reuse verdict |
|---|---|---|---|---|---|---|---|---|---|
| **LLaVA-1.5 GQA** | `results/thesis_main/gqa/testdev_dense_honest_bs1_20260605_010314/` | **12,578** | GQA exact | **61.42%** (ref 62.0, −0.58) | Yes — `metrics.prompt_suffix="\nAnswer the question using a single word or phrase."`, image_pad=True, greedy mnt=64 | testdev_balanced (full) | **Yes** (`testdev_balanced_predictions.json` + `results.jsonl`) | **Yes** (`questionId`) | **Final-eligible** after a scorer parity re-check (full split, canonical scorer, per-sample saved) |
| **LLaVA-1.5 TextVQA** | `results/thesis_main/gqa/textvqa_dense_full_20260605_125126/` | **5,000** | M4C soft | **57.65%** OCR (ref 58.2, −0.55); no-OCR 46.73 is a separate dir | Yes — prompt stored per sample in `predictions.json` (incl. OCR block) | val (full), use_ocr=on | **Yes** (`predictions.json` + `per_sample_scores.json`) | **Yes** (`question_id`=image_id) | **Final-eligible** (full split, per-sample saved, prompt recoverable) |
| **LLaVA-1.5 VQAv2** | `results/thesis_main/vqav2/dense_pad/generation_eval_10k.json` | **10,000** | VQA consensus | **76.44%** | Partial — prompt not stored per row; reconstructable from `generate_and_score` (`{q} Answer…`, mnt=10) | val2014, **10k first-N truncation** (val loader does not stratify; seed unused for val) | **Yes** (`generation.predictions`: q_id+image_id+pred+raw_answers) | **Yes** | **REFERENCE ONLY** — the **final VQAv2 subset is LOCKED to 25,000 stratified validation questions (seed 42)**, stratified by the repo's 4 question-type buckets (`DENSE_DECISION_LOCK.md` #2); 10k was too small (and only truncated), so the final dense number will be **re-run on the 25k subset** (prompt/mnt also differ from GQA/TextVQA) |
| **Qwen DocVQA** | `results/thesis_main/highres/qwen25_dense_docvqa.json` (n=50) **and** `qwen_kcurve_docvqa.json` `full` (n=200) | **50 / 200** | ANLS | **94.85% (n=50)** / **97.19% (n=200)** (ref 95.7) | Yes — `{q}`+INSTR, mnt=32, max_pixels=1003520 (n=50 file) / 1280·28² (kcurve) | DocVQA validation, first-N | **No** (aggregate only) | **No** | **REFERENCE ONLY** — no per-sample preds/IDs, small & inconsistent n, n=200 dense **above** published (small-n optimism); **must re-run with predictions+IDs at larger n for a final number** |

---

## 3. Cross-cutting gaps to fix before the final dense phase (no code written yet — this is the checklist)

1. **4 missing cells need new harnesses/adapters:** LLaVA-1.5 × DocVQA (loader exists, wire it to a generation+ANLS harness); Qwen × {GQA, TextVQA, VQAv2} (new `load_bench` branches + scorer dispatch — the Qwen pruner/dense scripts only know DocVQA/ChartQA today).
2. **Standardize the dense protocol** across models so cells are comparable: pick one instruction string + one `max_new_tokens` policy (currently GQA/TextVQA use `\n…`/mnt=64, VQAv2 uses ` …`/mnt=10, Qwen uses optional-instr/mnt=32).
3. **Standardize saved artifacts** to one schema for every cell: `{sample_id, prompt, pred, gold(s), per_sample_score, n_visual_tokens, n_text_tokens, seq_len}` + an aggregate block. Today Qwen-DocVQA saves none of the per-sample fields.
4. **Record tokens + FLOPs in the run output** (not only separately): all LLaVA cells omit token counts; no cell writes FLOPs. Add the analytical FLOPs (`flops*.py`) into each dense result JSON so dense is the FLOPs reference for every method.
5. **Fix Qwen DocVQA dense provenance:** reconcile n=50 vs n=200, save per-sample predictions+IDs, and re-run at a larger n (the n=200 97.19 > published 95.7 is not safe to cite as the final dense ceiling).
6. **Subset manifests:** GQA = full testdev (deterministic); VQAv2 old 10k = val first-N truncation (the loader stratifies only `train`); TextVQA and Qwen rely on file order. For final, emit an explicit version-controlled `configs/final_scope/sample_ids/{dataset}.json` per dataset so dense and every pruning method are provably on the **same** samples — and the 25k VQAv2 manifest must apply the repo's question-type stratification to the val split explicitly.

## 4. Bottom line per cell (for `CLEAN_EXPERIMENT_MATRIX.md` alignment)
- **READY (reuse as final after a parity re-check + add tokens/FLOPs to output):** LLaVA-1.5 × {GQA 61.42, TextVQA 57.65/46.73, VQAv2 76.44}.
- **PARTIAL (re-run for per-sample artifacts + larger n):** Qwen × DocVQA (97.19 reference only).
- **TODO (build):** LLaVA-1.5 × DocVQA; Qwen × {GQA, TextVQA, VQAv2}.

*Source files read for this audit: `run_dense_testdev.py`, `run_textvqa.py`, `qwen25_dense_eval.py`,
`qwen_kcurve.py`, `generate_and_score.py`, `static.py`, `qwen_pruner.py`, `src/data/{vqav2,docvqa}.py`,
the four old result JSON/dirs above, and grep sweeps confirming Qwen has no GQA/TextVQA/VQAv2 loader and no
active LLaVA-1.5 DocVQA harness.*
