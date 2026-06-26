# Full Repository Understanding Report

*Read-only audit produced 2026-06-26 on branch `method-migration`. Nothing was moved, edited, deleted,
renamed, or run (no training/eval was launched). Every claim below traces to a file that was opened and
read, or to a directory whose size/schema/count was inspected directly. Where a number comes from the
project's own docs (the evidence ledger / prior audits) rather than from re-deriving it, that is stated.*

This document is a standalone map of the whole repository: what it is, how the pipeline fits together,
folder by folder, the configs, the methods, the datasets, the training/eval flow, the metrics, the
existing evidence, and an honest read on readiness, risks, and next steps. It complements (does not
replace) the three authoritative docs: `THESIS_MASTER_PLAN.md` (story), `THESIS_EVIDENCE_LEDGER.md`
(every number), and the two prior read-only audits (`THESIS_AUDIT.md`, `THESIS_VERIFICATION.md`).

---

## A. Repository Purpose

This is a **Master's thesis codebase** on **efficient Vision–Language Models (VLMs)** via **visual-token
pruning**. VLMs spend most of their prefill compute on visual tokens (576 for LLaVA-1.5 @336px; hundreds
to a few thousand for high-resolution / Qwen models), and self-attention is quadratic in sequence length,
so cutting visual tokens cuts cost super-linearly.

A pruner makes two decisions per sample:
1. **Selection** — *which* visual tokens to keep, and
2. **Budget** — *how many* to keep per sample (a dynamic, per-input token count).

**The thesis finding is "Selection over Budget":** *which* tokens you keep — chosen by a
**question-conditioned, mid-layer** signal — is the dominant efficiency lever, while *how many* per
sample is essentially a mirage (its honest, noise-corrected, FLOPs-matched headroom is small and a
trained predictor captures almost none of it).

Crucially, the thesis is an **honest-measurement / negative-results-aware diagnostic paper, not a SOTA
method paper.** It explicitly does *not* claim to beat the state of the art, does *not* train the backbone
(all headline results are on **frozen** models), and packages a deployable-but-below-SOTA distilled
selector as the "method" payoff. The registered topic was *Dynamic, Question-Conditioned Visual Token
Pruning for Efficient VLMs*; the work pivoted from the original hypothesis ("dynamic budget beats static")
to a rigorous decomposition that lands on selection.

Backbones used (all frozen at evaluation): **LLaVA-1.5-7B** (low-res, 576 tokens), **LLaVA-1.6-7B**
(AnyRes high-res, ~2302 tokens), **Qwen-2.5-VL-7B/3B/32B** (native dynamic resolution).

---

## B. High-Level Architecture (the full pipeline)

The project is organized **by method** under `src/`, run as `python -m src.<...>` from the repo root.
The pipeline, end-to-end:

### B.1 Datasets & preprocessing
- **VQAv2** (LLaVA-1.5 budget track): COCO images, `expand2square` padding to a square with CLIP-mean
  background (`src/data/vqav2/image_transforms.py`), then the HF LLaVA processor. Stratified subsampling
  across 4 heuristic question types (yes/no, attribute, counting, spatial) in
  `src/data/vqav2/vqav2.py`. Answers normalized lightly; classification labels via an answer vocab.
- **GQA** (LLaVA-1.5 diagnostic): testdev_balanced (12,578) / val_balanced loaders in `src/data/gqa.py`,
  with semantic-type metadata (obj/attr/rel/cat/global) and an optional region-supervision variant.
- **DocVQA / ChartQA / InfoVQA** (high-res / Qwen): loaded from the HuggingFace hub at runtime
  (`src/data/docvqa.py` plus `datasets.load_dataset` calls inside the Qwen scripts).
- **TextVQA, POPE, ScienceQA-IMG**: local files used by the LLaVA-1.5 eval harnesses.
- **LLaVA-665K instruction mix** (`src/data/llava_mix.py`): only used by the *excluded* elastic Stage-1
  training; source-stratified, first-turn (q,a) extraction.

### B.2 Two regimes (the spine of the thesis)
The same diagnostic instrument is applied across two regimes — **this is the whole argument**, and the
axis separating them is *resolution × task token-demand × which-layer signal*, **not training**:

| Regime | Model(s) | Tokens | Role | Finding |
|---|---|---|---|---|
| **Frozen low-res diagnostic ("v1")** | LLaVA-1.5-7B | 576 | builds the measurement framework | budget ties static (L2); naive Q-cond selection fails (L4) |
| **High-res / SOTA main result ("v2")** | LLaVA-1.6, Qwen-2.5-VL | ~2302 / hundreds–thousands | the headline | selection dominates (L5–L7); budget still a mirage (L8–L9); cheap student (L11–L12) |

### B.3 Method components
1. **Dense baseline** — full visual tokens, frozen backbone. Two flavors: a classification answer-head
   proxy (`src/models/dense/`, *retired*) and the canonical generation-protocol path.
2. **Static pruning** — training-free physical token removal at fixed K, scored by CLIP CLS→patch
   attention (VisionZip "dominant"), with `random`/`spatial_uniform`/`l2_norm`/`fastv_style` baselines and
   a faithful VisionZip (dominant + contextual merge). Core class: `StaticPrunedLlava`
   (`src/models/static/static.py`).
3. **Dynamic budget** — a learned per-sample/per-type budget controller on top of CLS ranking
   (`src/models/dynamic_budget/`), trained with `train_dynamic.py`. The honest result: it ties static.
4. **Question-conditioned selection** — score visual tokens by relevance to the question. On LLaVA-1.5
   via LM-attention (`question_cond.py`) or CLIP-space cosine (`clip_select.py`) — both *fail* on frozen
   low-res. On Qwen via **mid-layer (L16) question→visual attention** (`qwen_pruner.py`) — this *dominates*.
5. **Budget oracle + predictor** — `qwen_oracle.py` / `qwen_oracle_qc.py` (monotone noise-corrected
   oracle), `qwen_budget_data.py` (feature collection), `qwen_budget_eval.py` (5-fold-CV predictor),
   `qwen_budget_robust.py` (hard-tail AUC). The "mirage" evidence.
6. **Distillation** — cache the expensive L16 teacher map (`cache_teacher.py`), distil a cheap
   no-decoder-forward student (`CheapQCSelector`, `train_student.py`), gate on held-out DocVQA.
7. **Confidence cascade** — a realizable budget mechanism (run base-K, escalate low-confidence samples);
   `cascade_pass.py` / `cascade_sweep.py`. It only *traces* the static frontier.
8. **Elastic Stage-1** — LoRA + projector trained with random-K CLS pruning (`elastic_wrapper.py`).
   *Excluded* from the thesis (the only "trained backbone" piece; archived).

### B.4 Training
- **Cached answer-head trainer** (`train_cached.py`): frozen backbone → cache pooled features once →
  train only the MLP head. The retired classification proxy.
- **Dynamic trainer** (`train_dynamic.py`): trains selector + budget controller + head on a frozen LLM.
- **Student distillation** (`cache_teacher.py` → `train_student.py`): KL + top-K BCE to the teacher map.
- **Elastic Stage-1** (`train_stage1.py`): teacher-forced answer-token LM loss, LoRA+projector only.

### B.5 Evaluation & metrics
The **canonical protocol is generation** (`model.generate()`, answer head bypassed), scored by the
official scorer per benchmark (`src/metrics/`): GQA `strip().rstrip('.').lower()` exact match, TextVQA
M4C soft-acc, POPE accuracy/F1, DocVQA ANLS, ChartQA relaxed accuracy, VQAv2 consensus min(matches/3,1).
The "locked honest protocol" = expand2square padding, official prompt+scorer, greedy, bs=1, no
min-new-tokens / no repetition penalty.

### B.6 Analysis / results generation
FLOPs (`src/analysis/flops.py`, `flops_vqav2.py`, `qwen_flops.py` — FastV Eq.5 prefill, prune-before-LLM),
oracle-noise decomposition (`oracle_decomposition.py`), instance headroom (`instance_headroom.py`),
cascade sweep (`cascade_sweep.py`), latency (`llava_latency.py`), figure makers. Results land in
`results/` (git-ignored); every number is registered in `THESIS_EVIDENCE_LEDGER.md`.

---

## C. Folder-by-Folder Understanding

```
vlm-thesis/
  README.md  requirements.txt  .gitignore
  configs/   data/  docs/  experiments/  results/  scripts/  src/
```

The repo was reorganized on 2026-06-24 (branch `method-migration`) from three historical track folders
(`GQA/`, `VQA_V2/`, `v2/`) + `outputs/` into a single method-based `src/` tree. The "v1/v2" labels now
survive **only** in the docs as historical shorthand.

### `src/` — all source code, by method (130 `.py`)
- **`src/data/`** — loaders. `vqav2/` (dataset + collator + `image_transforms` expand2square +
  `vqav2_answers` normalization), `gqa.py` (val/train/region-supervised + collators), `docvqa.py`
  (hub validation/test + direct parquet train shards), `llava_mix.py` (665K mix, pure-parsing +
  dataset). Matters: this is where preprocessing decisions (padding, stratification) live.
- **`src/metrics/`** — the scorers, one canonical copy each: `official_score.py` (GQA),
  `metrics.py` (GQA per-type + `extract_short_answer`), `textvqa_score.py` + `m4c_evaluator.py`,
  `pope_score.py` + `eval_pope_official.py`, `docvqa_score.py` (ANLS), `chartqa_score.py` (relaxed).
  Matters: these define "correct" — the integrity of every headline number.
- **`src/models/`** — `dense/`, `static/`, `dynamic_budget/`, `distillation/`, `elastic/`. Each has its
  own wrapper + (where relevant) selector + answer head. The two most load-bearing files are
  `static/static.py` (`StaticPrunedLlava`, training-free physical removal — the engine for GQA/TextVQA/
  POPE/SQA static + the parent of the QC/VisionZip probes) and `dynamic_budget/*` (the budget controller).
- **`src/pruning/`** — the selection methods. `static/visionzip.py`, `dynamic_budget/` (Qwen budget
  oracle/data/eval/robust + LLaVA budget data), `question_conditioned_selection/` (`qwen_pruner.py` =
  the Qwen prune-before-LLM harness with the mid-layer teacher; `question_cond.py` = LLaVA LM-attention
  probe; `clip_select.py` = CLIP-space probe).
- **`src/evaluation/`** — per-task entrypoints: `gqa/` (dense/static/visionzip/qcond/clip/speculative
  testdev runners), `vqa/` (generate_and_score = **the** canonical evaluator, cascade, instance
  headroom, per-type), `textvqa/` (incl. `HighResPruner` for LLaVA-1.6), `docvqa/` (Qwen kcurve/control/
  layer-sweep, gate/control distillation eval), `pope/`, `scienceqa/`.
- **`src/analysis/`** — FLOPs, latency, oracle decomposition, cascade sweep, FLOPs frontier (embeds the
  measured high-res numbers directly), figure makers.
- **`src/training/`** — `train_cached`, `train_dynamic`, `train_student`, `cache_teacher`, `train_stage1`.
- **`src/utils/`** — `config` (YAML + `_base_` inheritance), `seed`, `logger`, `io`, `checkpoint`,
  `device`. Single shared copy.

### `configs/` — method-based YAML (27 files)
`dense/` (2: 150k/443k), `static/` (21: K∈{64,96,128,144,160,192,219,255,265,275,276,288,334,357,432}
for the 150k set + a 6-K subset for 443k), `dynamic_budget/` (3: full 150k, gate_2k, gate_smoke),
`stage1_elastic.yaml`. Dataset paths point at the flat git-ignored `data/`; output dirs were repointed to
`results/`.

### `scripts/` — runnable launchers
`scripts/data/` (download, feature caching `cache_features.py`, vocab build, Qwen downloads, per-K cache
`.sh`), `scripts/training/` (`.sh` launchers). The README warns that some legacy `.sh` files still
reference the old run layout internally and should be reviewed before re-running.

### `experiments/` — thin index only (README), no code. Points back to `src.*` modules.

### `docs/` — thesis documentation
Active set: `THESIS_MASTER_PLAN.md`, `THESIS_EVIDENCE_LEDGER.md`, `PAPER_PUBLICATION_PLAN.md`,
`THESIS_AUDIT.md`, `THESIS_VERIFICATION.md`, `FINAL_REPOSITORY_CLEANUP_REPORT.md`,
`REPOSITORY_CLEANUP_LOG.md`. Subfolders: `literature/` (related work), `source_findings/` (per-track
FINDINGS/PAPER drafts — raw material for the paper), `migration_history/` (superseded plans, kept for
provenance). **Note:** `docs/` is git-ignored (private research material).

### `results/` — experiment results (git-ignored; only README + INDEX tracked, 681M, 302 files)
`thesis_main/{gqa,vqav2,highres}/` (headline evidence by track), `paper_candidates/` (raw inputs for the
L10 recompute), `archived/` (elastic Stage-1 checkpoints). Main-vs-appendix is carried by the **ledger's
Placement column**, not the folder.

### `data/` — datasets (git-ignored, flat, ~81G)
`vqav2/` (20G), `gqa/` (21G), `llava_mix/` (34G), `textvqa/` (6.7G), `scienceqa/` (125M), `pope/` (1.1M),
`budget_oracle/` (24M — diagnostic JSONs from the retired phase). See §F.

---

## D. Config Understanding

All configs share the same skeleton: `dataset`, `model` (frozen LLaVA-1.5-7B, fp16, sdpa/eager,
`vision_feature_layer=-2`), `token_selection`, `training`, `optimizer` (AdamW), `scheduler` (cosine +
warmup), `evaluation`, `system`, `logging`. Common to all: backbone fully frozen
(`freeze_vision_encoder/projector/llm: true`), trained head/selector only, `image_aspect_ratio: pad`,
answer vocab `./data/vqav2/answer_vocab_full.json` (**see §K — this file is missing on disk**).

### dense (`configs/dense/`)
`token_selection.mode: dense` (no removal). 150k vs 443k differ only in `max_samples` (150000 vs full).
`attn_implementation: sdpa`. These drive the retired classification answer-head proxy; the canonical
dense number comes from generation eval, not these heads.

### static (`configs/static/`)
`token_selection.mode: static_cls_attention`, `keep_tokens: K`, `attn_implementation: eager` (CLIP
attention maps must be exposed). 21 K-values for the 150k set; the extra fine-grained K's
(96,160,219,265,275,276,334,357) exist for the **per-type / matched-K instance-headroom** analysis
(`instance_headroom.py`). These configs feed the classification/cached pipeline; the headline static
frontier comes from the generation harnesses (`StaticPrunedLlava`).

### dynamic_budget (`configs/dynamic_budget/`)
`token_selection.mode: dynamic` with the key knobs: `scoring_mode` (`cls_only` | `learned_only` |
`cls_prior`), `budget_strategy` (`fixed` | `learned`), `min/max_keep_tokens` (64..432, spanning the
static curve for matched-avg-K), `train/eval_selection_mode` (soft/hard). The **headline config**
(`llava_dynamic_150k_10k_fullvocab.yaml`) uses `scoring_mode: cls_only` — i.e. token *ranking is plain
CLS*, and the only learned thing is the per-type **budget** (`budget_loss_type: question_type_target`,
target ratios `[0.38,0.48,0.58,0.62]` → K targets 219/276/334/357). This deliberately isolates the budget
effect → the "+0.05pp ties static" result (L2). The `gate_2k`/`gate_smoke` configs are small-data
smoke tests; `gate_2k` notably bumps `question_type_emb_dim` 16→64 to fix a smoke failure where the
type signal was drowned by question memorization. Two documented bugs are fixed in the wrapper:
`budget_loss_type` key handling and an all-ignored-CE NaN guard.

### stage1_elastic (`configs/stage1_elastic.yaml`)
The *only* training-the-backbone config: LoRA (r=16, α=32) on the Llama attention+MLP projections +
trainable projector, random-K ladder `[32,64,128,256,576]`, bf16, gradient checkpointing, LLaVA-665K mix.
**Excluded** from the thesis. A separate `data:`/`model:`/`training:` schema (not the dataset/model/...
skeleton of the others).

**Key cross-config differences:** dense vs static = removal mode + sdpa vs eager; static vs dynamic =
fixed K vs learned budget controller; all VQAv2 configs share frozen backbone + expand2square + full-vocab
classification; stage1 is the odd-one-out (generation LM loss, trainable LoRA).

---

## E. Model and Method Understanding

### E.1 Dense path
`LlavaDenseVQAModel` (`src/models/dense/llava_wrapper.py`): frozen LLaVA-1.5, processor with chat
template ("...Answer the question using a single word or phrase."), two modes — generation (sanity) and
classification (pool last valid hidden state → `AnswerHeadMLP` → vocab logits, CE with `ignore_index=-1`).
`DenseTokenSelector` is a no-op reporting token stats. The canonical dense **number** comes from running
`generate()` (answer head bypassed), not from the head.

### E.2 Static CLS-attention selection (the strong, training-free baseline)
`StaticPrunedLlava` (`src/models/static/static.py`) is the engine. Pipeline: CLIP vision tower forward →
score 576 patches → keep top-K → **physically remove** the dropped tokens by rebuilding
`inputs_embeds = [prefix | K projected visual | suffix]` → `backbone.generate(inputs_embeds=...)`. Methods:
`none` (K=576 sanity, must reproduce ~67.7% dense), `cls_attn` (CLS→patch attention at CLIP layer −2,
head-averaged — VisionZip "dominant"; the strong baseline), `l2_norm`, `random` (per-sample seeded),
`spatial_uniform` (deterministic 24×24 stride sets), `fastv_style` (LM-layer-2 received attention, eager;
included to demonstrate RoPE position bias). Honest-protocol knobs: `image_pad` (square), `honest` suffix
(LLaVA official) + `rstrip('.').lower()` post-processing. A separate classification wrapper
(`static/llava_wrapper.py`, `LlavaStaticVQAModel`) carries an answer head for the retired proxy and
re-sorts selected tokens into spatial order at the `<image>` position (the layout LLaVA-1.5 was trained
with).

Position-bias rationale is documented in the file: CLIP-side scoring (cls_attn, l2_norm) uses learned 2D
position embeddings → content scores; LM-side scoring (fastv_style) uses RoPE → raster-scan bottom bias.

### E.3 Dynamic budget / gating
`DynamicTokenSelector` (`src/models/dynamic_budget/token_selector.py`) = `QuestionConditionedTokenScorer`
(scores each patch by `[v, q, v·q, |v−q|]` against a mean-pooled question embedding) +
`BudgetController` (MLP over `question_projected` + 7 score-stats + optional 4-type embedding →
sigmoid → keep-ratio → K in [min,max]). Scoring modes: `cls_only` (rank by CLS — used for the headline
"isolate budget" run), `learned_only`, `cls_prior` (CLS + α·learned). Selection: `soft` (sigmoid weights,
training) / `hard` (top-K, eval). Losses (`dynamic_budget/llava_wrapper.py`): CE + `budget_loss`
(target / upper_bound / **question_type_target** = per-sample MSE to per-type ratio / none) +
entropy + budget_diversity (= −std(K), inactive at bs=1). The headline result trains the per-type budget
on CLS ranking and ties static (L2).

### E.4 Question-conditioned selection (the win, and the earlier failures)
- **Qwen mid-layer teacher** (`qwen_pruner.py`): one full decoder forward with `output_attentions`,
  take question→visual attention at layer 16, mean over heads/query-rows → per-token score → top-K, then
  prune (keep tokens **and their M-RoPE positions**) and greedy-decode. `selector ∈ {full, norm, uniform,
  textattn}`. `full` reproduces stock `generate()` exactly (correctness gate). **This is the headline
  selector — but it needs a full LLM forward to score (E4: an analysis/upper-bound, not deployable).**
- **LLaVA-1.5 LM-attention probe** (`question_cond.py`): subclasses `StaticPrunedLlava`, scores
  question→visual attention at LLM layers {2,5,8}. **Loses to CLS** on frozen low-res (L4).
- **LLaVA-1.5 CLIP-space probe** (`clip_select.py`): cosine(CLIP visual-projection patch, CLIP text
  feature), with a padding-patch mask fairness fix. **Fails badly** (L4).
- **High-res** (`HighResPruner` in `evaluate_textvqa_highres_kcurve.py`): LLaVA-1.6 AnyRes, prune the
  packed sequence; selectors `full`/`base`/`uniform`/`attn`(CLS)/`textattn`(question, early layer 3).

### E.5 Cache-based MLP / student training
- **Cached answer head** (`train_cached.py`): reads `pooled_features.npy` (frozen-backbone last hidden
  at the answer position) and trains only the `AnswerHeadMLP` (input_dim=4096). One epoch in <30 min.
- **Student distillation** (`cache_teacher.py` → `train_student.py`): cache the L16 teacher map once;
  `CheapQCSelector` (cross-attention head, ~8–20M params, **pre-LLM features only, no decoder forward**)
  is trained with listwise KL(teacher‖student) + top-K BCE. Feature cache on disk so epochs 2+ are ~2 min.

### E.6 FLOPs / latency
`src/analysis/flops.py` — FastV Eq.5 per-layer `4nD² + 2n²D + 2nDM` summed over T=32 layers, with
n = K + n_text; "attention-only" = `2·T·n²·D`. Per-benchmark `n_text` constants are measured
(testdev=34, TextVQA-OCR=86/noOCR=32, POPE=21, SQA=108). Prune-before-LLM means all 32 layers see K, so
static and QC at the same K have **identical** FLOPs → the QC>blind gap is a clean matched-FLOPs gap.
`flops_vqav2.py` (n_text=35) and `qwen_flops.py` (T=28, d=3584, m=18944, n_text=40) are the per-model
variants. **FLOPs are LLM-prefill only** — they exclude the CLIP/ViT encoder, the projector, decode, and
critically the QC teacher's own scoring forward (E4). Latency (`llava_latency.py`) is measured on
LLaVA-1.6, decode-only, n=40.

### E.7 How answers/metrics are computed
Generation → official scorer (§B.5). The retired classification path uses the answer head argmax → vocab
string → VQA consensus. The thesis uses **only** the generation numbers.

---

## F. Dataset Understanding

| Dataset | On disk | Files present | Schema / how used | Notes |
|---|---|---|---|---|
| **VQAv2** | 20G | `v2_OpenEnded_mscoco_{train,val}2014_questions.json` (214,354 val Q), `v2_mscoco_{train,val}2014_annotations.json` (214,354 val ann), `answer_vocab_top3500.json`, COCO `train2014/` (82,783 jpg) + `val2014/` (40,504 jpg) | questions = `{image_id, question, question_id}`; annotations carry `multiple_choice_answer` + `question_type`; loader stratifies by 4 heuristic types, expand2square, 10k val subset | **`answer_vocab_full.json` is MISSING** (configs reference it; regenerable via `scripts/data/build_answer_vocab_full.py`). Only the retired classification path needs it. |
| **GQA** | 21G | `testdev_balanced_questions.json` (12,578), `val_balanced_questions.json` (113MB), `images/images/` (148,854 jpg), `readme.txt` | per-Q dict with `question, answer, imageId, types{semantic,structural}`; semantic types → per-type tables | testdev is canonical; full-split eval |
| **DocVQA / ChartQA / InfoVQA** | (hub) | none local (loaded via `datasets.load_dataset`) | `{image, question, answers/answer}`; DocVQA train via direct parquet shards (`docvqa.py`) | the high-res headline datasets; pilot leakage-guarded by disjoint val slices |
| **TextVQA** | 6.7G | `TextVQA_0.5.1_val.json`, `TextVQA_Rosetta_OCR_v0.2_val.json`, `llava_textvqa_val_v051_ocr.jsonl`, `train_images/` (25,119 jpg) | OCR and no-OCR variants, M4C soft-acc; n=5000 | the "naive Q-cond selection fails" dataset (L4) |
| **POPE** | 1.1M | `coco/coco_pope_{random,popular,adversarial}.json` | JSONL: `{question_id, image, text, label}` yes/no; uses COCO val2014 images | accuracy/F1; appendix dense-reproduction + band |
| **ScienceQA** | 125M | `sqa_img_test.parquet` (130MB) | parquet via pyarrow; CQM-A prompt | appendix only |
| **LLaVA mix** | 34G | `llava_v1_5_mix665k.json` (1.03GB), `images/{coco,gqa,ocr_vqa,textvqa,vg}` (226,532 jpg) | conversation records; first-turn (q,a) | **only the excluded elastic Stage-1** uses it |
| **budget_oracle** | 24M | binary-routing / qtype-oracle / stage6c / exp2–5 diagnostics + `val_budget_*` labels/features | retired VQAv2 budget-predictor diagnostics | **excluded; generator code not in repo** (per audit) — historical evidence only |

**Likely assumptions / missing pieces:** `answer_vocab_full.json` absent (see §K); DocVQA/ChartQA/InfoVQA
require network + HF cache at run time; the high-res numbers depend on Qwen/LLaVA-1.6 weights downloaded
via `scripts/data/`.

---

## G. Training and Execution Flow

All runs are `python -m src.<module>` from the repo root. The key flows:

**Retired classification proxy (VQAv2):**
1. `scripts/data/cache_features.py --model-type {dense,static} --keep-tokens K ...` → caches pooled
   features to `feature_cache/`.
2. `python -m src.training.train_cached --train-cache ... --val-cache ... --config configs/...` → trains
   the `AnswerHeadMLP`.
3. `python -m src.evaluation.vqa.generate_and_score --config ... --checkpoint ... --model-type {dense,
   static,dynamic}` → the **canonical** generation eval (this is what the thesis cites).

**Dynamic budget (VQAv2):**
`python -m src.training.train_dynamic --config configs/dynamic_budget/llava_dynamic_150k_10k_fullvocab.yaml
--output-dir results/...` → trains selector+budget+head, reports per-epoch K-distribution + a small
generation-accuracy subset (the PRIMARY metric). Headline = `dynamic_150k_clsonly/RESULT_SUMMARY.json`.

**Static / QC / cascade (GQA, TextVQA, POPE, SQA — LLaVA-1.5):**
`python -m src.evaluation.gqa.run_static --method cls_attn --keep_k 288 ...` (and
`run_static_testdev`, `run_visionzip_testdev`, `run_qcond_probe`, `run_clip_probe`,
`run_speculative_testdev`); analogous `run_textvqa`, `run_pope`, `run_sqa`. All subclass/share
`StaticPrunedLlava` and the locked honest protocol; resumable via `predictions_partial.json`.

**High-res / Qwen (the headline):**
- Selection K-curve: `python -m src.evaluation.docvqa.qwen_kcurve` (and `qwen_control`,
  `qwen_layer_sweep`, `qwen25_dense_eval`).
- Budget mirage: `qwen_oracle` / `qwen_oracle_qc` → `qwen_budget_data` → `qwen_budget_eval` /
  `qwen_budget_robust`.
- LLaVA-1.6: `evaluate_textvqa_highres*`, `evaluate_highres_docchart`, `evaluate_gqa_highres`.
- Distillation: `cache_teacher` → `train_student` → `eval_gate` / `eval_control`.

**Excluded:** `python -m src.training.train_stage1 --config configs/stage1_elastic.yaml` (elastic LoRA).

**Dependencies / environment:** two conda envs are implied — `vlm_env` (torch 2.3.0+cu121,
transformers 4.46.3 — result-critical pins) for LLaVA tracks; a separate `qwen_env` (transformers 4.51,
`qwen_vl_utils`) for the Qwen scripts (their docstrings hardcode `~/miniconda3/envs/qwen_env/bin/python`).
Hardware: 2× RTX 6000 Ada (48GB), per the memory notes.

---

## H. Evaluation and Metrics (all implemented scorers)

| Metric | File | Definition | Used by |
|---|---|---|---|
| GQA exact match | `metrics.py`, `official_score.py` | `strip().rstrip('.').lower()` strict equality; per-type obj/attr/rel/cat/global | GQA static/dense/QC/cascade |
| VQA consensus | `vqav2_answers.py` + inline in `generate_and_score.py`/`train_cached.py` | `min(matches/3, 1)` over normalized raw answers | VQAv2 generation eval |
| TextVQA M4C soft-acc | `textvqa_score.py`, `m4c_evaluator.py` | M4C answer processing + averaged soft accuracy | TextVQA |
| POPE | `pope_score.py`, `eval_pope_official.py` | yes/no accuracy + F1 over random/popular/adversarial | POPE |
| DocVQA ANLS | `docvqa_score.py` | max over golds of `1 − lev/max(len)`, zeroed below τ=0.5 | DocVQA, Qwen budget |
| ChartQA relaxed | `chartqa_score.py` | numeric within 5% rel. tol., else exact text match | ChartQA |
| `extract_short_answer` | `metrics.py` | heuristic 1–3 word extraction (legacy, non-honest path) | legacy / non-honest generation |

Efficiency metrics (kept **separate**, never conflated — see ledger R3): token reduction (measured from
avg_tokens), FLOP reduction (analytical prefill), measured latency. Oracle/headroom analyses:
`instance_headroom.py` (Lagrangian sweep oracle frontier), `oracle_decomposition.py` /
`qwen_oracle*.py` (naive vs **monotone** noise-corrected oracle), `cascade_sweep.py` (confidence cascade
frontier).

---

## I. Existing Results and Evidence (what is saved, what it supports)

All numbers below are **registered in `THESIS_EVIDENCE_LEDGER.md`** and were spot-verified against the
saved JSONs during this audit. Ledger IDs in brackets.

**Frozen low-res diagnostic (LLaVA-1.5):**
- **[L1] Dense reproduces published** — GQA 61.42/62.0, TextVQA-OCR 57.65/58.2, POPE-F1 85.78/85.9,
  SQA 65.15/66.8 (`results_frozen/all.json`). The trust anchor.
- **[L2] Budget ties static** — dynamic 75.76 vs static-K265 75.71 = **+0.05pp**, avg K=264.3, VQAv2 10k
  (`dynamic_150k_clsonly/RESULT_SUMMARY.json`). Mechanically works (held-out K-std 44.9, correct per-type
  ordering) but Jensen caps the gain.
- **[L3] Thin oracle band** — first-correct-K band 4.93–9.13% across POPE/TextVQA/SQA/GQA
  (`all.json` "bands"). Static frontier (K=64..432) present on all four benchmarks.
- **[L4] Naive frozen Q-cond selection fails** — CLIP-Qcond −32.04pp, LM-Qcond −5.58pp vs CLS at K=64
  TextVQA-noOCR (`week1_all_numbers.json`).

**High-res / Qwen main result:**
- **[L5] Selection dominates** — Qwen DocVQA K=128: textattn 92.18 vs uniform 32.19 = **+59.99pp**
  (`qwen_kcurve_docvqa.json`, n=200). ChartQA K=128 QC 83.5 vs uniform 61.0.
- **[L6] Mid-layer signal** — L2 44.6 → L16 93.2 → L20 94.9 (`qwen_layer_sweep.json`).
- **[L7] Genuinely question-conditioned** — real 93.84 / mismatched 36.98 / blind 32.19 → 92.2%
  question-driven (`qwen_control_docvqa.json`).
- **[L8] Small honest oracle** — monotone FLOPs-matched **+7.50pp** DocVQA / +2.66pp ChartQA
  (`qwen_oracle_docvqa_qc.json`).
- **[L9] Predictor captures ≈0** — oracle 4.99 → MLP +0.09pp (1.8%), hard-tail AUC 0.643
  (`qwen_budget_eval.json`, `qwen_budget_robust.json`).
- **[L11/L12] Cheap student** — dense 95.0 / teacher 91.1 / student 65.9 / blind 31.2 → **57.9% recovery**
  (PASS), 99.2% question-driven, data>epochs (`distill/gate_12k.json`, `control_12k.json`).
- **[E1–E3] Efficiency** — 86.8% token / 87.9% FLOP reduction @K128, 3.31× latency (separate quantities).

**Paper candidates / not-yet-results:**
- **[L10] Cross-model generality** — **narrative-only**: raw data exists
  (`paper_candidates/qwen_budget_data_{docvqa_3b,docvqa_32b,infovqa,docvqa_7b_n1000}.json`) but the
  per-model eval gains are **not saved in any JSON**. Must be recomputed before writing.

**Archived/excluded:** `results/archived/stage1_*` (elastic checkpoints), `data/budget_oracle/*` (retired
VQAv2 budget diagnostics — generator code not in repo).

---

## J. Thesis / Paper Readiness

**Already strong (artifact-backed, thesis-ready now):**
- The full diagnostic foundation: L1 (dense reproduction), L2 (budget ties static), L3 (thin band),
  L4 (naive selection fails) — all on frozen LLaVA-1.5, full splits.
- The honest measurement methodology: the oracle-headroom band + **monotone oracle-noise correction** +
  matched-FLOPs win criterion. This *is* the methodological contribution and it is fully implemented.
- The Qwen budget mirage on DocVQA-7B (L8, L9) with paired oracle/predictor and the hard-tail AUC.
- A coherent end-to-end "diagnose → decompose → build" arc culminating in the distilled student (L11–L12).
- The repo itself is clean, compiles, imports, and is organized to write from (per the cleanup report:
  compileall exit 0, 0 stale imports, ledger paths 22/22 resolve).

**Incomplete / needs work before paper (not blocking a thesis):**
- Headline cells L5/L7/L9 are **n=100–300** — need full-validation reruns.
- L5 baselines are **blind floors** (uniform/norm), not strong published methods; no FEATHER/CDPruner/
  HiRED/VisionSelector or faithful layer-split FastV is implemented.
- **L10 generality is narrative-only** — the single highest-leverage cheap fix (raw data already exists;
  rerun `qwen_budget_eval.py` per model and save per-model JSONs).
- The student is DocVQA-only (~69% retention, below SOTA ~90%).

**Risky claims the docs themselves flag (do not write unqualified):**
- "+60pp selection win" is measured vs the *weakest* baseline (uniform); a fair CLS-equivalent Qwen
  baseline would likely shrink it toward +20–30pp (`THESIS_VERIFICATION.md` §4.3). **Qwen has no
  CLS-attention baseline implemented.**
- "+7.50pp oracle headroom" is a **FLOPs-matched** number; the pure-accuracy ceiling over the best fixed
  K is only **+0.81pp** (`THESIS_VERIFICATION.md` §1.4). State the matched-budget qualifier.
- "training fixed it" — false; all headline wins are frozen (R5).
- "matches/beats SOTA" — no strong baseline was run.

**Bottom line (from the project's own docs, corroborated here):** research is ~85% done and honest; the
deliverable is mostly **writing** plus one focused round of full-scale numbers/baselines for the paper.
The realistic paper is an **honest measurement/analysis paper** (workshop/short now; ACL/EMNLP Findings or
TMLR after the gaps close), not a SOTA-method paper.

---

## K. Bugs, Risks, or Inconsistencies (found in this audit)

**Concrete code/data gaps:**
1. **`data/vqav2/answer_vocab_full.json` is missing** but is referenced by ~16 configs
   (`dense/*`, `static/*`, `dynamic_budget/llava_dynamic_150k_*`). Any run of the classification/cached
   head pipeline or `LlavaDynamicVQAModel` will fail at vocab load until it is rebuilt
   (`scripts/data/build_answer_vocab_full.py`). Low impact (the classification proxy is retired and the
   headline numbers use generation), but the configs are not runnable as-shipped.
2. **`make_output_dir(base_dir="outputs", ...)`** in `run_static.py` (and likely siblings) writes to a
   top-level `outputs/` that the migration removed — saved results were later moved to
   `results/thesis_main/gqa/`. Re-running these GQA harnesses will recreate `outputs/` (git-ignored) rather
   than write under `results/`. Cosmetic but a path inconsistency vs the new layout.
3. **Two FLOPs calculators, two conventions.** `flops.py` (GQA/TextVQA/POPE) reports FastV-full TFLOPs;
   the saved VQAv2 baseline reports attention-only GFLOPs; `qwen_flops.py` uses Qwen constants. Internally
   consistent per dataset, but as-saved headline FLOPs are **not directly comparable across datasets**
   (audit F4/§3). VQAv2 needs re-extraction under one convention for a cross-dataset frontier.
4. **`distillation_loss_weight` in the dynamic config is a no-op** (commented in the config: not consumed
   by `forward()`). `budget_diversity_weight` is inactive at bs=1 (std of a scalar = 0). Both are
   documented in-line, but a reader could mistake them for active.

**Methodology caveats the project documents itself (must be carried into any writeup):**
5. **The QC teacher selector is not FLOPs-cheap** — it needs a full LLM forward to score (E4); the
   reported FLOP reductions exclude it. The deployable path is the (below-SOTA) student.
6. **Cross-model baseline asymmetry** — Qwen QC vs uniform (+60pp) vs LLaVA-1.6 QC vs CLS (+27pp) vs
   LLaVA-1.5 QC vs CLS (≈tie/negative). The +60pp is partly a weak-baseline artifact.
7. **Hard-tail AUC wording** — saved `qwen_budget_robust.json` = **0.643**, which crosses the script's own
   0.6 "might help" threshold, yet the FINDINGS narrative rounds to "0.59 ≈ unpredictable." The conclusion
   (realized gain ≈0) holds, but the *mechanism* phrasing should be "weakly identifiable but unfixable,"
   not "unpredictable."
8. **Latency claim provenance** — `llava_latency.json` (3.31×) is **LLaVA-1.6, n=40, decode-only**, and
   its "kept" tokens are `torch.arange(first-K)` (sequence-length curve, **not the real selector**). It is
   a different model from the accuracy/FLOPs headline (ledger R3).
9. **LLaVA-1.6 QC-layer inconsistency** — `HighResPruner` hardcodes `textattn_layer=3` (early), but
   `eval_highres_docvqa.json` records `qc_layer:16`; which harness produced the DocVQA headline is
   uncertain (verify before citing "mid-layer" for LLaVA-1.6).
10. **`data/budget_oracle/*` has no generator code in the repo** (grep-confirmed by the prior audit) —
    those numbers are historical/non-reproducible and on a different (classification-era, threshold-1/3)
    protocol; do not place on the same axis as the June generation numbers.

**Duplicated/entangled logic:**
11. `StaticPrunedLlava` (in-scope, canonical) and the retired classification `LlavaStaticVQAModel` live in
    the same `src/models/static/` package; `token_scorer.py` (the *learned QC scorer*, conceptually
    Idea-1) is stranded inside the `dynamic_budget` (Idea-2) package. `generate_and_score.py` lazily
    imports `LlavaDynamicVQAModel`, so archiving the budget package would break the in-scope `dynamic`
    branch. (Flagged in `THESIS_AUDIT.md` §5.)
12. Three eval files (`evaluation/gqa/evaluate_gqa.py`, `evaluation/textvqa/evaluate_textvqa.py`,
    `evaluation/test_generate.py`) import `ElasticPrunedLlava` — they would break if elastic is archived.

**Tests:** there are a few `test_*.py` files (`src/data/test_llava_mix.py`,
`src/models/elastic/test_*.py`, `src/training/test_train_stage1.py`, `src/evaluation/test_generate.py`)
but no test runner/CI config and no coverage of the headline scorers or the Qwen/distillation paths.

---

## L. Recommended Next Steps (prioritized)

### 1. Verify first (cheap, no GPU)
- `python -m compileall src scripts` and the import smoke from the README (confirm the migrated tree still
  loads; the cleanup report claims it does).
- Re-run the pure-arithmetic analyses to confirm they still read their inputs:
  `python -m src.analysis.oracle_decomposition`, `src.analysis.cascade_sweep`, `src.analysis.flops`,
  `src.analysis.flops_frontier`, `src.pruning.dynamic_budget.qwen_budget_eval/robust`.
- Decide whether to rebuild `data/vqav2/answer_vocab_full.json` (only needed if the classification/dynamic
  VQAv2 path will ever be re-run).
- Confirm the Qwen env exists and the Qwen weights/datasets are reachable before trusting any re-run.

### 2. Publication-worthy results (what's already strong)
- **Diagnostic foundation (L1–L4)** and the **monotone oracle-noise methodology** — fully supported, full
  splits, write as-is.
- **Qwen DocVQA budget mirage (L8, L9)** + **selection dominance (L5–L7)** + **student (L11–L12)** —
  write *with the stated qualifiers* (n, model, baseline=blind-floor, +7.50pp is FLOPs-matched).
- **Highest-leverage cheap upgrade:** recompute + **save** the L10 generality JSONs (raw data already in
  `results/paper_candidates/`). This converts the single "risky narrative-only" claim into evidence.

### 3. Clean before GitHub / paper / thesis
- Rebuild or remove the dangling `answer_vocab_full.json` reference so configs are runnable.
- Repoint `make_output_dir(base_dir="outputs")` → `results/...` (or document it) so re-runs land in the
  new layout.
- Pick **one** FLOPs convention and re-extract VQAv2 under it (note: `docs/` and `results/` are
  git-ignored, so a public GitHub push exposes only code/configs/scripts + the two tracked result
  index files — confirm nothing private is referenced from tracked files).
- Extract `token_scorer.py` (learned QC scorer) out of the `dynamic_budget` package and guard the lazy
  `LlavaDynamicVQAModel` import before any archiving, so the in-scope `dynamic` eval branch keeps working.
- Pin all numbers to ledger canon (oracle = +7.50pp FLOPs-matched / +0.81pp pure-accuracy; predictor =
  +0.09pp/1.8%; drop +4.2/+3.9; AUC = 0.643 with corrected "unfixable" wording).

### 4. Do NOT touch yet
- The frozen `results/thesis_main/*` evidence JSONs and the `distill/*.pt` checkpoints (the thesis
  numbers; the ledger pins exact paths).
- `results/paper_candidates/*` (raw inputs for the L10 recompute — needed before it can be deleted).
- `data/` (81G, regenerable but expensive; nothing here is wrong).
- The elastic Stage-1 code/checkpoints and `data/budget_oracle/*` (excluded, but preserved as provenance).
- Any code/folder reorganization beyond §3 — the master plan explicitly defers further reorg until a
  complete draft exists.

---

### Appendix: one-paragraph mental model

Frozen VLM backbones. A pruner decides *which* tokens (selection) and *how many* (budget). On frozen
low-res LLaVA-1.5 the project built an honest measurement instrument (dense reproduces published; oracle
band is thin; naive question-conditioned selection loses to CLS saliency) → the negatives. On
high-res/Qwen the *same* instrument shows the negatives are regime/signal-specific: a **mid-layer (L16)
question→visual** selection signal dominates (vastly outperforms blind selection, retains ~95% of dense
at ~88% prefill-FLOP cut, genuinely question-driven), while the **per-sample budget stays a mirage** (its
honest FLOPs-matched oracle is small and a trained predictor captures ≈0 because good selection already
fits most samples in the minimum budget and the residual tail is unfixable). The teacher signal needs a
full LLM forward, so it is distilled into a cheap no-decoder-forward student that recovers ~58% of the
gain — deployable but below SOTA. Conclusion: **invest in selection; per-sample budgeting is not worth
it.** All wins are frozen; the contribution is the rigorous decomposition + honest methodology, not a
leaderboard number.
