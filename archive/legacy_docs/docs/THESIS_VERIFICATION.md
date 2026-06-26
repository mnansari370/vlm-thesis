# Thesis Deep-Verification Audit — Idea 2 strength, oracle ceiling, cross-model scope

> **Read-only.** Nothing moved, edited, deleted, or run. Baseline = [`THESIS_AUDIT.md`](THESIS_AUDIT.md)
> (result inventory); this document does **not** repeat it. Every number/path verified against an actual file.
> Where I could not verify, I write **uncertain**. Judgments are deliberately blunt, per request.

## TL;DR (the uncomfortable truths)

1. **Idea 2 attempts were mostly weak-to-standard (strength 1–3/5), but a *strong* version would still not pay off
   for accuracy.** The reason is not the predictors — it is the **oracle ceiling**, and the ceiling is genuinely small
   once you read it correctly (see #2). Two months on a strong image-conditioned predictor → realistic best case
   **≈ +1 to +2pp** at matched compute on Qwen-DocVQA, **≈ +0.5 to +1.5pp** on VQAv2/GQA, and plausibly **0 or
   negative** (the unfixable tail). Not worth it for accuracy.
2. **The headline "+7.5pp oracle headroom" is a FLOPs-matched-at-low-budget number, not an accuracy ceiling.** The
   pure-accuracy gain of a perfect per-sample budget over the *best fixed K* is **+0.81pp** (monotone 98.0 vs
   best-static 97.19 in [qwen_oracle_docvqa_qc.json](../results/thesis_main/highres/qwen_oracle_docvqa_qc.json)). The
   +7.5 compares the oracle (98.0% @ ~90 tokens) to a static curve *also forced down to ~90 tokens* (~90.5%). It is a
   real efficiency number; it is **not** "7.5 accuracy points are sitting on the table."
3. **The "hard tail is unpredictable" claim is shakier than written.** The saved artifact
   [qwen_budget_robust.json](../results/thesis_main/highres/qwen_budget_robust.json) reports **AUC 0.643**, which
   *crosses the script's own 0.6 "a budget method MIGHT help" threshold* — yet the FINDINGS narrative rounds it to
   "0.59 ≈ unpredictable." The mirage holds because realized gain is ~0 *despite* AUC 0.64, not because the tail is
   un-identifiable. Honest framing matters here.
4. **The cross-model comparison is currently unfair and it inflates the Qwen headline.** Qwen's "+60pp selection win"
   is measured **vs `uniform` (the weakest floor)**; LLaVA-1.6's same method vs a **CLS-attention** blind baseline is
   only **+27pp**; on frozen LLaVA-1.5, question-conditioned selection **loses/ties** CLS-attention. Qwen has **no
   CLS-attention baseline implemented at all**. A fair Qwen baseline would shrink +60pp substantially.
5. **No image features were ever fed to a *learned per-sample* budget predictor.** The only image-derived inputs
   appear in the retired VQAv2 *binary-route* experiment (hand-crafted CLS/patch summary stats), whose **generator
   code is not in the repo** (external/archived → not reproducible).

---

# Section 1 — Was the Idea 2 approach a serious attempt?

## 1.1 — Catalog of every Idea-2 component

### (A) `BudgetController` — the LLaVA-1.5 / VQAv2 per-sample budget head
```
NAME:        BudgetController  (src/models/dynamic_budget/budget_controller.py)
             driven by token_selector.py + llava_wrapper.py; trained by train_dynamic.py / budget_variance_gate.py
PURPOSE:     per-sample keep-ratio -> K (continuous in [min/N, max/N], rounded). In practice supervised toward
             per-question-TYPE target ratios, so effectively per-type K.
INPUTS:      question_projected (512-d, projected mean LLM text embedding) + 7 score-statistics
             (mean/std/max/min/top5/top10/entropy of the token SCORES) + optional question_type embedding (16-d).
             NO raw image features. (Score-stats derive from CLS-attn scores, so only indirectly image-influenced.)
ARCH:        LayerNorm -> Linear(d->256) -> GELU -> Dropout -> Linear(256->128) -> GELU -> Dropout -> Linear(128->1)
             -> sigmoid -> affine to [min,max]. ~small (tens-to-hundreds of K params). 3-layer MLP.
TRAIN DATA:  VQAv2 train; the decisive "gate_real" run = 6,000 train / 3,000 held-out val (stratified).
             Headline dynamic_150k_clsonly = trained on 150k, reported at EPOCH 1.
TRAIN OBJ:   CE (answer head) + budget_loss (question_type_target = MSE(keep_ratio, per-type target))
             + entropy_loss + budget_diversity_loss (= -std(K), INACTIVE at bs=1 since std of a scalar = 0).
TRAIN DUR:   gate_real = 6 epochs, AdamW, budget_loss_weight 0.5; budget_loss converged 0.0033 -> 0.00016.
EVAL RESULT: dynamic 75.76 vs static-K265 75.71 = +0.05pp at matched avg K=264.3
             (results/thesis_main/vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json).
FAILURE MODE: not a collapse — it WORKS mechanically (held-out std(K)=44.3, correct per-type ordering
             yes/no 219 < attr 277 < count 336 < spatial 345, generalizes). It simply doesn't help: per-type
             accuracy curves are near-identical concave shapes, so reallocating a fixed average budget across types
             cannot beat uniform (Jensen). It only varies COUNT; token RANKING stayed plain CLS (cls_only).
```

### (B) Qwen per-sample budget MLP — the "predictor captures ~0%" result (L9)
```
NAME:        cv_mlp predictor  (src/pruning/dynamic_budget/qwen_budget_eval.py)
PURPOSE:     per-sample sufficiency-K (smallest ladder level robustly correct), evaluated vs best fixed K.
INPUTS:      5 QC-attention-distribution features ONLY (norm_entropy, eff_support, top64/128/256_mass);
             features built in qwen_budget_data.py:feats() from the L16 question->visual attention vector.
             NO image features.
ARCH:        Linear(5->32) -> ReLU -> Linear(32->L)  (L = #ladder levels). Trivial 2-layer MLP.
TRAIN DATA:  DocVQA n=200 (eval json) / data collected at n=400; 5-fold CV.
TRAIN OBJ:   cross-entropy on the sufficiency-K class; 300 steps Adam lr 1e-2 wd 1e-3.
EVAL RESULT: oracle_gain +4.99pp, MLP gain +0.09pp = 1.8% of oracle, rule +0.72pp, best_fixed 93.0
             (results/thesis_main/highres/qwen_budget_eval.json).
FAILURE MODE: collapses to "minimum-for-all" — 81% already correct at the 32-token minimum, so predicting more
             tokens for the 19% tail mostly mis-fires.
```

### (C) Qwen spread-rule predictor — `qwen_budget_eval.py`
```
NAME: spread rule | PURPOSE: K via eff_support quantile bins | INPUTS: 1 feature (eff_support) | ARCH: quantile
binning, no learning | RESULT: +0.72pp | FAILURE: a 1-feature heuristic; near-noise.
```

### (D) Qwen hard-tail predictability test — `qwen_budget_robust.py` (a DIAGNOSTIC, not a budget head)
```
NAME:        hard-tail classifier  (src/pruning/dynamic_budget/qwen_budget_robust.py)
PURPOSE:     can "needs > 32 tokens" be identified at all? (upper bound on any predictor's reach)
INPUTS:      6 attention-dist features (nvis + the 5 above). NO image features.
ARCH:        Linear(6->16)->ReLU->Linear(16->1), 5-fold logistic CV, 400 steps.
RESULT:      base rate 0.195; CV-AUC 0.643 (saved). FINDINGS/figure quote 0.59. **Discrepancy — see 1.4/§6.**
FAILURE/NOTE: this is the cleanest piece of the Idea-2 work, but its own verdict logic prints "somewhat
             predictable -> a budget method MIGHT help" when AUC>=0.6 — i.e. at 0.643 the artifact technically
             says "might help," contradicting the narrative.
```

### (E) Binary route predictor — VQAv2, retired phase (`data/budget_oracle/`)
```
NAME:        binary route (k144 vs dense-576)  — summaries in binary_routing_eval_summary.json,
             features in val_budget_visual_features_binary.jsonl, labels in val_budget_labels_binary.json
PURPOSE:     route each sample to small (144) or large (dense) budget.
INPUTS:      *** THE ONLY image-derived predictor *** — per-sample CLS-attention distribution stats
             (cls_attn_mean/std/top1/top2/margin/top5/10/25/entropy/effective_tokens) AND patch-norm stats
             (patch_norm_mean/std/top1.../entropy), plus tfidf-visual and qtype variants. Hand-crafted summary
             stats, NOT raw patch embeddings / a learned image encoder.
ARCH:        threshold rule on a routing score (tfidf_visual_thr_* / qtype_thr_*). Not a trained deep net.
TRAIN DATA:  VQAv2 val 10k, 2k test split. Dated May 2026 = early classification-head phase.
RESULT:      best routes leave +2.8 to +7.0pp accuracy gap to dense at 30-50% token saving
             (e.g. tfidf_visual_thr_0.30: 63.2% @ 362 tok, gap 4.0pp; qtype_thr_0.30: 64.4% @ 401 tok, gap 2.8pp).
             Never closes the gap = never beats static.
FAILURE/NOTE: binary (not fine-grained K); threshold (not learned); classification-era metric (~60-66%, NOT the
             75.76% generation protocol). **GENERATOR CODE IS NOT IN THE REPO** (grep finds no producer) -> these
             numbers are not reproducible from current code. Uncertain provenance of exact scoring.
```

### (F) Per-question-type budget maps — `qtype_oracle_summary.json`, `dynamic_stage6c_{6,7}type`, `exp2–exp5_7type`
```
NAME:        per-type budget allocation diagnostics (data/budget_oracle/*type*.json/.txt)
PURPOSE:     map question TYPE -> K (6-type / 7-type taxonomies; exp2-5 iterate the taxonomy/thresholds).
INPUTS:      question type label ONLY. NO image, NO per-sample features.
ARCH:        deterministic type->K table.
RESULT:      ~62.3% overall @ ~211-220 avg tokens (classification-era). qtype_oracle shows oracle token-NEED rises
             with type (yes/no ~194 tok, what/which ~368, where ~401, how ~450) BUT hard types have low
             resolved_rate (where 43%, how 32%, why 31%) -> their extra-token need is mostly UNFIXABLE.
FAILURE/NOTE: type is a coarse signal; per-type oracle (~62%) << per-sample oracle (77% naive / 81% ceiling).
             **GENERATOR CODE NOT IN REPO.** exp2-5 differ only in the type taxonomy; all land ~62%.
```

### (G) Confidence cascade — realizable budget mechanism (GQA/POPE/VQAv2)
```
NAME:        speculative/confidence cascade (run_speculative_testdev.py, run_pope_speculative.py,
             cascade_pass.py, cascade_sweep.py)
PURPOSE:     run base K; escalate low-confidence samples to higher K (an early-exit on token budget).
INPUTS:      the decoded answer's CONFIDENCE at base K (label-free, realizable). NOT a learned head.
RESULT:      traces the static frontier, never robustly above it (GQA tau-sweep 58.4->60.5 as K grows, all on the
             static curve; results_frozen/all.json "cascade"). VQAv2: best +0.10pp, evaporates under
             compute-matched accounting (RESULT_SUMMARY.json realizability_confidence_cascade).
FAILURE/NOTE: confidence predicts "is it correct NOW," not "will more tokens fix it"; low-confidence is dominated
             by the never-correct (~19% VQAv2) not the token-starved (~10%).
```

## 1.2 — Strength assessment (blunt)

- **(A) BudgetController — 3/5.** *Strong:* a real, properly-run experiment — held-out 3k val, correct per-type
  ordering, non-zero within-type std, generalization checked, two genuine bugs found and fixed (budget_loss_type key;
  all-ignored-CE NaN). This is a *standard, competent* attempt. *Weak:* it only adapts **count by question type**, a
  coarse 4-bucket signal; the token *ranking* stayed plain CLS (so it never tested learned QC selection + budget
  jointly); bs=1 disables the diversity loss; the headline used an **epoch-1** checkpoint. *Stronger version work?*
  **No, not meaningfully** — the per-type accuracy curves are near-identical concave shapes; Jensen caps the gain
  regardless of head quality.
- **(B) Qwen per-sample MLP — 2/5.** *Strong:* honest 5-fold CV, paired with the oracle it's measuring against,
  correct "captures 1.8%" accounting. *Weak:* a **5-feature, 32-hidden** net on **attention-distribution stats
  only** — no image, no question semantics, no learned representation; n=200; tiny. This is barely above a sanity
  check. *Stronger version?* **Architecturally yes, outcome no** — bounded by an oracle whose pure-accuracy ceiling
  is +0.81pp (see 1.4).
- **(C) spread rule — 1/5.** A one-feature quantile heuristic. Smoke-test grade. No.
- **(D) hard-tail AUC test — 3/5 as a diagnostic** (not a predictor). *Strong:* the right question ("is the tail even
  identifiable?"), clean rank-AUC, direction-agnostic single-feature scan. *Weak:* features are attention-stats only,
  and **the saved AUC (0.643) contradicts the "unpredictable" narrative** (§1.4). Honest but under-powered (n=200).
- **(E) binary route — 2/5.** *Strong:* the only attempt that fed **image-derived features**, and it tried both
  visual and qtype signals. *Weak:* binary not fine-grained, **threshold not learned**, summary-stats not a learned
  image encoder, retired classification-era metric, **non-reproducible (no code)**. *Stronger version?* This is the
  one direction with a non-trivial "maybe" — see 1.3 — but its own result already shows a persistent 2.8–7pp gap.
- **(F) per-type maps — 2/5.** Useful as a *negative diagnostic* (type is weak), not as a method. Classification-era.
- **(G) cascade — 3/5.** A legitimate, *realizable*, label-free budget mechanism (more honest than a label-trained
  predictor). *Weak:* it ties; mechanism explained (confidence ≠ token-need). Correctly reported as a tie, not a win.

**Overall:** the Idea-2 program is **competent at proving the negative** (BudgetController gate, cascade, hard-tail
AUC, oracle decomposition are all real, careful work) but **never mounted a strong *positive* attempt** — no learned
per-sample K head with real image features, no joint selector+budget training, no modern optimization. The strongest
*positive* try (BudgetController) attacked the provably-hardest-to-win axis (per-type count).

## 1.3 — Idea-2 approaches NEVER tried

> Oracle ceilings to keep in mind (explicit): **Qwen-DocVQA** pure-accuracy ceiling over best-fixed = **+0.81pp**
> (FLOPs-matched +7.5pp at ~90 tok); **VQAv2** instance oracle ceiling 81.13% vs uniform +5.4pp (noise-inflated),
> realizable token-need band only ~10.4%; **GQA** band 9.13%; **POPE** 4.93%; **TextVQA** ~6.1–6.2%.

| approach (not attempted) | why it might beat what was tried | effort | realistic ceiling |
|---|---|---|---|
| **Learned image-conditioned per-sample-K head** (small ViT/merger-feature encoder → K regressor), trained on actual visual embeds not summary stats | every learned predictor here used only attention/score stats; raw visual content is strictly more information | medium (2–4 wk) | **bounded by oracle**: Qwen-DocVQA ≤ +0.8pp pure-acc / a slice of the +7.5 FLOPs-matched; VQAv2 ≤ ~+5pp but mostly unfixable tail → likely +1–2pp |
| **Joint training of selector + budget** (vs the repo's fixed CLS selector + learned budget) | a better selector collapses the budget spread (proven on Qwen: good selection cut avg 337→91); co-training might find a budget that exploits the *residual* | large (4–8 wk) | low — good selection *removes* budget headroom by construction; this fights itself |
| **RL / non-differentiable budget head** with a compute-penalized reward | the discrete K + downstream-accuracy objective is RL-shaped; current MSE-to-type-target is a weak proxy | large | bounded by same oracle; RL variance likely ≥ the +0.8–5pp signal |
| **Difficulty proxy from a pretrained model** (e.g. an external question-difficulty / readability scorer as a feature) | adds signal orthogonal to attention stats | small–medium | small — the tail is *unfixable*, not just *unidentified* (resolved_rate is the wall) |
| **Multi-task budget across VQAv2+GQA+TextVQA+DocVQA** (shared head) | more data/labels; better-calibrated K | medium | small; each dataset's band is independently thin |
| **Cascade with a LEARNED escalation policy** (vs the current confidence threshold) | confidence-threshold is a 1-D rule; a learned policy on confidence+features might beat it | small–medium | low — RESULT_SUMMARY already shows the cascade win evaporates under compute-matched accounting |

**The honest pattern:** every untried approach is **bounded by the same oracle**, and the oracle's headroom is small
*and dominated by unfixable samples* (low resolved_rate). The one genuinely-untried lever with real information —
a **learned image-conditioned head** — is the only thing worth even a short spike, and only on the dataset with the
largest *realizable* band, **but its ceiling is still single-digit and mostly efficiency, not accuracy.**

## 1.4 — Is the oracle ceiling real? (walk-through)

**Code logic ([qwen_oracle_qc.py](../src/pruning/dynamic_budget/qwen_oracle_qc.py)):**
1. Per sample, compute QC (L16) attention ONCE; evaluate ANLS at every K in the **discrete ladder
   `{32,64,128,256,512,768,1024}` + full**; binary-correct at ANLS ≥ 0.5 (`THRESH`).
2. `best_static` = max over K of the mean correctness (= 97.19, achieved at K≥512/full).
3. `naive_oracle` = mean over samples of max-over-K (per-sample best K; **exploits noise** — a sample correct at K=32
   but wrong at full still scores) = 99.01 → naive band +1.82.
4. `monotone` = a sample counts at the smallest K where it is correct **at K and all larger K** (noise-free) = 98.0,
   at avg `mono_tok` = 90.62 tokens.
5. `honest_flops_matched` = `monotone − interp(static curve at mono_tok)` = 98.0 − ~90.5 = **+7.5pp**.

**K-set:** discrete, 7 rungs + full. A **finer grid** (e.g. 16/24/48/96 near the steep 32–128 region) would let the
oracle place samples more precisely → slightly higher monotone accuracy and lower mono_tok → a **modestly larger
+7.5-style number**. It would **not** change the pure-accuracy ceiling.

**Monotone vs naive:** the **monotone** (+7.5 / 98.0) is the honest one — the naive (+1.82 band / 99.01) credits
accidental correctness at a budget where the answer later breaks. (Note: the two coexisting numbers in the audit are
+7.5 [`qwen_oracle_qc.json`, monotone, vs static@mono-budget] and +4.99 [`qwen_budget_eval.json`, sufficiency-K vs
best-fixed]; both honest, different baselines.)

**The reframing that matters (this is the key correction):** the +7.5pp compares the oracle (98.0% @ **90 tokens**)
to static **also dragged down to 90 tokens** (~90.5%), i.e. on the steep part of the curve. The **pure-accuracy
headroom of perfect per-sample budgeting over the best fixed K is only `98.0 − 97.19 = +0.81pp`.** So:
- "How much *accuracy* can dynamic budget buy over just using K=512?" → **≈ +0.8pp.**
- "How much *compute* can it save at ~iso-accuracy?" → the efficiency story (98% at 90 vs static needing ~512 tok).
- The +7.5 is the **second** thing dressed as the first. It is **not an artifact** (the computation is correct) but it
  is **routinely mis-read** as "+7.5 accuracy points available." It is not.

**Larger-headroom reframings (honest options):** (a) finer K-grid → marginally larger; (b) oracle over
**(selector, K) jointly** rather than K with a fixed QC selector → larger, but then you're crediting *selection*, which
is the thing the thesis already says is the real lever; (c) different correctness threshold (lower τ) → inflates via
noise (the naive trap). None of these turn +0.8pp pure-accuracy into a large number.

**Verdict:** the ceiling is **real and small**. The +7.5pp is a legitimate *FLOPs-matched* figure; the *accuracy*
ceiling over the best fixed budget is **~+0.8pp** on Qwen-DocVQA. On VQAv2 the naive +5.4pp is **noise-inflated** and
the realizable token-need band is ~10.4% dominated by unfixable samples. **Not an artifact — but the number that
sounds big answers an efficiency question, not an accuracy question.**

## 1.5 — Final verdict on Idea 2

**If Nafees spends two more months on a stronger Idea-2 approach (image-conditioned head, larger architecture, better
loss), the realistic best case is ≈ +1 to +2pp accuracy over the best fixed-K baseline at matched compute on
Qwen-DocVQA, and ≈ +0.5 to +1.5pp on VQAv2/GQA — with a real chance of ≈ 0.** Justification: (i) the pure-accuracy
oracle ceiling over the best fixed budget is **+0.81pp (Qwen-DocVQA)**; a realistic predictor captures a *fraction* of
even the generous FLOPs-matched +7.5, and the best honest attempt captured **1.8%** (+0.09pp); (ii) the band is
dominated by **unfixable** samples (low resolved_rate; VQAv2 never-correct 19% vs token-starved 10%), so identifying
the tail (even at AUC 0.64) doesn't convert to accuracy; (iii) the only untried lever with new information (learned
image features) attacks a ceiling that is small to begin with. **The defensible Idea-2 deliverable is an *efficiency*
result (iso-accuracy at fewer average tokens), and static pruning already captures most of that.** Recommendation
stands: **commit to Idea 1; report Idea 2 as a rigorously-characterized negative.**

---

# Section 2 — LLaVA-1.6: what it is and whether to keep it

## 2.1 — Factual basics
- **Resolution / tokens:** AnyRes tiling; saved results show **avg ≈ 2302 visual tokens** (`eval_base.json`
  `avg_visual_tokens: 2302`), **2252** in the latency run, **max 2928**. The audit's "~2880" is the **max**, not
  typical — **correct to ≈ 2300 avg.** Model = `llava-hf/llava-v1.6-vicuna-7b-hf` (same CLIP-336 tower + Vicuna-7B as
  LLaVA-1.5; the only change is AnyRes tiling).
- **Drivers (functional, produced saved results):** `evaluate_textvqa_highres{,_kcurve,_spread}.py`,
  `evaluate_gqa_highres.py`, `evaluate_highres_docchart.py`, `evaluate_highres_spread_docchart.py`,
  `textattn_layer_sweep.py`, `control_question_conditioning.py`, [llava_latency.py](../src/analysis/llava_latency.py),
  [llava_budget_data.py](../src/pruning/dynamic_budget/llava_budget_data.py), `verify_dense.py`. Pruner class =
  `HighResPruner` (in `evaluate_textvqa_highres_kcurve.py`).
- **Results that exist:** DocVQA ([eval_highres_docvqa.json](../results/thesis_main/highres/eval_highres_docvqa.json),
  n=300: full 67.19, blind-attn K128 27.76, QC K128 55.13; n=1000 variant exists), TextVQA (highres full/kcurve/spread),
  ChartQA, GQA-highres, the latency curve, the budget-mirage data (`--tag docvqa_llava`). K-values vary by file
  (64/128/256/512/1152…).

## 2.2 — Role in the v2 work
LLaVA-1.6 was the **bridge model** between frozen low-res (LLaVA-1.5) and SOTA (Qwen): it (a) established that
high-res creates real pruning room (the "resolution × task" finding), (b) is the **sole source of the "3.3× latency"
claim** ([llava_latency.json](../results/thesis_main/highres/llava_latency.json): 3.31× @K128, decode-only, n=40 — and
note the keep set is `torch.arange(first-K)`, i.e. **not a real selection**, just sequence-length-vs-latency), and (c)
provided the **cross-family** budget-mirage replication (L10). It is a *second model family*, not abandoned.

## 2.3 — Keep or archive?
**Archive from the *thesis* scope, but retain two numbers.** Reasoning: under "Qwen headline + LLaVA-1.5 contrast,"
LLaVA-1.6 is a third backbone that muddies the two-model story; its DocVQA dense (67.19) is far below Qwen (97.19),
so it can't share an accuracy axis. **But** it is the **only** source of (i) the measured **3.3× latency** (E3) and
(ii) the cross-family budget-mirage check (L10). If the thesis cites either, keep those JSONs (and `llava_latency.py`)
even if the rest of the LLaVA-1.6 code is archived. **Caveat for honesty:** if you drop LLaVA-1.6, the latency claim
must either be dropped or **re-measured on Qwen/LLaVA-1.5**, because it is currently a *different model from the
accuracy/FLOPs headline* (already flagged in the ledger R3).

---

# Section 3 — Can Qwen run on VQAv2 / GQA / TextVQA?

## 3.1 — Pipeline readiness
**None of the three are supported today.** The Qwen harness ([qwen_pruner.py](../src/pruning/question_conditioned_selection/qwen_pruner.py)
+ [qwen_kcurve.py](../src/evaluation/docvqa/qwen_kcurve.py)) has a `load_bench` that handles **only `docvqa` and
`chartqa`** (via `datasets.load_dataset`), and scores with **ANLS / relaxed-acc only**. A grep for `vqav2|gqa|textvqa`
across the Qwen files returns nothing relevant. What's missing per dataset:

| dataset | loader | prompt | scorer | images |
|---|---|---|---|---|
| VQAv2 | **new** `load_bench` branch (read `data/vqav2/*questions*` + `*annotations*`, COCO val2014 jpgs) | Qwen chat template (exists) | **wire in** VQA-consensus ([vqav2_answers.py](../src/data/vqav2/vqav2_answers.py)) | local files (PIL → `qwen_pruner._inputs` accepts PIL ✓) |
| GQA | **new** branch (read `data/gqa/testdev_balanced_questions.json` + images) | template ✓ | **wire in** [official_score.py](../src/metrics/official_score.py) | local ✓ |
| TextVQA | **new** branch (read `data/textvqa/*.jsonl` + train_images) | template ✓ | **wire in** [textvqa_score.py](../src/metrics/textvqa_score.py)/m4c | local ✓ |

The scorers all **exist** but are wired to the LLaVA harness; the Qwen harness only knows `anls`/`relaxed_correct`.
So the work is **adapter + scorer-dispatch plumbing**, not a new model harness — `qwen_pruner.generate(image, question,
selector, K)` already returns a decoded string and works on arbitrary PIL images.

## 3.2 — Effort estimate
- **VQAv2 / GQA / TextVQA each: "adapt existing pattern," small-medium ≈ 0.5–1.5 days** (a `load_bench` branch + a
  scorer call + matching the prompt/instruction to the LLaVA-1.5 protocol for comparability). **No new harness.**
- The genuinely-non-trivial add is a **fair Qwen baseline** (§4.3), which is *medium* (not part of the loader work).

## 3.3 — Cost estimate (GPU-hours)
Timing extrapolated from saved `minutes` fields (n=200): full ≈ 2.3 min/200 ≈ **0.7 s/sample**; textattn ≈ 2.6 min/200
≈ **0.8 s/sample** (encode_qc reused across K). **Uncertain** (single-GPU, n=200 extrapolation).

| dataset | eval N | ~time/sample | one (method,K) cell | 5 K × 2 methods (blind+QC)* |
|---|---|---|---|---|
| VQAv2 (10k subset, to match LLaVA-1.5) | 10,000 | 0.8 s | ~2.2 GPU-h | ~22 GPU-h (QC reuses encode across K → less) |
| VQAv2 (full val ~214k) | 214,354 | 0.8 s | ~48 GPU-h | **prohibitive — use 10k** |
| GQA testdev | 12,578 | 0.8 s | ~2.8 GPU-h | ~28 GPU-h |
| TextVQA val | 5,000 | 0.8 s | ~1.1 GPU-h | ~11 GPU-h |

\*QC is cheaper than 5× because `encode_qc` runs once/sample and all K reuse it (the oracle pattern). Realistic total
for **Qwen × {VQAv2-10k, GQA, TextVQA} × dense+static+QC × {72,144,288,432}** ≈ **40–80 GPU-h** (order-of-magnitude).

## 3.4 — New infrastructure required
1. **Dataset adapters + scorer dispatch** (3.1).
2. **A fair Qwen blind baseline** — currently only uniform/norm exist (§4). **This is the important one.**
3. **FLOPs `N_TEXT` per dataset** — [qwen_flops.py](../src/analysis/qwen_flops.py) hardcodes `N_TEXT=40` (a DocVQA
   estimate); VQAv2/GQA/TextVQA prompts are shorter → re-measure N_TEXT (minor, ~0 GPU).
4. **No per-K caching needed** — Qwen prunes at runtime; one encode per sample. (Unlike the LLaVA-1.5 VQAv2 cached path.)
5. **Prompt note:** Qwen uses its own chat template + the same "Answer using a single word or phrase" instruction
   (`INSTR` in qwen_pruner) — comparable to LLaVA-1.5's suffix, but **not identical tokenization** (inherent, accept it).

---

# Section 4 — Baseline strength per model

## 4.1 — LLaVA-1.5 strongest static baseline
**CLS-attention (VisionZip "dominant")** via [static.py](../src/models/static/static.py) `cls_attn`, *plus* a faithful
**VisionZip** (dominant + contextual merge) in [visionzip.py](../src/pruning/static/visionzip.py). Also implemented:
`fastv_style` (early-LLM attn, weak/position-biased), `l2_norm`, `random`, `spatial_uniform`. **The strong baseline is
CLS-attn**; VisionZip ≈ CLS-attn (marginal). This is a **genuinely strong, literature-standard** blind baseline.

## 4.2 — Qwen image-only baselines
From [qwen_pruner.py:97-118](../src/pruning/question_conditioned_selection/qwen_pruner.py#L97-L118), `generate`
supports exactly: `full`, `norm` (visual-embed L2-norm top-K), `uniform` (linspace stride), `textattn` (the QC method).
**There is NO CLS-attention baseline, NO VisionZip, not even `random`.** `norm` is *weaker than uniform* on DocVQA
(K128: norm 20.98 < uniform 32.19). So **Qwen's only blind baselines are content-blind floors** — the *weakest* class
of baseline. (Mechanistically, Qwen's ViT+merger has no single CLS→patch map like CLIP, so a CLS-attn analog needs the
ViT attention or merger weights — **not implemented**.)

## 4.3 — Cross-model fairness — **confirmed unfair, and it inflates Qwen**
This asymmetry is real and currently active:

| model | QC selector | blind baseline it's compared to | reported margin |
|---|---|---|---|
| **Qwen-2.5-VL-7B** | textattn L16 | **uniform / norm (weakest floor)** | **+59.99pp** (92.18 vs 32.19, DocVQA K128) |
| **LLaVA-1.6-7B** | textattn | **`attn` = CLS→patch attention (strong)** | **+27.4pp** (55.13 vs 27.76, DocVQA K128, n=300) |
| **LLaVA-1.5-7B** | qcond LM-attn | **CLS-attn (strong)** | **−5.58pp (TextVQA-noOCR) / +3.5pp (GQA), K64** |

The same method's margin **collapses from +60 → +27 → ~0** as the blind baseline strengthens from uniform → CLS-attn.
**So the "+60pp selection win" is substantially a weak-baseline artifact.** To make the cross-model comparison fair,
Qwen needs a **CLS-attention-equivalent strong blind baseline** (extract Qwen ViT CLS/aggregate attention or use the
merger's attention) — *medium* effort. **Expectation (stated honestly): a fair Qwen baseline will shrink +60pp toward
the +20–30pp range seen on LLaVA-1.6**, and the headline must be re-stated as "QC vs CLS-attn," not "vs uniform."
This is the single most important correction before any cross-model claim.

---

# Section 5 — Cross-model experimental plan

Scope assumed: **{LLaVA-1.5, Qwen-2.5-VL} × {VQAv2, GQA, TextVQA} + DocVQA (Qwen only)**, methods
**{dense, static-best-baseline, QC-teacher, QC-student}**, K∈{72,144,288,432}.

## 5.1 — Readiness matrix
Legend: ✅ exists · ♻ needs re-run (protocol/K) · 🧱 needs new infra · 🏋 needs training · ➖ N/A.
"static-best" = CLS-attn (LLaVA) / **must-build CLS-equivalent** (Qwen). "QC-teacher" = LM/mid attn full-forward.

| model | dataset | dense | static-best | QC-teacher | QC-student |
|---|---|---|---|---|---|
| LLaVA-1.5 | VQAv2 | ✅ K576 | ✅ 144/288/432, ♻ K72 | 🧱+♻ (only K64/probe exists; none at grid; full-forward cost) | 🏋 (no LLaVA-1.5 student exists) |
| LLaVA-1.5 | GQA | ✅ | ✅ 144/288/432, ♻ K72 | ♻ (K64 probe only; run grid, full split) | 🏋 |
| LLaVA-1.5 | TextVQA | ✅ (OCR+noOCR) | ✅ 144/288/432, ♻ K72 | ♻ (K64/96 probe; **QC loses to CLS**) | 🏋 |
| Qwen-2.5-VL | VQAv2 | 🧱 (no loader) | 🧱 (no CLS baseline) | 🧱 (loader+grid) | 🏋+🧱 |
| Qwen-2.5-VL | GQA | 🧱 | 🧱 | 🧱 | 🏋+🧱 |
| Qwen-2.5-VL | TextVQA | 🧱 | 🧱 | 🧱 | 🏋+🧱 |
| Qwen-2.5-VL | DocVQA | ✅ (97.19) | 🧱 (only uniform/norm; **build CLS-equiv**) | ✅ (92.18 @K128, n=200; ♻ full-val + K72) | ✅ student exists (gate_12k.json, 57.9% recovery) |

**Reading the matrix:** the LLaVA-1.5 **static** row is nearly done (only K72). Everything **QC** is largely missing or
probe-only. Every **Qwen-on-{VQAv2,GQA,TextVQA}** cell needs new infra. **QC-student exists only for Qwen-DocVQA.**

## 5.2 — Total work estimate (dependency-ordered)
1. **Decide K72-vs-K64 and the QC selector** (blocking, 0 GPU) — see audit §6 + this §6.
2. **Build the fair Qwen CLS-equivalent baseline** (eng ~3–5 days; ~5–10 GPU-h to evaluate on DocVQA) — *gates every
   Qwen comparison's credibility.*
3. **Qwen dataset adapters + scorers** for VQAv2/GQA/TextVQA (eng ~1.5–4 days total).
4. **Qwen eval matrix** (dense+static+QC × 4 K × 3 datasets, 10k VQAv2) ≈ **40–80 GPU-h**.
5. **LLaVA-1.5 K72 static** ×3 datasets (eng ~0.5 day to relax K constraint; ~6–10 GPU-h) — or **skip via K64**.
6. **LLaVA-1.5 QC-at-grid** (if QC arm kept): eng small if using the LM-attn teacher probe; **but full-forward scoring
   cost** ≈ a dense pass/sample → ~3× the static eval time; ≈ 15–30 GPU-h ×3 datasets.
7. **QC-student training** (if deployable selector wanted per model/dataset): 🏋 each = days + GPU.
- **Rough total (eng):** ~2–3 weeks. **Rough total (GPU):** ~80–160 GPU-h, dominated by the Qwen matrix + QC full-forward scoring. **Uncertain ±50%** (timing extrapolated from n=200).

## 5.3 — Risk register
- **R1 [HIGH] Qwen-on-VQAv2/GQA may show QC barely helps** — these are low-resolution-friendly reasoning tasks where,
  per the thesis's own resolution×task finding, there is *little to prune*; the +60pp DocVQA result may not transfer.
  Plausible outcome: QC ≈ static on VQAv2/GQA, echoing the LLaVA-1.5 negative. **This could undercut the headline.**
- **R2 [HIGH] The fair Qwen baseline shrinks +60pp to ~+20–30pp** (§4.3). Expected, but it weakens the "dominates"
  language and must be pre-empted in the writing.
- **R3 [MED] QC-teacher full-forward cost** makes the Qwen efficiency story circular on these datasets unless the
  *student* is trained per dataset (🏋) — and the student only recovers ~58% (DocVQA) and is below SOTA.
- **R4 [MED] LLaVA-1.5 QC at the grid may stay negative** (TextVQA already −5.58pp at K64) → the "question-conditioned
  selection" arm on the contrast model is a *negative*, which is fine **if framed as such** but not if sold as a method.
- **R5 [LOW] VQAv2 10k-subset vs full-val** parity questions from a committee; mitigate by stating the subset protocol.
- **R6 [LOW] N_TEXT / FLOPs convention drift** across datasets/models (two calculators, two conventions) → pick one.

---

# Section 6 — Things you didn't ask about but should know

1. **AUC discrepancy undercuts the "unpredictable tail" wording.** Saved
   [qwen_budget_robust.json](../results/thesis_main/highres/qwen_budget_robust.json) = **0.643**; the script itself
   prints *"somewhat predictable → a budget method MIGHT help"* at AUC ≥ 0.6; the FINDINGS/figure say **0.59
   "≈ unpredictable."** The robust *conclusion* still holds (realized gain ~0), but the *stated mechanism*
   ("unpredictable") is not what the artifact says. **Fix the wording**: the tail is *weakly* identifiable (AUC ~0.64)
   but *unfixable* (low resolved-rate), so the predictor still captures ~0%.
2. **The +7.5pp is an efficiency number masquerading as accuracy** (§1.4). Pure-accuracy ceiling over best-fixed =
   **+0.81pp**. If the thesis says "+7.5pp headroom" without the matched-budget qualifier, a careful reader will catch it.
3. **Cross-model baseline asymmetry (§4.3) is the biggest threat to the headline** — +60pp is vs the weakest possible
   baseline; the fair number is likely ~+20–30pp. Treat as must-fix, not optional.
4. **Numbers that look too good (small-n optimism):** Qwen DocVQA **full = 97.19 > published 95.7** (n=200); textattn
   **K512 = 97.16 ≈ dense** (selection at 512/1286 tokens fully recovers). Plausible but **n=200** — re-run at full-val
   before trusting to 0.1pp.
5. **Latency claim caveats:** [llava_latency.py](../src/analysis/llava_latency.py) measures on **LLaVA-1.6, n=40,
   decode-only**, and the "kept" tokens are `torch.arange(first-K)` — **not the actual selector's tokens**. It's a valid
   sequence-length→latency curve, but it is *not* the deployed method's end-to-end latency and is a *different model*
   from the accuracy/FLOPs headline (ledger R3 already flags the model-mismatch; the arange detail is additional).
6. **LLaVA-1.6 QC layer inconsistency:** `HighResPruner` hardcodes `textattn_layer = 3` (an **early** layer, FastV-like)
   in [evaluate_textvqa_highres_kcurve.py](../src/evaluation/textvqa/evaluate_textvqa_highres_kcurve.py), yet
   [eval_highres_docvqa.json](../results/thesis_main/highres/eval_highres_docvqa.json) records `qc_layer: 16`. The DocVQA
   numbers likely came from a *different* harness (`evaluate_highres_docchart.py`) at layer 16 — **uncertain**; verify
   which file produced the LLaVA-1.6 DocVQA headline before citing "mid-layer" for LLaVA-1.6.
7. **Reproducibility gap:** the entire `data/budget_oracle/` line (binary routing, qtype oracle, stage6c, exp2-5,
   visual-feature jsonl) has **no generator code in the repo** (grep-confirmed). Those numbers cannot be regenerated
   from current code — treat as historical/archived evidence only, not as live results.
8. **Idea-2 metric-regime mixing:** the `data/budget_oracle/` accuracies (~60–66%, threshold 1/3, May, classification
   era) and the June generation numbers (75.76%) are **different protocols**. Do not place them on the same axis.
9. **Inherited dependency landmines (re-confirming/extending the prior audit):**
   [generate_and_score.py:68](../src/evaluation/vqa/generate_and_score.py#L68) lazily imports `LlavaDynamicVQAModel`
   (budget = archive candidate) → archiving `models/dynamic_budget` breaks the in-scope VQAv2 `model_type=="dynamic"`
   branch; and `token_scorer.py` (the *learned QC scorer*, conceptually **Idea 1**) is stranded inside the budget
   package — extract before archiving.
10. **`norm` baseline is below `uniform`** on Qwen DocVQA (20.98 < 32.19). Reporting "QC beats norm by +71pp" uses the
    *worst* baseline; prefer uniform, and ultimately the fair CLS-equivalent.

---

# Section 7 — Honest gaps

- **Read in full / verified this round:** `qwen_budget_data.py`, `qwen_budget_robust.py`, `qwen_oracle_qc.py` (+ its
  result json), `qwen_budget_eval.json`, `qwen_budget_robust.json`, `qwen_oracle_docvqa_qc.json`, `qwen_layer_sweep.json`,
  `eval_highres_docvqa.json`, `llava_latency.py`, `llava_budget_data.py`, `visionzip.py` (scoring half),
  `evaluate_textvqa_highres.py`, the HighResPruner selector block in `evaluate_textvqa_highres_kcurve.py` (via grep),
  plus the budget-oracle summaries (`binary_routing`, `qtype_oracle`, `val_oracle_static_dense`, `stage6c`/`exp5` txt),
  the visual-feature jsonl header, and grep sweeps for generators, Qwen dataset support, and LLaVA-1.6 token counts.
- **Skimmed / inferred, not line-by-line:** the full bodies of `evaluate_textvqa_highres_kcurve.py`,
  `evaluate_highres_docchart.py`, `evaluate_highres_spread_docchart.py`, `textattn_layer_sweep.py`,
  `control_question_conditioning.py` (read their selector/headers and saved outputs, not every line);
  `visionzip.py` second half (the merge math); the per-run `metrics.json` files (trusted aggregates);
  the `exp2/exp3/exp4` json bodies (read `exp5`/`6type` txt as representatives).
- **Could not verify (need you / external):** the **generator code** for `data/budget_oracle/*` (binary routing,
  stage6c, exp2-5, visual features) — **not in the repo**, so I audited those only from saved JSON; exact training
  details of that retired phase are **uncertain**. Which LLaVA-1.6 file produced the `qc_layer:16` DocVQA headline
  (kcurve hardcodes layer 3) — **uncertain**. Qwen per-sample timing is **extrapolated from n=200** `minutes` fields,
  not measured at scale (±50%). Whether a Qwen ViT CLS/merger-attention baseline is cleanly extractable was
  **reasoned, not prototyped**.

*No files were moved, edited, deleted, or run. Awaiting your decisions before proposing any action.*
