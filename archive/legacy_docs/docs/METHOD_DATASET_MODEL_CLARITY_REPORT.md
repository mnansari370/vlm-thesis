# Method / Dataset / Model Clarity Report

*Read-only analysis produced 2026-06-26 on branch `method-migration`. Nothing was modified, deleted, moved,
or run. Every number below was read directly from a saved result JSON (path given) or from the code; where a
number comes only from the project's own docs it is marked. Companion to
`docs/FULL_REPOSITORY_UNDERSTANDING_REPORT.md` (repo-wide map) and `docs/THESIS_EVIDENCE_LEDGER.md`
(authoritative number ledger, IDs L1–L12 / E1–E5).*

**Purpose:** make the four method families unambiguous — **dense**, **static pruning**, **dynamic WHICH
(question-conditioned, fixed-K selection)**, and **dynamic COUNT (per-sample budget)** — with their exact
models, datasets, numbers, and what each one proves. Written so you can lift it directly into a methodology
+ results chapter.

---

## 0. The four things, in one table (read this first)

| # | Name | Decides… | K per sample | Uses the question? | Verdict in your thesis |
|---|---|---|---|---|---|
| 1 | **Dense** | nothing (keep all) | all (576 / ~1286 / ~2302) | n/a | Ceiling + reproduction anchor |
| 2 | **Static pruning** | *which* by **image saliency** | **fixed** | **No** | Strong, training-free frontier; the baseline to beat |
| 3 | **Dynamic WHICH** (question-conditioned selection, fixed K) | *which* by **the question** | **fixed** | **Yes** | **THE WIN** (high-res/Qwen); fails on frozen low-res |
| 4 | **Dynamic COUNT** (per-sample budget) | *how many* | **varies** | indirectly | **THE MIRAGE** (ties static; oracle small; predictor ≈0) |

The single sentence the whole thesis defends: **for the *which* decision, conditioning on the question is the
lever (#3 ≫ #2 in the right regime); for the *how many* decision, adapting per sample buys essentially
nothing (#4 ≈ #2).** "Dynamic" survives as *question-dependent selection*, not as a per-sample token count.

A crucial subtlety you must keep straight: in the VQAv2 "dynamic" experiment, #3 and #4 are **deliberately
separated** — the token *ranking* is plain CLS (image-only, NOT question-conditioned), and only the *count*
is learned. That run is an **Approach-4 experiment**, not Approach-3. The genuine Approach-3 (question
chooses which tokens) lives in the GQA/TextVQA probes and the high-res/Qwen track.

---

# 1. DENSE BASELINE

**Definition in this repo:** keep all visual tokens, no pruning, frozen backbone, generation protocol
(`model.generate()`, answer head bypassed), official scorer. It is the reference everything else is measured
against (retention = method/dense).

## 1.A Models used for dense
- **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`) — 576 visual tokens @336px. The low-res diagnostic backbone.
- **LLaVA-1.6-7B** (`llava-hf/llava-v1.6-vicuna-7b-hf`) — AnyRes tiling, ~2302 visual tokens avg. The bridge model.
- **Qwen-2.5-VL-7B** (`Qwen/Qwen2.5-VL-7B-Instruct`) — native dynamic resolution, ~1286 tokens avg on DocVQA. The SOTA backbone (headline).
- **Qwen-2.5-VL-3B / 32B** — only used for the budget-mirage generality (raw data; L10, not as a dense result).
- *(Elastic LoRA LLaVA-1.5 exists but is **excluded** — the only non-frozen model.)*

## 1.C Dense results (model × dataset)

| Model | Dataset | File | Metric | Dense score | n | Vis tokens | Role |
|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | VQAv2 | `vqav2/dense_pad/generation_eval_10k.json` | VQA acc | **76.44** | 10,000 | 576 | baseline for static/dynamic-budget |
| LLaVA-1.5 | GQA testdev | `gqa/testdev_dense_honest_bs1_*/metrics.json` | exact | **61.42** (pub 62.0) | 12,578 | 576 | reproduction anchor (L1) |
| LLaVA-1.5 | TextVQA-OCR | `gqa/textvqa_analysis_ocr.json` | M4C soft | **57.65** (pub 58.2) | 5,000 | 576 | reproduction anchor (L1) |
| LLaVA-1.5 | TextVQA-noOCR | `gqa/textvqa_analysis_noocr.json` | M4C soft | **46.73** | 5,000 | 576 | "can't read" floor |
| LLaVA-1.5 | POPE | `gqa/week1_all_numbers.json` `pope_dense` | acc / F1 | **84.64 / 85.78** (F1 pub 85.9) | 9,000 | 576 | reproduction anchor (L1) |
| LLaVA-1.5 | ScienceQA-IMG | (ledger / `sqa_dense_full_*`) | acc | **65.15** (pub 66.8) | 2,017 | 576 | reproduction anchor (L1) |
| LLaVA-1.6 | TextVQA-noOCR | `highres/eval_textvqa_highres_selcompare.json` | M4C soft | **61.40** | 300 | ~2302 | "high-res reads better" (+19pp vs LLaVA-1.5) |
| LLaVA-1.6 | DocVQA | `highres/eval_highres_docvqa.json` | ANLS | **67.19** (n=300) / 67.89 (n=1000) | 300/1000 | ~2302 | high-res ceiling |
| LLaVA-1.6 | ChartQA | `highres/eval_highres_chartqa.json` | relaxed | **50.33** | 300 | ~2302 | high-res ceiling |
| LLaVA-1.6 | GQA (highres) | `highres/eval_gqa_highres.json` | exact | (diagnostic) | — | ~2302 | resolution×task context |
| **Qwen-2.5-VL-7B** | **DocVQA** | `highres/qwen_kcurve_docvqa.json` (`full`) | ANLS | **97.19** (n=200); 94.85 @ n=50; pub 95.7 | 200 | ~1286 | **headline dense ceiling** |
| Qwen-2.5-VL-7B | ChartQA | `highres/qwen_kcurve_chartqa.json` (`full`) | relaxed | **85.5** (n=200); 76.0 @ n=100; pub 87.3 | 200 | ~508 | second headline dataset |
| Qwen-2.5-VL-7B | InfoVQA | (ledger / raw data) | ANLS | **74.7** (pub 82.6) | ~400 | — | generality (L10, not saved) |

**FLOPs/latency for dense** (kept separate, see §6.E): Qwen-7B DocVQA dense = **7.302 TFLOPs** @ 1286 tokens
(`qwen_flops_summary.json`, FastV Eq.5, T=28/D=3584/M=18944/N_TEXT=40). LLaVA-1.6 dense generate() latency =
**355 ms** @ 2252 tokens (`llava_latency.json`, n=40, decode-only). LLaVA-1.5 GQA dense = 3.17 TFLOPs.

## 1.D What dense is used for
1. **Reproduction check (the trust anchor, L1).** The frozen LLaVA-1.5 dense numbers land within ≤1.65pp of
   published (GQA −0.58, TextVQA-OCR −0.55, POPE-F1 −0.12, SQA −1.65). This is what makes the *negatives*
   (budget ties, naive selection fails) trustworthy — they aren't a broken pipeline.
2. **Ceiling / retention denominator.** Every pruning result is reported as "retains X% of dense."
3. **Baseline for pruning.** Static, dynamic-which, and dynamic-count are all measured as deltas from dense.
4. **NOT a contribution.** Dense is not a result; it is the reference. (Note: Qwen DocVQA dense 97.19 at n=200
   is *above* published 95.7 — small-n optimism; rerun at full-val before quoting to 0.1pp.)

---

# 2. STATIC PRUNING

**Definition:** remove visual tokens using an **image-only / question-independent** signal, at a **fixed K**.
Engine = `StaticPrunedLlava` (`src/models/static/static.py`); it **physically removes** dropped tokens
(rebuilds `inputs_embeds = [prefix | K projected visual | suffix]`) so the LLM processes a genuinely shorter
sequence (prune-before-LLM, FlashAttention-compatible, real speedup).

## 2.A / 2.B Methods implemented

| Method | File | Uses question? | Physically removes? | Signal | Models |
|---|---|---|---|---|---|
| `none` (K=576 sanity) | `static.py` | No | — | keep all | LLaVA-1.5 |
| **`cls_attn`** (VisionZip "dominant") | `static.py` | **No** | Yes | CLIP CLS→patch attention @ layer −2, head-avg, top-K | LLaVA-1.5 |
| `l2_norm` | `static.py` | No | Yes | L2 norm of CLIP patch features | LLaVA-1.5 |
| `random` | `static.py` | No | Yes | per-sample seeded uniform subset | LLaVA-1.5 |
| `spatial_uniform` | `static.py` | No | Yes | deterministic 24×24 stride sets | LLaVA-1.5 |
| `fastv_style` | `static.py` | No (LM-side) | Yes | LM layer-2 received attention (RoPE-biased; ablation only) | LLaVA-1.5 |
| **VisionZip** (dominant + contextual **merge**) | `pruning/static/visionzip.py` | No | Yes | CLS-attn dominant + cosine-merge of the rest | LLaVA-1.5 |
| `uniform` / `norm` (Qwen blind) | `pruning/.../qwen_pruner.py` | No | Yes | stride / visual-embed L2-norm | Qwen-2.5-VL |
| `attn` (CLS) / `base` / `uniform` (LLaVA-1.6) | `evaluate_textvqa_highres_kcurve.py` | No | Yes | CLIP CLS-attn / first-576 / stride | LLaVA-1.6 |

**K values tested:** LLaVA-1.5 static frontier `{64,128,144,192,288,432,576}` (VQAv2) and
`{144,192,288,432,576}` (GQA/TextVQA/POPE/SQA); plus fine-grained `{96,160,219,265,275,276,334,357}` on
VQAv2 for the per-type / matched-K instance-headroom analysis. Qwen/LLaVA-1.6 blind baselines run
`{32,64,128,256,512,...}`.

## 2.C / 2.D Static results

### LLaVA-1.5 static frontier (CLS-attn, the strong baseline)

| Dataset | Dense | K=144 | K=192 | K=288 | K=432 | n | File |
|---|---|---|---|---|---|---|---|
| VQAv2 (gen acc) | 76.44 | 74.44 | 75.31 | 75.82 | 76.27 | 10k | `vqav2/static_k*_pad/generation_eval_10k.json` (also K64 71.02, K128 74.27) |
| GQA testdev (exact) | 61.42 | 58.15 | 59.19 | 60.53 | 61.53 | 12,578 | `gqa/testdev_static_cls_attn_k*/metrics.json` |
| TextVQA-OCR (soft) | 57.65 | 55.97 | 56.55 | 56.40 | 57.39 | 5,000 | `gqa/textvqa_analysis_ocr.json` |
| TextVQA-noOCR (soft) | 46.73 | 44.80 | 45.36 | 45.69 | 46.59 | 5,000 | `gqa/textvqa_analysis_noocr.json` |
| POPE (mean F1) | 85.78 | 85.71 | 85.41 | 85.06 | 85.76 | 9,000 | `gqa/week1_all_numbers.json` `pope_static` |
| ScienceQA-IMG | 65.15 | (band only saved) | — | — | — | 2,017 | `gqa/sqa_analysis.json` (band 7.73%) |

**Reading it:** static CLS-attn is a *near-oracle, training-free* frontier. On GQA, K=288 retains **98.5%** of
dense (60.53 vs 61.42) at ~48% prefill-FLOP cut; K=432 actually equals dense (retention 100.2%). VisionZip ≈
CLS-attn (marginal). This is why the thesis treats CLS-attn as a **genuinely strong blind baseline**, not a
straw man.

### High-res blind static (the floor the QC selector must beat)

| Model | Dataset | Dense | blind K=128 | File |
|---|---|---|---|---|
| Qwen-7B | DocVQA | 97.19 | uniform **32.19** / norm 20.98 | `qwen_kcurve_docvqa.json` |
| Qwen-7B | ChartQA | 85.5 | uniform 61.0 / norm 54.5 | `qwen_kcurve_chartqa.json` |
| LLaVA-1.6 | DocVQA | 67.19 | CLS-attn **27.76** | `eval_highres_docvqa.json` |
| LLaVA-1.6 | TextVQA-noOCR | 61.4 | CLS-attn 44.2 | `eval_textvqa_highres_selcompare.json` |

## 2.E What static pruning proves
1. **A strong fixed-budget frontier exists for free.** No training, image-only saliency, near-dense at
   moderate K. This is the bar.
2. **It is the comparison point for BOTH dynamic axes.** Dynamic-count (#4) is measured against static at the
   *same average K*; dynamic-which (#3) is measured against static/blind at the *same fixed K* (so identical
   prune-before-LLM FLOPs → a clean matched-FLOPs *content* comparison).
3. **It exposes the position-bias trap.** `fastv_style` (LM-side, RoPE) is dominated by CLS-attn (CLIP-side,
   learned 2D positions), motivating CLIP-side scoring.
4. **Caveat for the high-res story:** Qwen has **no CLS-attention baseline implemented** — only `uniform`/
   `norm` floors (and `norm` < `uniform`). So Qwen's "selection win" margins are vs the *weakest* baseline
   (see §3.D risk).

---

# 3. DYNAMIC PRUNING — the two meanings, separated

> **This is the part to get exactly right.** "Dynamic" is overloaded. Below, Approach 1 = *which* tokens
> change with the question (count fixed); Approach 2 = *how many* tokens change with the sample (count
> varies). They are different experiments, different files, and **opposite conclusions**.

---

## APPROACH 1 — Dynamic WHICH tokens, FIXED count (question-conditioned selection)

**Meaning:** K is fixed (e.g. 128 for every sample); the **question** decides *which* K tokens survive.
This is the registered thesis idea and the one that **wins** — but only in the right regime.

### 3.1.A / 3.1.B Methods

| Method | File | Model | Dataset | Scoring signal | Needs full LLM forward to score? | Deployable? |
|---|---|---|---|---|---|---|
| **LM-attention Q-cond** | `pruning/.../question_cond.py` (`run_qcond_probe.py`) | LLaVA-1.5 | GQA, TextVQA | question→visual attention at LLM layers {2,5,8} | **Yes** (1 dense prefill) | No (teacher/upper-bound) |
| **CLIP-space Q-cond** | `pruning/.../clip_select.py` (`run_clip_probe.py`) | LLaVA-1.5 | GQA, TextVQA | cosine(CLIP visual-proj patch, CLIP text feature) | No (cheap) | Yes but **fails** |
| **Qwen mid-layer Q-cond** (`textattn`) | `pruning/.../qwen_pruner.py` | Qwen-2.5-VL-7B/3B | DocVQA, ChartQA | question→visual attention at **decoder layer 16** | **Yes** | No (teacher) → distilled |
| **LLaVA-1.6 high-res Q-cond** | `evaluate_textvqa_highres_kcurve.py` (`HighResPruner`, `textattn`) | LLaVA-1.6 | DocVQA, ChartQA, TextVQA | question→visual attention (DocVQA/Chart: L16; TextVQA kcurve hardcodes **L3 = early**) | **Yes** | No (teacher) |
| **Cheap student selector** | `models/distillation/student_selector.py` + `train_student.py` | Qwen-7B | DocVQA | cross-attn on **pre-LLM** features (ViT embeds + question embeds), distilled from L16 | **No (deployable!)** | **Yes** (below SOTA) |

### 3.1.C / 3.1.D Results — where it FAILS and where it WINS

**FAILS on frozen low-res LLaVA-1.5 (L4)** — naive question signals lose to image saliency:

| Dataset | K | Random | CLS (blind) | LM-attn Q-cond (best layer) | CLIP-space Q-cond | File |
|---|---|---|---|---|---|---|
| TextVQA-noOCR | 64 | 24.33 | **43.23** | 37.65 (L8) → **−5.58** | 11.21 → **−32.04** | `qcond_textvqa_noocr_*/results.json`, `week1_all_numbers.json` |
| GQA | 64 | 56.90 | 54.90 | 58.43 (L8) → +3.5 *(but CLS < random here)* | 45.73 → −9.2 | `qcond_gqa_*/results.json`, `week1_all_numbers.json` |

**WINS on high-res / Qwen (L5–L7)** — a mid-layer question signal dominates blind selection:

| Model | Dataset | K | Blind | **Q-cond (textattn)** | Δ vs blind | Dense | n | File |
|---|---|---|---|---|---|---|---|---|
| Qwen-7B | DocVQA | 64 | uniform 23.93 | **84.73** | **+60.8** | 97.19 | 200 | `qwen_kcurve_docvqa.json` |
| Qwen-7B | DocVQA | 128 | uniform 32.19 | **92.18** | **+59.99** | 97.19 | 200 | `qwen_kcurve_docvqa.json` |
| Qwen-7B | DocVQA | 256 | uniform 61.12 | **94.72** | +33.6 | 97.19 | 200 | `qwen_kcurve_docvqa.json` |
| Qwen-7B | ChartQA | 128 | uniform 61.0 | **83.5** | +22.5 | 85.5 | 200 | `qwen_kcurve_chartqa.json` |
| LLaVA-1.6 | DocVQA | 128 | CLS 27.76 | **55.13** | +27.4 | 67.19 | 300 | `eval_highres_docvqa.json` |
| LLaVA-1.6 | DocVQA | 128 | CLS 23.11 | **54.02** | +30.9 | 67.89 | 1000 | `eval_highres_docvqa_n1000.json` |
| LLaVA-1.6 | ChartQA | 128 | CLS 25.67 | **43.0** | +17.3 | 50.33 | 300 | `eval_highres_chartqa.json` |

**Retention / efficiency for the headline (Qwen-7B DocVQA, teacher selector):** K=128 → **94.8% of dense at
86.8% token / 87.9% FLOP reduction** (`qwen_flops_summary.json`); K=512 ≈ dense (97.16). At K=128, blind
retains only 22–33%.

**It is genuinely the question, not disguised saliency (L7):** real-question selection 93.84 vs
mismatched-question 36.98 vs blind 32.19 → **92.2% of the gain is question-driven** (`qwen_control_docvqa.json`).

**The signal is mid-layer (L6):** Qwen-7B layer sweep (K=128, DocVQA) — L2 44.6, L4 45.7, L8 47.7, L12 73.7,
**L16 93.2**, L20 94.9, L24 91.2 (`qwen_layer_sweep.json`); 3B peaks at L24/28 (~88). Early-layer attention
(FastV-style) is far below — the signal *emerges* with depth.

### Why it fails low-res but wins high-res (the thesis bridge, R5)

| | LLaVA-1.5 (576 tok, low-res) | High-res / Qwen (1286–2302 tok) |
|---|---|---|
| Are there redundant tokens to drop? | Few — 576 tokens are already a tight global view | Many — most high-res tokens are background/blank |
| Naive signal | frozen CLIP-space cosine / **early-layer** LM attention | **mid-layer (L16)** question→visual attention |
| Result | Q-cond **loses** to CLS saliency (−5.58 to −32pp) | Q-cond **dominates** blind (+17 to +60pp) |

**The axis is resolution × task token-demand × which-layer-signal — NOT training.** Both regimes are
**frozen**. The low-res negative is not "selection never works"; it is "the *naive* signal on a token-tight
model doesn't beat saliency." High-res gives real redundancy to exploit, and the *mid-layer* signal is the
one that captures cross-modal relevance. **Never write "training fixed it."**

### The deployable payoff — the distilled student (L11–L12)

The winning selector is a **teacher**: it needs a full LLM forward to read L16 attention, so it is an
analysis/upper-bound, not a deployable method (E4). The student (`CheapQCSelector`) is distilled to predict
the L16 map from **pre-LLM features only (no decoder forward)**:

| K | dense | teacher | **student** | blind | recovery | gate | File |
|---|---|---|---|---|---|---|---|
| 64 | 95.03 | 84.16 | 51.13 | 20.73 | 47.9% | fail | `distill/gate_K64.json` |
| **128** | 95.0 | 91.1 | **65.9** | 31.2 | **57.9%** | **PASS** | `distill/gate_12k.json` |
| 256 | 95.03 | 92.41 | 72.32 | 55.88 | 45.0% | fail | `distill/gate_K256.json` |

Student is **99.2% question-driven** (`control_12k.json`), and **data > epochs** (3K→12K data lifts
recovery 46→58%; 10→25 epochs overfits 58→52%, `gate_12k_25ep.json`). The "recovery law": ~58% of the
mid-layer signal is cheaply recoverable, ~42% is irreducibly mid-LLM. Honest gap: student ~69% retention vs
SOTA cheap selectors ~90%.

---

## APPROACH 2 — Dynamic COUNT / per-sample budget

**Meaning:** the system decides *how many* tokens to keep per sample (easy → fewer, hard → more). The token
*ranking* in the main experiment is **CLS (image-only)** — so this is purely a *count* experiment, isolated
from Approach 1 on purpose.

### 3.2.A / 3.2.B Methods

| Method | File | Model | Dataset | Inputs to budget decision | Predicts | K range | Ranking | Status |
|---|---|---|---|---|---|---|---|---|
| **BudgetController** (learned) | `models/dynamic_budget/{budget_controller,token_selector,llava_wrapper}.py`; trained by `train_dynamic.py` | LLaVA-1.5 | VQAv2 | question embed (512-d) + 7 score-stats + 4-type embed; **no raw image** | keep-ratio → K | 64–432 | **`cls_only`** | thesis-main (the budget negative, L2) |
| Question-type target | same (`budget_loss_type=question_type_target`) | LLaVA-1.5 | VQAv2 | question type → target ratio `[.38,.48,.58,.62]` | per-type K | 64–432 | cls_only | thesis-main |
| **Budget oracle** (monotone, noise-corrected) | `pruning/dynamic_budget/qwen_oracle{,_qc}.py` | Qwen-7B | DocVQA, ChartQA | labels (per-sample best K, noise-removed) | upper-bound K | ladder | uniform / QC | thesis-main (L8) |
| **Budget predictor** (MLP, 5-fold CV) | `pruning/dynamic_budget/qwen_budget_eval.py` (+ `qwen_budget_data.py`) | Qwen-7B | DocVQA | 5 QC-attention-distribution features; **no image** | sufficiency-K | ladder | QC | thesis-main (L9) |
| Hard-tail predictability | `pruning/dynamic_budget/qwen_budget_robust.py` | Qwen-7B | DocVQA | 6 attention features | "needs >32 tok?" | — | QC | diagnostic |
| **Confidence cascade** | `evaluation/.../cascade_pass.py`, `run_speculative_testdev.py`, `cascade_sweep.py` | LLaVA-1.5 | GQA/POPE/VQAv2/TextVQA/SQA | decoded-answer confidence at base-K | escalate or not | {144→288} | cls | realizable mechanism (ties) |
| Binary route / qtype maps | `data/budget_oracle/*` (generator NOT in repo) | LLaVA-1.5 | VQAv2 | CLS/patch summary stats (image-derived) or type | k144 vs dense | binary | cls | **retired / excluded** |

### 3.2.C / 3.2.D Results & conclusion

**(i) Learned budget ties tuned static (L2), VQAv2:**

| | gen acc 10k | avg K | K-std | File |
|---|---|---|---|---|
| Dynamic (learned per-type budget, CLS ranking) | **75.76** | 264.3 | 44.9 | `vqav2/dynamic_150k_clsonly/RESULT_SUMMARY.json` |
| Static uniform K=265 (matched cost) | 75.71 | 265 | 0 | static frontier |
| **Δ** | **+0.05pp** | | | |

The controller *works mechanically* (held-out std 44.9, correct per-type ordering yes/no 219 < attr 277 <
count 336 < spatial 345, generalizes) — it simply **doesn't help**: the per-type accuracy curves are
near-identical concave shapes, so reallocating a fixed average budget across types can't beat uniform (Jensen).

**(ii) The oracle is small — and it's mostly an efficiency number, not accuracy (L8):**

| Selector | Dataset | best static | naive oracle (band) | **monotone (honest)** | @ avg tok | FLOPs-matched gain | File |
|---|---|---|---|---|---|---|---|
| **QC (good)** | DocVQA | 97.19 | 99.01 (+1.82) | 98.0 | 90.6 | **+7.50pp** | `qwen_oracle_docvqa_qc.json` |
| uniform (blind) | DocVQA | 97.19 | 98.74 (+1.56) | 98.0 | 337.5 | **+20.66pp** | `qwen_oracle_docvqa_uniform.json` |
| QC (good) | ChartQA | 86.0 | 92.0 (+6.0) | 85.5 | 121.5 | +2.66pp | `qwen_oracle_chartqa_qc.json` |

Two things to read here:
- **The "+7.50pp" is FLOPs-matched** (oracle 98.0% @ 90 tok vs static *dragged down to 90 tok* ~90.5%). The
  **pure-accuracy** headroom over the best *fixed* K is only **+0.81pp** (98.0 − 97.19). State the matched-budget
  qualifier or a careful reader will catch it (`THESIS_VERIFICATION.md` §1.4).
- **Budget benefit is a symptom of weak selection.** With blind selection the oracle is +20.66pp @ 337 tok;
  a good QC selector *collapses the per-sample spread* (everything fits in fewer tokens) → the oracle drops to
  +7.50pp @ 90 tok. So the apparent "dynamic budget pays off" shrinks exactly as selection improves.

**(iii) The realizable predictor captures ≈ none of even that small oracle (L9):**

| | gain vs best fixed | File |
|---|---|---|
| Oracle (sufficiency-K) | +4.99pp | `qwen_budget_eval.json` |
| **Trained MLP (5-fold CV)** | **+0.09pp = 1.8% of oracle** | `qwen_budget_eval.json` |
| Spread rule | +0.72pp | `qwen_budget_eval.json` |
| Hard-tail AUC ("needs >32 tok") | 0.643 (base rate 0.195) | `qwen_budget_robust.json` |

Why it fails: 81% of samples are already correct at the 32-token minimum, and the hard tail is **weakly
identifiable but unfixable** (more tokens don't fix it — it fails at full resolution too). *(Honest wording
fix: the saved AUC is 0.643, which crosses the script's own 0.6 "might help" threshold; the FINDINGS text's
"0.59 ≈ unpredictable" is slightly off. The conclusion holds because realized gain ≈0 despite AUC 0.64 —
say "weakly identifiable but unfixable," not "unpredictable.")*

**(iv) The realizable mechanism (confidence cascade) only traces the static frontier** — never robustly above
it (`results_frozen/all.json` "cascade"; VQAv2 best +0.10pp, evaporates under compute-matched accounting).

### How to explain "budget is a mirage" honestly in your thesis
> *Per-sample budgeting has three strikes. (1) The honest, noise-corrected oracle headroom over the best
> fixed budget is small — about +0.8 accuracy points on Qwen-DocVQA (the +7.5pp figure is a FLOPs-matched
> efficiency number, not accuracy). (2) That headroom shrinks as selection improves, because a good selector
> already fits most samples in the minimum budget — so the budget benefit is largely a symptom of weak
> selection. (3) A realistic trained predictor captures ≈1.8% of the oracle, because the budget-sensitive
> tail is small and dominated by unfixable samples. A learned per-sample-K head with raw image features was
> never tried, but the ceiling it would chase is single-digit and mostly efficiency, not accuracy. Conclusion:
> invest in selection; report per-sample budgeting as a rigorously characterized negative.*

---

# 4. MODEL-BY-MODEL UNDERSTANDING

### LLaVA-1.5-7B (frozen) — the diagnostic foundation ("v1")
- **Where:** `configs/{dense,static,dynamic_budget}/*`, `src/models/{dense,static,dynamic_budget}`, all
  `src/evaluation/{gqa,vqa,textvqa,pope,scienceqa}`, `results/thesis_main/{gqa,vqav2}`.
- **Datasets:** VQAv2, GQA, TextVQA, POPE, ScienceQA.
- **Methods tested:** dense ✓, static (cls_attn/random/spatial/l2/fastv/VisionZip) ✓, dynamic-which (LM-attn
  & CLIP-space probes) ✓, dynamic-count (BudgetController + cascade) ✓. No student.
- **Results:** dense reproduces published (L1); static frontier near-oracle; budget ties static (L2,
  +0.05pp); naive Q-cond selection fails (L4).
- **Contribution:** builds the measurement instrument + the two negatives (budget capped, naive selection
  fails). **Main thesis evidence.**
- **Caveats:** 576 tokens (token-tight → little to prune); the LM-attn Q-cond "winner" on GQA needs a full
  forward (not FLOPs-cheap) and CLS there is below random; VQAv2 is a 10k subset while others are full-split.

### LLaVA-1.6-7B (frozen) — the bridge model
- **Where:** `src/evaluation/textvqa/evaluate_textvqa_highres*`, `evaluate_highres_docchart.py`,
  `evaluate_gqa_highres.py`, `llava_latency.py`, `results/thesis_main/highres/eval_*highres*`.
- **Datasets:** TextVQA, DocVQA, ChartQA, GQA (high-res).
- **Methods:** dense ✓, static blind (CLS-attn/base/uniform) ✓, dynamic-which (textattn) ✓. No budget head;
  no student.
- **Results:** high-res reads +19pp better (TextVQA 42→61); Q-cond beats blind +17–31pp; the **only** source
  of the 3.31× latency number and the cross-family budget-mirage check (L10).
- **Contribution:** establishes "resolution × task creates real pruning room"; bridges low-res→SOTA.
  **Appendix evidence + two retained numbers (latency E3, L10 cross-family).**
- **Caveats:** different model from the Qwen accuracy headline (can't share an axis: dense 67 vs 97);
  the **TextVQA kcurve hardcodes `textattn_layer=3` (early)** so its textattn *underperforms* blind attn
  (42.0 vs 44.2 @K128) — the DocVQA win used a layer-16 harness; latency is decode-only, n=40, and "kept"
  tokens are `arange(first-K)`, not the real selector.

### Qwen-2.5-VL-7B (frozen) — the SOTA headline ("v2")
- **Where:** `src/pruning/question_conditioned_selection/qwen_pruner.py`,
  `src/pruning/dynamic_budget/qwen_*`, `src/evaluation/docvqa/qwen_*`, `src/analysis/qwen_flops.py`,
  `src/models/distillation/*`, `results/thesis_main/highres/qwen_*` + `distill/*`.
- **Datasets:** DocVQA, ChartQA (InfoVQA raw only).
- **Methods:** dense ✓, static blind (uniform/norm only) ✓, dynamic-which (textattn L16, the WIN) ✓,
  dynamic-count (oracle + predictor, the MIRAGE) ✓, distillation (student) ✓.
- **Results:** L5 (+59.99pp selection), L6 (mid-layer), L7 (92.2% question-driven), L8 (+7.50pp oracle),
  L9 (predictor +0.09pp), L11–L12 (student 57.9% recovery). **Primary main thesis evidence.**
- **Caveats:** runs in a **separate `qwen_env`** (transformers 4.51); headline cells are **n=200**; only
  `uniform`/`norm` blind baselines exist (**no CLS-attn equivalent**), so "+60pp" is vs the weakest floor —
  a fair baseline likely shrinks it to +20–30pp; teacher needs a full forward (FLOPs exclude it).

### Qwen-2.5-VL-3B / 32B (frozen) — generality only
- **Where:** `qwen_layer_sweep_3b.json`, `results/paper_candidates/qwen_budget_data_docvqa_{3b,32b}.json`,
  `scripts/data/dl_qwen{3b,32b}.py`.
- **Methods/Results:** budget-mirage data collected (raw); 3B layer sweep (peak L24/28 ~88). **The per-model
  eval gains are NOT saved** → L10 is narrative-only until recomputed.
- **Contribution:** would make the budget mirage a *cross-scale/family* phenomenon. **Appendix (→ main only
  if recomputed + saved).** 32B was 4-bit + reduced resolution to fit one GPU.

### Elastic LoRA LLaVA-1.5 — excluded
- The only *trained-backbone* model; `results/archived/stage1_*`. **Ignore** for the thesis (it would muddy
  the "all wins are frozen" message). Keep code/checkpoints for provenance.

---

# 5. DATASET-BY-DATASET UNDERSTANDING

| Dataset | Stored / loaded | Models | Methods run | Metric | Headline result | Role |
|---|---|---|---|---|---|---|
| **DocVQA** | HF hub (`docvqa.py`, parquet) | Qwen-7B/3B/32B, LLaVA-1.6 | dense, static-blind, **Q-cond**, budget oracle/predictor, **distill** | ANLS | selection +59.99pp @K128; budget mirage; student 57.9% | **MAIN** (the selection win + budget mirage live here) |
| **VQAv2** | local 20G (COCO + Q/A JSON) | LLaVA-1.5 | dense, static frontier, **dynamic-count** | VQA consensus | budget ties static +0.05pp | **MAIN** (the budget negative, L2) |
| **GQA** | local 21G (148,854 imgs) | LLaVA-1.5 (+1.6 diag) | dense, static, Q-cond probe, oracle band, cascade | exact | dense 61.42 (L1); band 9.13% (L3); naive Q-cond fails (L4) | **MAIN** (reproduction + band + selection negative) |
| **TextVQA** | local 6.7G (OCR + noOCR) | LLaVA-1.5, LLaVA-1.6 | dense, static, Q-cond probe | M4C soft | naive Q-cond −5.58pp (L4); high-res +19pp | **MAIN** (the clearest naive-selection-fails case) |
| **ChartQA** | HF hub | Qwen-7B, LLaVA-1.6 | dense, static-blind, Q-cond, oracle | relaxed | selection +22.5pp; oracle +2.66pp | **Appendix** (selection generalizes, smaller gain) |
| **POPE** | local 1.1M (3 subsets) | LLaVA-1.5 | dense, static, cascade | acc/F1 | dense F1 85.78 (L1); band 4.93% | **Appendix** (4-benchmark reproduction + band ordering) |
| **ScienceQA-IMG** | local 125M parquet | LLaVA-1.5 | dense, static | acc | dense 65.15 (L1); band 7.73% | **Appendix** (reproduction + band ordering) |
| **InfoVQA** | HF hub (raw data only) | Qwen-7B | budget data (eval not saved) | ANLS | (narrative) most token-hungry; mirage holds | **Appendix→main if recomputed** (L10) |
| **LLaVA-665K mix** | local 34G (226,532 imgs) | elastic LoRA | Stage-1 training | — | (excluded) | **Ignore** (only the excluded elastic uses it) |
| **budget_oracle/** | local 24M diagnostics | LLaVA-1.5 (retired) | binary route, qtype maps | classification-era | (excluded; no generator code) | **Ignore** (historical) |

### Recommendations
- **A. Main thesis datasets:** **DocVQA** (selection win + budget mirage), **VQAv2** (budget ties static),
  **GQA** (reproduction + oracle band + selection negative), **TextVQA** (naive selection fails / resolution
  effect). These four tell the whole "Selection over Budget" arc.
- **B. Appendix:** ChartQA (selection generalizes), POPE + ScienceQA (4-benchmark reproduction + band
  ordering), LLaVA-1.6 high-res (resolution×task), InfoVQA (generality, *after* recompute).
- **C. Can be ignored as thesis evidence:** the LLaVA-665K mix and `data/budget_oracle/*` (retired; the
  latter has no generator code and a different protocol).
- **D. Deletion guidance (recommendation only — DELETE NOTHING NOW):**
  - **Do NOT delete yet:** `results/paper_candidates/*` (raw inputs for the L10 recompute, the single
    highest-leverage paper fix); all `results/thesis_main/*` + `distill/*.pt` (the cited numbers/checkpoints);
    `data/{vqav2,gqa,textvqa,pope,scienceqa}` (all four main + two appendix datasets depend on them).
  - **Safe to delete later (large, only the excluded track / regenerable):** `data/llava_mix/` (34G, elastic
    only), `results/archived/stage1_*` (elastic checkpoints), any `feature_cache/` from the retired
    classification pipeline. `data/budget_oracle/` (24M) is small — keep as provenance even though excluded.

---

# 6. FINAL SYNTHESIS

## 6.A The clean final story
Visual tokens dominate VLM prefill cost. A pruner answers two questions per sample — *which* tokens and *how
many*. On **frozen** backbones across two regimes (low-res LLaVA-1.5 → high-res LLaVA-1.6 / SOTA Qwen-2.5-VL),
I measure both honestly (matched-FLOPs, strong/floor baselines, a dense pipeline that reproduces published
numbers, and a monotone oracle-noise correction) and find an **asymmetry: *which* is the lever, *how many* is
a mirage.** Question-conditioned, **mid-layer** selection retains ~95% of dense accuracy at ~88% fewer prefill
FLOPs (Qwen DocVQA K=128), while per-sample budgeting only ties a tuned static budget and its honest oracle
headroom is small and unrealizable. The expensive teacher signal is distilled into a cheap, no-decoder-forward
student that recovers ~58% of the gain (deployable, below SOTA).

## 6.B The exact baselines
- **Dense** (ceiling / retention denominator).
- **Static CLS-attention** (LLaVA) / **uniform** (Qwen) — the question-*independent* fixed-K frontier; the
  thing both dynamic axes must beat. *(Limitation: Qwen has no CLS-attn baseline; its blind floor is `uniform`.)*
- **Static at matched average K** — the baseline for the dynamic-count comparison.
- **Random / spatial / norm / fastv_style** — floors and ablations.

## 6.C Your actual method (the contribution)
A **rigorous selection-vs-budget decomposition** of question-conditioned visual-token pruning, plus the
honest-measurement methodology (the **monotone oracle-noise correction** + matched-FLOPs criterion), showing
**selection dominates and per-sample budget is a mirage** across regimes — and a **cheap distilled
question-conditioned selector** that realizes most of the expensive teacher signal. *Not* a new pruning
mechanism and *not* a SOTA number — the contribution is the decomposition, the budget negative, and the
recoverability law.

## 6.D The three things, distinguished precisely

| | Static pruning | Q-conditioned fixed-K (dynamic WHICH) | Dynamic budget (dynamic COUNT) |
|---|---|---|---|
| Question used? | No | **Yes (chooses which)** | No for ranking; question-*type* only for count |
| K per sample | Fixed | Fixed | **Varies** |
| Signal | CLIP CLS-attn (saliency) | mid-layer question→visual attention | learned keep-ratio MLP / oracle |
| Cost to score | free | full LLM forward (teacher) → cheap student | cheap |
| Result | strong frontier (the bar) | **+17 to +60pp over blind (high-res)** | **+0.05pp over static (ties)** |
| Verdict | baseline | **the win** | **the mirage** |

## 6.E What results support each claim

| Claim | Evidence (file) |
|---|---|
| Dense reproduces published (trustworthy negatives) | `results_frozen/all.json` → 61.42/57.65/85.78/65.15 |
| Static is a near-oracle fixed frontier | `testdev_frontier_analysis.json`, VQAv2 static curve |
| Q-cond selection dominates (high-res) | `qwen_kcurve_docvqa.json` (+59.99pp), `eval_highres_docvqa.json` (+27–31pp) |
| The signal is mid-layer | `qwen_layer_sweep.json` (L16 93.2 ≫ L2 44.6) |
| Selection is genuinely question-driven | `qwen_control_docvqa.json` (92.2%) |
| Naive Q-cond selection fails on frozen low-res | `qcond_textvqa_noocr_*` (−5.58), `week1_all_numbers.json` (CLIP −32.04) |
| Per-sample budget ties static | `dynamic_150k_clsonly/RESULT_SUMMARY.json` (+0.05pp) |
| Oracle budget headroom is small | `qwen_oracle_docvqa_qc.json` (+7.50pp FLOPs-matched / +0.81pp accuracy) |
| Predictor captures ≈ none | `qwen_budget_eval.json` (+0.09pp = 1.8%), `qwen_budget_robust.json` (AUC 0.643) |
| Budget benefit is a symptom of weak selection | `qwen_oracle_docvqa_uniform.json` (+20.66pp) vs `_qc` (+7.50pp) |
| Cheap student recovers majority | `distill/gate_12k.json` (57.9% recovery, PASS) |
| Efficiency (kept separate) | `qwen_flops_summary.json` (87.9% FLOP @K128), `llava_latency.json` (3.31×) |

## 6.F Claims to avoid (verbatim from the project's own audits)
- ❌ "training fixed it" — every headline win is **frozen** (R5).
- ❌ "we beat / match SOTA" — no strong published baseline was run (no FEATHER/CDPruner/HiRED/VisionSelector
  code in repo).
- ❌ unqualified "+60pp selection win" — it is vs the **weakest** baseline (uniform); state model/task/K/
  baseline/n; a fair CLS-equivalent Qwen baseline likely shrinks it to **+20–30pp**.
- ❌ "+7.50pp budget headroom" without the matched-budget qualifier — the **pure-accuracy** ceiling over the
  best fixed K is only **+0.81pp**.
- ❌ "budget mirage generalizes across families/scales" as proven — **L10 is narrative-only** until recomputed
  and the per-model JSONs are saved.
- ❌ unqualified "88% FLOP reduction" — analytical prefill, **excludes** the teacher selector's own forward.
- ❌ "the hard tail is unpredictable" — say **"weakly identifiable (AUC 0.64) but unfixable."**
- ❌ citing the LLaVA-1.6 latency as the method's latency — it's a different model, decode-only, n=40,
  `arange` tokens.

## 6.G The final thesis contribution (one paragraph)
> *I contribute a rigorous, honest decomposition of question-conditioned visual-token pruning into its two
> levers — selection (which tokens) and per-sample budget (how many) — and show, across a frozen low-resolution
> diagnostic model and high-resolution / SOTA backbones, that **selection is the lever and per-sample budget
> is a mirage**. The methodological contribution is the measurement framework: a dense pipeline that reproduces
> published accuracy, matched-FLOPs comparison, and a monotone oracle-noise correction that separates real
> headroom from evaluation noise. The empirical contributions are (i) a mid-layer question-conditioned
> selection signal that retains ~95% of dense accuracy at ~88% fewer prefill FLOPs where blind selection
> collapses; (ii) a budget negative — the honest oracle headroom is small, shrinks as selection improves, and
> a trained predictor captures ≈0; and (iii) a cheap distilled selector that realizes ~58% of the expensive
> teacher signal with no decoder forward, plus the recoverability law that ~42% of the signal is irreducibly
> mid-LLM. The result is practical guidance — invest in selection, not per-sample budgeting — backed by a
> reusable diagnostic, with all headline wins obtained on frozen models.*

---

### Appendix: the one diagram to draw

```
                     keep ALL tokens
   DENSE  ─────────────────────────────────────────►  ceiling (576 / 1286 / 2302 tok)
            │
            │ fix K, drop the rest
            ▼
   STATIC  ─ image saliency (CLS-attn) ─► strong frontier  ◄── the bar to beat
            │
   ┌────────┴──────────────────────────────┐
   │ WHICH (fix count, question chooses)    │ COUNT (vary count per sample)
   ▼                                        ▼
  DYNAMIC-WHICH                            DYNAMIC-COUNT
  low-res naive  → FAILS (−5 to −32pp)     learned budget → TIES static (+0.05pp)
  high-res L16   → WINS (+17 to +60pp)     oracle small (+0.8pp acc) → predictor ≈0
  → distill to cheap student (57.9%)       → "budget is a mirage"
  ════════════ THE LEVER ════════════      ════════════ THE MIRAGE ════════════
```
