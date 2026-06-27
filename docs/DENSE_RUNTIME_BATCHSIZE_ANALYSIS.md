# DENSE RUNTIME & BATCH-SIZE ANALYSIS (Phase 1 decision support)

*Planning analysis — written 2026-06-27. No code changed, no experiments run, no GPU used. Feeds the
VQAv2-subset decision and the batch-size decision in `docs/DENSE_DECISION_LOCK.md`. All runtimes are
**estimates** grounded in measured per-sample timings from the saved runs (cited in §1.4); treat Qwen-on-VQAv2
as ±50% because that harness does not exist yet.*

> **DECISION (2026-06-27): VQAv2 final subset = 25,000 stratified validation questions, seed 42 — LOCKED**
> (`DENSE_DECISION_LOCK.md` #2; manifest `configs/final_scope/sample_ids/vqav2.json`). **20k and 30k below are
> retained only as documented alternatives**, not as open choices. The §1.7 recommendation and §4 table are kept
> as the rationale that led to the 25k choice.

---

## 1. VQAv2 final subset size — 20k vs 25k vs 30k

### 1.1 Coverage (% of full VQAv2 validation = 214,354 questions)
| Subset | % of full val | vs current 10k |
|---|---|---|
| 10k (current, reference only) | 4.67% | — |
| **20k** | 9.33% | 2× |
| **25k** | 11.66% | 2.5× |
| **30k** | 14.00% | 3× |
| full (214,354) | 100% | 21× (out — too costly) |

### 1.2 Statistical reliability (95% CI half-width on the accuracy, p≈0.76)
`SE = sqrt(p(1−p)/n)`, CI ≈ ±1.96·SE.
| Subset | ± 95% CI (absolute acc) | Per-type cell (~¼ of n) ± CI |
|---|---|---|
| 10k | **±0.84pp** | ±~1.7pp |
| 20k | **±0.59pp** | ±~1.2pp |
| 25k | **±0.53pp** | ±~1.1pp |
| 30k | **±0.48pp** | ±~1.0pp |
| full | ±0.18pp | ±0.36pp |

**Two things to read here:**
- **Diminishing returns:** the big gain is 10k→20k (±0.84→±0.59). 20k→25k→30k tightens only ±0.59→±0.53→±0.48.
- **Method-vs-method comparisons are PAIRED** (all methods run the *same* sample IDs), so the precision that
  matters for "does WHICH beat static at the same budget" is the **per-sample disagreement rate**, not the
  marginal CI above — that paired power is already strong at 10k. **The larger subset mainly buys (a) a tighter
  *absolute* dense number and (b) reliable *per-question-type* breakdowns** (yes/no / attribute / counting /
  spatial), where counting/spatial cells are thin at 10k. If per-type tables matter, lean ≥25k; if only the
  overall number matters, 20k is already plenty.

### 1.3 Estimated runtime per dense run (one model, one pass, bs=1, mnt=64)
| Subset | LLaVA-1.5 (@~0.15 s/sample) | Qwen-2.5-VL-7B (@~0.6 s/sample, est.) |
|---|---|---|
| 20k | ~0.83 h | ~3.3 h |
| 25k | ~1.04 h | ~4.2 h |
| 30k | ~1.25 h | ~5.0 h |

### 1.4 Grounding (measured s/sample from saved runs)
- LLaVA-1.5 dense, bs=1: GQA **0.117 s/sample** (12,578, mnt=64), TextVQA **0.157 s/sample** (5,000, mnt=64) →
  use ~0.15 s/sample for VQAv2.
- Qwen-2.5-VL-7B dense, bs=1: DocVQA full **0.69 s/sample** (n=200). VQAv2 COCO images carry fewer visual
  tokens than DocVQA documents → use ~0.6 s/sample, **±50%** (new harness, not yet measured).

### 1.5 How many VQAv2 runs this multiplies into (downstream, per model)
| Method | VQAv2 passes (per model) |
|---|---|
| Dense (100%) | 1 |
| Static at 15/25/35/50/75% | 5 |
| Dynamic-WHICH at 15/25/35/50/75% | 5 |
| Dynamic-COUNT (final pass; oracle sweep reuses cached per-sample scores) | ~1 |
| **Per model** | **~12** |
| **× 2 models (LLaVA-1.5 + Qwen-7B)** | **~24** |

### 1.6 Total cost impact (both models, ~12 passes each)
Using the dense pass as the cost unit (note: static is *cheaper* — prune-before-LLM shortens the sequence;
the dynamic-WHICH **QC teacher is ~2×** because it adds a scoring forward; these roughly average to ≈ the dense
unit). Qwen dominates the bill.
| Subset | LLaVA-1.5 total (~12 passes) | Qwen-7B total (~12 passes) | **Combined VQAv2 (all methods, both models)** |
|---|---|---|---|
| 20k | ~10 h | ~40 h | **~50 h** (±50%) |
| 25k | ~12.5 h | ~50 h | **~62 h** (±50%) |
| 30k | ~15 h | ~60 h | **~75 h** (±50%) |

(These are VQAv2-only; GQA/TextVQA/DocVQA add their own, but those subset sizes are already locked.)

### 1.7 Recommendation → **chosen: 25k (LOCKED 2026-06-27)**
**25k is the balanced choice** *if* ~62 GPU-h of VQAv2 work across the full method matrix is acceptable: it
gives reliable per-type tables (±~1.1pp per type) at a modest step over 20k. **20k** is the cost-saver
(~50 h, ±0.59pp overall) and is fully defensible if per-type precision isn't a headline. **30k** buys little
over 25k (±0.53→±0.48) for ~+13 h — pick it only if you specifically want the per-type cells as tight as
possible. **The deciding question is per-type-table precision vs ~12–25 extra GPU-h on Qwen.**

---

## 2. Batch size for dense evaluation

### 2.1 Current batch-size support
| Harness | Model | Batch support today |
|---|---|---|
| `run_dense_testdev.py` (GQA) | LLaVA-1.5 | DataLoader `--batch_size` (default 4); `LlavaTestdevEval.generate` batches a padded `processor(...)` call → **supports bs>1**, but the accepted honest number used **bs=1** |
| `run_textvqa.py` (TextVQA) | LLaVA-1.5 | DataLoader `batch_size=1` **hardcoded** → bs=1 (the model could batch, the loader doesn't) |
| `generate_and_score.py` (VQAv2) | LLaVA-1.5 | `eval_batch_size` from config (default 1); loop batches → **supports bs>1** |
| `qwen25_dense_eval.py` (DocVQA) | Qwen-2.5-VL | per-sample loop (`for i in range(n)`) → **bs=1 by construction** |
| `qwen_pruner.py` (DocVQA full) | Qwen-2.5-VL | per-sample `_encode` + manual greedy decode with explicit M-RoPE → **bs=1 by construction** |

**LLaVA can batch; Qwen cannot (today)** — Qwen's manual M-RoPE greedy decode and ragged per-image
`image_grid_thw` make batching non-trivial; the harness is one-sample-at-a-time on purpose.

### 2.2 Can batching change predictions? (yes, possibly)
For decoder-only generation, batching requires **left-padding**. Padded positions + the attention mask, plus
floating-point accumulation order changing with batch shape, mean greedy decoding can **flip on near-ties**.
So bs>1 may yield a few predictions different from bs=1 → a small score delta. This is why the locked honest
protocol fixed **bs=1** for accepted numbers.

### 2.3 Risks
- **Padding/generation:** left-padding mishandling can leak pad tokens into attention; HF handles it, but FP
  nondeterminism across batch shapes remains.
- **Memory:** LLaVA-1.5 fp16 (~14 GB weights) at bs=4 with 576 visual + ~40 text + mnt=64 fits comfortably on a
  48 GB RTX 6000 Ada. Qwen-7B bf16 (~16 GB) batching is harder: ragged visual lengths spike memory to the
  largest image in the batch, and M-RoPE batching isn't implemented → **don't batch Qwen**.

### 2.4 Exact pilot test to compare bs=1 vs bs=2/4 (LLaVA only)
On the **same first n=200 VQAv2 IDs**: run dense at bs=1, bs=2, bs=4. Compare:
1. **Exact prediction-string match rate** bs>1 vs bs=1 (per sample).
2. **Aggregate score delta** (pp).
3. **Wall-clock speedup**.
**Accept bs>1 only if** prediction match rate is 100% (or score delta < ~0.05pp *and* you record the bs in the
result). Otherwise keep bs=1.

### 2.5 Decision rule
- **Accepted final numbers: bs=1** (matches the locked protocol; D8). Qwen is bs=1 regardless (harness).
- **bs>1 allowed only as a verified speed optimization for LLaVA**, gated by the §2.4 pilot showing identical
  predictions. Never mix batch sizes across methods at the same (model, dataset) without re-verifying.

---

## 3. Training vs evaluation (clearing up the confusion)

- **Why old experiments trained for many epochs on a train/val split:** the *retired* VQAv2 work trained small
  learnable modules on top of the frozen backbone — the **classification answer head** and the **dynamic
  BudgetController** — using a supervised loop: a **train split**, a held-out **val split**, and **multiple
  epochs** (repeated passes over the train data with gradient updates). That is standard supervised learning;
  the epochs exist because you are *fitting parameters*. (The numbers like "train on ~60k / val on ~10k over
  many epochs" describe that fitting loop, not an evaluation.)
- **Why the final dense generation baseline has NO training and NO epochs:** dense is a **frozen-model
  generation evaluation**. There are **no trainable parameters**, so there is **nothing to fit and no epoch
  concept**. You run the model **once** over the evaluation subset and score the outputs.
- **What "one evaluation pass" means:** load the frozen model → for each of the N eval samples, build the
  prompt, `generate()` greedily, score the answer → aggregate. One traversal of the data, no gradients, no
  repetition.
- **Which later methods bring training back:** **dense, static, and the dynamic-WHICH *teacher*** (mid-layer
  attention selection) are all **training-free** single passes. Training reappears only for **(a) the
  deployable distilled *student* selector** (dynamic-WHICH's cheap version) and **(b) any learned per-sample
  budget predictor** in the dynamic-COUNT redesign. Those train a small head; the backbone stays frozen.

---

## 4. Final decision table (subset size)

| Subset | Runtime (VQAv2, all methods, both models) | Statistical strength | Risk | Recommendation | Choose it when… |
|---|---|---|---|---|---|
| **20k** | ~50 GPU-h (±50%) | overall ±0.59pp; per-type ±~1.2pp | low cost; per-type cells a bit thin | viable cost-saver | budget-tight; overall number is the headline, per-type is secondary |
| **25k** *(default candidate)* | ~62 GPU-h (±50%) | overall ±0.53pp; per-type ±~1.1pp | balanced | **balanced pick** | you want reliable per-type tables without paying for 30k |
| **30k** | ~75 GPU-h (±50%) | overall ±0.48pp; per-type ±~1.0pp | highest cost, marginal gain over 25k | only if per-type must be tightest | per-type breakdowns are a headline result and the extra ~13 h is fine |

**Bottom line:** the statistical step from 20k→25k is real (per-type), 25k→30k is marginal; the cost step is
~+12 h then ~+13 h, almost entirely on Qwen. **Decision: 25k is LOCKED** (`DENSE_DECISION_LOCK.md` #2). 20k and
30k stay in the table above only as documented alternatives. Freeze the manifest
(`configs/final_scope/sample_ids/vqav2.json`) and reuse it for dense, static, dynamic-WHICH, and dynamic-COUNT.

### Manifest the locked subset must record (`configs/final_scope/sample_ids/vqav2.json`)
`{ "dataset": "vqav2", "n": 25000, "seed": 42,
   "stratification": "repo 4 question-type buckets (yes/no, attribute, counting, spatial) via _stratified_sample/_question_type_id in src/data/vqav2/vqav2.py — proportional; NOT VQAv2 answer_type",
   "question_ids": [ ... ], "sha256": "<hash of the ordered id list>" }`
— version-controlled under `configs/`; every VQAv2 run (all methods, both models) reads this one file. **Note:**
the existing val loader truncates (stratifies only `train`), so the builder must apply this stratification to the
val split explicitly when constructing the 25k manifest.
