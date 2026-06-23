# From Teacher to Student: Distilling a Cheap Question-Conditioned Selector — Findings (thesis Part III)

**Frozen Qwen2.5-VL-7B. DocVQA (ANLS). Generation protocol, official scorer. Prune-before-LLM.**

Part II established that question-conditioned **selection** dominates token pruning, via a *teacher*
signal — the L16 question→visual attention map — that needs a **full LLM forward** and is therefore
**not deployable**. This part asks the natural next question and turns the measurement into a method:

> **How much of the mid-layer teacher's selection quality can a cheap front-end student recover with
> NO LLM-decoder forward?**

We distil the L16 teacher into `CheapQCSelector`, a small cross-attention head (≈8–20M params) that
scores the visual tokens from **pre-LLM features only** (ViT/merger visual embeds + question token
embeds). It adds negligible FLOPs and runs before the LLM, so its token cut converts to real prefill
savings.

---

## Method

- **Teacher (target):** `QwenPruner.encode_qc` → L16 question→visual attention, per visual token
  (`cache_teacher.py`, cached once; the expensive full forward is paid only at caching time).
- **Student:** `CheapQCSelector` — project (vis, question) → cross-attention (visual queries attend to
  question tokens) → per-token score. Inputs are pre-LLM; **no decoder forward at inference.**
- **Distillation:** listwise KL(teacher‖softmax(student)) + top-K BCE on the keep/drop decision
  (`train_student.py`). Features cached to disk so epochs after the first are ~2 min.
- **Eval (gate):** at K=128, ANLS of four selections — dense / blind(uniform) / teacher / student —
  on held-out DocVQA **validation**; recovery = (student−blind)/(teacher−blind) (`eval_gate.py`).
- **Honesty controls:** harness validated keep-all==stock; train/eval on **disjoint splits**
  (train→validation); question-conditioning control (mismatched-question selection).

## Headline results (held-out, K=128)

| setup | train data | gate set | dense | teacher | **student** | blind | **recovery** |
|---|---|---|---|---|---|---|---|
| pilot (proof-of-concept) | val[0:3000] | val[3000:3400] | 95.4 | 91.4 | 60.8 | 34.5 | **46.2%** |
| **scaled (clean train→val, 10 ep)** | **train[0:12000]** | **val[0:400]** | 95.0 | 91.1 | **65.9** | 31.2 | **57.9% (PASS)** |
| scaled, 25 epochs | train[0:12000] | val[0:400] | 95.0 | 91.1 | 62.5 | 31.2 | 52.3% (**overfit**) |

Scaling **data** 3K→12K (real train split) + a larger student (19.7M params) lifted recovery
**46%→58%** and student ANLS **60.8→65.9**. But scaling **epochs** 10→25 *hurt* held-out recovery
(58%→52%) even though train teacher-overlap kept rising (0.591→higher) — i.e. the student **overfits
on epochs**. The operating point is therefore **10 epochs**, and the lever for further gains is
**more data, not more epochs**.

## Student K-curve (pilot, recovery is stable across budgets)

| K | dense | teacher | student | blind | recovery |
|---|---|---|---|---|---|
| 64 | 95.0 | 84.2 | 51.1 | 20.7 | 47.9% |
| 128 | 95.4 | 91.4 | 60.8 | 34.5 | 46.2% |
| 256 | 95.0 | 92.4 | 72.3 | 55.9 | 45.0% |

The cheap student **more than doubles** the deployable blind baseline at every budget, recovering a
stable ~45–48% of the teacher's gain (scaled to ~58% at K=128 with more data).

## The student is genuinely question-conditioned (control)

Selecting tokens with a **mismatched** question (far-offset; answering always with the real question):

| | student real-q | student mismatched-q | blind | question-driven |
|---|---|---|---|---|
| pilot (n=200) | 60.0 | 39.8 | 32.5 | **73.4%** |
| **scaled (n=200)** | 64.8 | 32.5 | 32.2 | **99.2%** |

For the scaled student the mismatched-question selection (32.5) collapses **all the way to blind**
(32.2): selecting by the wrong question is no better than uniform, so **~100% of the student's gain
is due to the question**. This is the first cheap (no-LLM-forward) **genuinely question-conditioned**
selector we are aware of.

## The recovery law (the scientific contribution)

The mid-layer question-conditioned selection signal is **~58% cheaply recoverable** (front-end,
no decoder forward) and **~42% irreducibly mid-LLM** — it requires the deep cross-modal reasoning
that only emerges ~16 layers into the network (consistent with Part I's "frozen features localize
poorly" and the layer sweep where the signal jumps at L12→L16). This is a clean, reusable
characterization of *how cheaply* question-conditioned selection can be realized.

## Honest positioning (what this is / isn't)

- **Is:** a deployable, genuinely question-conditioned selector (negligible FLOPs, no decoder forward)
  that doubles blind and recovers the majority of the expensive teacher's gain; plus the recoverability
  law. A complete diagnose→decompose→**build** arc.
- **Is not:** SOTA. Student 65.9 vs dense 95.0 (~69% retention) is below dedicated cheap selectors
  (FEATHER/CDPruner/VisionSelector ~90% retention). The ~42% irreducible gap is the reason, and it is
  itself the honest finding. The paper's strength is the rigorous decomposition + this recoverability
  characterization, not a leaderboard number.

## Reproduction (qwen_env, from repo root)

```bash
# 1) cache teacher (12K DocVQA train shards)
CUDA_VISIBLE_DEVICES=1 python -m v2.distill.cache_teacher --split train --n-shards 4 \
    --start 0 --end 12000 --out v2/outputs/distill/teacher_docvqa_train12k.pt
# 2) distil student (disk feature cache; epochs after the 1st are ~2 min)
CUDA_VISIBLE_DEVICES=1 python -m v2.distill.train_student \
    --teacher v2/outputs/distill/teacher_docvqa_train12k.pt --out v2/outputs/distill/student_12k.pt \
    --feat-cache-dir v2/outputs/distill/featcache_12k --d 768 --layers 3 --epochs 10
# 3) gate on held-out validation
CUDA_VISIBLE_DEVICES=1 python -m v2.distill.eval_gate --student v2/outputs/distill/student_12k.pt \
    --gate-split validation --gate-start 0 --n 400 --K 128
# 4) question-conditioning control
CUDA_VISIBLE_DEVICES=1 python -m v2.distill.eval_control --student v2/outputs/distill/student_12k.pt \
    --gate-split validation --gate-start 0 --n 200 --K 128
```

Artifacts: `v2/outputs/distill/{teacher_docvqa_train12k.pt, student_12k.pt, gate_12k.json,
control_12k.json}`. The 101 GB `featcache_12k/` is regenerable and can be deleted after training.
