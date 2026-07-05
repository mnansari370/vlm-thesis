# Dynamic Question Conditioned Visual Token Pruning for Efficient Vision Language Models

**Author:** MO Nafees
**Supervisor:** Decebal Constantin Mocanu
**Advisor:** Boqian Wu

> **Status: experiments complete; thesis text in preparation.** The full evaluation matrix — dense, static, Dynamic-WHICH, and Dynamic-COUNT (DC-D and DC-C) on both models and all four datasets — is finished and validated. Every number below comes from a run that passed the full evaluation protocol; the complete result tables live in `results/final_scope/tables/` (tracked in this repository).

## 1. What this project studies

Vision language models such as LLaVA and Qwen2.5 VL spend most of their compute on visual tokens. A single image can occupy hundreds or thousands of positions in the language model sequence, while the answer to a question often depends on a small part of the image. Pruning visual tokens is therefore a natural way to make inference cheaper.

Any pruning method quietly makes two separate decisions:

* **Which** visual tokens to keep (selection)
* **How many** visual tokens to keep per sample (budget)

This thesis decomposes visual token pruning along exactly these two axes and measures each one honestly, on frozen backbones, under matched token budgets, with reproduced dense baselines as the reference point. The central question is which of the two decisions actually pays off when the comparison is fair.

## 2. Experimental setup

**Models (both frozen, no backbone training):**

* LLaVA 1.5 7B, a fixed grid of 576 visual tokens at 336 px
* Qwen2.5 VL 7B, a native dynamic visual token count that varies per image

**Datasets:** GQA (testdev balanced, exact match), TextVQA (val, M4C soft accuracy, OCR tokens in the prompt), DocVQA (val, ANLS), and VQAv2 (a fixed subset of 25,000 validation questions, stratified by the official answer type with seed 42, VQA consensus metric).

**Budgets:** every pruning method is evaluated at 15, 25, 35, 50 and 75 percent of the dense visual token count, plus the dense reference at 100 percent. For LLaVA this maps to fixed token counts (86, 144, 202, 288, 432, 576). For Qwen the budget is applied per sample, as a fraction of that image's own dense token count.

**Fairness protocol.** All methods, both models, and every budget run on the exact same sample lists, which are version controlled with sha256 checksums under `configs/final_scope/sample_ids/`. Decoding is greedy with batch size 1 and identical prompts across methods. Every run writes per sample predictions plus an aggregate record, counts tokens per sample, computes analytical prefill FLOPs per sample before averaging, and must pass an automatic fairness gate before it is accepted. Token reduction and FLOP reduction are always reported separately.

## 3. Methods

### 3.1 Dense baseline

The dense runs keep all visual tokens and serve two purposes: they are the accuracy ceiling that every pruning result is measured against, and they are the reproduction anchor that shows the evaluation harness is trustworthy (the LLaVA numbers match the published references closely). The dense per sample outputs also provide the reference token counts that the Qwen budgets are computed from.

Accepted dense results (accuracy in percent, full evaluation sets):

| Model | GQA | TextVQA | VQAv2 | DocVQA |
|---|---|---|---|---|
| LLaVA 1.5 7B | 61.42 | 57.65 | 77.33 | 21.53 |
| Qwen2.5 VL 7B | 60.96 | 81.06 | 84.27 | 94.76 |

The low LLaVA score on DocVQA is expected and is itself informative: 576 low resolution tokens cannot read a dense document page, which illustrates how strongly the resolution and the task shape what pruning can achieve.

### 3.2 Static pruning

Static pruning selects tokens from the image alone, without looking at the question, at a fixed budget. It is the honest floor that any question aware method has to beat at the same budget.

* **LLaVA:** CLS attention saliency from the vision encoder (the VisionZip style dominant token criterion). The selected features are physically removed before the language model, so the shorter sequence is real compute savings, not masking.
* **Qwen:** an activation norm criterion over the merged visual embeddings, with a uniform stride selector kept as a control.

The static frontier is complete for both models on all four datasets at all five budgets. The broad picture: static pruning is remarkably strong on GQA and VQAv2, where even a 75 percent budget matches dense and a 15 percent budget loses only a few points. It degrades gracefully on TextVQA for LLaVA, and it collapses on document understanding, where Qwen falls from 94.76 dense to about 30 at the tightest budget. How much a task suffers under pruning is strongly task dependent, which matters for everything that follows.

### 3.3 Dynamic WHICH: question conditioned selection

Dynamic WHICH keeps the budget identical to static and changes only which tokens survive. The selector, called `textsim`, scores every visual token by its maximum cosine similarity to the question token embeddings, keeps the top K, and restores the original spatial order. It is training free, needs no extra forward pass through the language model, and composes the same frozen generation code as the static runs, so any accuracy difference is attributable purely to the selection.

The headline positive result is Qwen2.5 VL on TextVQA (full 5,000 validation questions, all methods on the same samples):

| Budget | Dynamic WHICH | Static | Dense | Gain vs static |
|---|---|---|---|---|
| 15% | 60.56 | 53.06 | 81.06 | +7.50 |
| 25% | 67.53 | 59.17 | 81.06 | +8.36 |
| 35% | 71.36 | 63.57 | 81.06 | +7.79 |
| 50% | 76.08 | 70.14 | 81.06 | +5.94 |
| 75% | 79.80 | 78.43 | 81.06 | +1.37 |

Dense keeps all visual tokens, so its value is the same at every budget row and serves as the ceiling that both pruning methods trade accuracy against.

The gain is largest exactly where the budget is tightest, which is the intended behavior of question conditioned selection. This result was additionally validated by an independent clean room reimplementation of the selector that reproduces the predictions exactly.

Just as important, the same selector was evaluated on the full matrix of both models and all four datasets, and it does **not** win everywhere. On GQA, VQAv2 and DocVQA it lands below the static floor by roughly one to fourteen points depending on the cell. The pattern is consistent: question conditioned selection helps when the answer lives in a small, question locatable region (scene text in TextVQA) and hurts when the task needs broad or relational coverage of the image (GQA relations, natural scenes in VQAv2, full page text in DocVQA). A dedicated failure analysis confirmed these negatives are properties of the task regime, not implementation bugs. The full matrix is being completed so that positive and negative cells are reported on the same footing.

### 3.4 Dynamic COUNT: adaptive per sample budget (complete — a documented negative)

Dynamic COUNT asks the complementary question: keeping the selector fixed, can choosing a different budget per sample beat the best single fixed budget at the same average cost?

An oracle analysis first bounded the headroom: a perfect budget router would gain roughly two to six accuracy points over the best fixed budget while removing 61 to 83 percent of visual tokens. Two real mechanisms were then built and evaluated on all eight model×dataset cells, with controllers calibrated on the first 20 percent of each manifest and scored on the held-out 80 percent, under honest multi-pass accounting (an escalated sample pays for both its cheap probe pass and its second pass):

* **DC-D**, a discrete confidence-gated cascade over the budget anchors, and
* **DC-C**, the main method: a calibrated controller (a transparent rule controller, plus a small ridge variant) that predicts a per-sample **integer** token count, executed with a real second pass at exactly that budget.

The probes that record the confidence signals reproduce the frozen static and WHICH predictions byte-for-byte (a built-in reproduction gate, zero mismatches across all 18 probe runs), so the substrate is exact. The outcome, against the static accuracy-versus-FLOPs curve at matched average compute: **DC-D wins once** (LLaVA TextVQA, +0.75 points) and otherwise near-ties or loses; **DC-C never wins** (best +0.48, a near-tie; the steep-curve cells lose by up to eleven points); and running COUNT on top of the successful WHICH selector adds **no increment** over WHICH alone. The oracle headroom is real but not harvestable by input-side confidence signals under honest accounting — a carefully documented negative result.

## 4. What the results say

The evidence supports a clear selection-over-budget reading. The dense ceilings are reproduced; the static floors are strong and cheap (within about one point of dense at the 75 percent budget on almost every cell); question conditioned selection is genuinely valuable but only in a specific task regime (Qwen2.5-VL on TextVQA — validated by an exact clean-room reproduction); and the adaptive budget axis, despite visible oracle headroom, does not survive honest matched-compute evaluation. Per cell, the best method is the static floor on five of eight cells, Dynamic WHICH on Qwen TextVQA (and nominally on a collapsed low-budget corner of Qwen DocVQA), and DC-D on LLaVA TextVQA by a small margin. All headline claims come from frozen models under matched budgets, and negative results are reported alongside the positive ones — see `results/final_scope/tables/final_thesis_results_summary.md` for the complete per-cell verdicts.

## 5. Repository layout (after the 2026-07-05 cleanup)

* `src/final_scope/` shared evaluation infrastructure: sample manifests, unified output schema, token and FLOP accounting, fairness gate, and the dense, static, dynamic WHICH and dynamic COUNT runner cores
* `src/models/static/static.py` the frozen LLaVA engine (physical token removal before the language model)
* `src/pruning/` the method code: the `textsim` dynamic WHICH selectors, the clean room reference implementation, the dynamic COUNT probes and controllers, the frozen Qwen engine (`question_conditioned_selection/qwen_pruner.py`), and the VisionZip baseline
* `src/metrics/` the official scorers (GQA exact match, M4C, ANLS, VQA consensus)
* `scripts/final_scope/` all runnable launchers, audits, validators and table generators for the final experiment matrix
* `configs/final_scope/sample_ids/` the frozen, sha256 verified evaluation subsets (tracked)
* `results/final_scope/` run outputs — per cell one aggregate JSON plus one per sample JSONL (git ignored, local evidence); **`results/final_scope/tables/`** holds all final result tables and reports (tracked); `results/final_scope/dynamic_count_configs/` holds the fitted COUNT controllers (tracked)
* `archive/` **all retired legacy code, configs, scripts, logs and docs** — the pre-final-scope pipelines (classification heads, the old budget controller, the distillation study, legacy evaluation harnesses, out-of-scope models and datasets) were moved here during the staged cleanup and are preserved, not deleted. Do not import from `archive/`; every move is recorded in `archive/migration_manifests/`.
* `docs/` the final documentation set (local-only): thesis scope, repository map, protocol, one document per method, results summary, reproducibility, and limitations

### Where each method lives

The active code is organized around the four method families. Each row is one method, from
implementation to committed evidence:

| Method | Runner core (CPU) | Selection / generation | Launcher | Result basename |
|---|---|---|---|---|
| Dense | `src/final_scope/dense_pilot.py` | `models/static/static.py` (`none`) · `pruning/.../qwen_pruner.py` (`full`) | `run_dense_pilot.py` | `dense_final` |
| Static | `src/final_scope/static_eval.py` | LLaVA `cls_attn` / Qwen `norm` (inside the engines) | `run_static_eval.py` | `static_final_{sel}_p{b}` |
| Dynamic-WHICH | `src/final_scope/dynamic_which_eval.py` | `src/pruning/dynamic_which/` (+ `_ref` clean-room) | `run_dynamic_which_eval.py` | `dynamic_which_final_textsim_p{b}` |
| Dynamic-COUNT (DC-D, DC-C) | `src/final_scope/dynamic_count_eval.py` | `src/pruning/dynamic_count/` | `run_dynamic_count_{probe,discrete,continuous}.py` | `dynamic_count_{probe,dcd,dcc}_*` |

`src/final_scope/` holds the shared fairness toolkit (`sample_ids`, `output_writer`,
`schema_validator`, `token_flops`) that every method reuses. The launchers live under
`scripts/final_scope/` (grouped by method in `scripts/README.md`); the per-cell result files and the
committed tables live under `results/final_scope/` (see `results/README.md`).

## 6. Running the code

Each model runs in its own environment: LLaVA in `vlm_env` (torch 2.3, transformers 4.46.3) and Qwen in `qwen_env` (transformers 4.51 with `qwen_vl_utils`). The pinned versions matter for reproducing the numbers.

Validating the completed results without a GPU (self-tests, then the final-matrix validators and audits — expected: ALL PASSED, ALL_FINAL_VALID=True, ALL_DC_VALID=True, 40/40 WHICH cells and 8/8 COUNT cells complete):

```bash
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
python -m scripts.final_scope.validate_dynamic_which_final
python -m scripts.final_scope.audit_dynamic_which_full_final_matrix
python -m scripts.final_scope.validate_dynamic_count_final
python -m scripts.final_scope.audit_dynamic_count_full_matrix
```

Evaluations are launched per cell, for example:

```bash
python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset gqa --full
python -m scripts.final_scope.run_static_eval --model llava15 --dataset gqa --budget-pct 25 --full
python -m scripts.final_scope.run_dynamic_which_eval --model qwen25vl7b --dataset textvqa --budget-pct 25 --selector textsim --full
```

Static runs require the matching dense final to exist first, and dynamic WHICH runs additionally require the matching static final, because deltas and per sample budgets are computed against those references.

## 7. Acknowledgements

This thesis is carried out under the supervision of **Decebal Constantin Mocanu** and with the guidance of advisor **Boqian Wu**. I am grateful for their feedback and direction throughout the project.
