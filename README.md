# Dynamic Question Conditioned Visual Token Pruning for Efficient Vision Language Models

**Author:** MO Nafees
**Supervisor:** Decebal Constantin Mocanu
**Advisor:** Boqian Wu

> **Status: work in progress.** This is an active master's thesis project. Experiments are still running and the numbers below will be extended and refined. Only results that have already passed the full evaluation protocol are reported here.

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

### 3.4 Dynamic COUNT: adaptive per sample budget (in progress)

Dynamic COUNT asks the complementary question: keeping the selector fixed, can choosing a different budget per sample beat the best single fixed budget at the same average cost?

The first step is an oracle analysis that bounds what a perfect budget router could achieve. Using the accepted dense and static per sample results, an oracle that picks the cheapest sufficient budget per sample gains roughly two to six accuracy points over the best fixed budget while removing 60 to 83 percent of visual tokens, and it can match dense accuracy for the vast majority of samples at a fraction of the tokens. This is explicitly an upper bound, not a method: a real controller must predict the budget from the input alone. Designing and honestly evaluating such a controller against matched average compute is the current work in this project, and earlier attempts in the literature and in this project realized almost none of the oracle headroom, so the outcome may well be a carefully documented negative.

## 4. What the results say so far

The evidence so far supports a selection over budget reading. The dense ceilings are reproduced, the static floors are strong and cheap, question conditioned selection is genuinely valuable but only in a specific task regime, and the adaptive budget axis has visible oracle headroom whose practical reachability is still open. All headline claims come from frozen models under matched budgets, and negative results are reported alongside the positive ones.

## 5. Repository layout

* `src/final_scope/` shared evaluation infrastructure: sample manifests, unified output schema, token and FLOP accounting, fairness gate, and the dense, static and dynamic WHICH runner cores
* `src/models/` frozen model wrappers, including the physical token removal path for LLaVA
* `src/pruning/` selection methods: static criteria, the `textsim` dynamic WHICH selectors, the clean room reference implementation, and the earlier budget analysis code
* `src/metrics/` the official scorers (GQA exact match, M4C, ANLS, VQA consensus)
* `src/data/` dataset loaders
* `scripts/final_scope/` runnable launchers, audits and table generators for the final experiment matrix
* `configs/final_scope/sample_ids/` the frozen, sha256 verified evaluation subsets
* `results/final_scope/` run outputs, git ignored, one aggregate JSON plus one per sample JSONL per cell
* `docs/` the locked scope, the evidence ledger, the protocol documents and the run plan

## 6. Running the code

Each model runs in its own environment: LLaVA in `vlm_env` (torch 2.3, transformers 4.46.3) and Qwen in `qwen_env` (transformers 4.51 with `qwen_vl_utils`). The pinned versions matter for reproducing the numbers.

A quick check without a GPU:

```bash
python -m compileall src scripts
python -m src.final_scope.test_final_scope
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
