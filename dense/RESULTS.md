# Dense — results

Source: `results/tables/final_dense_static_dynamic_comparison.md` (section A). Full
manifests, all fairness gates passed.

| Model | Dataset | n | Score | Avg visual tokens | Prefill TFLOPs/sample |
|---|---|--:|--:|--:|--:|
| LLaVA-1.5-7B | GQA | 12,578 | **61.42** | 576 | 3.17 |
| LLaVA-1.5-7B | VQAv2 | 25,000 | **77.33** | 576 | 3.15 |
| LLaVA-1.5-7B | TextVQA | 5,000 | **57.65** | 576 | 3.45 |
| LLaVA-1.5-7B | DocVQA | 5,349 | **21.53** | 576 | 3.17 |
| Qwen2.5-VL-7B | GQA | 12,578 | **60.96** | 358.6 | 2.12 |
| Qwen2.5-VL-7B | VQAv2 | 25,000 | **84.27** | 358.6 | 2.11 |
| Qwen2.5-VL-7B | TextVQA | 5,000 | **81.06** | 963.6 | 5.73 |
| Qwen2.5-VL-7B | DocVQA | 5,349 | **94.76** | 1,229.1 | 6.99 |

Reading:

- LLaVA-1.5 always carries exactly 576 visual tokens (hard-asserted per sample); its TFLOPs vary
  only through prompt length (the TextVQA OCR block lengthens the text).
- Qwen2.5-VL's token count is per image: natural scenes (GQA/VQAv2, ~359 tokens) are cheap; scene
  text (TextVQA, ~964) and document pages (DocVQA, ~1,229) are expensive. This spread is why the
  Qwen budgets are computed per sample.
- **LLaVA-1.5 × DocVQA = 21.53 is expected**: 576 low-resolution tokens cannot read a dense document
  page. It is an informative resolution × task data point, not a harness failure.
- The VQAv2 numbers use the locked 25,000-question stratified subset (the older 10k/76.44 run was
  truncation-sampled and is reference-only).
