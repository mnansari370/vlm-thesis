"""Source tree for the thesis "Dynamic Question-Conditioned Visual Token Pruning for
Efficient Vision-Language Models".

Method map (docs/ holds the full documentation; the method folders dense/, static/,
dynamic_which/, dynamic_count/ hold the reader-facing navigation):
  shared evaluation core (manifests, schema, gate, FLOPs)      -> src/common/
  dense / static / dynamic-WHICH / dynamic-COUNT runner cores  -> src/{dense,static,dynamic_which,dynamic_count}/
  LLaVA-1.5 frozen engine (physical prune-before-LLM)          -> src/models/static/static.py
  Qwen2.5-VL frozen engine (M-RoPE-preserving pruning)         -> src/pruning/question_conditioned_selection/qwen_pruner.py
  Dynamic-WHICH selectors (+ clean-room reference)             -> src/pruning/dynamic_which{,_ref}/
  Dynamic-COUNT probes/controllers/wrappers                    -> src/pruning/dynamic_count/
  Canonical scorers                                            -> src/metrics/
  FLOPs formulas (FastV Eq. 5 prefill)                         -> src/analysis/{flops,qwen_flops}.py
"""
