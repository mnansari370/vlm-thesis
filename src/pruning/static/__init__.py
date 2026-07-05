"""Static (image-only) pruning baselines beyond the engines' built-in selectors.

The primary static selectors live inside the frozen engines (LLaVA cls_attn in
src/models/static/static.py; Qwen norm/uniform in
src/pruning/question_conditioned_selection/qwen_pruner.py). This package holds the
additional published-method baseline: visionzip.py (dominant + contextual merge).
"""
