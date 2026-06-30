"""Fresh Dynamic-WHICH (fixed-budget, question-conditioned) visual-token selectors.

These wrappers COMPOSE the frozen generators (StaticPrunedLlava, QwenPruner) and reuse their
validated build/generate seams; only the SCORING is new. No training, no answer head, no dense
LM prefill. GPU code — imported lazily by scripts/final_scope/run_dynamic_which_eval.py.
"""
