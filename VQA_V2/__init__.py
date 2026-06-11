"""
VQA_V2 — canonical VQAv2 track (generation protocol) on frozen LLaVA-1.5-7B.

Layout:
    dense/    full-576-token baseline (model + configs)
    static/   CLS-attention top-K pruning (model + per-K configs)
    dynamic/  question-conditioned scorer + budget controller + trainer
    shared/   datasets, evaluation, training utilities, scripts, experiments

All thesis/paper claims from this track use the GENERATION protocol
(greedy decode + VQA consensus). See ../docs/vqav2_findings.md.
Run modules from the repo root, e.g.:
    python -m VQA_V2.shared.evaluation.generate_and_score --help
"""
