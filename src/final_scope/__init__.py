"""
Final-scope (post-2026-06-27 scope lock) shared infrastructure for the dense → static →
dynamic experiment matrix. See docs/DENSE_PROTOCOL.md and docs/DENSE_DECISION_LOCK.md.

Modules (CPU-only utilities; no model/generation code lives here):
  sample_ids       — sample-ID manifest builder helpers + loader/validator (sha256-verified)
  output_writer    — unified per-sample JSONL + aggregate JSON writer
  schema_validator — per-sample/aggregate schema + fairness-gate checks
  token_flops      — per-sample token stats + per-sample FLOPs (wraps src/analysis/flops*)
"""
