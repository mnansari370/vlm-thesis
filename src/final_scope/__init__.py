"""
Final-scope shared infrastructure for the dense → static → dynamic-WHICH → dynamic-COUNT
experiment matrix (protocol: docs/03_final_scope_protocol.md).

Fairness toolkit (CPU-only; no model/generation code lives here):
  sample_ids       — sample-ID manifest builder helpers + loader/validator (sha256-verified;
                     one locked manifest per dataset, shared by EVERY method and both models)
  output_writer    — unified per-sample JSONL + aggregate JSON writer (one schema everywhere)
  schema_validator — per-sample/aggregate schema + the fairness gate a run must pass to be accepted
  token_flops      — per-sample token stats + per-sample-then-averaged prefill FLOPs
                     (per-sample first because the FastV formula's n² term is convex)

Runner cores (data/scoring/schema/deltas; generation is injected by scripts/final_scope/*):
  dense_pilot        — dense runs + the per-dataset adapters/prompts/scorers all methods reuse
  static_eval        — static runs; loads the dense final as the same-sample reference
  dynamic_which_eval — WHICH runs; loads dense AND the same-budget static final (the floor)
  dynamic_count_eval — COUNT probe/DC-D/DC-C; compares against the static curve at matched FLOPs

  test_final_scope   — CPU self-checks for all of the above (run via python -m)
"""
