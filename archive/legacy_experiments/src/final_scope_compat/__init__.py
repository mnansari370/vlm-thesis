"""Backward-compatibility layer (thin shims only — no implementation).

The shared evaluation core moved to method-oriented packages during the 2026-07-05
restructure. This package re-exports the old module names so older commands and provenance
references keep working. New code imports from the real locations:

  old  src.final_scope.sample_ids / output_writer / schema_validator / token_flops
  new  src.common.*

  old  src.final_scope.test_final_scope        new  src.common.test_evaluation_core
  old  src.final_scope.dense_pilot             new  src.dense.evaluate_dense
  old  src.final_scope.static_eval             new  src.static.evaluate_static
  old  src.final_scope.dynamic_which_eval      new  src.dynamic_which.evaluate_dynamic_which
  old  src.final_scope.dynamic_count_eval      new  src.dynamic_count.evaluate_dynamic_count
"""
