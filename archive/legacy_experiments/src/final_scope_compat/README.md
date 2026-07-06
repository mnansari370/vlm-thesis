# `src/final_scope/` — backward-compatibility layer

This folder contains **thin re-export shims only**, kept for backward compatibility after the
2026-07-05 method-oriented restructure. The tested implementation now lives in method-named
packages; a reader should start there:

| Old import | New location |
|---|---|
| `src.final_scope.sample_ids`, `output_writer`, `schema_validator`, `token_flops` | `src/common/` (the shared evaluation core) |
| `src.final_scope.test_final_scope` | `src/common/test_evaluation_core.py` |
| `src.final_scope.dense_pilot` | `src/dense/evaluate_dense.py` |
| `src.final_scope.static_eval` | `src/static/evaluate_static.py` |
| `src.final_scope.dynamic_which_eval` | `src/dynamic_which/evaluate_dynamic_which.py` |
| `src.final_scope.dynamic_count_eval` | `src/dynamic_count/evaluate_dynamic_count.py` |

Nothing here should be edited; change the real modules in the packages above.
