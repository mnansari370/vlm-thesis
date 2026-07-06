# Backward-compatibility shim. The implementation moved to src.dynamic_count.evaluate_dynamic_count during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.dynamic_count_eval`
# importable for older commands and provenance. New code imports from src.dynamic_count.evaluate_dynamic_count directly.
from src.dynamic_count.evaluate_dynamic_count import *  # noqa: F401,F403
