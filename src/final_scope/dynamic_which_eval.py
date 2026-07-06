# Backward-compatibility shim. The implementation moved to src.dynamic_which.evaluate_dynamic_which during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.dynamic_which_eval`
# importable for older commands and provenance. New code imports from src.dynamic_which.evaluate_dynamic_which directly.
from src.dynamic_which.evaluate_dynamic_which import *  # noqa: F401,F403
