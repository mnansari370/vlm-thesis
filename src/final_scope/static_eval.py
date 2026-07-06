# Backward-compatibility shim. The implementation moved to src.static.evaluate_static during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.static_eval`
# importable for older commands and provenance. New code imports from src.static.evaluate_static directly.
from src.static.evaluate_static import *  # noqa: F401,F403
