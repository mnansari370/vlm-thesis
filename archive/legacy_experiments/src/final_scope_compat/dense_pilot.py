# Backward-compatibility shim. The implementation moved to src.dense.evaluate_dense during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.dense_pilot`
# importable for older commands and provenance. New code imports from src.dense.evaluate_dense directly.
from src.dense.evaluate_dense import *  # noqa: F401,F403
