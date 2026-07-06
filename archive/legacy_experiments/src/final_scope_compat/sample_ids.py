# Backward-compatibility shim. The implementation moved to src.common.sample_ids during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.sample_ids`
# importable for older commands and provenance. New code imports from src.common.sample_ids directly.
from src.common.sample_ids import *  # noqa: F401,F403
