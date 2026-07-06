# Backward-compatibility shim. The implementation moved to src.common.token_flops during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.token_flops`
# importable for older commands and provenance. New code imports from src.common.token_flops directly.
from src.common.token_flops import *  # noqa: F401,F403
