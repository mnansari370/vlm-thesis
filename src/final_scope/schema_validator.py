# Backward-compatibility shim. The implementation moved to src.common.schema_validator during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.schema_validator`
# importable for older commands and provenance. New code imports from src.common.schema_validator directly.
from src.common.schema_validator import *  # noqa: F401,F403
