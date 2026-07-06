# Backward-compatibility shim. The implementation moved to src.common.output_writer during the
# 2026-07-05 method-oriented restructure; this re-export keeps `src.final_scope.output_writer`
# importable for older commands and provenance. New code imports from src.common.output_writer directly.
from src.common.output_writer import *  # noqa: F401,F403
