# Backward-compatibility shim. The self-checks moved to src/common/test_evaluation_core.py
# during the 2026-07-05 method-oriented restructure; this keeps the old command
# `python -m src.final_scope.test_final_scope` working. New command:
# `python -m src.common.test_evaluation_core`.
import sys

from src.common.test_evaluation_core import *  # noqa: F401,F403
from src.common.test_evaluation_core import main

if __name__ == "__main__":
    sys.exit(main())
