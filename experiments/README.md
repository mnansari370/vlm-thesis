# `experiments/` — per-method experiment entrypoints

The runnable experiment/evaluation code now lives under **`src/`** (organized by method), and is invoked
from the repo root, e.g.:

```
python -m src.evaluation.docvqa.qwen_kcurve --help     # selection K-curve (L5)
python -m src.evaluation.gqa.run_static_testdev --help # static frontier (L2/L3)
python -m src.training.train_student --help            # distilled selector (L11/L12)
```

The method subfolders here (`diagnostic_foundation/`, `static_pruning/`, `dynamic_budget/`,
`question_conditioned_selection/`, `distillation/`, `ablations/`) are kept as a curated index of which
`src.*` entrypoints + `configs/*` belong to each thesis section. The historical track folders they
replaced (`GQA/`, `VQA_V2/`, `v2/`) no longer exist. See `docs/THESIS_MASTER_PLAN.md` for the mapping of
experiments to chapters.
