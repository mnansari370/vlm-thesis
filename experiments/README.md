# `experiments/` — per-method experiment entrypoints

The runnable experiment/evaluation code now lives under **`src/`** (organized by method), and is invoked
from the repo root, e.g.:

```
python -m src.evaluation.docvqa.qwen_kcurve --help     # selection K-curve (L5)
python -m src.evaluation.gqa.run_static_testdev --help # static frontier (L2/L3)
python -m src.training.train_student --help            # distilled selector (L11/L12)
```

This folder is a thin index, not a code location: every experiment is an `src.*` module run via
`python -m`, with its config in `configs/` and its results under `results/`. The mapping of experiments
to thesis sections lives in `docs/THESIS_MASTER_PLAN.md`; the exact evidence paths are in
`docs/THESIS_EVIDENCE_LEDGER.md`. The historical track folders this replaced (`GQA/`, `VQA_V2/`, `v2/`)
no longer exist.
