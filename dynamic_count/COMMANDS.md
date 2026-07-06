# Dynamic-COUNT — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: self-checks + the COUNT validators/audits, prints where results live
bash dynamic_count/scripts/validate_dynamic_count.sh

# or directly:
python -m compileall -q src scripts
python -m src.common.test_evaluation_core
python -m scripts.validation.validate_dynamic_count     # every probe/DC-D/DC-C aggregate
python -m scripts.validation.audit_dynamic_count  # all 8 cells complete
```

## Oracle bound + table generation (CPU, from saved results)

```bash
python -m scripts.dynamic_count.analyze_oracle      # the adaptive-budget upper bound
python -m scripts.tables.make_dynamic_count_tables
python -m scripts.tables.make_final_thesis_tables
```

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

The five stages run in order; probes and DC-C need GPUs, calibration and DC-D are CPU. Every stage
is skip-safe (existing outputs are never overwritten). The probes require the frozen static/WHICH
finals they reproduce.

```bash
# 1. probes (GPU 0 = LLaVA, GPU 1 = Qwen), with a byte-exact reproduction gate:
bash scripts/dynamic_count/run_probe_full_matrix.sh

# 2. fit controllers on the first-20% calibration split (CPU):
python -m scripts.dynamic_count.make_controller_calibration

# 3. compose DC-D from the frozen finals (CPU):
python -m scripts.dynamic_count.compose_discrete

# 4. DC-C real second passes (GPU 0 = LLaVA, GPU 1 = Qwen):
bash scripts/dynamic_count/run_continuous_full_matrix.sh
```
