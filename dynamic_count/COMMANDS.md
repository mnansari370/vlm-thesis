# Dynamic-COUNT — commands

All commands run from the repository root. The CPU commands are safe and read-only over results.

## CPU validation (safe, no GPU, no overwrite)

```bash
# method-facing wrapper: self-checks + the COUNT validators/audits, prints where results live
bash dynamic_count/scripts/validate_dynamic_count.sh

# or directly:
python -m compileall -q src scripts
python -m src.final_scope.test_final_scope
python -m scripts.final_scope.validate_dynamic_count_final     # every probe/DC-D/DC-C aggregate
python -m scripts.final_scope.audit_dynamic_count_full_matrix  # all 8 cells complete
```

## Oracle bound + table generation (CPU, from saved results)

```bash
python -m scripts.final_scope.analyze_dynamic_count_oracle      # the adaptive-budget upper bound
python -m scripts.final_scope.make_dynamic_count_final_tables
python -m scripts.final_scope.make_final_thesis_tables
```

## GPU rerun — EXPENSIVE, NOT needed unless reproducing the experiment

The five stages run in order; probes and DC-C need GPUs, calibration and DC-D are CPU. Every stage
is skip-safe (existing outputs are never overwritten). The probes require the frozen static/WHICH
finals they reproduce.

```bash
# 1. probes (GPU 0 = LLaVA, GPU 1 = Qwen), with a byte-exact reproduction gate:
bash scripts/final_scope/run_dynamic_count_probe_full_matrix.sh

# 2. fit controllers on the first-20% calibration split (CPU):
python -m scripts.final_scope.make_dynamic_count_controller_calibration

# 3. compose DC-D from the frozen finals (CPU):
python -m scripts.final_scope.compose_dynamic_count_discrete

# 4. DC-C real second passes (GPU 0 = LLaVA, GPU 1 = Qwen):
bash scripts/final_scope/run_dynamic_count_continuous_full_matrix.sh
```
