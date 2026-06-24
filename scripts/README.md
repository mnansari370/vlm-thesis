# `scripts/` — runnable launchers

Shell launchers for data prep and training, organized by purpose:

```
scripts/
  data/       dataset download / feature caching / vocab building (*.py and *.sh)
  training/   training & evaluation run launchers (*.sh)
```

These are convenience launchers; the importable code they call lives in `src/`. Some legacy `.sh`
launchers (HPC/run helpers carried over from the historical tracks) still reference the old run layout
internally and should be reviewed before re-running — they are not needed for the lightweight checks or
for any thesis/paper evidence. See `docs/FINAL_REPOSITORY_CLEANUP_REPORT.md`.
