# Methods restructure plan

> **Superseded (2026-07-05).** The deeper restructure was subsequently executed — see
> `deep_structure_plan_20260705.md` and `archive/migration_manifests/archive_manifest_20260705.md`.
> The internal paths were renamed to method-oriented names (`src/common` + `src/{method}`,
> `scripts/{method,validation,tables,data}`, `results/{runs,tables,configs}`,
> `configs/sample_ids`), with `src/final_scope/` kept only as a thin backward-compatibility
> shim. The risk analysis below is retained as the record of why the move was done carefully,
> in isolated groups, each validated before the next.


This repository is now organized for a reader around the four thesis methods, through the top-level
folders `dense/`, `static/`, `dynamic_which/`, and `dynamic_count/`. Each contains a README, a code
map, a command sheet, a results summary, and a safe method-facing wrapper script. This document
records whether a *deeper* refactor — physically renaming the internal runtime — is worthwhile, and
why the recommendation is to leave the tested runtime paths stable.

## Current situation

The tested runtime uses three internal paths that still carry the experimentation-era label
`final_scope`:

- `src/final_scope/` — the shared evaluation core (fairness toolkit + the four runner cores).
- `scripts/final_scope/` — the launchers, validators, audits, and table generators.
- `results/runs/` — the saved per-cell outputs, the committed tables, and the fitted
  Dynamic-COUNT controllers.

For a reader, that label is noise. The method folders solve the *navigation* problem without
touching the runtime: they name each method, point to the exact implementation files, give safe
commands, and summarize the real results.

## Would a deeper rename help?

A future refactor could rename the three internal paths to something method-neutral (for example a
`core/` evaluation package, `run/` launchers, and `output/` results). It is technically possible but
carries real, broad risk for cosmetic benefit.

### Risks of physically renaming the runtime

1. **Import-path breakage.** `src.final_scope` is imported at roughly 27 call sites (every launcher,
   validator, audit, and table generator, plus the runner cores that import each other). A rename
   means editing every import and every `python -m` invocation together, in one commit, or the
   package stops importing. The two model environments (vlm_env, qwen_env) would both need
   re-verification.
2. **Result-basename breakage.** Static, Dynamic-WHICH, and Dynamic-COUNT resolve their dense /
   static / frozen references by **reconstructing file basenames** under `results/runs/`.
   Moving or renaming that tree, or the basenames within it, silently breaks reference resolution,
   the same-ids static-curve builder, and the reproduction gates — with no error until a run
   produces wrong deltas.
3. **Audit / table-script breakage.** The audits and table generators hardcode
   `results/runs/...` and `results/tables/...` paths. A rename requires updating
   all of them and re-confirming the committed tables regenerate byte-identically.
4. **Log / provenance-path breakage.** The final-run logs, the two archive manifests, and the
   committed tables all reference `final_scope` paths as the record of how the results were
   produced. Renaming the live tree desynchronizes that provenance from the frozen evidence.

### Benefit

Purely cosmetic: a tidier internal name. A reader already gets the method view from the top-level
folders, so the rename would not improve navigation — only the internal label.

## Recommendation

**Keep the tested runtime paths stable** (`src/final_scope/`, `scripts/final_scope/`,
`results/runs/`) and use the method-facing folders for clarity. The runtime is fully
validated and its result chain is frozen; a rename would put that chain at risk to change a name a
reader never has to see. The method folders, the per-method wrappers, and the README navigation
deliver the method-oriented structure without any of the four risks above.

If a rename is ever pursued, do it as an isolated, separately-approved pass with: a full backup of
`results/runs/`, a single atomic commit updating every import + `python -m` path + hardcoded
result path, the complete CPU validation suite plus both-environment import smokes re-run, and a
manifest recording the old→new path map. It is not recommended for the thesis submission.
