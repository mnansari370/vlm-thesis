# Dynamic-COUNT (adaptive per-sample budget)

## What the method does

Dynamic-COUNT keeps the selector fixed (the static selectors) and changes *how many* tokens each
sample keeps, aiming to beat the best single fixed budget at the same average compute. Two variants:

- **DC-D** (discrete cascade, baseline): a cheap probe pass, then escalate to a fixed budget anchor
  when a calibrated confidence signal says the sample is unsure. Composed offline from the frozen
  finals.
- **DC-C** (continuous, the main method): a calibrated controller predicts a per-sample **integer**
  token count K_i = clamp(round(fraction(risk)·N_i), 32, N_i) — a genuine continuum, not one of the
  five anchors — and runs one real second pass at exactly K_i when the probe budget is insufficient.
  Two controllers: a transparent rule controller and a small ridge regression.

Accounting is honest: every executed prefill is charged, so an escalated sample pays for both its
probe and its second pass ("double-pay"). Controllers are fitted on the first 20% of each manifest
and scored on the held-out 80%; the comparison target is the static accuracy-vs-FLOPs curve
interpolated at matched average compute.

## Why it exists in the thesis

This is the *how many tokens* axis. An oracle bound shows real headroom (+2 to +6 pp over the best
fixed budget), so the honest question is whether any input-side controller can harvest it.

## What code implements it

The method-facing entry point is this folder; exact files are in [CODE_MAP.md](CODE_MAP.md). The
probe wrappers mirror the frozen engines byte-for-byte (a reproduction gate enforces this); the
controllers are small, transparent, and CPU-fitted.

## What scripts run it

- Method-facing wrapper (safe, CPU): [`scripts/validate_dynamic_count.sh`](scripts/validate_dynamic_count.sh)
- Full commands (the 5-stage pipeline): [COMMANDS.md](COMMANDS.md)

## Where results are stored

Per cell: `results/final_scope/{model}/{dataset}/dynamic_count_{probe,dcd,dcc}_*[...].{json,jsonl}`.
Fitted controllers: `results/final_scope/dynamic_count_configs/{model}_{dataset}[_textsim].json`.
Committed summaries: `results/final_scope/tables/dynamic_count_{dc_d,dc_c,win_loss,oracle}_summary.md`;
reproduced in [RESULTS.md](RESULTS.md).

## Final conclusion

Dynamic-COUNT is a fully-tested, honestly-accounted **negative result**. DC-D wins once (LLaVA
TextVQA, +0.75 pp) and otherwise near-ties or loses; DC-C never wins (best +0.48, a near-tie; the
steep-curve cells lose by up to −11.07). Running COUNT on top of the successful WHICH selector adds
**no increment** over WHICH alone (−0.41 to −2.30 vs the textsim curve). The confidence signals are
informative (47–73% wrong-capture), but under honest double-pay accounting the oracle headroom is
not harvestable: flat curves leave no room, and steep curves punish every wasted probe.
