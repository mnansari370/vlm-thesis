# `GQA/` — oracle-headroom diagnostic (GQA + TextVQA + POPE + ScienceQA)

Frozen LLaVA-1.5-7B. **Runnable.** All evaluation uses the locked honest protocol
(image_pad, official prompt/scorer per benchmark, greedy, max_new_tokens=64, no min_new_tokens,
no repetition_penalty, bs=1). See `../docs/reproducibility.md`.

## Layout

```
GQA/
  dense/         dense (K=576) baseline
    run_dense_testdev.py   honest-protocol dense eval; also defines GQATestdevDataset/collate
                           reused by the static & dynamic runners
                           (the legacy val_balanced dense runner was removed; testdev is canonical)
  static/        static pruning + question-conditioned selection probes (both negatives)
    run_static.py, run_static_testdev.py            CLS-Attn / random / spatial / L2 at fixed K
    run_visionzip_testdev.py, visionzip.py          VisionZip baseline (~= static)
    run_clip_probe.py, clip_select.py               CLIP-space relevance selector (-32pp)
    run_qcond_probe.py, question_cond.py            LM-attention Q-conditioned selector (-5.58)
    clip_visual_check.py
  dynamic/       confidence cascade (traces along the static frontier, never above)
    run_speculative_testdev.py, run_pope_speculative.py, cascade_sweep.py
  eval_runners/  per-benchmark eval entrypoints
    run_textvqa.py, run_pope.py, run_sqa.py
  analysis/      no-GPU regeneration of figures/tables from saved ../outputs/
    analyze_frontier.py, analyze_textvqa.py, analyze_pope.py
    build_figures.py, build_master_table.py, build_latex.py, build_results.py, build_oracle.py, gather_week1.py
  shared/        shared building blocks (imported everywhere; the old `common/`)
    static.py            StaticPrunedLlava — physical token removal (CLS-Attn/random/spatial/L2)
    flops.py             FastV Eq.5 FLOPs, per-benchmark n_text constants
    metrics.py           answer extraction / accuracy helpers
    official_score.py    canonical GQA scorer (normalize = strip.rstrip('.').lower())
    textvqa_score.py / m4c_evaluator.py   TextVQA M4C scorer
    pope_score.py / eval_pope_official.py POPE accuracy/F1
    dataset.py           GQA loader
    utils/               logger/checkpoint/config helpers
```

Datasets live in the top-level `../data/`; experiment outputs in `../outputs/` (both git-ignored).

## Regenerate everything (no GPU)

```bash
python -m GQA.dynamic.cascade_sweep
python -m GQA.analysis.build_master_table
python -m GQA.analysis.build_figures      # F1-F5 + tables.md
python -m GQA.analysis.build_latex        # tables -> paper/ + thesis/
```

GPU runs (only if re-deriving from scratch): `GQA.dense.run_dense_testdev`,
`GQA.static.run_static_testdev`, `GQA.eval_runners.run_{textvqa,pope,sqa}` — all bs=1.

## Superseded experiments

A series of learned dynamic-budget predictors (budget-MLP, QV, region, generative) and
LoRA/dense fine-tuning attempts were explored and all failed to beat the static CLS-attention
frontier; those negative results motivated the oracle-headroom diagnostic that this track
implements. Only the final, verified pipeline is kept here.
