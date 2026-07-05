# `configs/`

```
configs/
  final_scope/sample_ids/{gqa,textvqa,docvqa,vqav2}.json
```

The **sample-ID manifests** are the only configs the final experiment matrix uses: version-controlled,
sha256-verified ordered ID lists that define the exact evaluation subset per dataset (GQA 12,578 ·
TextVQA 5,000 · DocVQA 5,349 · VQAv2 25,000 stratified by official `answer_type`, seed 42). Every
run — dense, static, Dynamic-WHICH, Dynamic-COUNT, both models — reads the same manifest and records
its path + sha256 in the output aggregate, so all methods are provably scored on identical samples.
Built once by `scripts/final_scope/build_sample_manifests.py`; loaded/validated by
`src/final_scope/sample_ids.py`. **Never regenerate or reorder these files** — the sha256 is pinned
in every saved result.

The final-scope runners are argparse-driven (see `scripts/final_scope/`), so there are no method
YAMLs here. The old classification-era YAMLs (`dense/`, `static/`, `dynamic_budget/`) were archived
in cleanup Pass 1 (2026-07-05) to `archive/legacy_experiments/configs/`; see
`archive/migration_manifests/archive_manifest_20260705.md`.
