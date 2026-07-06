# `configs/`

> **For method-specific navigation, start from [`dense/`](../dense/), [`static/`](../static/),
> [`dynamic_which/`](../dynamic_which/), or [`dynamic_count/`](../dynamic_count/).** All methods share
> the sample manifests below.

```
configs/
  sample_ids/{gqa,textvqa,docvqa,vqav2}.json
```

The **sample-ID manifests** are the only configs the final experiment matrix uses: version-controlled,
sha256-verified ordered ID lists that define the exact evaluation subset per dataset (GQA 12,578 ·
TextVQA 5,000 · DocVQA 5,349 · VQAv2 25,000 stratified by official `answer_type`, seed 42). Every
run — dense, static, Dynamic-WHICH, Dynamic-COUNT, both models — reads the same manifest and records
its path + sha256 in the output aggregate, so all methods are provably scored on identical samples.
Built once by `scripts/data/build_sample_manifests.py`; loaded/validated by
`src/common/sample_ids.py`. **Never regenerate or reorder these files** — the sha256 is pinned
in every saved result.

The evaluation runners are argparse-driven (see the `scripts/` method folders), so there are no
method YAMLs here. The old classification-era YAMLs (`dense/`, `static/`, `dynamic_budget/`) were
archived in cleanup Pass 1 (2026-07-05) to `archive/legacy_experiments/configs/`; see
`archive/migration_manifests/archive_manifest_20260705.md`.
