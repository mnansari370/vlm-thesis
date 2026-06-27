"""
test_final_scope.py — CPU-only self-checks for the Phase-2 dense infrastructure.

No GPU, no model, no large data files: all checks use tiny synthetic inputs. Run:
    python -m src.final_scope.test_final_scope
Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import sys

from src.final_scope.sample_ids import (
    compute_ids_sha256,
    proportional_quota,
    proportional_stratified_sample,
    assemble_manifest,
    validate_manifest,
    save_manifest,
    assert_strata_subset,
    assert_strata_exact,
    SampleManifest,
)

# convenience: a non-empty source_fingerprints for toy manifests (required field now)
_FP = {"toy_src": "deadbeef"}
from src.final_scope.schema_validator import (
    validate_per_sample,
    validate_visual_tokens,
    fairness_gate,
)
from src.final_scope.output_writer import per_sample_record, aggregate_record
from src.final_scope import token_flops as tf


_failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _failures.append(name)


def test_sha_deterministic():
    a = compute_ids_sha256([3, 1, 2])
    b = compute_ids_sha256([3, 1, 2])
    c = compute_ids_sha256([1, 2, 3])
    check("sha256 deterministic (same input → same hash)", a == b)
    check("sha256 order-sensitive (different order → different hash)", a != c)
    check("sha256 hex length 64", len(a) == 64)


def test_quota_and_sample():
    sizes = {"yes/no": 80541, "number": 28134, "other": 105679}
    q = proportional_quota(sizes, 25000)
    check("quota sums to exactly 25000", sum(q.values()) == 25000)
    check("quota within capacity", all(q[k] <= sizes[k] for k in sizes))
    # expected largest-remainder result for the real VQAv2 pool (yes/no has the largest
    # fractional remainder 0.4566 so it gets the +1): yes/no=9394, number=3281, other=12325.
    check("quota matches expected yes/no=9394", q["yes/no"] == 9394)
    check("quota matches expected number=3281", q["number"] == 3281)
    check("quota matches expected other=12325", q["other"] == 12325)

    strata = {"a": list(range(0, 1000)), "b": list(range(1000, 1200)), "c": list(range(1200, 1300))}
    ids1, c1 = proportional_stratified_sample(strata, 250, seed=42)
    ids2, c2 = proportional_stratified_sample(strata, 250, seed=42)
    ids3, _ = proportional_stratified_sample(strata, 250, seed=43)
    check("sample is exactly N", len(ids1) == 250)
    check("sample has no duplicates", len(set(ids1)) == 250)
    check("sample deterministic (same seed → identical)", ids1 == ids2 and c1 == c2)
    check("sample changes with seed", ids1 != ids3)
    check("sample sorted canonically", ids1 == sorted(ids1))


def test_manifest_loader_and_validation():
    m = assemble_manifest(
        dataset="toy", split="val", ordered_ids=[10, 20, 30], seed=42,
        selection_method="full", source_files=["x"], source_fingerprints=_FP, id_field="qid", notes="",
    )
    check("fresh manifest validates clean", validate_manifest(m) == [])
    sm = SampleManifest(m)
    check("loader exposes ordered ids", sm.ids == [10, 20, 30])
    check("loader verifies sha", sm.verify_sha())

    # tampered sha is caught
    bad = dict(m); bad["sha256"] = "deadbeef"
    check("validate catches wrong sha", "sha256 does not match ordered_ids" in validate_manifest(bad))
    # duplicate ids caught
    dup = assemble_manifest(dataset="toy", split="val", ordered_ids=[1, 1, 2], seed=1,
                            selection_method="x", source_files=["y"], source_fingerprints=_FP, id_field="qid")
    check("validate catches duplicate ids", any("duplicate" in p for p in validate_manifest(dup)))
    # n mismatch caught
    nm = dict(m); nm["n"] = 99
    check("validate catches n mismatch", any("!= len" in p for p in validate_manifest(nm)))
    # empty source_fingerprints caught for ANY manifest (not just row_index)
    nofp = assemble_manifest(dataset="toy", split="val", ordered_ids=[1, 2], seed=1,
                             selection_method="x", source_files=["y"], id_field="qid")  # no fingerprints
    check("validate catches empty source_fingerprints (all manifests)",
          any("source_fingerprints must be non-empty" in p for p in validate_manifest(nofp)))


def test_schema_validator():
    rec = per_sample_record(
        sample_id=1, image_id="img42", dataset="gqa", model="llava15", method="dense", budget_pct=100,
        question="q", prompt="p", pred_raw="yes", pred_norm="yes", gold=["yes"],
        per_sample_score=1.0, n_visual_tokens=576, n_text_tokens=34,
    )
    check("good record passes", validate_per_sample([rec]) == [])
    check("seq_len derived correctly", rec["seq_len"] == 610)

    missing = dict(rec); del missing["pred_raw"]
    check("validator catches missing field", any("missing" in p for p in validate_per_sample([missing])))

    empty = dict(rec); empty["pred_raw"] = ""
    check("validator catches empty prediction", any("empty prediction" in p for p in validate_per_sample([empty])))

    # image_id None / empty string is rejected (media/document id required)
    img_none = dict(rec); img_none["image_id"] = None
    check("validator catches image_id=None", any("empty image_id" in p for p in validate_per_sample([img_none])))
    img_empty = dict(rec); img_empty["image_id"] = ""
    check("validator catches image_id=''", any("empty image_id" in p for p in validate_per_sample([img_empty])))

    # ID order mismatch
    recs = [dict(rec, sample_id=1), dict(rec, sample_id=2)]
    out_of_order = validate_per_sample(recs, manifest_ids=[2, 1])
    check("validator catches ID ORDER mismatch", any("ORDER differs" in p for p in out_of_order))
    set_mismatch = validate_per_sample(recs, manifest_ids=[1, 3])
    check("validator catches ID SET mismatch", any("set mismatch" in p for p in set_mismatch))

    # image_id is required and its absence is caught
    no_imgid = dict(rec); del no_imgid["image_id"]
    check("validator catches missing image_id", any("missing" in p for p in validate_per_sample([no_imgid])))

    # visual-token check is EXPLICIT-expected, not always-576:
    #  - dense expects 576 → a 512 record is rejected
    dense_bad = dict(rec, n_visual_tokens=512, seq_len=512 + 34)
    check("dense LLaVA wrong visual tokens rejected (expected=576)",
          validate_visual_tokens([dense_bad], expected_visual_tokens=576) != [])
    #  - static K=144 record is NOT rejected when no expected is given (the old always-576 bug)
    static_rec = dict(rec, method="static", budget_pct=25, n_visual_tokens=144, seq_len=144 + 34)
    check("static LLaVA (144 tokens) NOT rejected by default (expected=None)",
          validate_visual_tokens([static_rec], expected_visual_tokens=None) == [])
    #  - and IS checked when its own expected K is supplied
    check("static LLaVA checked against its own K=144",
          validate_visual_tokens([static_rec], expected_visual_tokens=144) == [])

    # full fairness gate (need an aggregate)
    agg = aggregate_record(
        model="llava15", dataset="gqa", split="testdev_balanced", method="dense", budget_pct=100,
        n=1, metric="gqa_exact", score_pct=100.0, visual_tokens={"avg": 576, "max": 576},
        text_tokens_avg=34, seq_len_avg=610, flops_prefill_TFLOPs_avg=3.17,
        flops_convention="x", token_reduction_pct=0.0, flop_reduction_pct=0.0,
        prompt_template="t", max_new_tokens=64, decoding="greedy bs=1", image_pad=True,
        sample_ids_path="configs/final_scope/sample_ids/gqa.json", sample_ids_sha256="abc",
    )
    gate = fairness_gate([rec], agg, manifest_ids=[1], manifest_sha="abc", expected_visual_tokens=576)
    check("fairness gate passes a clean dense result", gate["ok"])
    gate_bad = fairness_gate([rec], agg, manifest_ids=[1], manifest_sha="WRONG")
    check("fairness gate catches sha mismatch", not gate_bad["ok"])

    # per-record vs aggregate identity mismatch (method/model/dataset/budget)
    mism = dict(rec, method="static", budget_pct=25)  # agg says dense/100
    g_mism = fairness_gate([mism], agg, manifest_ids=[1], manifest_sha="abc")
    check("fairness gate catches method/budget mismatch vs aggregate",
          (not g_mism["ok"]) and any("method=" in p for p in g_mism["issues"]))


def test_save_manifest_and_fingerprints():
    import os, tempfile
    # save_manifest must REFUSE a manifest whose sha doesn't match its ids
    good = assemble_manifest(dataset="toy", split="val", ordered_ids=[1, 2, 3], seed=1,
                             selection_method="full", source_files=["x"], source_fingerprints=_FP,
                             id_field="qid")
    bad = dict(good); bad["sha256"] = "deadbeef"
    refused = False
    try:
        with tempfile.TemporaryDirectory() as d:
            save_manifest(os.path.join(d, "m.json"), bad)
    except ValueError:
        refused = True
    check("save_manifest refuses an invalid (bad-sha) manifest", refused)
    # and writes a good one
    wrote_ok = False
    with tempfile.TemporaryDirectory() as d:
        save_manifest(os.path.join(d, "m.json"), good)
        wrote_ok = os.path.exists(os.path.join(d, "m.json"))
    check("save_manifest writes a valid manifest", wrote_ok)

    # row_index manifest WITHOUT source_fingerprints must be flagged
    ri_bad = assemble_manifest(dataset="textvqa", split="val", ordered_ids=[0, 1, 2], seed=1,
                               selection_method="full", source_files=["f.jsonl"], id_field="row_index")
    check("row_index manifest without fingerprints is rejected",
          any("source_fingerprints" in p for p in validate_manifest(ri_bad)))
    # WITH a fingerprint it validates
    ri_ok = assemble_manifest(dataset="textvqa", split="val", ordered_ids=[0, 1, 2], seed=1,
                              selection_method="full", source_files=["f.jsonl"], id_field="row_index",
                              source_fingerprints={"f.jsonl": "abc123"})
    check("row_index manifest with fingerprints validates", validate_manifest(ri_ok) == [])


def test_answer_type_strict():
    # exact answer_type set is accepted
    ok = True
    try:
        assert_strata_subset(["yes/no", "number", "other"], ("yes/no", "number", "other"))
    except ValueError:
        ok = False
    check("assert_strata_subset accepts the official 3 answer_type values", ok)
    # an unexpected value hard-fails
    raised = False
    try:
        assert_strata_subset(["yes/no", "number", "other", "color"], ("yes/no", "number", "other"))
    except ValueError:
        raised = True
    check("assert_strata_subset hard-fails on unexpected answer_type", raised)

    # EXACT check: the present set must equal {yes/no, number, other}
    ok_exact = True
    try:
        assert_strata_exact(["other", "number", "yes/no"], ("yes/no", "number", "other"))
    except ValueError:
        ok_exact = False
    check("assert_strata_exact accepts the exact official set", ok_exact)

    missing_raised = False
    try:
        assert_strata_exact(["yes/no", "number"], ("yes/no", "number", "other"))  # 'other' missing
    except ValueError:
        missing_raised = True
    check("assert_strata_exact hard-fails on a MISSING stratum", missing_raised)

    extra_raised = False
    try:
        assert_strata_exact(["yes/no", "number", "other", "color"], ("yes/no", "number", "other"))
    except ValueError:
        extra_raised = True
    check("assert_strata_exact hard-fails on an EXTRA stratum", extra_raised)


def test_token_flops():
    # LLaVA wrapper equals the underlying repo formula
    from src.analysis.flops import fastv_full_flops
    f = tf.per_sample_prefill_flops("llava15", 576, 34)
    check("llava per-sample flops == fastv_full_flops", f == float(fastv_full_flops(576, 34)))

    # convexity: mean(FLOPs(n_i)) > FLOPs(mean n_i) for varying token counts (Qwen)
    ns = [(100, 40), (2000, 40)]
    per = [tf.per_sample_prefill_flops("qwen25vl7b", v, t) for v, t in ns]
    avg_visual = sum(v for v, _ in ns) / len(ns)
    flops_of_mean = tf.per_sample_prefill_flops("qwen25vl7b", int(avg_visual), 40)
    check("mean(FLOPs) > FLOPs(mean) for varying tokens (convex n^2)",
          tf.mean_flops(per) > flops_of_mean)

    stats = [tf.TokenStats(576, 34), tf.TokenStats(576, 34)]
    summ = tf.summarize_tokens_and_flops("llava15", stats)
    check("summary visual avg == 576", summ["visual_tokens_avg"] == 576)
    check("summary reports TFLOPs avg > 0", summ["flops_prefill_TFLOPs_avg"] > 0)
    check("reduction_pct(dense vs dense) == 0", tf.reduction_pct(10.0, 10.0) == 0.0)


def main():
    print("=== final_scope CPU self-checks ===")
    for t in (test_sha_deterministic, test_quota_and_sample, test_manifest_loader_and_validation,
              test_save_manifest_and_fingerprints, test_answer_type_strict,
              test_schema_validator, test_token_flops):
        print(f"\n# {t.__name__}")
        t()
    print(f"\n{'ALL PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
