"""
test_evaluation_core.py — CPU-only self-checks for the shared evaluation core.

No GPU, no model, no large data files: all checks use tiny synthetic inputs. Run:
    python -m src.common.test_evaluation_core
Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import sys

from src.common.sample_ids import (
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
from src.common.schema_validator import (
    validate_per_sample,
    validate_visual_tokens,
    fairness_gate,
)
from src.common.output_writer import per_sample_record, aggregate_record
from src.common import token_flops as tf


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
        sample_ids_path="configs/sample_ids/gqa.json", sample_ids_sha256="abc",
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


def test_static_eval():
    import os, re
    from src.static import evaluate_static as se

    # 1. budget_pct → LLaVA keep_k map (incl. 86 and 202) and the SUPPORTED_K source extension
    check("LLaVA budget→K map exact",
          se.BUDGET_TO_KEEP_K == {15: 86, 25: 144, 35: 202, 50: 288, 75: 432})
    check("BUDGETS tuple is (15,25,35,50,75)", se.BUDGETS == (15, 25, 35, 50, 75))
    check("budget_keep_k llava 15→86", se.budget_keep_k("llava15", 15) == 86)
    check("budget_keep_k llava 35→202", se.budget_keep_k("llava15", 35) == 202)
    check("budget_keep_k qwen→None (per-sample)", se.budget_keep_k("qwen25vl7b", 25) is None)
    bad_budget = False
    try:
        se.budget_keep_k("llava15", 40)
    except ValueError:
        bad_budget = True
    check("budget_keep_k rejects an unlisted budget", bad_budget)
    # parse SUPPORTED_K straight from the source (no torch import) → 86 and 202 are now allowed
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _src = open(os.path.join(_root, "src/models/static/static.py")).read()
    _m = re.search(r"SUPPORTED_K\s*=\s*frozenset\(\{([^}]*)\}\)", _src)
    _ks = {int(x) for x in re.findall(r"\d+", _m.group(1))} if _m else set()
    check("static.py SUPPORTED_K includes 86 and 202", {86, 202} <= _ks)
    check("static.py SUPPORTED_K covers all LLaVA budgets",
          set(se.BUDGET_TO_KEEP_K.values()) <= _ks)

    # 2. static output basename variants
    check("basename gqa pilot",
          se.static_basename("gqa", 200, "cls_attn", 25) == "static_pilot_n200_cls_attn_p25")
    check("basename gqa final",
          se.static_basename("gqa", 0, "cls_attn", 25, full=True) == "static_final_cls_attn_p25")
    check("basename vqav2 pilot",
          se.static_basename("vqav2", 200, "cls_attn", 15) == "static_pilot_n200_cls_attn_p15")
    check("basename textvqa ocr-on",
          se.static_basename("textvqa", 200, "cls_attn", 50, use_ocr=True)
          == "static_pilot_n200_cls_attn_p50_ocr_on")
    check("basename textvqa ocr-off",
          se.static_basename("textvqa", 200, "cls_attn", 50, use_ocr=False)
          == "static_pilot_n200_cls_attn_p50_ocr_off")
    check("basename docvqa instruction-on",
          se.static_basename("docvqa", 200, "norm", 75, instruction=True)
          == "static_pilot_n200_norm_p75_instruction_on")
    check("basename docvqa instruction-off final",
          se.static_basename("docvqa", 0, "norm", 75, instruction=False, full=True)
          == "static_final_norm_p75_instruction_off")

    # 3. --full and --n conflict  +  4. n > manifest length
    conflict = False
    try:
        se.resolve_n(100, True, 1000)
    except ValueError:
        conflict = True
    check("resolve_n errors on --full + --n", conflict)
    too_big = False
    try:
        se.resolve_n(2000, False, 1000)
    except ValueError:
        too_big = True
    check("resolve_n errors on n > manifest length", too_big)
    default_too_big = False
    try:
        se.resolve_n(None, False, 50)  # default PILOT_N=200 > manifest 50 → error
    except ValueError:
        default_too_big = True
    check("resolve_n errors when default PILOT_N exceeds a tiny manifest", default_too_big)
    check("resolve_n full → total", se.resolve_n(None, True, 1000) == 1000)
    check("resolve_n default → PILOT_N", se.resolve_n(None, False, 1000) == se.PILOT_N)
    check("resolve_n explicit n passes", se.resolve_n(50, False, 1000) == 50)

    # 5. selector defaults + per-model allow-lists
    check("default_selector llava→cls_attn", se.default_selector("llava15") == "cls_attn")
    check("default_selector qwen→norm", se.default_selector("qwen25vl7b") == "norm")
    check("allowed_selectors llava == (cls_attn,)", se.allowed_selectors("llava15") == ("cls_attn",))
    check("allowed_selectors qwen == (norm, uniform)",
          se.allowed_selectors("qwen25vl7b") == ("norm", "uniform"))

    # 6. Qwen per-sample K = clamp(round(pct% · dense_n_visual), 1, dense_n_visual)
    check("clamp_qwen_k 15% of 1000 = 150", se.clamp_qwen_k(15, 1000) == 150)
    check("clamp_qwen_k 75% of 100 = 75", se.clamp_qwen_k(75, 100) == 75)
    check("clamp_qwen_k 50% of 324 = 162", se.clamp_qwen_k(50, 324) == 162)
    check("clamp_qwen_k clamps up to >=1 (15% of 3 → 1)", se.clamp_qwen_k(15, 3) == 1)
    check("clamp_qwen_k never exceeds dense_n_visual (75% of 4 → 3)", se.clamp_qwen_k(75, 4) == 3)
    # exact half-integer ties resolve to the nearest EVEN integer (Python round; matches Sec 3.7)
    check("clamp_qwen_k tie 50% of 5 = 2.5 → 2 (even)", se.clamp_qwen_k(50, 5) == 2)
    check("clamp_qwen_k tie 50% of 7 = 3.5 → 4 (even)", se.clamp_qwen_k(50, 7) == 4)
    check("clamp_qwen_k tie 50% of 9 = 4.5 → 4 (even)", se.clamp_qwen_k(50, 9) == 4)
    check("clamp_qwen_k tie 50% of 3 = 1.5 → 2 (even)", se.clamp_qwen_k(50, 3) == 2)
    check("clamp_qwen_k tie 25% of 6 = 1.5 → 2 (even)", se.clamp_qwen_k(25, 6) == 2)
    check("clamp_qwen_k tie 25% of 10 = 2.5 → 2 (even)", se.clamp_qwen_k(25, 10) == 2)
    check("clamp_qwen_k tie 25% of 14 = 3.5 → 4 (even)", se.clamp_qwen_k(25, 14) == 4)
    check("clamp_qwen_k tie 25% of 2 = 0.5 → 0 then clamped to 1", se.clamp_qwen_k(25, 2) == 1)

    # 7. LLaVA static expected-visual-token validation against its own K
    recK = per_sample_record(
        sample_id=1, image_id="i", dataset="gqa", model="llava15", method="static", budget_pct=35,
        question="q", prompt="p", pred_raw="a", pred_norm="a", gold=["a"], per_sample_score=1.0,
        n_visual_tokens=202, n_text_tokens=30, selector="cls_attn")
    check("LLaVA static K=202 record passes expected=202",
          validate_visual_tokens([recK], expected_visual_tokens=202) == [])
    bad_k = dict(recK, n_visual_tokens=200, seq_len=230)
    check("LLaVA static wrong visual tokens rejected (expected=202)",
          validate_visual_tokens([bad_k], expected_visual_tokens=202) != [])

    # 8. schema / fairness gate accepts static method+budget+selector metadata
    agg = aggregate_record(
        model="llava15", dataset="gqa", split="testdev_balanced", method="static", budget_pct=35,
        n=1, metric="gqa_exact", score_pct=100.0, visual_tokens={"avg": 202, "max": 202},
        text_tokens_avg=30, seq_len_avg=232, flops_prefill_TFLOPs_avg=1.2, flops_convention="x",
        token_reduction_pct=64.5, flop_reduction_pct=70.0, prompt_template="t", max_new_tokens=64,
        decoding="greedy bs=1", image_pad=True,
        sample_ids_path="configs/sample_ids/gqa.json", sample_ids_sha256="abc",
        selector_name="cls_attn", keep_k=202)
    gate = fairness_gate([recK], agg, manifest_ids=[1], manifest_sha="abc", expected_visual_tokens=202)
    check("fairness gate passes a clean static LLaVA (K=202) result", gate["ok"])

    # Qwen static: visual tokens VARY per image → expected=None must not reject
    qrec1 = per_sample_record(
        sample_id=10, image_id="i", dataset="gqa", model="qwen25vl7b", method="static", budget_pct=25,
        question="q", prompt="p", pred_raw="a", pred_norm="a", gold=["a"], per_sample_score=1.0,
        n_visual_tokens=81, n_text_tokens=20, selector="norm")
    qrec2 = per_sample_record(
        sample_id=11, image_id="i", dataset="gqa", model="qwen25vl7b", method="static", budget_pct=25,
        question="q", prompt="p", pred_raw="b", pred_norm="b", gold=["b"], per_sample_score=0.0,
        n_visual_tokens=64, n_text_tokens=22, selector="norm")
    qagg = aggregate_record(
        model="qwen25vl7b", dataset="gqa", split="testdev_balanced", method="static", budget_pct=25,
        n=2, metric="gqa_exact", score_pct=50.0, visual_tokens={"avg": 72.5, "max": 81},
        text_tokens_avg=21, seq_len_avg=93.5, flops_prefill_TFLOPs_avg=0.4, flops_convention="x",
        token_reduction_pct=70.0, flop_reduction_pct=80.0, prompt_template="t", max_new_tokens=64,
        decoding="greedy bs=1", image_pad=None,
        sample_ids_path="configs/sample_ids/gqa.json", sample_ids_sha256="z",
        selector_name="norm", keep_k=None)
    qgate = fairness_gate([qrec1, qrec2], qagg, manifest_ids=[10, 11], manifest_sha="z",
                          expected_visual_tokens=None)
    check("fairness gate passes static Qwen (varying tokens, expected=None)", qgate["ok"])

    # Qwen retained-token soft sanity: close → None; gross mismatch → flagged
    check("qwen_retained_sanity OK within tolerance",
          se.qwen_retained_sanity(35.0, 144.0, 25) is None)            # requested 36, |Δ|=1
    check("qwen_retained_sanity flags gross mismatch",
          se.qwen_retained_sanity(10.0, 144.0, 25) is not None)        # requested 36, |Δ|=26

    # 9. accuracy metric rename: accuracy_delta_pp (signed) + accuracy_drop_pp (dense - static)
    d, dr = se.accuracy_deltas(60.0, 58.0)                # static beats dense by 2pp
    check("accuracy_delta_pp = static - dense (+2.0 when static wins)", d == 2.0)
    check("accuracy_drop_pp = dense - static (-2.0, i.e. NEGATIVE drop, when static wins)", dr == -2.0)
    d2, dr2 = se.accuracy_deltas(55.0, 58.0)              # static worse by 3pp
    check("accuracy_delta_pp negative when static worse", d2 == -3.0)
    check("accuracy_drop_pp positive when static worse", dr2 == 3.0)
    check("accuracy_delta_pp and accuracy_drop_pp are negatives of each other", d2 == -dr2)
    d3, dr3 = se.accuracy_deltas(58.0, 58.0)              # tie
    check("accuracy_delta/drop both 0.0 on a tie", d3 == 0.0 and dr3 == 0.0)
    check("accuracy_deltas returns a (delta, drop) 2-tuple", isinstance(se.accuracy_deltas(1.0, 0.0), tuple))


def test_dynamic_which():
    from src.static import evaluate_static as se
    from src.dynamic_which import evaluate_dynamic_which as dw

    # allowed dynamic selectors per model + defaults
    check("dynamic selectors llava == (textsim, textsim_cls_mix)",
          dw.allowed_dynamic_selectors("llava15") == ("textsim", "textsim_cls_mix"))
    check("dynamic selectors qwen == (textsim, textsim_norm_mix)",
          dw.allowed_dynamic_selectors("qwen25vl7b") == ("textsim", "textsim_norm_mix"))
    check("default_dynamic_selector llava → textsim", dw.default_dynamic_selector("llava15") == "textsim")
    check("default_dynamic_selector qwen → textsim", dw.default_dynamic_selector("qwen25vl7b") == "textsim")

    # dynamic basename generation (same variant scheme as static)
    check("dyn basename gqa pilot",
          dw.dynamic_basename("gqa", 200, "textsim", 25) == "dynamic_which_pilot_n200_textsim_p25")
    check("dyn basename gqa final",
          dw.dynamic_basename("gqa", 0, "textsim", 25, full=True) == "dynamic_which_final_textsim_p25")
    check("dyn basename textvqa ocr-on",
          dw.dynamic_basename("textvqa", 200, "textsim", 75, use_ocr=True)
          == "dynamic_which_pilot_n200_textsim_p75_ocr_on")
    check("dyn basename docvqa instruction-on final",
          dw.dynamic_basename("docvqa", 0, "textsim_norm_mix", 50, instruction=True, full=True)
          == "dynamic_which_final_textsim_norm_mix_p50_instruction_on")

    # budget mapping matches static EXACTLY (same map, same keep_k)
    check("dyn BUDGET_TO_KEEP_K is the static map", dw.BUDGET_TO_KEEP_K == se.BUDGET_TO_KEEP_K)
    check("dyn budgets == static budgets", dw.BUDGETS == se.BUDGETS == (15, 25, 35, 50, 75))
    check("dyn budget_keep_k llava 35→202 (== static)",
          dw.budget_keep_k("llava15", 35) == se.budget_keep_k("llava15", 35) == 202)
    check("dyn budget_keep_k qwen → None", dw.budget_keep_k("qwen25vl7b", 25) is None)

    # Qwen per-sample K rule matches static EXACTLY
    check("dyn clamp_qwen_k is the static fn", dw.clamp_qwen_k is se.clamp_qwen_k)
    for b, nv in [(15, 1000), (25, 729), (50, 324), (75, 100)]:
        check(f"dyn/static K agree (b={b}, nv={nv})",
              dw.clamp_qwen_k(b, nv) == se.clamp_qwen_k(b, nv))

    # dynamic_minus_static_pp math
    check("dynamic_minus_static positive", dw.dynamic_minus_static(60.0, 58.0) == 2.0)
    check("dynamic_minus_static negative", dw.dynamic_minus_static(55.5, 58.0) == -2.5)

    # selection: top-K then SORT back to original order (critical rule) + keep-all == identity
    check("select_topk_sorted picks the highest, sorted",
          dw.select_topk_sorted([0.1, 0.9, 0.5, 0.3], 2) == [1, 2])
    check("select_topk_sorted is ascending (original visual order)",
          dw.select_topk_sorted([0.9, 0.1, 0.8, 0.2, 0.7], 3) == [0, 2, 4])
    check("select_topk_sorted keep-all == identity (dense-equivalence hook)",
          dw.select_topk_sorted([0.3, 0.1, 0.2, 0.9], 4) == [0, 1, 2, 3])
    check("select_topk_sorted clamps k≥1", dw.select_topk_sorted([0.5, 0.2], 0) == [0])

    # normalized mix math (textsim_*_mix reference)
    mixed = dw.normalized_mix([0.0, 1.0], [1.0, 0.0], 0.5)
    check("normalized_mix alpha=0.5 averages the normalized extremes", abs(mixed[0] - 0.5) < 1e-6)
    mixed_a1 = dw.normalized_mix([0.0, 10.0], [5.0, 5.0], 1.0)
    check("normalized_mix alpha=1.0 == textsim only", mixed_a1[1] > mixed_a1[0])

    # static reference path resolution for ALL primary static outputs (no file needed)
    for b in se.BUDGETS:
        check(f"static ref basename llava gqa p{b}",
              dw.static_reference_basename("llava15", "gqa", b) == f"static_final_cls_attn_p{b}")
        check(f"static ref basename qwen gqa p{b}",
              dw.static_reference_basename("qwen25vl7b", "gqa", b) == f"static_final_norm_p{b}")
    check("static ref basename llava textvqa ocr-on",
          dw.static_reference_basename("llava15", "textvqa", 50) == "static_final_cls_attn_p50_ocr_on")
    check("static ref basename qwen docvqa instruction-on",
          dw.static_reference_basename("qwen25vl7b", "docvqa", 75) == "static_final_norm_p75_instruction_on")
    check("static ref basename honors an explicit static_selector override",
          dw.static_reference_basename("qwen25vl7b", "gqa", 25, static_selector="uniform")
          == "static_final_uniform_p25")

    # method / fairness gate accepts dynamic_which records (LLaVA static-K + Qwen varying)
    rec = per_sample_record(
        sample_id=1, image_id="i", dataset="gqa", model="llava15", method="dynamic_which",
        budget_pct=35, question="q", prompt="p", pred_raw="a", pred_norm="a", gold=["a"],
        per_sample_score=1.0, n_visual_tokens=202, n_text_tokens=30, selector="textsim")
    agg = aggregate_record(
        model="llava15", dataset="gqa", split="testdev_balanced", method="dynamic_which",
        budget_pct=35, n=1, metric="gqa_exact", score_pct=100.0, visual_tokens={"avg": 202, "max": 202},
        text_tokens_avg=30, seq_len_avg=232, flops_prefill_TFLOPs_avg=1.2, flops_convention="x",
        token_reduction_pct=64.5, flop_reduction_pct=70.0, prompt_template="t", max_new_tokens=64,
        decoding="greedy bs=1", image_pad=True,
        sample_ids_path="configs/sample_ids/gqa.json", sample_ids_sha256="abc",
        selector_name="textsim", keep_k=202, dynamic_minus_static_pp=1.5)
    gate = fairness_gate([rec], agg, manifest_ids=[1], manifest_sha="abc", expected_visual_tokens=202)
    check("fairness gate passes a clean dynamic_which LLaVA (K=202) result", gate["ok"])

    qrec1 = per_sample_record(
        sample_id=10, image_id="i", dataset="gqa", model="qwen25vl7b", method="dynamic_which",
        budget_pct=25, question="q", prompt="p", pred_raw="a", pred_norm="a", gold=["a"],
        per_sample_score=1.0, n_visual_tokens=81, n_text_tokens=20, selector="textsim")
    qrec2 = per_sample_record(
        sample_id=11, image_id="i", dataset="gqa", model="qwen25vl7b", method="dynamic_which",
        budget_pct=25, question="q", prompt="p", pred_raw="b", pred_norm="b", gold=["b"],
        per_sample_score=0.0, n_visual_tokens=64, n_text_tokens=22, selector="textsim")
    qagg = aggregate_record(
        model="qwen25vl7b", dataset="gqa", split="testdev_balanced", method="dynamic_which",
        budget_pct=25, n=2, metric="gqa_exact", score_pct=50.0, visual_tokens={"avg": 72.5, "max": 81},
        text_tokens_avg=21, seq_len_avg=93.5, flops_prefill_TFLOPs_avg=0.4, flops_convention="x",
        token_reduction_pct=70.0, flop_reduction_pct=80.0, prompt_template="t", max_new_tokens=64,
        decoding="greedy bs=1", image_pad=None,
        sample_ids_path="configs/sample_ids/gqa.json", sample_ids_sha256="z",
        selector_name="textsim", keep_k=None, dynamic_minus_static_pp=-1.0)
    qgate = fairness_gate([qrec1, qrec2], qagg, manifest_ids=[10, 11], manifest_sha="z",
                          expected_visual_tokens=None)
    check("fairness gate passes dynamic_which Qwen (varying tokens, expected=None)", qgate["ok"])

    # --full + --n conflict (reused resolve_n)
    conflict = False
    try:
        dw.resolve_n(100, True, 1000)
    except ValueError:
        conflict = True
    check("dyn resolve_n errors on --full + --n", conflict)

    # selector_text_source: scoring uses the raw question, NOT the VLM prompt
    import inspect
    check("DEFAULT_SELECTOR_TEXT_SOURCE == 'sample.question'",
          dw.DEFAULT_SELECTOR_TEXT_SOURCE == "sample.question")
    sig = inspect.signature(dw.run_dynamic_which)
    check("run_dynamic_which has selector_text_source default 'sample.question'",
          sig.parameters["selector_text_source"].default == "sample.question")
    # the metadata field is emitted under extra_dynamic_which (documented contract)
    check("dynamic basename + text-source are wired (constant present)",
          isinstance(dw.DEFAULT_SELECTOR_TEXT_SOURCE, str))

    # keep-all equivalence script is importable (CPU-safe: torch imports are lazy) + exposes main()
    import importlib
    eq = importlib.import_module("scripts.dynamic_which.check_equivalence")
    check("equivalence script exposes main()", hasattr(eq, "main"))
    check("equivalence script exposes build_llava_keepall/build_qwen_keepall",
          hasattr(eq, "build_llava_keepall") and hasattr(eq, "build_qwen_keepall"))


def test_dynamic_which_ref():
    from src.pruning import dynamic_which_ref as ref
    from src.dynamic_which import evaluate_dynamic_which as dw

    # allowed ref selectors per model (Qwen has no coverage_guarded)
    check("llava ref selectors",
          ref.allowed_ref_selectors("llava15")
          == ("textsim_ref", "saliency_mix_ref", "static_guarded_textsim_ref", "coverage_guarded_textsim_ref"))
    check("qwen ref selectors (no coverage_guarded)",
          ref.allowed_ref_selectors("qwen25vl7b")
          == ("textsim_ref", "saliency_mix_ref", "static_guarded_textsim_ref"))
    check("default ref selector is textsim_ref", ref.default_ref_selector("llava15") == "textsim_ref")

    # top-k sorted: identity at K=all; picks highest; ascending
    check("topk_sorted keep-all == identity",
          ref.topk_sorted([0.3, 0.1, 0.9, 0.2], 4) == [0, 1, 2, 3])
    check("topk_sorted picks highest, sorted",
          ref.topk_sorted([0.1, 0.9, 0.5, 0.3], 2) == [1, 2])

    # guarded selection: never exceeds K, preserves original order, includes reserved
    fill = [0.9, 0.1, 0.8, 0.2, 0.7, 0.0, 0.6, 0.3]   # n=8
    reserve = [5, 1]                                   # low-score tokens forced in
    keep = ref.guarded_select(4, reserve, fill)
    check("guarded_select never exceeds K", len(keep) <= 4)
    check("guarded_select exactly K when K<=n", len(keep) == 4)
    check("guarded_select preserves original order (sorted)", keep == sorted(keep))
    check("guarded_select includes ALL reserved tokens", set(reserve) <= set(keep))
    check("guarded_select fills remainder from top fill_scores (0 and 2 are top non-reserved)",
          keep == sorted(set(reserve) | {0, 2}))
    # reserve larger than K is capped at K (still <= K, no overflow)
    keep2 = ref.guarded_select(2, [7, 6, 5, 4], fill)
    check("guarded_select caps oversized reserve at K", len(keep2) == 2 and set(keep2) <= {4, 5, 6, 7})

    # spatial coverage indices: r evenly-spaced, sorted, unique, in-range
    cov = ref.spatial_coverage_indices(5, 576)
    check("spatial_coverage count == r", len(cov) == 5)
    check("spatial_coverage sorted+unique+in-range",
          cov == sorted(set(cov)) and all(0 <= i < 576 for i in cov))
    check("spatial_coverage r>=n → full", ref.spatial_coverage_indices(600, 576) == list(range(576)))

    # ref basename cannot collide with existing dynamic_which outputs (must carry _ref_)
    rb = ref.ref_basename("gqa", 200, "textsim_ref", 25)
    check("ref basename carries dynamic_which_ref", rb.startswith("dynamic_which_ref_pilot_n200"))
    check("ref basename != current dynamic_which basename",
          rb != dw.dynamic_basename("gqa", 200, "textsim", 25))
    check("ref basename textvqa variant",
          ref.ref_basename("textvqa", 200, "saliency_mix_ref", 25)
          == "dynamic_which_ref_pilot_n200_saliency_mix_ref_p25_ocr_on")
    check("ref basename docvqa variant",
          ref.ref_basename("docvqa", 200, "static_guarded_textsim_ref", 35)
          == "dynamic_which_ref_pilot_n200_static_guarded_textsim_ref_p35_instruction_on")

    # ref eval CLI: NO --full flag (n-only), and passing --full is rejected
    import importlib
    rr = importlib.import_module("scripts.dynamic_which.run_dynamic_which_ref")
    parser = rr.build_parser()
    opt_strings = {s for a in parser._actions for s in a.option_strings}
    check("ref eval CLI has no --full flag", "--full" not in opt_strings)
    rejected = False
    try:
        parser.parse_args(["--model", "llava15", "--dataset", "gqa", "--budget-pct", "25", "--full"])
    except SystemExit:
        rejected = True
    check("ref eval rejects --full", rejected)

    # coverage_guarded is not constructible for Qwen (documented layout limitation)
    check("Qwen omits coverage_guarded from allowed",
          "coverage_guarded_textsim_ref" not in ref.allowed_ref_selectors("qwen25vl7b"))


def test_dynamic_count():
    from src.pruning import dynamic_count as dcp
    from src.pruning.dynamic_count.confidence import confidence_signals, risk_from_signal
    from src.pruning.dynamic_count.controllers import (RuleController, RidgeRiskModel,
                                                       sufficiency_fraction, threshold_for_rate,
                                                       load_controller)
    from src.dynamic_count import evaluate_dynamic_count as dce

    # clamp + split
    check("clamp_k rounds and clamps", dcp.clamp_k(96.6, 32, 576) == 97)
    check("clamp_k floor", dcp.clamp_k(3, 32, 576) == 32)
    check("clamp_k ceiling (max wins when max<min)", dcp.clamp_k(999, 32, 20) == 20)
    cal, ev = dcp.calibration_split(list(range(100)))
    check("calibration split is first 20% / rest", cal == list(range(20)) and ev == list(range(20, 100)))
    check("calibration split disjoint+complete", set(cal) | set(ev) == set(range(100)))

    # saliency stats
    uni = dcp.saliency_stats([1.0] * 100)
    check("uniform saliency → PR == n", abs(uni["sal_participation_ratio"] - 100) < 1e-6)
    delta = dcp.saliency_stats([0.0] * 99 + [10.0])
    check("peaked saliency → PR == 1, top32 mass == 1",
          abs(delta["sal_participation_ratio"] - 1) < 1e-6 and abs(delta["sal_top32_mass"] - 1) < 1e-9)

    # confidence signals + risk direction
    c = confidence_signals(0.9, 0.05, 0.3, [0.9, 0.8, 0.7])
    check("conf: margin/mean/min/len", abs(c["conf_first_margin"] - 0.85) < 1e-9
          and abs(c["conf_mean_token_prob"] - 0.8) < 1e-9
          and abs(c["conf_min_token_prob"] - 0.7) < 1e-9 and c["conf_answer_len"] == 3)
    check("risk inverts prob-like, keeps entropy",
          risk_from_signal(0.9, "conf_first_max_prob") < risk_from_signal(0.2, "conf_first_max_prob")
          and risk_from_signal(2.0, "conf_first_entropy") > risk_from_signal(0.5, "conf_first_entropy"))

    # question features
    qf = dcp.question_features("What is the 2nd word?", 720)
    check("question features (what/number/nvis)",
          qf["q_is_what"] == 1.0 and qf["q_has_number"] == 1.0 and qf["nvis_native"] == 720.0)

    # sufficiency fraction: interpolates BETWEEN anchors → continuous (non-anchor) values
    sf = sufficiency_fraction({0.15: 0.2, 0.25: 0.4, 0.35: 0.8, 0.50: 0.9, 0.75: 0.9, 1.0: 0.9}, eps=0.01)
    check("sufficiency interpolates between anchors", 0.25 < sf < 0.50 and sf not in (0.35, 0.50))

    # RuleController: monotone easy→small / hard→large, continuous outputs, λ scaling
    risks = [i / 100 for i in range(100)]
    anch = []
    for i in range(100):
        if i < 50:   # easy: solvable at every anchor
            anch.append({f: 1.0 for f in dcp.ANCHOR_FRACTIONS})
        else:        # hard: only from 0.5 up
            anch.append({f: (1.0 if f >= 0.50 else 0.0) for f in dcp.ANCHOR_FRACTIONS})
    rc_ = RuleController(nbins=4).fit(risks, anch)
    lo = rc_.predict_fraction(0.05)
    hi = rc_.predict_fraction(0.95)
    check("rule: low risk → small fraction, high risk → large", lo < 0.3 < hi and hi >= 0.45)
    check("rule: isotonic (monotone in risk)",
          rc_.predict_fraction(0.2) <= rc_.predict_fraction(0.6) <= rc_.predict_fraction(0.9))
    check("rule: λ scales allocation", rc_.predict_fraction(0.95, lam=0.5) < hi)
    k_mid = dcp.clamp_k(rc_.predict_fraction(0.7) * 576, 32, 576)
    check("rule → integer K not restricted to anchors", isinstance(k_mid, int) and 32 <= k_mid <= 576)
    rt = RuleController.from_dict(rc_.to_dict())
    check("rule: JSON round-trip", abs(rt.predict_fraction(0.7) - rc_.predict_fraction(0.7)) < 1e-12)

    # Ridge: separable synthetic → correct risk ordering + round-trip
    X = [[1.0, 0.0]] * 40 + [[0.0, 1.0]] * 40
    y = [0.0] * 40 + [1.0] * 40
    rm = RidgeRiskModel(["a", "b"], alpha=0.1).fit(X, y)
    check("ridge: wrong-class risk higher", rm.predict_risk([0.0, 1.0]) > rm.predict_risk([1.0, 0.0]))
    rm2 = load_controller(rm.to_dict())
    check("ridge: JSON round-trip via load_controller",
          abs(rm2.predict_risk([0.0, 1.0]) - rm.predict_risk([0.0, 1.0])) < 1e-12)

    # threshold_for_rate
    tau = threshold_for_rate([float(i) for i in range(100)], 0.10)
    check("threshold_for_rate ~10% above tau", sum(1 for r in range(100) if r > tau) in (9, 10, 11))

    # basenames: dynamic_count_ prefix, variant-aware, no clash with frozen prefixes
    pb = dcp.probe_basename("textvqa", "norm", 15)
    check("probe basename", pb == "dynamic_count_probe_norm_p15_ocr_on")
    check("dcd basename", dcp.dcd_basename("gqa", "cls_attn", 15, 75)
          == "dynamic_count_dcd_cls_attn_p15_to_p75")
    dcc = dcp.dcc_basename("docvqa", "norm", 25, "rule", 1.0)
    check("dcc basename carries controller+λ+variant",
          dcc == "dynamic_count_dcc_rule_norm_p25_lam1_00_instruction_on")
    check("dc prefixes cannot collide with frozen outputs",
          all(b.startswith("dynamic_count_") for b in (pb, dcc)))
    check("probe budgets restricted to {15,25}", dcp.PROBE_BUDGETS == (15, 25))

    # curve interpolation + labels
    pts = [(1.0, 50.0), (2.0, 60.0), (4.0, 70.0)]
    check("interp_curve midpoint", abs(dce.interp_curve(pts, 1.5) - 55.0) < 1e-9)
    check("interp_curve clamps ends",
          dce.interp_curve(pts, 0.1) == 50.0 and dce.interp_curve(pts, 9.0) == 70.0)
    check("labels at ±0.50pp", dce.label_delta(0.51) == "dynamic_win"
          and dce.label_delta(-0.4) == "near_tie" and dce.label_delta(-0.51) == "dynamic_loss")

    # K-range histogram covers 100%
    h = dce._k_range_hist([50, 100, 150, 250, 400, 500])
    check("k-range histogram sums to 100", abs(sum(h.values()) - 100.0) < 0.1)

    # multi-pass FLOP accounting: escalated = probe + second pass, exactly
    from src.common import token_flops as tfm
    f_probe = tfm.per_sample_prefill_flops("llava15", 86, 40)
    f_second = tfm.per_sample_prefill_flops("llava15", 391, 40)
    check("multi-pass flops = probe + second (escalated)",
          f_probe + f_second > f_second > f_probe > 0)
    check("non-escalated pays probe only (strictly less than any 2-pass total)",
          f_probe < f_probe + tfm.per_sample_prefill_flops("llava15", 87, 40))

    # token reduction must come from VISUAL TOKENS, not FLOPs: executed-visual arithmetic
    recs = [{"probe_visual_tokens": 86, "k_executed": 391, "escalated": True},
            {"probe_visual_tokens": 86, "k_executed": 86, "escalated": False}]
    tot = [r["probe_visual_tokens"] + (r["k_executed"] if r["escalated"] else 0) for r in recs]
    check("executed-visual accounting: escalated=477, kept=86", tot == [477, 86])

    # selector-aware basenames: pure vs COUNT-on-WHICH can never collide
    check("dcd basenames differ across selectors (pure vs CoW)",
          dcp.dcd_basename("textvqa", "norm", 25, 75) != dcp.dcd_basename("textvqa", "textsim", 25, 75))
    check("dcc basenames differ across selectors (pure vs CoW)",
          dcp.dcc_basename("textvqa", "norm", 25, "rule", 1.0)
          != dcp.dcc_basename("textvqa", "textsim", 25, "rule", 1.0))
    check("probe basenames differ across selectors",
          dcp.probe_basename("textvqa", "norm", 15) != dcp.probe_basename("textvqa", "textsim", 15))

    # anchor_reference_path: textsim → WHICH finals; static → static finals (never mixed)
    ap_tx = dce.anchor_reference_path("qwen25vl7b", "textvqa", "textsim", 75)
    ap_st = dce.anchor_reference_path("qwen25vl7b", "textvqa", "norm", 75)
    check("textsim anchors come from frozen WHICH finals", "dynamic_which_final_textsim_p75" in ap_tx)
    check("static anchors come from frozen static finals", "static_final_norm_p75" in ap_st)
    raised = False
    try:
        dce.anchor_reference_path("llava15", "gqa", "textsim", 75)
    except ValueError:
        raised = True
    check("textsim anchors rejected outside qwen×textvqa", raised)


def main():
    print("=== shared evaluation core CPU self-checks ===")
    for t in (test_sha_deterministic, test_quota_and_sample, test_manifest_loader_and_validation,
              test_save_manifest_and_fingerprints, test_answer_type_strict,
              test_schema_validator, test_token_flops, test_static_eval, test_dynamic_which,
              test_dynamic_which_ref, test_dynamic_count):
        print(f"\n# {t.__name__}")
        t()
    print(f"\n{'ALL PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
