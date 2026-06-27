"""
build_sample_manifests.py — build the version-controlled sample-ID manifests for the
final-scope dense (and later static/dynamic) experiment matrix.

Writes `configs/final_scope/sample_ids/{gqa,textvqa,docvqa,vqav2}.json`. CPU-only: it only
reads dataset files / the HF cache to enumerate sample IDs — it does NOT load any model.

Per docs/DENSE_PROTOCOL.md + docs/DENSE_DECISION_LOCK.md:
  - VQAv2 : n=25000, seed=42, split=validation, PROPORTIONAL STRATIFIED SAMPLING by the
            OFFICIAL VQAv2 annotation `answer_type` (strata: yes/no, number, other).
            (NOT the repo's old `_question_type_id` heuristic.)
  - GQA   : full testdev_balanced (deterministic), id = questionId (dict key).
  - TextVQA: full val (5000), id = ROW INDEX (the jsonl `question_id` is the image id and is
             NOT unique), file order.
  - DocVQA: full validation from the HF dataset (lmms-lab/DocVQA), id = questionId. If the
            dataset is not locally available this is reported BLOCKED (never faked).

Usage (CPU):
  python -m scripts.final_scope.build_sample_manifests --datasets vqav2,gqa,textvqa
  python -m scripts.final_scope.build_sample_manifests --datasets docvqa     # needs HF cache
  python -m scripts.final_scope.build_sample_manifests --datasets all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope.sample_ids import (
    assemble_manifest,
    assert_strata_exact,
    canonical_sorted,
    compute_ids_sha256,
    proportional_stratified_sample,
    save_manifest,
    source_fingerprints_for,
)

OUT_DIR = "configs/final_scope/sample_ids"

VQAV2_QUESTIONS = "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
VQAV2_ANNOTATIONS = "data/vqav2/v2_mscoco_val2014_annotations.json"
GQA_TESTDEV = "data/gqa/testdev_balanced_questions.json"
TEXTVQA_JSONL = "data/textvqa/llava_textvqa_val_v051_ocr.jsonl"

VQAV2_N = 25000
VQAV2_SEED = 42
VQAV2_STRATA = ("yes/no", "number", "other")


# ── VQAv2 (answer_type stratified) ────────────────────────────────────────────

def build_vqav2() -> Dict[str, Any]:
    for p in (VQAV2_QUESTIONS, VQAV2_ANNOTATIONS):
        if not os.path.exists(p):
            raise FileNotFoundError(f"VQAv2 source missing: {p}")

    print(f"[vqav2] loading annotations {VQAV2_ANNOTATIONS} ...", flush=True)
    with open(VQAV2_ANNOTATIONS) as f:
        ann = json.load(f)["annotations"]
    # join: question_id -> official answer_type
    qid_to_atype: Dict[int, str] = {}
    for a in ann:
        qid_to_atype[int(a["question_id"])] = a["answer_type"]

    # (optional) heuristic question_type breakdown, ANALYSIS ONLY (not the sampling rule)
    qtype_of: Dict[int, str] = {int(a["question_id"]): a.get("question_type", "?") for a in ann}

    # bucket val question_ids by answer_type
    strata: Dict[str, List[int]] = defaultdict(list)
    n_missing = 0
    with open(VQAV2_QUESTIONS) as f:
        qs = json.load(f)["questions"]
    for q in qs:
        qid = int(q["question_id"])
        at = qid_to_atype.get(qid)
        if at is None:
            n_missing += 1
            continue
        strata[at].append(qid)
    # HARD-FAIL: every val question must have an official annotation/answer_type.
    if n_missing:
        raise ValueError(
            f"VQAv2: {n_missing} val questions have no annotation/answer_type — refusing to build "
            f"a manifest from an incomplete join (check {VQAV2_ANNOTATIONS})"
        )

    sizes = {k: len(v) for k, v in strata.items()}
    print(f"[vqav2] answer_type pool sizes: {sizes} (total {sum(sizes.values())})", flush=True)
    # STRICT: the official answer_type set must be EXACTLY {yes/no, number, other}
    # (no unexpected value AND no missing stratum) — hard-fail otherwise.
    assert_strata_exact(strata.keys(), VQAV2_STRATA)

    ordered_ids, counts = proportional_stratified_sample(dict(strata), VQAV2_N, VQAV2_SEED)
    # analysis-only question_type breakdown of the selected set
    qtype_breakdown = dict(Counter(qtype_of.get(i, "?") for i in ordered_ids))

    print(f"[vqav2] selected per answer_type: {counts} (n={len(ordered_ids)})", flush=True)
    return assemble_manifest(
        dataset="vqav2",
        split="validation",
        ordered_ids=ordered_ids,
        seed=VQAV2_SEED,
        selection_method="proportional stratified sampling (largest-remainder to exactly 25000), seed 42",
        source_files=[VQAV2_QUESTIONS, VQAV2_ANNOTATIONS],
        source_fingerprints=source_fingerprints_for([VQAV2_QUESTIONS, VQAV2_ANNOTATIONS]),
        id_field="question_id",
        notes="Official VQAv2 answer_type stratification (NOT the repo _question_type_id heuristic). "
              "question_type_breakdown is analysis-only.",
        stratification_method="official VQAv2 annotation answer_type",
        strata=list(VQAV2_STRATA),
        stratum_counts=counts,
        extra={"answer_type_pool_sizes": sizes,
               "question_type_breakdown_analysis_only": qtype_breakdown},
    )


# ── GQA (full testdev) ────────────────────────────────────────────────────────

def build_gqa() -> Dict[str, Any]:
    if not os.path.exists(GQA_TESTDEV):
        raise FileNotFoundError(f"GQA source missing: {GQA_TESTDEV}")
    print(f"[gqa] loading {GQA_TESTDEV} ...", flush=True)
    with open(GQA_TESTDEV) as f:
        raw = json.load(f)
    ordered = canonical_sorted(list(raw.keys()))  # numeric-aware sort of questionId strings
    print(f"[gqa] full testdev_balanced: {len(ordered)} questionIds", flush=True)
    return assemble_manifest(
        dataset="gqa",
        split="testdev_balanced",
        ordered_ids=ordered,
        seed=42,
        selection_method="full split (all questions), numeric-aware sort by questionId",
        source_files=[GQA_TESTDEV],
        source_fingerprints=source_fingerprints_for([GQA_TESTDEV]),
        id_field="questionId",
        notes="Full GQA testdev_balanced; no subsampling, so seed is nominal.",
    )


# ── TextVQA (full val; row-index ids) ─────────────────────────────────────────

def build_textvqa() -> Dict[str, Any]:
    if not os.path.exists(TEXTVQA_JSONL):
        raise FileNotFoundError(f"TextVQA source missing: {TEXTVQA_JSONL}")
    print(f"[textvqa] loading {TEXTVQA_JSONL} ...", flush=True)
    with open(TEXTVQA_JSONL) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    n = len(rows)
    n_unique_qid = len({r.get("question_id") for r in rows})
    ordered = list(range(n))  # row index = canonical unique id (question_id is image_id, not unique)
    print(f"[textvqa] full val: {n} rows ({n_unique_qid} unique question_id/image_id → using row index)", flush=True)
    return assemble_manifest(
        dataset="textvqa",
        split="val",
        ordered_ids=ordered,
        seed=42,
        selection_method="full split (all rows), file order",
        source_files=[TEXTVQA_JSONL],
        source_fingerprints=source_fingerprints_for([TEXTVQA_JSONL]),  # REQUIRED for row_index ids
        id_field="row_index",
        notes=f"row_index is the canonical sample id; the jsonl question_id is the image id and is "
              f"NOT unique ({n_unique_qid} unique / {n} rows). Eval must iterate the jsonl in file order.",
    )


# ── DocVQA (full validation from HF; BLOCKED if unavailable) ──────────────────

def build_docvqa() -> Tuple[Dict[str, Any] | None, str | None]:
    """Returns (manifest, None) or (None, blocked_reason)."""
    try:
        from datasets import load_dataset
    except Exception as e:
        return None, f"`datasets` not importable ({type(e).__name__}: {e})"
    try:
        print("[docvqa] loading lmms-lab/DocVQA validation (HF cache) ...", flush=True)
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        cols = ds.column_names
        id_field = "questionId" if "questionId" in cols else ("docId" if "docId" in cols else None)
        if id_field is None:
            return None, f"no questionId/docId column in DocVQA validation (cols={cols})"
        ids = list(ds[id_field])  # Arrow column access — does NOT decode images
        hf_source = "hf:lmms-lab/DocVQA:DocVQA:validation"
        # content fingerprint over the raw id column (pins the dataset content/order without files)
        fp = {hf_source: compute_ids_sha256(ids)}
        if id_field == "questionId":
            ordered = canonical_sorted(ids)
            id_used = "questionId"
            notes = "Full DocVQA validation; ordered by questionId."
        else:
            # docId is not unique per question → fall back to row index, store docId alongside
            ordered = list(range(len(ids)))
            id_used = "row_index"
            notes = "DocVQA validation has no questionId; using row_index (docId stored per row in eval)."
        print(f"[docvqa] full validation: {len(ordered)} rows (id_field={id_used})", flush=True)
        return assemble_manifest(
            dataset="docvqa",
            split="validation",
            ordered_ids=ordered,
            seed=42,
            selection_method="full split (all validation rows)",
            source_files=[hf_source],
            source_fingerprints=fp,
            id_field=id_used,
            notes=notes + " Pilot uses the first 200 ids of this manifest.",
        ), None
    except Exception as e:
        return None, f"could not load DocVQA validation ({type(e).__name__}: {e})"


# ── CLI ───────────────────────────────────────────────────────────────────────

BUILDERS = {"vqav2": build_vqav2, "gqa": build_gqa, "textvqa": build_textvqa}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all",
                    help="comma list of {vqav2,gqa,textvqa,docvqa} or 'all'")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="build + print summary, do not write")
    args = ap.parse_args()

    want = (["vqav2", "gqa", "textvqa", "docvqa"] if args.datasets == "all"
            else [d.strip() for d in args.datasets.split(",") if d.strip()])

    built: List[str] = []
    blocked: List[Tuple[str, str]] = []
    summary: Dict[str, Dict[str, Any]] = {}

    for d in want:
        try:
            if d == "docvqa":
                manifest, reason = build_docvqa()
                if manifest is None:
                    blocked.append((d, reason or "unknown"))
                    print(f"[{d}] BLOCKED: {reason}", flush=True)
                    continue
            elif d in BUILDERS:
                manifest = BUILDERS[d]()
            else:
                blocked.append((d, "unknown dataset"))
                continue
        except FileNotFoundError as e:
            blocked.append((d, str(e)))
            print(f"[{d}] BLOCKED: {e}", flush=True)
            continue

        out_path = os.path.join(args.out_dir, f"{d}.json")
        summary[d] = {"n": manifest["n"], "sha256": manifest["sha256"],
                      "stratum_counts": manifest.get("stratum_counts"), "path": out_path}
        if args.dry_run:
            print(f"[{d}] DRY-RUN ok  n={manifest['n']}  sha256={manifest['sha256']}", flush=True)
        else:
            save_manifest(out_path, manifest)
            print(f"[{d}] wrote {out_path}  n={manifest['n']}  sha256={manifest['sha256']}", flush=True)
            built.append(out_path)

    print("\n=== SUMMARY ===")
    for d, s in summary.items():
        extra = f"  strata={s['stratum_counts']}" if s.get("stratum_counts") else ""
        print(f"  {d:8s} n={s['n']:>6}  sha256={s['sha256']}{extra}")
    if blocked:
        print("  BLOCKED:")
        for d, r in blocked:
            print(f"    {d}: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
