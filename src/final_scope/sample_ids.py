"""
sample_ids.py — sample-ID manifest helpers + loader/validator (Phase-2 dense infra).

A *manifest* is the version-controlled, sha256-verified list of sample IDs that defines the
exact evaluation subset for a dataset. The SAME manifest is reused by dense, static,
dynamic-WHICH, and dynamic-COUNT so every method is provably scored on identical samples
(fairness). Manifests live under `configs/final_scope/sample_ids/{dataset}.json` (tracked).

This module is **pure / CPU-only**: it has the deterministic sampling math, the canonical
sha256, the manifest schema, and the loader. The actual building (which reads the dataset
files) lives in `scripts/final_scope/build_sample_manifests.py`, which imports from here.

Canonical sha256: `sha256( "\\n".join(str(id) for id in ordered_ids) )` — deterministic,
independent of JSON formatting, and stable across runs.

See docs/DENSE_PROTOCOL.md §1/§6/§7 and docs/DENSE_DECISION_LOCK.md #2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Sequence, Tuple


# Required top-level fields for every manifest (docs/DENSE_PROTOCOL.md §1).
REQUIRED_FIELDS: Tuple[str, ...] = (
    "dataset",
    "split",
    "n",
    "seed",
    "selection_method",
    "source_files",
    "source_fingerprints",
    "id_field",
    "ordered_ids",
    "sha256",
    "created_at",
    "notes",
)

# Fields additionally required for a stratified manifest (e.g. VQAv2).
STRATIFIED_EXTRA_FIELDS: Tuple[str, ...] = (
    "stratification_method",
    "strata",
    "stratum_counts",
)


# ── canonical hashing ─────────────────────────────────────────────────────────

def compute_ids_sha256(ordered_ids: Sequence[Any]) -> str:
    """Deterministic sha256 over the ORDERED id list (joined by newline as strings)."""
    payload = "\n".join(str(x) for x in ordered_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str, _chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes (chunked — safe for the large VQAv2 annotation file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprints_for(paths: Sequence[str]) -> Dict[str, str]:
    """Map each existing local file path -> its sha256 (skips non-existent / non-file sources)."""
    out: Dict[str, str] = {}
    for p in paths:
        if os.path.isfile(p):
            out[p] = file_sha256(p)
    return out


def canonical_sorted(ids: Sequence[Any]) -> List[Any]:
    """Canonical, numeric-aware sort (numeric strings sort numerically, else lexicographic)."""
    return sorted(ids, key=_sort_key)


def assert_strata_subset(present: Sequence[str], allowed: Sequence[str]) -> None:
    """Hard-fail if any present stratum value is outside the allowed set."""
    unexpected = set(present) - set(allowed)
    if unexpected:
        raise ValueError(
            f"unexpected stratum values {sorted(unexpected)} not in allowed {sorted(allowed)}"
        )


def assert_strata_exact(present: Sequence[str], required: Sequence[str]) -> None:
    """Hard-fail unless the present stratum set is EXACTLY `required` (no missing, no extra)."""
    p, r = set(present), set(required)
    if p != r:
        raise ValueError(
            f"strata must be exactly {sorted(r)}; missing={sorted(r - p)} unexpected={sorted(p - r)}"
        )


# ── proportional stratified sampling (largest-remainder, deterministic) ───────

def proportional_quota(stratum_sizes: Dict[str, int], n_total: int) -> Dict[str, int]:
    """
    Largest-remainder apportionment of `n_total` across strata proportional to their sizes.
    Deterministic: ties in the fractional remainder are broken by stratum name (sorted).
    Guarantees `sum(quota.values()) == n_total` and `quota[k] <= stratum_sizes[k]`.
    """
    total = sum(stratum_sizes.values())
    if total == 0:
        raise ValueError("empty strata")
    if n_total > total:
        raise ValueError(f"n_total={n_total} exceeds available {total}")

    raw = {k: n_total * sz / total for k, sz in stratum_sizes.items()}
    quota = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = n_total - sum(quota.values())

    # distribute the remainder by largest fractional part, tie-break by name (deterministic)
    order = sorted(stratum_sizes.keys(), key=lambda k: (-(raw[k] - quota[k]), k))
    i = 0
    while remainder > 0:
        k = order[i % len(order)]
        if quota[k] < stratum_sizes[k]:
            quota[k] += 1
            remainder -= 1
        i += 1
        if i > 10 * len(order):  # safety; should never trigger when n_total<=total
            raise RuntimeError("could not apportion remainder within strata capacity")
    return quota


def proportional_stratified_sample(
    strata: Dict[str, List[Any]],
    n_total: int,
    seed: int,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Proportional stratified sample of exactly `n_total` ids from `strata`
    (stratum_name -> list of ids), deterministic given `seed`.

    Procedure (reproducible):
      1. quota per stratum via largest-remainder apportionment.
      2. within each stratum (iterated in sorted stratum order, ids pre-sorted),
         draw `quota[k]` ids with a single seeded RNG.
      3. return the selected ids **sorted into a canonical order** (so the manifest order
         and its sha are independent of draw order) + the realized per-stratum counts.
    """
    quota = proportional_quota({k: len(v) for k, v in strata.items()}, n_total)
    rng = random.Random(seed)
    selected: List[Any] = []
    counts: Dict[str, int] = {}
    for k in sorted(strata.keys()):
        ids_sorted = sorted(strata[k], key=_sort_key)
        picked = rng.sample(ids_sorted, quota[k])
        selected.extend(picked)
        counts[k] = len(picked)
    selected_sorted = sorted(selected, key=_sort_key)
    assert len(selected_sorted) == n_total, (len(selected_sorted), n_total)
    assert len(set(map(str, selected_sorted))) == n_total, "duplicate ids in sample"
    return selected_sorted, counts


def _sort_key(x: Any):
    """Canonical sort: numeric when possible, else string."""
    if isinstance(x, bool):
        return (1, str(x))
    if isinstance(x, (int, float)):
        return (0, x)
    s = str(x)
    return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)


# ── manifest assembly + IO ─────────────────────────────────────────────────────

def assemble_manifest(
    *,
    dataset: str,
    split: str,
    ordered_ids: Sequence[Any],
    seed: int,
    selection_method: str,
    source_files: Sequence[str],
    id_field: str,
    notes: str = "",
    source_fingerprints: Dict[str, str] | None = None,
    stratification_method: str | None = None,
    strata: Sequence[str] | None = None,
    stratum_counts: Dict[str, int] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a manifest dict with the canonical schema + computed sha256."""
    from datetime import datetime, timezone

    ordered = list(ordered_ids)
    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "n": len(ordered),
        "seed": seed,
        "selection_method": selection_method,
        "source_files": list(source_files),
        # sha256 of each source FILE's bytes — pins content/order (esp. for row_index ids,
        # where the id list [0,1,2,...] alone does not prove the same file was used).
        "source_fingerprints": dict(source_fingerprints or {}),
        "id_field": id_field,
        "ordered_ids": ordered,
        "sha256": compute_ids_sha256(ordered),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    if stratification_method is not None:
        manifest["stratification_method"] = stratification_method
        manifest["strata"] = list(strata or [])
        manifest["stratum_counts"] = dict(stratum_counts or {})
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    # Validate before writing so we never persist a malformed manifest. validate_manifest
    # raises on hard schema breakage AND returns soft issues — we refuse to write on EITHER.
    issues = validate_manifest(manifest)
    if issues:
        raise ValueError(f"refusing to write invalid manifest to {path}: {issues}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ── validation + loader ─────────────────────────────────────────────────────────

def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of problems (empty == valid). Raises ValueError on hard schema breakage."""
    issues: List[str] = []
    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")

    ordered = manifest["ordered_ids"]
    if len(ordered) != manifest["n"]:
        issues.append(f"n ({manifest['n']}) != len(ordered_ids) ({len(ordered)})")
    if len(set(map(str, ordered))) != len(ordered):
        issues.append("ordered_ids contains duplicates")
    if compute_ids_sha256(ordered) != manifest["sha256"]:
        issues.append("sha256 does not match ordered_ids")

    # every manifest must record non-empty source_fingerprints (sha256 of each source file, or
    # the HF id-column hash) — this pins content/order, and is the ONLY content proof for
    # row_index manifests (where the id list [0,1,2,...] alone proves nothing).
    if not manifest.get("source_fingerprints"):
        issues.append("source_fingerprints must be non-empty")

    if "stratification_method" in manifest:
        for f in STRATIFIED_EXTRA_FIELDS:
            if f not in manifest:
                issues.append(f"stratified manifest missing '{f}'")
        sc = manifest.get("stratum_counts", {})
        if sc and sum(sc.values()) != manifest["n"]:
            issues.append(f"stratum_counts sum ({sum(sc.values())}) != n ({manifest['n']})")
    return issues


class SampleManifest:
    """Loaded manifest with sha-verified ordered IDs."""

    def __init__(self, manifest: Dict[str, Any], path: str | None = None):
        self.raw = manifest
        self.path = path
        problems = validate_manifest(manifest)
        if problems:
            raise ValueError(f"invalid manifest {path or ''}: {problems}")

    @property
    def ids(self) -> List[Any]:
        return list(self.raw["ordered_ids"])

    @property
    def n(self) -> int:
        return int(self.raw["n"])

    @property
    def sha256(self) -> str:
        return str(self.raw["sha256"])

    @property
    def dataset(self) -> str:
        return str(self.raw["dataset"])

    def verify_sha(self) -> bool:
        return compute_ids_sha256(self.raw["ordered_ids"]) == self.raw["sha256"]


def load_manifest(path: str) -> SampleManifest:
    with open(path, "r", encoding="utf-8") as f:
        return SampleManifest(json.load(f), path=path)
