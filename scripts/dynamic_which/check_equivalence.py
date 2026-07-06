"""
check_dynamic_which_equivalence.py — keep-all (K=all) dense-equivalence sanity for Dynamic-WHICH.

PURPOSE: confirm the fresh WHICH pipeline is a faithful SUPERSET of dense — i.e. when it keeps
ALL visual tokens, it reproduces the dense_final predictions exactly. If so, any difference at a
real budget is attributable to the SELECTION, not to a pipeline bug.

  * LLaVA : LlavaDynamicWhich(selector="textsim", keep_k=576)  — top-576 of 576 == identity
  * Qwen  : QwenDynamicWhich(selector="textsim", K=dense_n_visual_tokens) — top-nvis == identity

It runs NO official outputs and writes NO result JSON/JSONL. It only generates predictions for the
first n GQA samples and compares pred_raw against the accepted dense_final.jsonl, printing:
  - exact prediction-match count (ideally n/n),
  - dense score vs dynamic keep-all score (over the same n),
  - (Qwen) nkeep-vs-dense_n_visual mismatch count,
  - up to 10 differing examples.
Exit 0 iff fully equivalent (n/n matches and, for Qwen, 0 nkeep mismatches). A non-zero exit means
DO NOT proceed to pilots until the discrepancy is understood.

Envs / GPU (run each model under its own env):
  LLaVA : CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python \
              -m scripts.dynamic_which.check_equivalence --model llava15 --n 50
  Qwen  : CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python \
              -m scripts.dynamic_which.check_equivalence --model qwen25vl7b --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dense import evaluate_dense as dp
from src.common.sample_ids import load_manifest

DATASET = "gqa"


def load_dense_final(model: str):
    """Load results/runs/{model}/gqa/dense_final.jsonl, keyed by str(sample_id)."""
    path = os.path.join(dp.OUT_ROOT, model, DATASET, "dense_final.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dense reference not found: {path} (run the GQA dense final first)")
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lookup[str(r["sample_id"])] = r
    return path, lookup


def build_llava_keepall(max_new_tokens: int):
    """Returns gen(sample) -> (pred, nkeep). LLaVA keep-all == top-576 of 576 (identity)."""
    import torch  # noqa: F401
    from src.pruning.dynamic_which.llava_textsim import LlavaDynamicWhich

    model = LlavaDynamicWhich(selector="textsim", keep_k=576, add_instruction=True,
                              max_new_tokens=max_new_tokens)

    def gen(s: dp.Sample):
        # prompt = s.model_input_text (== question for GQA); selector_text = s.question
        return model.generate_one(s.image, s.model_input_text, s.question), 576

    return gen


def build_qwen_keepall(max_new_tokens: int, dense_lookup: dict):
    """Returns gen(sample) -> (pred, nkeep). Qwen keep-all uses K = dense_n_visual_tokens."""
    import torch  # noqa: F401
    from src.pruning.dynamic_which.qwen_textsim import QwenDynamicWhich

    model = QwenDynamicWhich(selector="textsim", max_new_tokens=max_new_tokens)

    def gen(s: dp.Sample):
        k = int(dense_lookup[str(s.sample_id)]["n_visual_tokens"])  # keep-all = dense nvis
        pred, nkeep = model.generate_one(s.image, s.model_input_text, s.question, k,
                                         instr=s.add_instruction)
        return pred, int(nkeep)

    return gen


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamic-WHICH keep-all dense-equivalence check (GQA).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--n", type=int, default=50, help="number of GQA samples (default 50)")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--show-diffs", type=int, default=10, help="how many differing examples to print")
    args = ap.parse_args()

    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{DATASET}.json"))
    if args.n > len(manifest.ids):
        ap.error(f"--n {args.n} exceeds GQA manifest length {len(manifest.ids)}")
    ids = manifest.ids[:args.n]
    dense_path, dense = load_dense_final(args.model)
    missing = [str(sid) for sid in ids if str(sid) not in dense]
    if missing:
        ap.error(f"{len(missing)} id(s) missing from {dense_path} (e.g. {missing[:5]})")

    adapter = dp.GQAAdapter()
    gen = (build_llava_keepall(args.max_new_tokens) if args.model == "llava15"
           else build_qwen_keepall(args.max_new_tokens, dense))

    print(f"[equiv] {args.model} × {DATASET}: keep-all vs {dense_path}  (n={len(ids)})", flush=True)
    exact = 0
    nkeep_mismatch = 0
    dyn_scores, dense_scores = [], []
    diffs = []
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        drec = dense[str(sid)]
        pred, nkeep = gen(s)
        _, dyn_score = adapter.score(pred, s)
        dyn_scores.append(float(dyn_score))
        dense_scores.append(float(drec.get("per_sample_score", 0.0)))
        dense_pred = str(drec["pred_raw"]).strip()
        if pred == dense_pred:
            exact += 1
        else:
            diffs.append((sid, s.question, dense_pred, pred))
        if args.model == "qwen25vl7b" and int(nkeep) != int(drec["n_visual_tokens"]):
            nkeep_mismatch += 1
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(ids)}  exact={exact}", flush=True)

    n = len(ids)
    dense_acc = round(sum(dense_scores) / n * 100, 2)
    dyn_acc = round(sum(dyn_scores) / n * 100, 2)
    print("\n" + "=" * 64)
    print(f"  exact pred_raw matches : {exact}/{n}")
    print(f"  dense score (same n)   : {dense_acc}%")
    print(f"  keep-all WHICH score   : {dyn_acc}%")
    if args.model == "qwen25vl7b":
        print(f"  nkeep != dense nvis    : {nkeep_mismatch}/{n}")
    if diffs:
        print(f"\n  differing examples (showing up to {args.show_diffs}):")
        for sid, q, dpred, ypred in diffs[:args.show_diffs]:
            print(f"   - id={sid}  q='{q[:48]}'  dense='{dpred}'  whichKeepAll='{ypred}'")
    ok = (exact == n) and (nkeep_mismatch == 0)
    print("\n  VERDICT:", "EQUIVALENT ✓ (safe to pilot)" if ok
          else "NOT EQUIVALENT ✗ — DO NOT proceed to pilots")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
