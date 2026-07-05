"""
check_dynamic_which_equivalence_extended.py — stronger equivalence checks for the CURRENT
Dynamic-WHICH pipeline, across datasets. Isolates whether the observed failures are a PIPELINE
bug or the SELECTION (textsim) itself. Writes NO official result files.

Two checks (both route through the EXISTING wrappers src/pruning/dynamic_which/*):

  A) keep-all dense equivalence
     Dynamic-WHICH with K = all visual tokens must reproduce dense_final predictions.
       LLaVA: LlavaDynamicWhich(textsim, keep_k=576)         (top-576 of 576 == identity)
       Qwen : QwenDynamicWhich(textsim, K=dense_n_visual)    (top-nvis == identity)

  B) static-selector equivalence
     A "reference mode" that selects the SAME tokens as static must reproduce static_final at the
     same budget. The static selection criterion is obtained INSIDE the WHICH pipeline via the mix
     selector at alpha=0 (score reduces to minmax(static-saliency) → top-K == static top-K):
       LLaVA: LlavaDynamicWhich(textsim_cls_mix, alpha=0)    → CLS-attention selection
       Qwen : QwenDynamicWhich(textsim_norm_mix, alpha=0)    → visual-norm selection
     Tested at p25 and p35 (configurable).

  If A and B both PASS, the WHICH selection→removal→generation→scoring pipeline is byte-faithful to
  dense/static, so any budget-level failure is the textsim SCORE (algorithmic), not an impl bug.

Supports: --model {llava15,qwen25vl7b}  --dataset {gqa,textvqa,docvqa,vqav2}  --n (default 50)
          --budgets (default "25,35")   --mode {all,keepall,static}
Prints per-check exact-match count, score comparison, token-mismatch count, up to 10 diffs, and a
clear PASS/FAIL verdict. Exit 0 iff every requested check passes.

Envs / GPU:
  LLaVA : CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python \
              -m scripts.final_scope.check_dynamic_which_equivalence_extended --model llava15 --dataset textvqa
  Qwen  : CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python \
              -m scripts.final_scope.check_dynamic_which_equivalence_extended --model qwen25vl7b --dataset gqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import static_eval as se
from src.final_scope.sample_ids import load_manifest


def _load_jsonl(path: str) -> dict:
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lookup[str(r["sample_id"])] = r
    return lookup


def _add_instruction(dataset: str) -> bool:
    if dataset in ("gqa", "vqav2"):
        return True
    if dataset == "textvqa":
        return False
    return True   # docvqa instruction-on default


def _dense_ref(model: str, dataset: str) -> tuple[str, dict]:
    base = dp.default_basename(dataset, 0, use_ocr=True, instruction=True, full=True)
    path = os.path.join(dp.OUT_ROOT, model, dataset, f"{base}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dense reference missing: {path}")
    return path, _load_jsonl(path)


def _static_ref(model: str, dataset: str, budget: int) -> tuple[str, dict]:
    sel = se.default_selector(model)
    base = se.static_basename(dataset, 0, sel, budget, use_ocr=True, instruction=True, full=True)
    path = os.path.join(dp.OUT_ROOT, model, dataset, f"{base}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"static reference missing: {path}")
    return path, _load_jsonl(path)


def _report(title: str, ids, gen, ref_lookup, model: str, adapter, *, expect_nvis=None,
            show_diffs: int = 10) -> bool:
    """gen(sample) -> (pred, nkeep). Compare pred to ref_lookup[sid]['pred_raw']; verify tokens."""
    exact = tok_mismatch = 0
    dyn_scores, ref_scores, diffs = [], [], []
    for sid in ids:
        s = adapter.sample(sid)
        rec = ref_lookup[str(sid)]
        pred, nkeep = gen(s)
        _, dyn_score = adapter.score(pred, s)
        dyn_scores.append(float(dyn_score))
        ref_scores.append(float(rec.get("per_sample_score", 0.0)))
        ref_pred = str(rec["pred_raw"]).strip()
        if pred == ref_pred:
            exact += 1
        else:
            diffs.append((sid, s.question, ref_pred, pred))
        want = expect_nvis if expect_nvis is not None else int(rec["n_visual_tokens"])
        if int(nkeep) != int(want):
            tok_mismatch += 1
    n = len(ids)
    print(f"\n=== {title} ===")
    print(f"  exact pred_raw matches : {exact}/{n}")
    print(f"  reference score        : {round(sum(ref_scores)/n*100,2)}%")
    print(f"  equiv-mode score       : {round(sum(dyn_scores)/n*100,2)}%")
    print(f"  n_visual mismatches    : {tok_mismatch}/{n}")
    for sid, q, rp, yp in diffs[:show_diffs]:
        print(f"   - id={sid} q='{str(q)[:44]}' ref='{rp}' got='{yp}'")
    ok = (exact == n) and (tok_mismatch == 0)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Extended Dynamic-WHICH equivalence checks (no result writes).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--budgets", default="25,35", help="comma budgets for the static check (default 25,35)")
    ap.add_argument("--mode", default="all", choices=["all", "keepall", "static"])
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    add_instr = _add_instruction(args.dataset)
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{args.dataset}.json"))
    if args.n > len(manifest.ids):
        ap.error(f"--n {args.n} exceeds {args.dataset} manifest length {len(manifest.ids)}")
    ids = manifest.ids[:args.n]
    adapter = dp.get_adapter(args.dataset, use_ocr=True, instruction=True)

    dense_path, dense = _dense_ref(args.model, args.dataset)
    for sid in ids:
        if str(sid) not in dense:
            ap.error(f"id {sid} missing from {dense_path}")

    results = []

    # ── A) keep-all dense equivalence ──
    if args.mode in ("all", "keepall"):
        import torch  # noqa: F401
        if args.model == "llava15":
            from src.pruning.dynamic_which.llava_textsim import LlavaDynamicWhich
            m = LlavaDynamicWhich(selector="textsim", keep_k=576, alpha=0.5,
                                  add_instruction=add_instr, max_new_tokens=args.max_new_tokens)

            def gen_keepall(s):
                return m.generate_one(s.image, s.model_input_text, s.question), 576
            results.append(_report(f"A) keep-all vs dense [{args.model}/{args.dataset}]",
                                   ids, gen_keepall, dense, args.model, adapter, expect_nvis=576))
        else:
            from src.pruning.dynamic_which.qwen_textsim import QwenDynamicWhich
            m = QwenDynamicWhich(selector="textsim", alpha=0.5, max_new_tokens=args.max_new_tokens)

            def gen_keepall(s):
                k = int(dense[str(s.sample_id)]["n_visual_tokens"])
                return m.generate_one(s.image, s.model_input_text, s.question, k, instr=add_instr)
            results.append(_report(f"A) keep-all vs dense [{args.model}/{args.dataset}]",
                                   ids, gen_keepall, dense, args.model, adapter, expect_nvis=None))

    # ── B) static-selector equivalence (alpha=0 mix reduces to the static criterion) ──
    if args.mode in ("all", "static"):
        import torch  # noqa: F401
        if args.model == "llava15":
            from src.pruning.dynamic_which.llava_textsim import LlavaDynamicWhich
            mB = LlavaDynamicWhich(selector="textsim_cls_mix", keep_k=576, alpha=0.0,
                                   add_instruction=add_instr, max_new_tokens=args.max_new_tokens)
            for b in budgets:
                K = se.budget_keep_k("llava15", b)
                sref_path, sref = _static_ref(args.model, args.dataset, b)
                mB.keep_k = K

                def gen_static(s, _K=K):
                    return mB.generate_one(s.image, s.model_input_text, s.question), _K
                results.append(_report(
                    f"B) static-equiv (CLS-attn) p{b} vs {os.path.basename(sref_path)}",
                    ids, gen_static, sref, args.model, adapter, expect_nvis=K))
        else:
            from src.pruning.dynamic_which.qwen_textsim import QwenDynamicWhich
            mB = QwenDynamicWhich(selector="textsim_norm_mix", alpha=0.0,
                                  max_new_tokens=args.max_new_tokens)
            for b in budgets:
                sref_path, sref = _static_ref(args.model, args.dataset, b)

                def gen_static(s, _b=b):
                    dn = int(dense[str(s.sample_id)]["n_visual_tokens"])
                    k = se.clamp_qwen_k(_b, dn)
                    return mB.generate_one(s.image, s.model_input_text, s.question, k, instr=add_instr)
                results.append(_report(
                    f"B) static-equiv (norm) p{b} vs {os.path.basename(sref_path)}",
                    ids, gen_static, sref, args.model, adapter, expect_nvis=None))

    ok = all(results)
    print("\n" + "=" * 66)
    print("  OVERALL:", "ALL EQUIVALENCE CHECKS PASS ✓ — failures are algorithmic (selection)"
          if ok else "SOME CHECK FAILED ✗ — investigate the pipeline before trusting results")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
