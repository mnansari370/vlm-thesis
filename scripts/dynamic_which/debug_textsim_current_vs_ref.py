"""
debug_textsim_current_vs_ref.py — no-result-write diagnostic comparing the CURRENT Dynamic-WHICH
textsim path vs the clean-room REFERENCE textsim_ref, for ONE model/dataset/budget.

Writes NO result JSON/JSONL. Two modes:

  --mode jsonl  (default, CPU, conclusive):
      Load the existing n=200 current textsim pilot and the n=200 ref textsim_ref pilot and compare
      per sample: same sample IDs (and order), same n_visual_tokens (K), same prompt_text, exact
      pred_raw match count, per-sample score differences, and up to 10 differing examples.
      NOTE: it hard-requires n==200 on BOTH sides (never an n20 smoke) — the exact confusion that
      produced the spurious "IMPL-SUSPECT" rows in the earlier compare.

  --mode live  (GPU, optional, self-contained):
      Re-derive the textsim score from the FROZEN base primitives (StaticPrunedLlava / QwenPruner)
      for the first n samples, then compare the TWO selection algorithms on IDENTICAL scores:
        current = torch `scores.topk(k).indices.sort().values`
        ref     = python `topk_sorted(scores, k)`
      Reports selected-token overlap and top-20 textsim index overlap, and confirms selector_text
      == sample.question for both. Selected indices are returned in memory only — nothing is written,
      and NO wrapper is modified.

Envs / GPU (live mode only):
  LLaVA : CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python ... --mode live
  Qwen  : CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python ... --mode live
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dense import evaluate_dense as dp
from src.static import evaluate_static as se
from src.dynamic_which import evaluate_dynamic_which as dw
from src.common.sample_ids import load_manifest
from src.pruning import dynamic_which_ref as ref


def _load_jsonl(path: str):
    d = {}
    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[str(r["sample_id"])] = r
                order.append(str(r["sample_id"]))
    return d, order


def _add_instruction(dataset: str) -> bool:
    return dataset in ("gqa", "vqav2", "docvqa")


def mode_jsonl(model, dataset, budget, n, use_ocr, instruction, show_diffs) -> int:
    cur_base = dw.dynamic_basename(dataset, 200, "textsim", budget,
                                   use_ocr=use_ocr, instruction=instruction, full=False)
    ref_base = ref.ref_basename(dataset, 200, "textsim_ref", budget,
                                use_ocr=use_ocr, instruction=instruction)
    cur_path = os.path.join(dp.OUT_ROOT, model, dataset, f"{cur_base}.jsonl")
    ref_path = os.path.join(dp.OUT_ROOT, model, dataset, f"{ref_base}.jsonl")
    for p in (cur_path, ref_path):
        if not os.path.exists(p):
            print(f"[error] missing n=200 output: {p}")
            return 2
    cur, cur_order = _load_jsonl(cur_path)
    rff, ref_order = _load_jsonl(ref_path)

    # HARD GUARD: both must be n=200 (never an n20 smoke)
    if len(cur_order) != 200 or len(ref_order) != 200:
        print(f"[error] expected n=200 on both sides; got current={len(cur_order)} ref={len(ref_order)}")
        return 2

    common = [s for s in cur_order if s in rff][:n] if n else [s for s in cur_order if s in rff]
    same_id_order = cur_order == ref_order
    pred_match = sum(1 for s in common if str(cur[s]["pred_raw"]) == str(rff[s]["pred_raw"]))
    prompt_match = sum(1 for s in common if cur[s]["prompt"] == rff[s]["prompt"])
    k_match = sum(1 for s in common if cur[s]["n_visual_tokens"] == rff[s]["n_visual_tokens"])
    score_diff = [s for s in common if float(cur[s]["per_sample_score"]) != float(rff[s]["per_sample_score"])]
    pred_diff = [s for s in common if str(cur[s]["pred_raw"]) != str(rff[s]["pred_raw"])]

    print(f"=== jsonl mode: {model}/{dataset} p{budget}  current textsim vs ref textsim_ref ===")
    print(f"  current : {os.path.basename(cur_path)}  (n={len(cur_order)})")
    print(f"  ref     : {os.path.basename(ref_path)}  (n={len(ref_order)})")
    print(f"  same sample-id ORDER      : {same_id_order}")
    print(f"  compared samples          : {len(common)}")
    print(f"  n_visual (K) match        : {k_match}/{len(common)}")
    print(f"  prompt_text exact match   : {prompt_match}/{len(common)}")
    print(f"  pred_raw exact match      : {pred_match}/{len(common)}")
    print(f"  per-sample score differ   : {len(score_diff)}/{len(common)}")
    print(f"  selector_text_source      : current={_sts(cur_path)}  ref={_sts(ref_path)}")
    for s in pred_diff[:show_diffs]:
        print(f"    DIFF id={s}  q='{str(cur[s]['question'])[:40]}'  "
              f"cur='{cur[s]['pred_raw']}'  ref='{rff[s]['pred_raw']}'")
    ok = same_id_order and pred_match == len(common) and k_match == len(common) \
        and prompt_match == len(common) and not score_diff
    print(f"\n  VERDICT: {'IDENTICAL ✓ (current textsim == ref textsim_ref)' if ok else 'DIVERGENT ✗'}")
    return 0 if ok else 1


def _sts(agg_jsonl_path: str):
    agg = agg_jsonl_path[:-1]  # .jsonl -> .json
    if os.path.exists(agg):
        r = json.load(open(agg))
        ex = r.get("extra_dynamic_which") or r.get("extra_dynamic_which_ref") or {}
        return ex.get("selector_text_source")
    return "?"


def mode_live(model, dataset, budget, n, use_ocr, instruction) -> int:
    """Self-contained: re-derive textsim from frozen primitives, compare torch-topk vs python topk_sorted."""
    import torch
    manifest = load_manifest(os.path.join(dp.MANIFEST_DIR, f"{dataset}.json"))
    ids = manifest.ids[:n]
    adapter = dp.get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)
    add_instr = _add_instruction(dataset)

    def textsim_scores_llava(base, image, prompt_text, selector_text):
        import torch.nn.functional as F
        img = base._pad_to_square(image)
        suffix = ""  # scoring path does not depend on suffix; K selection uses textsim only
        conv = [{"role": "user", "content": [{"type": "image"},
                 {"type": "text", "text": prompt_text.strip() + suffix}]}]
        prompt = base.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = base.processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
        inp = {k: (v.to(base.backbone.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        pv = inp["pixel_values"].to(base.backbone.dtype)
        raw = base._vt(pv, output_hidden_states=True).hidden_states[base._vis_layer][:, 1:, :]
        projected = base._proj(raw)[0].float()
        qi = base.processor.tokenizer(selector_text.strip(), return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(base.backbone.device)
        q = base._embed(qi)[0].float()
        sim = F.normalize(projected, dim=-1) @ F.normalize(q, dim=-1).t()
        return sim.max(dim=1).values

    def textsim_scores_qwen(m, image, prompt_text, selector_text):
        import torch.nn.functional as F
        emb, pos, attn, v0, nvis, vis = m._encode(image, prompt_text, add_instr)
        visf = vis.float()
        qi = m.proc.tokenizer(selector_text.strip(), return_tensors="pt",
                              add_special_tokens=False).input_ids.to(m.dev)
        q = m.embed(qi)[0].float()
        sim = F.normalize(visf, dim=-1) @ F.normalize(q, dim=-1).t()
        return sim.max(dim=1).values, nvis

    if model == "llava15":
        from src.models.static.static import StaticPrunedLlava
        K = se.budget_keep_k("llava15", budget)
        base = StaticPrunedLlava(method="cls_attn", keep_k=K, image_pad=True, honest=True)
        get_scores = lambda s: (textsim_scores_llava(base, s.image, s.model_input_text, s.question),
                                min(K, 576))
    else:
        from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
        m = QwenPruner()
        dense_path = os.path.join(dp.OUT_ROOT, model, dataset,
                                  dp.default_basename(dataset, 0, use_ocr=use_ocr,
                                                      instruction=instruction, full=True) + ".jsonl")
        dense, _ = _load_jsonl(dense_path)

        def get_scores(s):
            sc, nvis = textsim_scores_qwen(m, s.image, s.model_input_text, s.question)
            k = se.clamp_qwen_k(budget, int(dense[str(s.sample_id)]["n_visual_tokens"]))
            return sc, min(k, nvis)

    print(f"=== live mode: {model}/{dataset} p{budget}  torch-topk (current) vs python topk_sorted (ref) ===")
    tot_sel_ov, tot_top20_ov, nsamp = 0.0, 0.0, 0
    for sid in ids:
        s = adapter.sample(sid)
        scores, k = get_scores(s)
        scores_list = scores.tolist()
        cur_keep = set(scores.topk(k).indices.sort().values.tolist())      # current-wrapper selection
        ref_keep = set(ref.topk_sorted(scores_list, k))                    # ref-wrapper selection
        sel_ov = len(cur_keep & ref_keep) / max(len(cur_keep), 1)
        cur_top20 = set(scores.topk(min(20, len(scores_list))).indices.tolist())
        ref_top20 = set(ref.topk_sorted(scores_list, min(20, len(scores_list))))
        top20_ov = len(cur_top20 & ref_top20) / max(len(cur_top20), 1)
        tot_sel_ov += sel_ov
        tot_top20_ov += top20_ov
        nsamp += 1
        if sel_ov < 1.0 or top20_ov < 1.0:
            print(f"   id={sid} K={k} sel_overlap={sel_ov:.3f} top20_overlap={top20_ov:.3f} "
                  f"(selector_text='{str(s.question)[:32]}')")
    print(f"  samples                 : {nsamp}")
    print(f"  mean selected-token overlap : {tot_sel_ov/max(nsamp,1):.4f}")
    print(f"  mean top-20 overlap         : {tot_top20_ov/max(nsamp,1):.4f}")
    ok = abs(tot_sel_ov/max(nsamp, 1) - 1.0) < 1e-9
    print(f"\n  VERDICT: {'SELECTIONS IDENTICAL ✓' if ok else 'selection differs (tie-break?) — inspect above'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose current textsim vs clean-room textsim_ref (no writes).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--budget-pct", required=True, type=int, choices=list(dw.BUDGETS), dest="budget_pct")
    ap.add_argument("--n", type=int, default=20, help="samples to compare (live mode; jsonl uses all 200)")
    ap.add_argument("--mode", default="jsonl", choices=["jsonl", "live"])
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--show-diffs", type=int, default=10)
    args = ap.parse_args()
    use_ocr = not args.no_ocr
    instruction = not args.no_instruction
    if args.mode == "jsonl":
        return mode_jsonl(args.model, args.dataset, args.budget_pct, None, use_ocr, instruction,
                          args.show_diffs)
    return mode_live(args.model, args.dataset, args.budget_pct, args.n, use_ocr, instruction)


if __name__ == "__main__":
    raise SystemExit(main())
