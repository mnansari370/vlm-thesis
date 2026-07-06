"""
run_dynamic_count_continuous.py — run DC-C (THE main Dynamic-COUNT method) for ONE cell. GPU job.

Reads the probe output + calibrated controller config; predicts a per-sample INTEGER K_i
(continuous fraction × native token count, clamped to [32, N_i]); keeps the probe answer when
K_i ≤ K_probe, otherwise generates ONCE at exactly K_i through the frozen path. Honest double-pay
FLOPs. Evaluates the held-out 80% by default.

Envs / GPUs: LLaVA GPU0 vlm_env; Qwen GPU1 qwen_env; DocVQA offline env vars.
Usage:
  ... -m scripts.dynamic_count.run_continuous --model llava15 --dataset gqa \
        --controller rule --lam 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dense import evaluate_dense as dp
from src.dynamic_count import evaluate_dynamic_count as dce

CFG_DIR = os.path.join("results", "configs", "dynamic_count")


def build_second_pass(model: str, selector: str, add_instruction: bool, max_new_tokens: int):
    if model == "llava15":
        from src.pruning.dynamic_count.llava_dynamic_count import LlavaDynamicCount
        m = LlavaDynamicCount(add_instruction=add_instruction, max_new_tokens=max_new_tokens)

        def fn(s: dp.Sample, k_i: int):
            return m.generate_at_k(s.image, s.model_input_text, k_i)
        return fn
    from src.pruning.dynamic_count.qwen_dynamic_count import QwenDynamicCount
    m = QwenDynamicCount()

    def fn(s: dp.Sample, k_i: int):
        pred, _ = m.generate_at_k(s.image, s.model_input_text, s.question, k_i,
                                  selector=selector, instr=s.add_instruction,
                                  max_new_tokens=max_new_tokens)
        return pred
    return fn


def main() -> int:
    ap = argparse.ArgumentParser(description="DC-C continuous Dynamic-COUNT for one cell (GPU).")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--controller", default="rule", choices=["rule", "ridge"])
    ap.add_argument("--lam", type=float, default=None, help="compute knob (default: config's)")
    ap.add_argument("--count-on-which", action="store_true", dest="cow",
                    help="use the SEPARATE textsim (COUNT-on-WHICH) config — qwen25vl7b×textvqa only")
    ap.add_argument("--all-samples", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=None, dest="smoke_n",
                    help="SMOKE: first N eval samples only; output basename gets _smoke_n{N}")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-instruction", action="store_true")
    args = ap.parse_args()
    tag = f"{args.model}_{args.dataset}" + ("_textsim" if args.cow else "")
    cfg_path = os.path.join(CFG_DIR, f"{tag}.json")
    if not os.path.exists(cfg_path):
        print(f"[error] missing calibration config {cfg_path} — run calibration first")
        return 2
    full_cfg = json.load(open(cfg_path))
    ccfg = full_cfg["dcc_rule"] if args.controller == "rule" else full_cfg["dcc_ridge"]
    selector = full_cfg["dcd"]["selector"]
    probe_pct = full_cfg["dcd"]["probe_pct"]
    lam = args.lam if args.lam is not None else ccfg["lam_default"]
    instruction = not args.no_instruction
    add_instruction = (args.dataset in ("gqa", "vqav2")) or (args.dataset == "docvqa" and instruction)

    fn = build_second_pass(args.model, selector, add_instruction, args.max_new_tokens)
    js, jl, score, delta, ok = dce.run_dcc(
        model_name=args.model, dataset=args.dataset, selector=selector, probe_pct=probe_pct,
        controller_cfg=ccfg, lam=lam, second_pass_fn=fn, controller_tag=args.controller,
        use_ocr=not args.no_ocr, instruction=instruction,
        eval_split_only=not args.all_samples, smoke_n=args.smoke_n)
    print(f"\n[done] DC-C({args.controller}, lam={lam}) {args.model}×{args.dataset}: "
          f"score={score}% Δcurve={delta:+.2f}pp gate={'OK' if ok else 'ISSUES'}\n  {js}\n  {jl}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
