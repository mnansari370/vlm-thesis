"""
run_dynamic_count_probe.py — run ONE Dynamic-COUNT PROBE cell (model × dataset × selector ×
probe budget) with confidence/saliency/question signal recording. GPU job.

The probe reruns the cheap pass through wrappers that mirror the FROZEN generation paths exactly,
recording signals for free. BUILT-IN REPRODUCTION GATE: every prediction is compared against the
frozen reference final (static cls_attn/norm, or WHICH textsim for the Qwen TextVQA textsim probe);
ANY mismatch fails the gate and exits rc=2 — stop and debug before continuing to DC-D/DC-C.

Envs / GPUs:
  LLaVA : CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python
  Qwen  : CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python
DocVQA also needs HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1.

Usage (full-manifest probe):
  ... -m scripts.final_scope.run_dynamic_count_probe --model llava15 --dataset gqa --probe-pct 15 --full
  ... -m scripts.final_scope.run_dynamic_count_probe --model qwen25vl7b --dataset textvqa \
        --selector textsim --probe-pct 25 --full        # COUNT-on-WHICH probe
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp
from src.final_scope import dynamic_count_eval as dce
from src.pruning import dynamic_count as dc


def build_llava_probe(add_instruction: bool, max_new_tokens: int):
    from src.pruning.dynamic_count.llava_dynamic_count import LlavaDynamicCount
    from src.models.static.static import HONEST_SUFFIX
    m = LlavaDynamicCount(add_instruction=add_instruction, max_new_tokens=max_new_tokens)
    suffix = HONEST_SUFFIX if add_instruction else ""

    def probe_fn(s: dp.Sample, k_probe: int):
        pred, conf, sal = m.probe(s.image, s.model_input_text, k_probe)
        return pred, k_probe, conf, sal
    meta = {"suffix": suffix,
            "notes": ("LlavaDynamicCount probe: frozen cls_attn top-K path + output_scores "
                      "recording; honest greedy; reproduction-gated vs static_final")}
    return probe_fn, meta


def build_qwen_probe(selector: str, add_instruction: bool, max_new_tokens: int):
    from src.pruning.dynamic_count.qwen_dynamic_count import QwenDynamicCount
    from src.pruning.question_conditioned_selection.qwen_pruner import INSTR
    m = QwenDynamicCount()
    suffix = (" " + INSTR) if add_instruction else ""

    def probe_fn(s: dp.Sample, k_probe: int):
        pred, nkeep, conf, sal, _nvis = m.probe(
            s.image, s.model_input_text, s.question, k_probe, selector=selector,
            instr=s.add_instruction, max_new_tokens=max_new_tokens)
        return pred, nkeep, conf, sal
    meta = {"suffix": suffix,
            "notes": (f"QwenDynamicCount probe selector={selector}: frozen selection + _greedy "
                      "mirror with prob recording; reproduction-gated vs frozen finals")}
    return probe_fn, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamic-COUNT probe pass with signal recording.")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--selector", default=None, help="default: locked static selector per model; "
                                                     "'textsim' allowed for qwen25vl7b×textvqa")
    ap.add_argument("--probe-pct", required=True, type=int, choices=list(dc.PROBE_BUDGETS),
                    dest="probe_pct")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-instruction", action="store_true")
    args = ap.parse_args()
    if args.full and args.n is not None:
        ap.error("--n cannot be combined with --full")

    selector = args.selector or ("cls_attn" if args.model == "llava15" else "norm")
    if selector not in dc.PROBE_SELECTORS[args.model]:
        ap.error(f"--selector {selector} not valid for {args.model}; "
                 f"allowed: {dc.PROBE_SELECTORS[args.model]}")
    use_ocr, instruction = not args.no_ocr, not args.no_instruction
    add_instruction = (args.dataset in ("gqa", "vqav2")) or \
                      (args.dataset == "docvqa" and instruction)

    if args.model == "llava15":
        probe_fn, meta = build_llava_probe(add_instruction, args.max_new_tokens)
    else:
        probe_fn, meta = build_qwen_probe(selector, add_instruction, args.max_new_tokens)

    js, jl, score, equiv_ok, gate_ok = dce.run_probe(
        model_name=args.model, dataset=args.dataset, selector=selector, probe_pct=args.probe_pct,
        probe_fn=probe_fn, n=args.n, full=args.full, max_new_tokens=args.max_new_tokens,
        use_ocr=use_ocr, instruction=instruction,
        model_instruction_suffix=meta["suffix"], decode_notes=meta["notes"])
    print(f"\n[done] probe {args.model}×{args.dataset} {selector} p{args.probe_pct}: "
          f"score={score}% equivalence={'OK' if equiv_ok else 'FAILED'} "
          f"gate={'OK' if gate_ok else 'ISSUES'}\n  {js}\n  {jl}")
    return 0 if (equiv_ok and gate_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
