"""
run_dynamic_which_eval.py — run ONE Dynamic-WHICH cell (model × dataset × selector × budget)
under the final-scope schema. GPU job — do NOT run full evaluations until approved.

Dynamic-WHICH = fixed-budget, QUESTION-CONDITIONED visual-token selection (same K as static;
different WHICH tokens). It WRAPS fresh textsim selectors that COMPOSE the frozen generators
(no edits to static.py / qwen_pruner.py):
  * LLaVA-1.5  → LlavaDynamicWhich(selector="textsim"|"textsim_cls_mix", keep_k=K)
  * Qwen-2.5-VL → QwenDynamicWhich(selector="textsim"|"textsim_norm_mix"), per-sample K
The Qwen per-sample K is read from the dense FINAL JSONL
  K_sample = clamp(round(budget_pct/100 * dense_n_visual_tokens), 1, dense_n_visual_tokens)
(no extra encode to count visual tokens); n_text is reused from the dense row. Token counts,
per-sample-averaged FLOPs, dense+static referenced deltas, output writing, and the fairness gate
are handled by src/final_scope/dynamic_which_eval.py.

Each cell needs the matching dense FINAL *and* the matching STATIC FINAL (same model/dataset/
budget/variant) to already exist under results/runs/{model}/{dataset}/.

Environments (run each model under its own env + GPU):
  LLaVA  : /home/nafees/miniconda3/envs/vlm_env/bin/python  on GPU 0
           (torch 2.3.0+cu121, transformers 4.46.3)
  Qwen   : /home/nafees/miniconda3/envs/qwen_env/bin/python on GPU 1
           (transformers 4.51, qwen_vl_utils)
  DocVQA (either model) also needs `datasets` + the HF cache (offline OK).

GPU SANITY commands (small n, just to confirm it runs — NOT a full eval):
  CUDA_VISIBLE_DEVICES=0 /home/nafees/miniconda3/envs/vlm_env/bin/python \
      -m scripts.dynamic_which.run_dynamic_which \
      --model llava15 --dataset gqa --budget-pct 75 --selector textsim --n 50
  CUDA_VISIBLE_DEVICES=1 /home/nafees/miniconda3/envs/qwen_env/bin/python \
      -m scripts.dynamic_which.run_dynamic_which \
      --model qwen25vl7b --dataset gqa --budget-pct 75 --selector textsim --n 50

Flags: --selector (default textsim), --alpha (mix only, default 0.5), --n (default 200) | --full,
       --static-selector (default primary: llava→cls_attn, qwen→norm), --max-new-tokens (default 64),
       --no-ocr (TextVQA), --no-instruction (DocVQA).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dynamic_which import evaluate_dynamic_which as dw
from src.dense import evaluate_dense as dp


# ── LLaVA-1.5 dynamic-WHICH runner (wraps LlavaDynamicWhich) ──────────────────

def build_llava_dynamic_runner(*, selector: str, keep_k: int, alpha: float,
                               add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample, dense_record) ->
    (pred_raw, n_visual=keep_k, n_text, prompt). Loads the model."""
    import torch  # noqa: F401  (ensures the GPU stack is present at call time)
    from src.pruning.dynamic_which.llava_textsim import LlavaDynamicWhich
    from src.models.static.static import HONEST_SUFFIX

    model = LlavaDynamicWhich(selector=selector, keep_k=keep_k, alpha=alpha,
                              add_instruction=add_instruction, max_new_tokens=max_new_tokens)
    suffix = HONEST_SUFFIX if add_instruction else ""

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        # prompt = the EXACT VLM text (s.model_input_text); selector scoring uses ONLY s.question
        pred = model.generate_one(s.image, s.model_input_text, s.question)
        # pruning removes ONLY visual tokens → n_text is identical to the dense reference row
        n_text = int(dense_rec["n_text_tokens"])
        prompt = s.model_input_text.strip() + suffix
        return pred, keep_k, n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": True,
        "max_pixels": None,
        "decode_notes": (f"LlavaDynamicWhich selector={selector} K={keep_k}"
                         + (f" alpha={alpha}" if selector.endswith("_mix") else "")
                         + "; honest do_sample=False, pad_token_id=eos; VLM prompt = "
                         "sample.model_input_text; question-conditioned top-K scored on "
                         "sample.question only (textsim = cos(projected-visual, question-token "
                         "embeds)); physical removal via StaticPrunedLlava._build_inputs; "
                         "no LM prefill for scoring"),
    }
    return generate_and_count, meta


# ── Qwen-2.5-VL dynamic-WHICH runner (wraps QwenDynamicWhich) ──────────────────

def build_qwen_dynamic_runner(*, selector: str, budget_pct: int, alpha: float,
                              add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample, dense_record) ->
    (pred_raw, n_visual=nkeep, n_text, prompt). Loads the model."""
    import torch  # noqa: F401
    from src.pruning.dynamic_which.qwen_textsim import QwenDynamicWhich
    from src.pruning.question_conditioned_selection.qwen_pruner import INSTR

    model = QwenDynamicWhich(selector=selector, alpha=alpha, max_new_tokens=max_new_tokens)
    max_pixels = model.proc.image_processor.max_pixels if hasattr(model.proc, "image_processor") else None
    suffix = (" " + INSTR) if add_instruction else ""

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        # per-sample K from the dense reference visual count (no extra encode to count nvis)
        dense_nvis = int(dense_rec["n_visual_tokens"])
        k = dw.clamp_qwen_k(budget_pct, dense_nvis)
        # prompt = the EXACT VLM text (s.model_input_text); selector scoring uses ONLY s.question
        pred, nkeep = model.generate_one(s.image, s.model_input_text, s.question, k,
                                         instr=s.add_instruction)
        n_text = int(dense_rec["n_text_tokens"])
        prompt = s.model_input_text.strip() + suffix
        return pred, int(nkeep), n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": None,
        "max_pixels": max_pixels,
        "decode_notes": (f"QwenDynamicWhich selector={selector}"
                         + (f" alpha={alpha}" if selector.endswith("_mix") else "")
                         + f"; K_sample=clamp(round({budget_pct}%·dense_n_visual),1,dense_n_visual); "
                         "VLM prompt = sample.model_input_text; question-conditioned top-K scored on "
                         "sample.question only (textsim = cos(visual embeds, question-token embeds)); "
                         "reuses QwenPruner._encode/generate_keep; no dense LM prefill; "
                         "manual greedy argmax decode"),
    }
    return generate_and_count, meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Dynamic-WHICH eval under the final-scope schema.")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--budget-pct", required=True, type=int, choices=list(dw.BUDGETS),
                    dest="budget_pct")
    ap.add_argument("--selector", default=None,
                    choices=["textsim", "textsim_cls_mix", "textsim_norm_mix"],
                    help="default per model: textsim (both). Mixes: llava→textsim_cls_mix, "
                         "qwen→textsim_norm_mix")
    ap.add_argument("--alpha", type=float, default=dw.DEFAULT_ALPHA,
                    help="mix weight on textsim for the *_mix selectors (default 0.5)")
    ap.add_argument("--static-selector", default=None, dest="static_selector",
                    choices=["cls_attn", "norm", "uniform"],
                    help="static baseline to compare against (default primary: llava→cls_attn, qwen→norm)")
    ap.add_argument("--n", type=int, default=None,
                    help="pilot size (default 200). Cannot be combined with --full.")
    ap.add_argument("--full", action="store_true",
                    help="FINAL run on the entire manifest; output basename = dynamic_which_final[...].")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true", help="TextVQA: drop the OCR block (no-OCR setting)")
    ap.add_argument("--no-instruction", action="store_true", help="DocVQA: omit the single-word instruction")
    args = ap.parse_args()

    # guard: --n is a pilot knob; --full uses the whole manifest.
    if args.full and args.n is not None:
        ap.error("--n cannot be combined with --full (a final run uses the full manifest length)")

    # selector: per-model default + per-model validity.
    selector = args.selector if args.selector is not None else dw.default_dynamic_selector(args.model)
    if selector not in dw.allowed_dynamic_selectors(args.model):
        ap.error(f"--selector {selector} is not valid for {args.model}; "
                 f"allowed: {dw.allowed_dynamic_selectors(args.model)}")

    n = args.n if args.n is not None else dp.PILOT_N
    use_ocr = not args.no_ocr
    instruction = not args.no_instruction
    # per-dataset: whether the model should append the single-word instruction.
    if args.dataset in ("gqa", "vqav2"):
        add_instruction = True
    elif args.dataset == "textvqa":
        add_instruction = False
    else:  # docvqa
        add_instruction = instruction

    keep_k = dw.budget_keep_k(args.model, args.budget_pct)   # int (LLaVA) or None (Qwen)
    if args.model == "llava15":
        gen, meta = build_llava_dynamic_runner(
            selector=selector, keep_k=keep_k, alpha=args.alpha,
            add_instruction=add_instruction, max_new_tokens=args.max_new_tokens)
    else:
        gen, meta = build_qwen_dynamic_runner(
            selector=selector, budget_pct=args.budget_pct, alpha=args.alpha,
            add_instruction=add_instruction, max_new_tokens=args.max_new_tokens)

    res = dw.run_dynamic_which(
        model_name=args.model, dataset=args.dataset, selector=selector, budget_pct=args.budget_pct,
        generate_and_count=gen, n=(None if args.full else n), full=args.full, alpha=args.alpha,
        static_selector=args.static_selector, max_new_tokens=args.max_new_tokens,
        image_pad=meta["image_pad"], max_pixels=meta["max_pixels"],
        use_ocr=use_ocr, instruction=instruction,
        model_instruction_suffix=meta["model_instruction_suffix"], decode_notes=meta["decode_notes"],
    )
    mode = "FINAL" if args.full else "pilot"
    print(f"\n[done] {mode} dynamic_which: {args.model} × {args.dataset} {selector} p{args.budget_pct}: "
          f"n={res.n} score={res.score_pct}% (dense {res.dense_score_pct}%, static {res.static_score_pct}%; "
          f"dyn−static={res.dynamic_minus_static_pp:+.2f}pp) "
          f"tok-red={res.token_reduction_pct}% flop-red={res.flop_reduction_pct}% "
          f"gate={'OK' if res.gate_ok else 'ISSUES'}\n  {res.out_json}\n  {res.out_jsonl}")
    return 0 if res.gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
