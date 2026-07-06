"""
run_static_eval.py — run ONE static-pruning cell (model × dataset × selector × budget) under
the final-scope schema. GPU job — do NOT run until approved.

It WRAPS the already-validated static generation paths (does not modify them):
  * LLaVA-1.5  → StaticPrunedLlava(method="cls_attn", keep_k=K)       (image-only CLS-attn top-K)
  * Qwen-2.5-VL → QwenPruner.generate(selector="norm"|"uniform", K=K_sample)  (image-only floor)
and feeds them the SAME per-dataset prompt the dense protocol uses (reused from dense_pilot).
Token counts + per-sample-averaged FLOPs + dense-referenced reductions + output writing + the
fairness gate are handled by src/final_scope/static_eval.py.

KEY: the Qwen per-sample K is read from the accepted DENSE FINAL JSONL
  K_sample = clamp(round(budget_pct/100 * dense_n_visual_tokens), 1, dense_n_visual_tokens)
so there is NO extra encode just to count visual tokens; n_text is also reused from the dense
row (pruning removes only visual tokens, so the text-token count is identical to dense).

Environments (different stacks — run each model under its own env):
  LLaVA  : conda env `vlm_env`  (torch 2.3.0+cu121, transformers 4.46.3) + 1 GPU
  Qwen   : conda env `qwen_env` (transformers 4.51, qwen_vl_utils)        + 1 GPU
  DocVQA (either model) also needs `datasets` + the HF cache (offline OK).
Each static cell requires the matching dense FINAL of the same variant to already exist
(results/runs/{model}/{dataset}/dense_final[...].jsonl).

Usage (after approval; from repo root):
  # LLaVA-1.5 cells (vlm_env), selector defaults to cls_attn:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.static.run_static --model llava15 --dataset gqa --budget-pct 25
  CUDA_VISIBLE_DEVICES=0 python -m scripts.static.run_static --model llava15 --dataset gqa --budget-pct 15 --full

  # Qwen-2.5-VL cells (qwen_env), selector defaults to norm (uniform = optional control):
  CUDA_VISIBLE_DEVICES=0 python -m scripts.static.run_static --model qwen25vl7b --dataset gqa --budget-pct 25
  CUDA_VISIBLE_DEVICES=0 python -m scripts.static.run_static --model qwen25vl7b --dataset gqa --budget-pct 25 --selector uniform

Flags: --selector (default per model), --n (default 200) | --full, --max-new-tokens (default 64),
       --no-ocr (TextVQA), --no-instruction (DocVQA).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.static import evaluate_static as se
from src.dense import evaluate_dense as dp


# ── LLaVA-1.5 static runner (wraps StaticPrunedLlava method="cls_attn") ───────

def build_llava_static_runner(*, selector: str, keep_k: int, add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample, dense_record) ->
    (pred_raw, n_visual=keep_k, n_text, prompt). Loads the model."""
    import torch  # noqa: F401  (ensures the GPU stack is present at call time)
    from src.models.static.static import StaticPrunedLlava, HONEST_SUFFIX

    model = StaticPrunedLlava(
        method=selector, keep_k=keep_k, image_pad=True, honest=True,
        append_suffix=add_instruction,   # True → appends HONEST_SUFFIX (newline + the instruction)
    )
    suffix = HONEST_SUFFIX if add_instruction else ""   # actual model suffix (note leading "\n")

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        # static cls_attn top-K through the validated physical-removal path
        pred = model.generate_answers([s.image], [s.model_input_text], max_new_tokens=max_new_tokens)[0]
        # pruning removes ONLY visual tokens → n_text is identical to the dense reference row
        n_text = int(dense_rec["n_text_tokens"])
        prompt = s.model_input_text.strip() + suffix   # EXACT user-content text the model received
        return pred, keep_k, n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": True,
        "max_pixels": None,
        "decode_notes": (f"StaticPrunedLlava method={selector} K={keep_k} honest: do_sample=False, "
                         "pad_token_id=eos; physical-removal CLS-attn top-K; "
                         "no min_new_tokens, no repetition_penalty"),
    }
    return generate_and_count, meta


# ── Qwen-2.5-VL static runner (wraps QwenPruner.generate selector=norm|uniform) ─

def build_qwen_static_runner(*, selector: str, budget_pct: int, add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample, dense_record) ->
    (pred_raw, n_visual=nkeep, n_text, prompt). Loads the model."""
    import torch  # noqa: F401
    from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner, INSTR

    model = QwenPruner()
    max_pixels = model.proc.image_processor.max_pixels if hasattr(model.proc, "image_processor") else None
    suffix = (" " + INSTR) if add_instruction else ""   # actual model suffix (leading space)

    def generate_and_count(s: dp.Sample, dense_rec: dict):
        # per-sample K from the dense reference visual count (no extra encode to count nvis)
        dense_nvis = int(dense_rec["n_visual_tokens"])
        k = se.clamp_qwen_k(budget_pct, dense_nvis)
        pred, nkeep = model.generate(
            s.image, s.model_input_text, selector=selector, K=k,
            max_new_tokens=max_new_tokens, instr=s.add_instruction,
        )
        # n_text identical to dense (pruning removes only visual tokens) → reuse the dense row
        n_text = int(dense_rec["n_text_tokens"])
        prompt = s.model_input_text.strip() + suffix   # EXACT user-content text the model received
        return pred, int(nkeep), n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": None,
        "max_pixels": max_pixels,
        "decode_notes": (f"QwenPruner selector={selector}; "
                         f"K_sample=clamp(round({budget_pct}%·dense_n_visual),1,dense_n_visual); "
                         "manual greedy argmax decode; no min_new_tokens, no repetition_penalty"),
    }
    return generate_and_count, meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Static-pruning eval under the final-scope schema.")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--budget-pct", required=True, type=int, choices=list(se.BUDGETS),
                    dest="budget_pct")
    ap.add_argument("--selector", default=None, choices=["cls_attn", "norm", "uniform"],
                    help="default per model: llava15→cls_attn, qwen25vl7b→norm")
    ap.add_argument("--n", type=int, default=None,
                    help="pilot size (default 200). Cannot be combined with --full.")
    ap.add_argument("--full", action="store_true",
                    help="FINAL run on the entire manifest; output basename = static_final[...].")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true", help="TextVQA: drop the OCR block (no-OCR setting)")
    ap.add_argument("--no-instruction", action="store_true", help="DocVQA: omit the single-word instruction")
    args = ap.parse_args()

    # guard: --n is a pilot knob; --full uses the whole manifest.
    if args.full and args.n is not None:
        ap.error("--n cannot be combined with --full (a final run uses the full manifest length)")

    # selector: per-model default + per-model validity.
    selector = args.selector if args.selector is not None else se.default_selector(args.model)
    if selector not in se.allowed_selectors(args.model):
        ap.error(f"--selector {selector} is not valid for {args.model}; "
                 f"allowed: {se.allowed_selectors(args.model)}")

    n = args.n if args.n is not None else dp.PILOT_N
    use_ocr = not args.no_ocr
    instruction = not args.no_instruction
    # per-dataset: whether the model should append the single-word instruction.
    #   gqa/vqav2 → always; textvqa → never (already in the text); docvqa → the --instruction choice.
    if args.dataset in ("gqa", "vqav2"):
        add_instruction = True
    elif args.dataset == "textvqa":
        add_instruction = False
    else:  # docvqa
        add_instruction = instruction

    keep_k = se.budget_keep_k(args.model, args.budget_pct)   # int (LLaVA) or None (Qwen)
    if args.model == "llava15":
        gen, meta = build_llava_static_runner(
            selector=selector, keep_k=keep_k, add_instruction=add_instruction,
            max_new_tokens=args.max_new_tokens)
    else:
        gen, meta = build_qwen_static_runner(
            selector=selector, budget_pct=args.budget_pct, add_instruction=add_instruction,
            max_new_tokens=args.max_new_tokens)

    res = se.run_static(
        model_name=args.model, dataset=args.dataset, selector=selector, budget_pct=args.budget_pct,
        generate_and_count=gen, n=(None if args.full else n), full=args.full,
        max_new_tokens=args.max_new_tokens,
        image_pad=meta["image_pad"], max_pixels=meta["max_pixels"],
        use_ocr=use_ocr, instruction=instruction,
        model_instruction_suffix=meta["model_instruction_suffix"], decode_notes=meta["decode_notes"],
    )
    mode = "FINAL" if args.full else "pilot"
    print(f"\n[done] {mode} static: {args.model} × {args.dataset} {selector} p{args.budget_pct}: "
          f"n={res.n} score={res.score_pct}% (dense {res.dense_score_pct}%, Δ={res.score_pct - res.dense_score_pct:+.2f}pp) "
          f"tok-red={res.token_reduction_pct}% flop-red={res.flop_reduction_pct}% "
          f"gate={'OK' if res.gate_ok else 'ISSUES'}\n  {res.out_json}\n  {res.out_jsonl}")
    return 0 if res.gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
