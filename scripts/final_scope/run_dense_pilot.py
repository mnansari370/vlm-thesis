"""
run_dense_pilot.py — run ONE dense pilot cell (model × dataset, n=200) under the final-scope
schema. GPU job — do NOT run until approved.

It WRAPS already-validated generation paths (does not modify them):
  * LLaVA-1.5  → StaticPrunedLlava(method="none", keep_k=576)   (dense == K=576, validated)
  * Qwen-2.5-VL → QwenPruner.generate(selector="full")          (keep-all == stock, validated)
and feeds them the per-dataset prompt the dense protocol specifies (DENSE_PROTOCOL.md §2/§3/§4).
Token counts + per-sample-averaged FLOPs + output writing + the fairness gate are handled by
src/final_scope/{dense_pilot,token_flops,output_writer,schema_validator}.py.

Environments (different stacks — run each model under its own env):
  LLaVA  : conda env `vlm_env`  (torch 2.3.0+cu121, transformers 4.46.3) + 1 GPU
  Qwen   : conda env `qwen_env` (transformers 4.51, qwen_vl_utils)        + 1 GPU
  DocVQA (either model) also needs `datasets` + the HF cache (offline OK).

Usage (after approval; from repo root):
  # LLaVA-1.5 cells (vlm_env):
  CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset gqa
  CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset textvqa
  CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset vqav2
  CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 python -m scripts.final_scope.run_dense_pilot --model llava15 --dataset docvqa

  # Qwen-2.5-VL cells (qwen_env):
  CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset docvqa
  CUDA_VISIBLE_DEVICES=0 python -m scripts.final_scope.run_dense_pilot --model qwen25vl7b --dataset gqa
  ...

Flags: --n (default 200), --max-new-tokens (default 64), --no-ocr (TextVQA), --no-instruction (DocVQA).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.final_scope import dense_pilot as dp


# ── LLaVA-1.5 runner (wraps StaticPrunedLlava method="none") ──────────────────

def build_llava_runner(*, add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample) ->
    (pred_raw, n_visual=576, n_text, prompt). Loads the model."""
    import torch  # noqa: F401  (ensures the GPU stack is present at call time)
    from src.models.static.static import StaticPrunedLlava, HONEST_SUFFIX

    model = StaticPrunedLlava(
        method="none", keep_k=576, image_pad=True, honest=True,
        append_suffix=add_instruction,   # True → appends HONEST_SUFFIX (newline + the instruction)
    )
    suffix = HONEST_SUFFIX if add_instruction else ""   # actual model suffix (note: leading "\n")
    img_tok = model._image_token_id

    def _count_text_tokens(image, text: str) -> int:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text.strip() + suffix}]}]
        prompt = model.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = model.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        ids = inp["input_ids"][0]
        return int((ids != img_tok).sum().item())  # everything except the 576 image tokens

    def generate_and_count(s: dp.Sample):
        # honest dense generation through the validated physical-removal path (K=576 == dense)
        pred = model.generate_answers([s.image], [s.model_input_text], max_new_tokens=max_new_tokens)[0]
        n_text = _count_text_tokens(s.image, s.model_input_text)
        prompt = s.model_input_text.strip() + suffix   # EXACT user-content text the model received
        return pred, 576, n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": True,
        "max_pixels": None,
        "decode_notes": "StaticPrunedLlava method=none honest: do_sample=False, pad_token_id=eos, "
                        "no min_new_tokens, no repetition_penalty",
    }
    return generate_and_count, meta


# ── Qwen-2.5-VL runner (wraps QwenPruner.generate selector="full") ────────────

def build_qwen_runner(*, add_instruction: bool, max_new_tokens: int):
    """Returns (generate_and_count, meta). generate_and_count(sample) ->
    (pred_raw, n_visual, n_text, prompt). Loads the model."""
    import torch  # noqa: F401
    from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner, IMG_TOK, INSTR

    model = QwenPruner()
    max_pixels = model.proc.image_processor.max_pixels if hasattr(model.proc, "image_processor") else None
    suffix = (" " + INSTR) if add_instruction else ""   # actual model suffix (leading space)

    def generate_and_count(s: dp.Sample):
        pred, _nkeep = model.generate(
            s.image, s.model_input_text, selector="full", K=None,
            max_new_tokens=max_new_tokens, instr=s.add_instruction,
        )
        inp = model._inputs(s.image, s.model_input_text, instr=s.add_instruction)
        ids = inp["input_ids"][0]
        n_vis = int((ids == IMG_TOK).sum().item())
        n_text = int(ids.shape[0]) - n_vis
        prompt = s.model_input_text.strip() + suffix   # EXACT user-content text the model received
        return pred, n_vis, n_text, prompt

    meta = {
        "model_instruction_suffix": suffix,
        "image_pad": None,
        "max_pixels": max_pixels,
        "decode_notes": "QwenPruner manual greedy argmax decode (do_sample n/a); "
                        "no min_new_tokens, no repetition_penalty",
    }
    return generate_and_count, meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Dense pilot / final run under the final-scope schema.")
    ap.add_argument("--model", required=True, choices=list(dp.MODELS))
    ap.add_argument("--dataset", required=True, choices=list(dp.DATASETS))
    ap.add_argument("--n", type=int, default=None,
                    help="pilot size (default 200). Cannot be combined with --full.")
    ap.add_argument("--full", action="store_true",
                    help="FINAL run on the entire manifest; output basename = dense_final[...]. "
                         "Final defaults: TextVQA OCR-on, DocVQA instruction-on.")
    ap.add_argument("--max-new-tokens", type=int, default=dp.MAX_NEW_TOKENS)
    ap.add_argument("--no-ocr", action="store_true", help="TextVQA: drop the OCR block (no-OCR setting)")
    ap.add_argument("--no-instruction", action="store_true", help="DocVQA: omit the single-word instruction")
    args = ap.parse_args()

    # guard: --n is a pilot knob; --full uses the whole manifest.
    if args.full and args.n is not None:
        ap.error("--n cannot be combined with --full (a final run uses the full manifest length)")
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

    if args.model == "llava15":
        gen, meta = build_llava_runner(add_instruction=add_instruction, max_new_tokens=args.max_new_tokens)
    else:
        gen, meta = build_qwen_runner(add_instruction=add_instruction, max_new_tokens=args.max_new_tokens)

    res = dp.run_pilot(
        model_name=args.model, dataset=args.dataset, generate_and_count=gen,
        n=n, full=args.full, max_new_tokens=args.max_new_tokens,
        image_pad=meta["image_pad"], max_pixels=meta["max_pixels"],
        use_ocr=use_ocr, instruction=instruction,
        model_instruction_suffix=meta["model_instruction_suffix"], decode_notes=meta["decode_notes"],
    )
    mode = "FINAL" if args.full else "pilot"
    print(f"\n[done] {mode}: {args.model} × {args.dataset}: n={res.n} score={res.score_pct}% "
          f"gate={'OK' if res.gate_ok else 'ISSUES'}\n  {res.out_json}\n  {res.out_jsonl}")
    return 0 if res.gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
