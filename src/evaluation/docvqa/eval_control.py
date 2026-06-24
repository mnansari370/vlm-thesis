"""
the decisive cheap control: is the STUDENT genuinely
question-conditioned, or did it just learn a better blind saliency prior?

Per held-out sample (DocVQA val, disjoint from train), at K=128:
  - select tokens with the student using the REAL question        -> student_real
  - select tokens with the student using a MISMATCHED question    -> student_mis
  - blind (uniform)                                               -> blind
ALWAYS generate the answer with the REAL question; only the SELECTED visual tokens differ.

Mismatched question = a far-offset sample (offset 200) to avoid DocVQA's multiple-questions-
per-document collisions (a near neighbour can share the document => not a real mismatch).

VERDICT: question_fraction = (student_real - student_mis) / (student_real - blind).
  ~1.0 => the student's gain is DUE TO THE QUESTION (genuinely question-conditioned).
  ~0.0 => the student is just a learned blind prior (mismatch doesn't hurt).

Run in qwen_env, from repo root:
  CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen_env/bin/python -m src.evaluation.docvqa.eval_control \
      --student results/thesis_main/highres/distill/student.pt --n 200 --K 128
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.models.distillation.student_selector import CheapQCSelector
from src.data.docvqa import load_docvqa
from src.metrics.docvqa_score import anls


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gate-split", default=None,
                    help="split to evaluate on (default = student's train split). Use 'validation' "
                         "when the student trained on 'train'.")
    ap.add_argument("--gate-start", type=int, default=None)
    ap.add_argument("--mis-offset", type=int, default=200, help="far offset for the mismatched q")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/thesis_main/highres/distill/control_result.json")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    meta, cfg = ck["teacher_meta"], ck["cfg"]
    train_split = meta["split"]
    train_start, train_end = meta["start"], meta["end"]
    gate_split = args.gate_split or train_split
    if args.gate_start is not None:
        gate_start = args.gate_start
    else:
        gate_start = train_end if gate_split == train_split else 0

    ds = load_docvqa(gate_split, n_shards=(meta.get("n_shards") if gate_split == "train" else None))
    gate_end = min(gate_start + args.n, len(ds))
    if gate_split == train_split:
        assert gate_end <= train_start or gate_start >= train_end, (
            f"LEAKAGE: control [{gate_start}:{gate_end}) overlaps train [{train_start}:{train_end})")
        leak = "same split, ranges disjoint OK"
    else:
        leak = f"different splits OK (train={train_split}, gate={gate_split})"
    print(f"[ctrl] train {train_split}[{train_start}:{train_end}) | control {gate_split}"
          f"[{gate_start}:{gate_end}) mis_offset={args.mis_offset}  ({leak})", flush=True)

    pruner = QwenPruner(model_name=args.model, textattn_layer=meta["textattn_layer"],
                        max_pixels=meta["max_pixels"])
    dev = pruner.dev
    student = CheapQCSelector(d=cfg["d"], n_layers=cfg["layers"]).to(dev).float().eval()
    student.load_state_dict(ck["student_state"])

    S = {"real": [], "mis": [], "blind": []}
    t0 = time.time()
    for j in range(gate_start, gate_end):
        ex = ds[j]
        img, q, gold = ex["image"], ex["question"], [str(a) for a in ex.get("answers", [])]
        mis_q = ds[(j + args.mis_offset) % len(ds)]["question"]

        # real-question encode: vis (image-only) + real q context + pos/attn for generation
        emb, pos, attn, v0, nvis, vis = pruner._encode(img, q, instr=True)
        q_real = emb[0, v0 + nvis:].float()
        vis_f = vis.float()
        # mismatched-question embeds (same image -> same vis; we only need its q tokens)
        emb_m, _, _, v0m, nvism, _ = pruner._encode(img, mis_q, instr=True)
        q_mis = emb_m[0, v0m + nvism:].float()

        k = min(args.K, nvis)
        keep = {
            "real":  student(vis_f, q_real).topk(k).indices.sort().values,
            "mis":   student(vis_f, q_mis).topk(k).indices.sort().values,
            "blind": torch.linspace(0, nvis - 1, k, device=dev).round().long().unique(),
        }
        for name, kp in keep.items():
            # ALWAYS answer with the REAL question (emb/pos/attn); only selection differs
            pred, _ = pruner.generate_keep(emb, pos, attn, v0, nvis, kp, args.max_new_tokens)
            S[name].append(anls(pred, gold))

        done = j - gate_start + 1
        if done % args.log_every == 0:
            cur = {kk: 100 * sum(v) / len(v) for kk, v in S.items()}
            print(f"[ctrl] {done}/{gate_end-gate_start} real={cur['real']:.1f} "
                  f"mis={cur['mis']:.1f} blind={cur['blind']:.1f} ({(time.time()-t0)/60:.1f}m)",
                  flush=True)

    res = {k: round(100 * sum(v) / max(len(v), 1), 2) for k, v in S.items()}
    denom = res["real"] - res["blind"]
    qfrac = (res["real"] - res["mis"]) / denom if denom > 1e-6 else 0.0
    res.update({"K": args.K, "n": gate_end - gate_start, "mis_offset": args.mis_offset,
                "question_fraction": round(qfrac, 3)})
    print("\n" + "=" * 58)
    print(f"  STUDENT question-conditioning control — DocVQA @ K={args.K}, n={res['n']}")
    print(f"  student REAL q   : {res['real']:.2f}")
    print(f"  student MIS q    : {res['mis']:.2f}   (control)")
    print(f"  blind            : {res['blind']:.2f}")
    print(f"  -> {100*qfrac:.0f}% of the student's gain over blind is DUE TO THE QUESTION")
    print(f"  VERDICT: {'genuinely QUESTION-CONDITIONED' if qfrac >= 0.5 else 'mostly a learned blind prior'}")
    print("=" * 58)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[ctrl] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
