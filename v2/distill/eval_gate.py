"""
v2/distill/eval_gate.py — Step 3: the PILOT GO/NO-GO gate.

Question: can the cheap student (no decoder forward) recover most of the mid-layer
TEACHER's downstream selection accuracy at K=128 on a held-out set?

On DocVQA validation[gate_start : gate_start+n], at K=128, generate and ANLS-score four
selections per sample:
  - dense   : keep all tokens                         (ceiling)
  - blind   : uniform stride                          (question-blind floor)
  - teacher : top-K by L16 question->visual attention (the expensive upper bound)
  - student : top-K by the CHEAP CheapQCSelector      (our method)

GATE METRIC: recovery = (student - blind) / (teacher - blind).
  PASS (build the full method) if recovery > 0.50.
  If FAIL: try a shallow proxy (first 2-3 LLM layers) before declaring the negative.

LEAKAGE GUARD (hard): gate_start defaults to the teacher cache's end index and the script
ASSERTS the gate window does not overlap the student's training slice.

Run in qwen_env, from repo root:
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/qwen_env/bin/python -m v2.distill.eval_gate \
      --student v2/outputs/distill/student.pt --n 400 --K 128
Smoke:
  ... --student /tmp/student_smoke.pt --n 4 --gate-start 5000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from v2.qwen.qwen_pruner import QwenPruner
from v2.distill.student_selector import CheapQCSelector
from v2.distill.docvqa_data import load_docvqa
from v2.shared.docvqa_score import anls


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="student checkpoint from train_student.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n", type=int, default=400, help="gate sample count")
    ap.add_argument("--gate-split", default=None,
                    help="split to evaluate on (default = the student's train split). Use "
                         "'validation' when the student trained on 'train' (cleanest, no leakage).")
    ap.add_argument("--gate-start", type=int, default=None,
                    help="first gate index; default = train-end if same split as train, else 0.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="v2/outputs/distill/gate_result.json")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    meta = ck["teacher_meta"]
    cfg = ck["cfg"]
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
        # same split -> HARD leakage guard: gate window must not intersect the train slice.
        assert gate_end <= train_start or gate_start >= train_end, (
            f"LEAKAGE: gate [{gate_start}:{gate_end}) overlaps train [{train_start}:{train_end})")
        leak = "same split, ranges disjoint OK"
    else:
        leak = f"different splits OK (train={train_split}, gate={gate_split})"
    print(f"[gate] train {train_split}[{train_start}:{train_end})  |  "
          f"gate {gate_split}[{gate_start}:{gate_end})  ({leak})", flush=True)

    pruner = QwenPruner(model_name=args.model, textattn_layer=meta["textattn_layer"],
                        max_pixels=meta["max_pixels"])
    dev = pruner.dev
    student = CheapQCSelector(d=cfg["d"], n_layers=cfg["layers"]).to(dev).float().eval()
    student.load_state_dict(ck["student_state"])

    S = {"dense": [], "blind": [], "teacher": [], "student": []}
    t0 = time.time()
    for i in range(gate_start, gate_end):
        ex = ds[i]
        img, q, gold = ex["image"], ex["question"], [str(a) for a in ex.get("answers", [])]

        # one ViT forward (vis) + one decoder forward (teacher attn) — same as encode_qc
        emb, pos, attn, v0, nvis, vis = pruner._encode(img, q, instr=True)
        out = pruner.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                                 use_cache=False, output_attentions=True)
        A = out.attentions[pruner.textattn_layer][0].float().mean(0)
        teacher = A[v0 + nvis:, v0:v0 + nvis].mean(0)        # [nvis]
        del out, A
        q_embeds = emb[0, v0 + nvis:].float()
        s_scores = student(vis.float(), q_embeds)            # [nvis]

        k = min(args.K, nvis)
        sel = {
            "dense":   torch.arange(nvis, device=dev),
            "blind":   torch.linspace(0, nvis - 1, k, device=dev).round().long().unique(),
            "teacher": teacher.topk(k).indices.sort().values,
            "student": s_scores.topk(k).indices.sort().values,
        }
        for name, keep in sel.items():
            pred, _ = pruner.generate_keep(emb, pos, attn, v0, nvis, keep, args.max_new_tokens)
            S[name].append(anls(pred, gold))

        done = i - gate_start + 1
        if done % args.log_every == 0:
            cur = {k2: 100 * sum(v) / len(v) for k2, v in S.items()}
            print(f"[gate] {done}/{gate_end-gate_start}  "
                  f"dense={cur['dense']:.1f} teacher={cur['teacher']:.1f} "
                  f"student={cur['student']:.1f} blind={cur['blind']:.1f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)

    res = {k: round(100 * sum(v) / max(len(v), 1), 2) for k, v in S.items()}
    gain_teacher = res["teacher"] - res["blind"]
    recovery = (res["student"] - res["blind"]) / gain_teacher if gain_teacher > 1e-6 else 0.0
    res.update({"K": args.K, "n": gate_end - gate_start,
                "teacher_gain_over_blind": round(gain_teacher, 2),
                "student_gain_over_blind": round(res["student"] - res["blind"], 2),
                "recovery_frac": round(recovery, 3),
                "gate_pass": bool(recovery > 0.5)})

    print("\n" + "=" * 60)
    print(f"  DISTILLATION GATE — DocVQA val[{gate_start}:{gate_end}] @ K={args.K}")
    print(f"  dense   {res['dense']:.2f}")
    print(f"  teacher {res['teacher']:.2f}   (gain over blind +{gain_teacher:.2f})")
    print(f"  student {res['student']:.2f}   (gain over blind +{res['student']-res['blind']:.2f})")
    print(f"  blind   {res['blind']:.2f}")
    print(f"  RECOVERY = {100*recovery:.1f}%   -> {'PASS (build it)' if res['gate_pass'] else 'FAIL (try shallow proxy)'}")
    print("=" * 60)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import json
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[gate] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
