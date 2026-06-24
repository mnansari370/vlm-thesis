"""
v2/distill/train_student.py — Step 2: distill the cheap student from the cached teacher.

Loads the cached L16 teacher maps (cache_teacher.py), recomputes the cheap pre-LLM
features on the fly (QwenPruner._encode = ViT + merger + embed; NO decoder), and trains
CheapQCSelector to predict the teacher's visual-token importance.

Loss = listwise KL(teacher_dist || softmax(student))  +  lambda * top-K BCE
  - KL teaches the full ranking; the BCE directly optimizes the K=128 keep/drop decision.

Cheap proxy logged every epoch: mean top-128 set overlap between student and teacher
(selection quality without any generation). The real downstream gate is eval_gate.py.

Run in qwen_env, from repo root:
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/qwen_env/bin/python -m src.training.train_student \
      --teacher results/thesis_main/highres/distill/teacher_docvqa_train.pt \
      --out results/thesis_main/highres/distill/student.pt --epochs 8
Smoke:
  ... --teacher /tmp/teacher_smoke.pt --out /tmp/student_smoke.pt --epochs 2 --max-rows 2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.models.distillation.student_selector import CheapQCSelector
from src.data.docvqa import load_docvqa

TOPK = 128   # operating budget for the BCE aux + the overlap proxy


@torch.no_grad()
def encode_features(pruner, image, question):
    """Cheap pre-LLM features: vis embeds [nvis,D], question embeds [Lq,D]. No decoder."""
    emb, pos, attn, v0, nvis, vis = pruner._encode(image, question, instr=True)
    q_embeds = emb[0, v0 + nvis:]          # text tokens after the visual block
    return vis.float(), q_embeds.float(), nvis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="teacher cache .pt from cache_teacher.py")
    ap.add_argument("--out", default="results/thesis_main/highres/distill/student.pt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temp for student in KL")
    ap.add_argument("--bce-weight", type=float, default=1.0, help="top-K BCE aux weight")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=None, help="cap teacher rows (smoke)")
    ap.add_argument("--feat-cache-dir", default=None,
                    help="if set, cache (vis,q) features to DISK here (scales past RAM). "
                         "Keyed by DocVQA row idx; epoch 1 builds it, later epochs read it.")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    cache = torch.load(args.teacher, map_location="cpu", weights_only=False)
    meta, rows = cache["meta"], cache["rows"]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[train] teacher: {len(rows)} rows | split={meta['split']} "
          f"[{meta['start']}:{meta['end']}] L{meta['textattn_layer']} maxpix={meta['max_pixels']}",
          flush=True)

    # Reload the SAME split the teacher was cached from, to fetch images by idx.
    ds = load_docvqa(meta["split"], n_shards=meta.get("n_shards"))
    pruner = QwenPruner(model_name=args.model, textattn_layer=meta["textattn_layer"],
                        max_pixels=meta["max_pixels"])
    dev = pruner.dev

    student = CheapQCSelector(d=args.d, n_layers=args.layers).to(dev).float()
    print(f"[train] student params: {CheapQCSelector.num_params(student):,}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Feature cache: ViT features are deterministic, so compute each sample's (vis, q) ONCE
    # in epoch 1 and reuse — epochs 2+ skip the ViT forward. RAM dict by default; --feat-cache-dir
    # puts it on DISK (keyed by DocVQA row idx) so it scales past RAM. None = mismatch -> skip.
    feat_cache: dict = {}
    fc_dir = args.feat_cache_dir
    if fc_dir:
        os.makedirs(fc_dir, exist_ok=True)

    def get_feats(ri, r, teacher_numel):
        if fc_dir:
            fp = os.path.join(fc_dir, f"{r['idx']}.pt")
            if os.path.exists(fp):
                return torch.load(fp, map_location="cpu", weights_only=False)
            ex = ds[r["idx"]]
            vis_g, q_g, nvis = encode_features(pruner, ex["image"], ex["question"])
            d = (None if nvis != teacher_numel
                 else (vis_g.to("cpu", torch.float16), q_g.to("cpu", torch.float16)))
            torch.save(d, fp)
            return d
        if ri not in feat_cache:
            ex = ds[r["idx"]]
            vis_g, q_g, nvis = encode_features(pruner, ex["image"], ex["question"])
            feat_cache[ri] = (None if nvis != teacher_numel
                              else (vis_g.to("cpu", torch.float16), q_g.to("cpu", torch.float16)))
        return feat_cache[ri]

    order = list(range(len(rows)))
    for epoch in range(1, args.epochs + 1):
        import random
        random.Random(epoch).shuffle(order)
        student.train()
        tot_loss = tot_kl = tot_bce = tot_ov = 0.0
        nseen = 0
        t0 = time.time()
        for step, ri in enumerate(order):
            r = rows[ri]
            teacher = r["qscores"].float().to(dev)          # [nvis]
            entry = get_feats(ri, r, teacher.numel())         # RAM or disk cache
            if entry is None:                                 # resolution mismatch guard
                continue
            vis, q = entry[0].to(dev).float(), entry[1].to(dev).float()
            nvis = vis.shape[0]

            scores = student(vis, q)                          # [nvis]
            # listwise KL: teacher normalized to a distribution over visual tokens
            t_p = (teacher / teacher.sum().clamp_min(1e-8))
            s_logp = F.log_softmax(scores / args.temp, dim=-1)
            kl = F.kl_div(s_logp, t_p, reduction="sum")
            # top-K BCE on the keep/drop decision at the operating budget
            k = min(TOPK, nvis)
            tgt = torch.zeros(nvis, device=dev)
            tgt[teacher.topk(k).indices] = 1.0
            bce = F.binary_cross_entropy_with_logits(scores, tgt)
            loss = kl + args.bce_weight * bce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                ov = len(set(scores.topk(k).indices.tolist()) &
                         set(teacher.topk(k).indices.tolist())) / k
            tot_loss += float(loss); tot_kl += float(kl); tot_bce += float(bce); tot_ov += ov
            nseen += 1
            if (step + 1) % args.log_every == 0:
                print(f"[train] ep{epoch} {step+1}/{len(order)} loss={tot_loss/nseen:.4f} "
                      f"kl={tot_kl/nseen:.4f} bce={tot_bce/nseen:.4f} "
                      f"top{k}_overlap={tot_ov/nseen:.3f}", flush=True)

        m = max(nseen, 1)
        print(f"[train] === epoch {epoch} done: loss={tot_loss/m:.4f} "
              f"top{TOPK}_overlap={tot_ov/m:.3f}  ({(time.time()-t0)/60:.1f} min, n={nseen}) ===",
              flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"student_state": student.state_dict(),
                    "cfg": {"d": args.d, "layers": args.layers, "topk": TOPK},
                    "teacher_meta": meta, "epoch": epoch}, args.out)
    print(f"[train] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
