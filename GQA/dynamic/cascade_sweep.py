"""
Task 3 — Cascade frontier: sweep tau for the frozen cascade (K1=144 -> K2=288 if conf<tau).

For each benchmark, joins per-sample (by dataset order/index): K1 confidence, K1 score, K2 score.
cascade_score(tau) = K1 if conf>=tau else K2 ; rerun(tau)=mean(conf<tau);
honest cascade FLOPs = f(144)+rerun*f(288). Reports acc/avg_K/rerun/FLOPs/retention per tau.
All CPU (no GPU). Score = VQA soft-acc (TextVQA) or binary correct (GQA/POPE/SQA).

Usage: python -m GQA.dynamic.cascade_sweep
"""

import glob, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from GQA.shared.official_score import is_correct
from GQA.shared.flops import (flops_row, N_TEXT_TESTDEV, N_TEXT_TEXTVQA_OCR,
                             N_TEXT_TEXTVQA_NOOCR, N_TEXT_POPE, N_TEXT_SQA)

TAUS = [0.30, 0.45, 0.55, 0.65, 0.80]


def _latest(p):
    h = sorted(glob.glob(p)); return h[-1] if h else None


def _preds(run, key="predictions"):
    d = _latest(run)
    if not d:
        return None
    for fn in ["per_sample_scores.json", "per_sample.json", "predictions.json"]:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            raw = json.load(open(p))
            return raw if isinstance(raw, list) else raw.get(key, raw)
    return None


def load_gqa():
    k144 = _preds("outputs/testdev_static_cls_attn_k144_*")
    k288 = _preds("outputs/testdev_static_cls_attn_k288_*")
    spec = _preds("outputs/testdev_speculative_tau055_*")  # has confidence, order matches
    if not (k144 and k288 and spec):
        return None
    conf = [s["confidence"] for s in spec]
    s144 = [int(is_correct(p["pred_answer"], p["answer"])) for p in k144]
    s288 = [int(is_correct(p["pred_answer"], p["answer"])) for p in k288]
    n = min(len(conf), len(s144), len(s288))
    return conf[:n], s144[:n], s288[:n], N_TEXT_TESTDEV, "GQA"


def load_textvqa(tag, ntext):
    k144d = _latest(f"outputs/textvqa_cls_attn_k144_{tag}*") or _latest(f"outputs/textvqa_cls_attn_k144_2*") if tag == "ocr" else _latest(f"outputs/textvqa_cls_attn_k144_noocr*")
    # confidence in predictions.json, soft_acc in per_sample_scores.json (same order)
    def two(d):
        pr = json.load(open(os.path.join(d, "predictions.json")))["predictions"]
        ps = json.load(open(os.path.join(d, "per_sample_scores.json")))
        return [x.get("confidence") for x in pr], [x["soft_acc"] for x in ps]
    if tag == "ocr":
        d144 = _latest("outputs/textvqa_cls_attn_k144_2*"); d288 = _latest("outputs/textvqa_cls_attn_k288_2*")
    else:
        d144 = _latest("outputs/textvqa_cls_attn_k144_noocr*"); d288 = _latest("outputs/textvqa_cls_attn_k288_noocr*")
    if not (d144 and d288):
        return None
    conf, s144 = two(d144)
    s288 = [x["soft_acc"] for x in json.load(open(os.path.join(d288, "per_sample_scores.json")))]
    n = min(len(conf), len(s144), len(s288))
    return conf[:n], s144[:n], s288[:n], ntext, f"TextVQA-{tag}"


def load_pope():
    d144 = _latest("outputs/pope_cls_attn_k144_conf_*") or _latest("outputs/pope_k144_conf_*")
    d288 = _latest("outputs/pope_cls_attn_k288_*")
    if not (d144 and d288):
        return None
    p144 = json.load(open(os.path.join(d144, "per_sample.json")))
    p288 = json.load(open(os.path.join(d288, "per_sample.json")))
    conf, s144, s288 = [], [], []
    for sub in ["random", "popular", "adversarial"]:
        a, b = p144.get(sub, []), p288.get(sub, [])
        if not a or "confidence" not in a[0]:
            return None
        for i in range(min(len(a), len(b))):
            conf.append(a[i]["confidence"]); s144.append(int(a[i]["correct"])); s288.append(int(b[i]["correct"]))
    return conf, s144, s288, N_TEXT_POPE, "POPE"


def load_sqa():
    d144 = _latest("outputs/sqa_cls_attn_k144_*"); d288 = _latest("outputs/sqa_cls_attn_k288_*")
    if not (d144 and d288):
        return None
    a = json.load(open(os.path.join(d144, "per_sample.json")))
    b = json.load(open(os.path.join(d288, "per_sample.json")))
    if not a or "confidence" not in a[0]:
        return None
    n = min(len(a), len(b))
    return [a[i]["confidence"] for i in range(n)], [int(a[i]["correct"]) for i in range(n)], \
           [int(b[i]["correct"]) for i in range(n)], N_TEXT_SQA, "SQA"


def sweep(data):
    if data is None:
        return None
    conf, s144, s288, ntext, label = data
    n = len(conf)
    f144 = flops_row(144, n_text=ntext)["fastv_full_TFLOPs"]
    f288 = flops_row(288, n_text=ntext)["fastv_full_TFLOPs"]
    fdense = flops_row(576, n_text=ntext)["fastv_full_TFLOPs"]
    acc144 = sum(s144) / n * 100
    rows = []
    print(f"\n=== {label} (n={n}) — cascade K1=144 -> K2=288 ===")
    print(f"  static K=144: {acc144:.2f}%  ({f144:.3f}T)   static K=288: {sum(s288)/n*100:.2f}% ({f288:.3f}T)")
    print(f"  {'tau':>5} {'acc':>7} {'rerun%':>7} {'avgK':>6} {'cascadeT':>9} {'reduction%':>10}")
    for tau in TAUS:
        rr = sum(1 for c in conf if c < tau) / n
        acc = sum(s288[i] if conf[i] < tau else s144[i] for i in range(n)) / n * 100
        casc = f144 + rr * f288
        avgk = 144 + rr * 144  # final-budget view (rerun used K288)
        red = (1 - casc / fdense) * 100
        rows.append({"tau": tau, "acc": round(acc, 2), "rerun_pct": round(rr * 100, 1),
                     "avg_k": round(avgk, 1), "cascade_TFLOPs": round(casc, 4), "reduction_pct": round(red, 1)})
        print(f"  {tau:>5} {acc:>6.2f}% {rr*100:>6.1f}% {avgk:>6.0f} {casc:>8.3f}T {red:>9.1f}%")
    return {"label": label, "static_k144_acc": round(acc144, 2),
            "static_k144_TFLOPs": f144, "static_k288_TFLOPs": f288, "sweep": rows}


def main():
    out = {}
    loaders = [load_gqa, lambda: load_textvqa("ocr", N_TEXT_TEXTVQA_OCR),
               lambda: load_textvqa("noocr", N_TEXT_TEXTVQA_NOOCR), load_pope, load_sqa]
    for ld in loaders:
        try:
            data = ld()
        except Exception as e:
            print(f"[skip] {ld}: {e}")
            data = None
        r = sweep(data)
        if r:
            out[r["label"]] = r
    with open("outputs/cascade_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[saved] outputs/cascade_sweep.json")


if __name__ == "__main__":
    main()
