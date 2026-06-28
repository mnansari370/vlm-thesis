"""
dense_pilot.py — model-agnostic dense PILOT runner core + per-dataset data adapters.

Phase-3 dense pilots (n=200, dense / budget_pct=100, bs=1, greedy, max_new_tokens=64) for
{LLaVA-1.5-7B, Qwen-2.5-VL-7B} × {GQA, TextVQA, DocVQA, VQAv2}, written through the
final-scope schema (output_writer) and checked by the fairness gate (schema_validator),
with per-sample token counts + per-sample-averaged FLOPs (token_flops).

DESIGN — reuse, don't rewrite:
  * This module only does DATA (resolve manifest id -> image/question/gold), SCORING (the
    existing canonical scorers), and OUTPUT (final-scope schema + validation). It contains
    NO model/generation code.
  * The actual generation is supplied by a `ModelRunner` (see scripts/final_scope/run_dense_pilot.py)
    that wraps the already-validated paths: LLaVA → StaticPrunedLlava(method="none", K=576);
    Qwen → QwenPruner.generate(selector="full"). Those are not modified.

Heavy / optional deps (PIL, datasets) are imported lazily inside functions so this module
imports under a plain CPU Python for `compileall` and the self-checks.

Outputs (docs/DENSE_PROTOCOL.md §12), basename is variant-aware so prompt variants do not collide:
  results/final_scope/{model}/{dataset}/{basename}.jsonl   (per-sample, §6)
  results/final_scope/{model}/{dataset}/{basename}.json    (aggregate, §7)
  basename: GQA/VQAv2 = dense_pilot_n{n}; TextVQA = dense_pilot_n{n}_ocr_{on,off};
            DocVQA = dense_pilot_n{n}_instruction_{on,off}  (see default_basename)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.final_scope.sample_ids import load_manifest
from src.final_scope.output_writer import (
    per_sample_record,
    aggregate_record,
    write_per_sample_jsonl,
    write_aggregate_json,
)
from src.final_scope.schema_validator import fairness_gate
from src.final_scope import token_flops as tf

PILOT_N = 200
MAX_NEW_TOKENS = 64
INSTRUCTION = "Answer the question using a single word or phrase."


def describe_prompt(dataset: str, *, use_ocr: bool = True, instruction: bool = True) -> str:
    """Human-readable description of the actual prompt MODE for the aggregate (DENSE_PROTOCOL §2)."""
    if dataset in ("gqa", "vqav2"):
        return "raw question + single-word instruction"
    if dataset == "textvqa":
        return ("cached LLaVA TextVQA OCR prompt (question + OCR block + instruction) from jsonl"
                if use_ocr else "question + single-word instruction")
    if dataset == "docvqa":
        return "raw question + single-word instruction" if instruction else "raw question only"
    raise ValueError(f"unknown dataset {dataset!r}")


def default_basename(dataset: str, n: int, *, use_ocr: bool = True, instruction: bool = True) -> str:
    """Variant-aware output basename so prompt variants do NOT overwrite each other.

      GQA / VQAv2 : dense_pilot_n{n}
      TextVQA     : dense_pilot_n{n}_ocr_on   | dense_pilot_n{n}_ocr_off
      DocVQA      : dense_pilot_n{n}_instruction_on | dense_pilot_n{n}_instruction_off
    """
    base = f"dense_pilot_n{n}"
    if dataset == "textvqa":
        return f"{base}_ocr_{'on' if use_ocr else 'off'}"
    if dataset == "docvqa":
        return f"{base}_instruction_{'on' if instruction else 'off'}"
    return base
MANIFEST_DIR = "configs/final_scope/sample_ids"
OUT_ROOT = "results/final_scope"

DATASETS = ("gqa", "textvqa", "docvqa", "vqav2")
MODELS = ("llava15", "qwen25vl7b")


# ── per-sample data record passed to the model runner ─────────────────────────

@dataclass
class Sample:
    sample_id: Any            # the manifest id (questionId / question_id / row_index)
    image_id: Any            # dataset media/document id (image id / docId)
    image: Any               # PIL.Image (RGB), loaded lazily by the adapter
    question: str            # RAW question (for the record + scoring)
    model_input_text: str    # text fed to the model (raw question, or full OCR text for TextVQA)
    gold: Any                # gold answer(s) for scoring
    add_instruction: bool    # whether the model should append the single-word instruction


# ── dataset adapters ───────────────────────────────────────────────────────────

class DatasetAdapter:
    """Resolves manifest ids -> Sample, and scores predictions with the canonical scorer."""
    name: str = ""
    split: str = ""
    metric: str = ""

    def sample(self, sample_id: Any) -> Sample:  # pragma: no cover - overridden
        raise NotImplementedError

    def score(self, pred_raw: str, s: Sample) -> Tuple[str, float]:  # pragma: no cover
        raise NotImplementedError


def _pil(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


class GQAAdapter(DatasetAdapter):
    name, split, metric = "gqa", "testdev_balanced", "gqa_exact"
    QUESTIONS = "data/gqa/testdev_balanced_questions.json"
    IMAGE_DIR = "data/gqa/images/images"

    def __init__(self):
        with open(self.QUESTIONS) as f:
            self._q = json.load(f)

    def sample(self, sample_id: Any) -> Sample:
        rec = self._q[str(sample_id)]
        iid = rec["imageId"]
        q = rec["question"]
        return Sample(sample_id=str(sample_id), image_id=iid,
                      image=_pil(os.path.join(self.IMAGE_DIR, f"{iid}.jpg")),
                      question=q, model_input_text=q, gold=rec.get("answer", ""),
                      add_instruction=True)

    def score(self, pred_raw: str, s: Sample) -> Tuple[str, float]:
        from src.metrics.official_score import normalize, is_correct
        return normalize(pred_raw), (1.0 if is_correct(pred_raw, s.gold) else 0.0)


class VQAv2Adapter(DatasetAdapter):
    name, split, metric = "vqav2", "validation", "vqa_consensus"
    QUESTIONS = "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
    ANNOTATIONS = "data/vqav2/v2_mscoco_val2014_annotations.json"
    IMAGE_DIR = "data/vqav2/val2014"

    def __init__(self):
        with open(self.QUESTIONS) as f:
            self._q = {int(x["question_id"]): x for x in json.load(f)["questions"]}
        with open(self.ANNOTATIONS) as f:
            self._a = {int(x["question_id"]): x for x in json.load(f)["annotations"]}

    def sample(self, sample_id: Any) -> Sample:
        qid = int(sample_id)
        q = self._q[qid]
        ann = self._a[qid]
        iid = q["image_id"]
        path = os.path.join(self.IMAGE_DIR, f"COCO_val2014_{iid:012d}.jpg")
        raw = [a["answer"] for a in ann.get("answers", [])]
        return Sample(sample_id=qid, image_id=iid, image=_pil(path),
                      question=q["question"], model_input_text=q["question"],
                      gold=raw, add_instruction=True)

    def score(self, pred_raw: str, s: Sample) -> Tuple[str, float]:
        from src.data.vqav2.vqav2_answers import normalize_answer
        pn = normalize_answer(pred_raw)
        matches = sum(1 for a in s.gold if normalize_answer(a) == pn)
        return pn, min(1.0, matches / 3.0)


class TextVQAAdapter(DatasetAdapter):
    name, split, metric = "textvqa", "val", "m4c_soft"
    JSONL = "data/textvqa/llava_textvqa_val_v051_ocr.jsonl"
    IMAGE_DIR = "data/textvqa/train_images"
    ANNOTATION = "data/textvqa/TextVQA_0.5.1_val.json"

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        with open(self.JSONL) as f:
            self._rows = [json.loads(l) for l in f if l.strip()]
        self._ev = None
        self._gt = None  # lazy (m4c import + annotation load)

    def _ensure_scorer(self):
        if self._ev is None:
            import sys
            sys.path.insert(0, "src/metrics")
            from m4c_evaluator import TextVQAAccuracyEvaluator  # type: ignore
            self._ev = TextVQAAccuracyEvaluator()
            data = json.load(open(self.ANNOTATION))["data"]
            self._gt = {(a["image_id"], a["question"].lower()): a["answers"] for a in data}

    def sample(self, sample_id: Any) -> Sample:
        self._ensure_scorer()  # also loads the gold map so the record carries its gold
        r = self._rows[int(sample_id)]  # row_index
        question = r["text"].split("\n")[0]
        # OCR (primary): feed the full jsonl text (question + OCR + instruction).
        # no-OCR: feed question + instruction only. The instruction is already in the text either way.
        model_text = r["text"] if self.use_ocr else f"{question}\n{INSTRUCTION}"
        gold = self._gt.get((r["question_id"], question.lower()), [])
        return Sample(sample_id=int(sample_id), image_id=r["question_id"],
                      image=_pil(os.path.join(self.IMAGE_DIR, r["image"])),
                      question=question, model_input_text=model_text,
                      gold=gold, add_instruction=False)

    def score(self, pred_raw: str, s: Sample) -> Tuple[str, float]:
        self._ensure_scorer()
        pred = self._ev.answer_processor(pred_raw)
        if not s.gold:
            return pred, 0.0
        scores = self._ev._compute_answer_scores(s.gold)
        return pred, float(scores.get(pred, 0.0))


class DocVQAAdapter(DatasetAdapter):
    name, split, metric = "docvqa", "validation", "anls"

    def __init__(self, instruction: bool = True):
        self.instruction = instruction
        from datasets import load_dataset  # lazy (HF datasets)
        self._ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        self._qid_to_idx = {str(q): i for i, q in enumerate(self._ds["questionId"])}

    def sample(self, sample_id: Any) -> Sample:
        idx = self._qid_to_idx[str(sample_id)]
        row = self._ds[idx]
        return Sample(sample_id=str(sample_id), image_id=str(row.get("docId", sample_id)),
                      image=row["image"].convert("RGB"), question=row["question"],
                      model_input_text=row["question"],
                      gold=[str(a) for a in row.get("answers", [])],
                      add_instruction=self.instruction)

    def score(self, pred_raw: str, s: Sample) -> Tuple[str, float]:
        from src.metrics.docvqa_score import anls
        return pred_raw.strip().lower(), float(anls(pred_raw, s.gold))


def get_adapter(dataset: str, *, use_ocr: bool = True, instruction: bool = True) -> DatasetAdapter:
    if dataset == "gqa":
        return GQAAdapter()
    if dataset == "vqav2":
        return VQAv2Adapter()
    if dataset == "textvqa":
        return TextVQAAdapter(use_ocr=use_ocr)
    if dataset == "docvqa":
        return DocVQAAdapter(instruction=instruction)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


# ── pilot core ─────────────────────────────────────────────────────────────────

@dataclass
class PilotResult:
    out_jsonl: str
    out_json: str
    n: int
    score_pct: float
    gate_ok: bool
    gate_issues: List[str] = field(default_factory=list)


def run_pilot(
    *,
    model_name: str,
    dataset: str,
    generate_and_count: Callable[[Sample], Tuple[str, int, int, str]],
    n: int = PILOT_N,
    max_new_tokens: int = MAX_NEW_TOKENS,
    image_pad: Optional[bool] = None,
    max_pixels: Optional[int] = None,
    use_ocr: bool = True,
    instruction: bool = True,
    model_instruction_suffix: str = "",
    decode_notes: str = "",
    manifest_dir: str = MANIFEST_DIR,
    out_root: str = OUT_ROOT,
    out_basename: Optional[str] = None,
    log_every: int = 50,
) -> PilotResult:
    """
    Run a dense pilot for ONE (model, dataset) cell.

    `generate_and_count(sample) -> (pred_raw, n_visual_tokens, n_text_tokens, prompt)` is the
    model runner (GPU) supplied by the caller — it returns the EXACT prompt the model received
    (incl. the model-specific instruction suffix) so the per-sample `prompt` is faithful.
    Everything else (data, scoring, schema, FLOPs, validation, writing) is handled here. This
    function performs the GPU loop — do not call it until a run is approved.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if out_basename is None:
        out_basename = default_basename(dataset, n, use_ocr=use_ocr, instruction=instruction)
    manifest = load_manifest(os.path.join(manifest_dir, f"{dataset}.json"))
    ids = manifest.ids[:n]
    adapter = get_adapter(dataset, use_ocr=use_ocr, instruction=instruction)
    decoding = f"greedy bs=1 do_sample=False max_new_tokens={max_new_tokens}"

    records: List[Dict[str, Any]] = []
    stats: List[tf.TokenStats] = []
    for i, sid in enumerate(ids):
        s = adapter.sample(sid)
        pred_raw, n_vis, n_txt, prompt = generate_and_count(s)
        pred_norm, score = adapter.score(pred_raw, s)
        records.append(per_sample_record(
            sample_id=s.sample_id, image_id=s.image_id, dataset=dataset, model=model_name,
            method="dense", budget_pct=100, question=s.question, prompt=prompt,
            pred_raw=str(pred_raw).strip(), pred_norm=str(pred_norm), gold=s.gold,
            per_sample_score=float(score), n_visual_tokens=int(n_vis), n_text_tokens=int(n_txt),
        ))
        stats.append(tf.TokenStats(int(n_vis), int(n_txt)))
        if (i + 1) % log_every == 0:
            avg = sum(r["per_sample_score"] for r in records) / len(records)
            print(f"  [{dataset}/{model_name}] {i+1}/{len(ids)}  running_score={avg*100:.2f}%", flush=True)

    summ = tf.summarize_tokens_and_flops(model_name, stats)
    score_pct = round(sum(r["per_sample_score"] for r in records) / max(len(records), 1) * 100, 2)
    expected_vis = 576 if model_name == "llava15" else None

    agg = aggregate_record(
        model=model_name, dataset=dataset, split=adapter.split, method="dense", budget_pct=100,
        n=len(records), metric=adapter.metric, score_pct=score_pct,
        visual_tokens={"avg": summ["visual_tokens_avg"], "max": summ["visual_tokens_max"]},
        text_tokens_avg=summ["text_tokens_avg"], seq_len_avg=summ["seq_len_avg"],
        flops_prefill_TFLOPs_avg=summ["flops_prefill_TFLOPs_avg"],
        flops_convention=summ["flops_convention"],
        token_reduction_pct=0.0, flop_reduction_pct=0.0,
        prompt_template=describe_prompt(dataset, use_ocr=use_ocr, instruction=instruction),
        max_new_tokens=max_new_tokens, decoding=decoding,
        image_pad=image_pad, sample_ids_path=os.path.join(manifest_dir, f"{dataset}.json"),
        sample_ids_sha256=manifest.sha256,
        # prompt/decode provenance as TOP-LEVEL aggregate fields (faithful to what the model saw)
        protocol_instruction=INSTRUCTION,
        model_instruction_suffix=model_instruction_suffix,
        decode_notes=decode_notes,
        # run knobs only
        extra_pilot={"n_requested": n, "use_ocr": use_ocr if dataset == "textvqa" else None,
                     "instruction": instruction if dataset == "docvqa" else None,
                     "max_pixels": max_pixels},
    )

    gate = fairness_gate(records, agg, manifest_ids=ids, manifest_sha=manifest.sha256,
                         expected_visual_tokens=expected_vis)

    out_dir = os.path.join(out_root, model_name, dataset)
    out_jsonl = os.path.join(out_dir, f"{out_basename}.jsonl")
    out_json = os.path.join(out_dir, f"{out_basename}.json")
    write_per_sample_jsonl(out_jsonl, records, validate=True)
    write_aggregate_json(out_json, {**agg, "fairness_gate": gate}, validate=True)

    print(f"[{dataset}/{model_name}] score={score_pct}%  vis~{summ['visual_tokens_avg']:.0f} "
          f"text~{summ['text_tokens_avg']:.0f}  FLOPs~{summ['flops_prefill_TFLOPs_avg']:.3f}T  "
          f"gate={'OK' if gate['ok'] else 'ISSUES'}", flush=True)
    if not gate["ok"]:
        for p in gate["issues"][:8]:
            print(f"    [gate] {p}", flush=True)
    return PilotResult(out_jsonl, out_json, len(records), score_pct, gate["ok"], gate["issues"])
