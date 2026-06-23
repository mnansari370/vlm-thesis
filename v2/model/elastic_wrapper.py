"""
elastic_wrapper.py — Stage-1 elastic backbone for v2.

LLaVA-1.5-7B with:
  - vision tower (CLIP)            FROZEN
  - base LLM weights              FROZEN
  - LoRA adapters on the LLM      TRAINABLE   (peft; applied to the LLM ONLY, not CLIP)
  - multi-modal projector         TRAINABLE

Pruning (this stage): training-free CLS->patch attention selection (VisionZip "dominant"
scoring at CLIP layer vision_feature_layer), top-K, PHYSICAL removal before the LLM
(prune-before-LLM, all 32 layers see K visual tokens). K is sampled RANDOMLY each step
from a ladder, so one set of weights becomes good at every budget ("elastic").

Objective: teacher-forced next-token LM loss on the ANSWER tokens only (generation-style;
prompt + visual tokens are masked with -100). No answer classification anywhere.

This stage uses the FIXED CLS selector (no learned question-conditioned selector yet — that
is Stage 2). It exists to (a) remove v1's "model collapses below K=144" wall and (b) provide
the dense ceiling + the entire static frontier + the pilot measurement from ONE model.

NOTE: self-contained — does not import from GQA/ or VQA_V2/ (v1 stays frozen).
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
N_PATCHES = 576
ELASTIC_K_LADDER = (32, 64, 128, 256, 576)
# CLIP ViT-L/14 image mean (LLaVA-1.5 expand2square background), as 0-255 RGB.
_CLIP_MEAN_RGB = (122, 116, 104)
# Llama attention + MLP projections (NOT CLIP — peft is applied to the LLM submodule only).
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_INSTR = "Answer the question using a single word or phrase."


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class ElasticPrunedLlava(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: Tuple[str, ...] = _LORA_TARGETS,
        dtype: str = "bfloat16",
        image_pad: bool = True,
        k_ladder: Tuple[int, ...] = ELASTIC_K_LADDER,
        seed: int = 42,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_pad = image_pad
        self.k_ladder = tuple(k_ladder)
        self._rng = random.Random(seed)
        self.compute_dtype = _torch_dtype(dtype)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Elastic] loading {model_name} ({dtype}) ...", flush=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.compute_dtype, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.dev)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"   # for the batched processor call
        vc = getattr(self.model.config, "vision_config", None)
        self.processor.patch_size = getattr(vc, "patch_size", 14)
        self.processor.vision_feature_select_strategy = "default"
        self.processor.num_additional_image_tokens = 0

        self._image_token_id = int(getattr(self.model.config, "image_token_index", 32000))
        self._vis_layer = int(getattr(self.model.config, "vision_feature_layer", -2))
        self._eos_id = self.processor.tokenizer.eos_token_id
        self._pad_id = self.processor.tokenizer.pad_token_id or 0

        # sub-modules
        self._vt = self.model.vision_tower
        self._proj = self.model.multi_modal_projector
        self._lm = self.model.language_model   # LlamaForCausalLM

        # 1) freeze EVERYTHING first
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) LoRA on the LLM only (peft injects adapters in-place into self._lm)
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_targets), bias="none", task_type="CAUSAL_LM",
        )
        self.peft_lm = get_peft_model(self._lm, lora_cfg)   # LoRA params -> requires_grad=True

        # 3) projector trainable
        for p in self._proj.parameters():
            p.requires_grad = True

        self._embed = self._lm.get_input_embeddings()   # base embed_tokens (untouched by LoRA)

        # 4) gradient checkpointing (memory ↓ for larger batches).
        # Safe here because inputs_embeds requires grad via the trainable projector
        # (no enable_input_require_grads needed); use_reentrant=False preserves the
        # RNG so LoRA dropout matches between forward and recompute (exact grads).
        self.grad_checkpointing = bool(gradient_checkpointing)
        if self.grad_checkpointing:
            self._lm.config.use_cache = False
            self._lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[Elastic] gradient checkpointing ON (use_reentrant=False, use_cache=False)",
                  flush=True)

        s = self.param_summary()
        print(f"[Elastic] trainable: {s['trainable']:,} / {s['total']:,} "
              f"({100*s['trainable']/s['total']:.3f}%)  "
              f"[lora={s['lora']:,}  projector={s['projector']:,}]", flush=True)

    # ── param audit ───────────────────────────────────────────────────────────
    def param_summary(self) -> Dict[str, int]:
        total = lora = proj = other_train = 0
        for n, p in self.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                if "lora_" in n:
                    lora += p.numel()
                elif "multi_modal_projector" in n:
                    proj += p.numel()
                else:
                    other_train += p.numel()
        return {"total": total, "trainable": lora + proj + other_train,
                "lora": lora, "projector": proj, "other_trainable": other_train}

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # ── image / scoring helpers ───────────────────────────────────────────────
    @staticmethod
    def _pad_to_square(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        out = Image.new("RGB", (side, side), _CLIP_MEAN_RGB)
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out

    def sample_k(self) -> int:
        return self._rng.choice(self.k_ladder)

    # ── shared train/eval encoder: pad → prompt → CLIP → CLS-attn top-K → project
    def _encode_and_select(
        self, images: List[Image.Image], questions: List[str], K: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per-sample (prefix_emb, vis_emb, suffix_emb). Used by BOTH _build_batch
        (training, appends the answer) and generate_answers (eval, generates) so the
        prompt format + CLS top-K selection are GUARANTEED identical between train and
        eval. Not @no_grad: training keeps grad through the trainable projector; the
        eval caller wraps it in torch.no_grad()."""
        B = len(images)
        if self.image_pad:
            images = [self._pad_to_square(im) for im in images]

        # Use the question text AS-IS. The LLaVA mix already formats every sample per task:
        # short-answer tasks (VQAv2/GQA/OCR, ~39%) carry the "...single word or phrase."
        # instruction; caption/region/conversation tasks (~23% long answers) do NOT. At EVAL,
        # the caller appends the benchmark's own instruction. (The first full training appended
        # _INSTR to ALL samples — including the long-answer ones — which taught the model the
        # instruction was meaningless, so it answered verbosely. That bug is fixed here.)
        convs = [[{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q.strip()}]}]
            for q in questions]
        prompts = [self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                   for c in convs]
        inp = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inp = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inp.items()}

        pv = inp["pixel_values"].to(self.compute_dtype)
        clip_out = self._vt(pv, output_attentions=True, output_hidden_states=True)
        raw_feat = clip_out.hidden_states[self._vis_layer][:, 1:, :]            # [B,576,Dclip]
        scores = clip_out.attentions[self._vis_layer][:, :, 0, 1:].mean(dim=1).float()  # [B,576]
        projected = self._proj(raw_feat)                                       # [B,576,Dlm]

        pieces = []
        for i in range(B):
            ids_i = inp["input_ids"][i][inp["attention_mask"][i].bool()]       # strip left-pad
            img_pos = (ids_i == self._image_token_id).nonzero(as_tuple=True)[0]
            img_start = int(img_pos[0]); img_end = int(img_pos[-1]) + 1
            prefix_ids = ids_i[:img_start]
            suffix_ids = ids_i[img_end:]                                       # question + "ASSISTANT:"

            keep = scores[i].topk(K).indices.sort().values                     # spatial order
            vis_emb = projected[i, keep, :]                                    # [K,Dlm]

            prefix_emb = self._embed(prefix_ids)
            suffix_emb = self._embed(suffix_ids)
            pieces.append((prefix_emb, vis_emb.to(prefix_emb.dtype), suffix_emb))
        return pieces

    # ── build one training batch (prune at K, teacher-force the answer) ────────
    def _build_batch(
        self, images: List[Image.Image], questions: List[str], answers: List[str], K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        pieces = self._encode_and_select(images, questions, K)

        seqs, labels_list, masks = [], [], []
        for i, (prefix_emb, vis_emb, suffix_emb) in enumerate(pieces):
            ans_ids = self.processor.tokenizer(
                answers[i].strip(), add_special_tokens=False).input_ids
            ans_ids = torch.tensor(ans_ids + [self._eos_id], dtype=torch.long, device=self.dev)
            ans_emb = self._embed(ans_ids)
            seq = torch.cat([prefix_emb, vis_emb, suffix_emb, ans_emb], dim=0)

            n_pre = prefix_emb.size(0) + vis_emb.size(0) + suffix_emb.size(0)
            lab = torch.cat([torch.full((n_pre,), -100, dtype=torch.long, device=self.dev),
                             ans_ids], dim=0)
            seqs.append(seq); labels_list.append(lab)
            masks.append(torch.ones(seq.size(0), dtype=torch.long, device=self.dev))

        # right-pad to max length (training convention; pad labels -100, mask 0)
        Lmax = max(s.size(0) for s in seqs)
        emb_b, lab_b, mask_b = [], [], []
        for seq, lab, m in zip(seqs, labels_list, masks):
            pad = Lmax - seq.size(0)
            if pad > 0:
                pad_emb = self._embed(torch.full((pad,), self._pad_id, dtype=torch.long, device=self.dev))
                seq = torch.cat([seq, pad_emb.to(seq.dtype)], dim=0)
                lab = torch.cat([lab, torch.full((pad,), -100, dtype=torch.long, device=self.dev)], dim=0)
                m = torch.cat([m, torch.zeros(pad, dtype=torch.long, device=self.dev)], dim=0)
            emb_b.append(seq); lab_b.append(lab); mask_b.append(m)

        info = {"K": K, "n_visual": K, "seq_len": Lmax, "batch": len(pieces)}
        return (torch.stack(emb_b), torch.stack(mask_b), torch.stack(lab_b), info)

    # ── training forward → loss ────────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any], K: Optional[int] = None) -> Dict[str, Any]:
        if K is None:
            K = self.sample_k()
        emb, mask, labels, info = self._build_batch(
            batch["images"], batch["questions"], batch["answers"], K)
        out = self._lm(inputs_embeds=emb, attention_mask=mask, labels=labels)
        return {"loss": out.loss, "info": info}

    # ── eval: prune at K, then GENERATE the answer (greedy) ────────────────────
    @torch.no_grad()
    def generate_answers(
        self, images: List[Image.Image], questions: List[str], K: int,
        max_new_tokens: int = 20,
    ) -> List[str]:
        """Generation-protocol inference. Mirrors training's prune+splice exactly
        (via _encode_and_select), then greedily generates instead of teacher-forcing.
        Generates per sample (bs=1) — matches the locked honest protocol. With
        inputs_embeds, generate() returns only the new tokens."""
        self.eval()
        self._lm.config.use_cache = True
        pieces = self._encode_and_select(images, questions, K)
        answers: List[str] = []
        for (prefix_emb, vis_emb, suffix_emb) in pieces:
            seq = torch.cat([prefix_emb, vis_emb, suffix_emb], dim=0).unsqueeze(0)  # [1,L,D]
            mask = torch.ones(1, seq.size(1), dtype=torch.long, device=self.dev)
            out = self._lm.generate(
                inputs_embeds=seq, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=self._eos_id, use_cache=True,
            )
            text = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
            answers.append(text.strip())
        return answers

    # ── load a trained Stage-1 checkpoint for eval ─────────────────────────────
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, gradient_checkpointing: bool = False):
        """Rebuild the model from the config stored in the checkpoint, then load the
        trained LoRA adapter + projector. Returns an eval-ready model."""
        ck = torch.load(ckpt_path, map_location="cpu")
        mc = ck["config"]["model"]
        m = cls(
            model_name=mc.get("model_name", MODEL_NAME),
            lora_rank=int(mc.get("lora_rank", 16)), lora_alpha=int(mc.get("lora_alpha", 32)),
            lora_dropout=float(mc.get("lora_dropout", 0.05)), dtype=str(mc.get("dtype", "bfloat16")),
            image_pad=bool(mc.get("image_pad", True)),
            k_ladder=tuple(mc.get("k_ladder", ELASTIC_K_LADDER)),
            gradient_checkpointing=gradient_checkpointing,
        )
        set_peft_model_state_dict(m.peft_lm, ck["lora_state"])
        m._proj.load_state_dict(ck["projector_state"])
        m.eval()
        print(f"[Elastic] loaded checkpoint {ckpt_path} (trained step {ck.get('step')})", flush=True)
        return m
