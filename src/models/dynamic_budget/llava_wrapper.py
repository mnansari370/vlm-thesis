import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

from .answer_head import AnswerHeadMLP
from .token_selector import DynamicTokenSelector


def _torch_dtype_from_string(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None

    name = str(dtype_name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported torch dtype string: {dtype_name}")


class LlavaDynamicVQAModel(nn.Module):
    """
    Dynamic-pruning LLaVA-style VQA model.

    This version supports:
    - CLS-only dynamic pipeline check
    - CLS-prior + learned question-conditioned correction
    - fixed or learned budget
    - soft training / hard validation
    - optional initialization of answer head from static K288 checkpoint
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.dataset_cfg = cfg["dataset"]
        self.model_cfg = cfg["model"]
        self.token_cfg = cfg["token_selection"]
        self.training_cfg = cfg["training"]

        self.answer_mode = self.dataset_cfg["answer_mode"]
        self.use_answer_head = bool(self.model_cfg.get("use_answer_head", False))
        self.training_mode = self.training_cfg.get("mode", "train_dynamic_answer_head")

        if not (self.answer_mode == "classification" and self.use_answer_head):
            raise ValueError(
                "LlavaDynamicVQAModel currently supports classification mode with answer head."
            )

        self.model = self._build_backbone()
        print(f"[Info] Backbone device after load: {self._get_model_device()}", flush=True)

        self.processor = self._build_processor()

        self.hidden_size = self._infer_hidden_size()
        self.visual_hidden_size = self._infer_visual_hidden_size()
        self.num_visual_tokens = self._estimate_num_visual_tokens()

        self.id_to_answer = self._load_id_to_answer_if_available()

        self.token_selector = self._build_token_selector()
        self.answer_head = self._build_answer_head_if_needed()

        self._freeze_requested_modules()
        self._move_trainable_modules_to_backbone_device()
        self._load_optional_answer_head_init()

    def _build_backbone(self):
        if LlavaForConditionalGeneration is None:
            raise ImportError(
                "LlavaForConditionalGeneration is not available in this transformers installation."
            )

        model_name = self.model_cfg["pretrained_model_name_or_path"]
        low_cpu_mem_usage = bool(self.model_cfg.get("low_cpu_mem_usage", True))
        load_in_4bit = bool(self.model_cfg.get("load_in_4bit", False))
        attn_implementation = self.model_cfg.get("attn_implementation", None)
        torch_dtype = _torch_dtype_from_string(self.model_cfg.get("torch_dtype", None))

        quantization_config = None
        if load_in_4bit:
            compute_dtype = torch_dtype if torch_dtype is not None else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model_kwargs = {
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }

        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        if load_in_4bit:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if not load_in_4bit:
            use_cuda = bool(self.cfg.get("system", {}).get("use_cuda", True))
            if use_cuda and torch.cuda.is_available():
                model = model.to("cuda")

        model.config.use_cache = False
        return model

    def _build_processor(self):
        processor_name = self.model_cfg["processor_name"]
        processor = AutoProcessor.from_pretrained(processor_name)

        if hasattr(processor, "tokenizer"):
            padding_side = self.model_cfg.get("padding_side")
            if padding_side is not None:
                processor.tokenizer.padding_side = padding_side

        vision_config = getattr(self.model.config, "vision_config", None)
        patch_size = getattr(vision_config, "patch_size", 14)
        processor.patch_size = getattr(processor, "patch_size", patch_size)

        processor.vision_feature_select_strategy = self.model_cfg.get(
            "vision_feature_select_strategy",
            getattr(self.model.config, "vision_feature_select_strategy", "default"),
        )

        processor.num_additional_image_tokens = 0
        if processor.vision_feature_select_strategy == "full":
            processor.num_additional_image_tokens = 1

        return processor

    def _get_vision_tower(self):
        vision_module = getattr(self.model, "vision_tower", None)
        if vision_module is None and hasattr(self.model, "model"):
            vision_module = getattr(self.model.model, "vision_tower", None)

        if vision_module is None:
            raise ValueError("Could not locate vision tower inside LLaVA model.")
        return vision_module

    def _get_projector(self):
        projector_module = getattr(self.model, "multi_modal_projector", None)
        if projector_module is None and hasattr(self.model, "model"):
            projector_module = getattr(self.model.model, "multi_modal_projector", None)

        if projector_module is None:
            raise ValueError("Could not locate multimodal projector inside LLaVA model.")
        return projector_module

    def _get_language_model(self):
        lm_module = getattr(self.model, "language_model", None)
        if lm_module is None and hasattr(self.model, "model"):
            lm_module = getattr(self.model.model, "language_model", None)

        if lm_module is None:
            raise ValueError("Could not locate language model inside LLaVA model.")
        return lm_module

    def _freeze_module(self, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_requested_modules(self) -> None:
        vision_module = self._get_vision_tower()
        projector_module = self._get_projector()
        lm_module = self._get_language_model()

        if self.model_cfg.get("freeze_vision_encoder", False):
            self._freeze_module(vision_module)

        if self.model_cfg.get("freeze_projector", False):
            self._freeze_module(projector_module)

        if self.model_cfg.get("freeze_llm", False):
            self._freeze_module(lm_module)

    def _get_model_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_trainable_modules_to_backbone_device(self) -> None:
        device = self._get_model_device()

        if self.token_selector is not None:
            self.token_selector.to(device)

        if self.answer_head is not None:
            self.answer_head.to(device)

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            return int(self.model.config.text_config.hidden_size)

        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)

        lm_module = self._get_language_model()
        if hasattr(lm_module, "config") and hasattr(lm_module.config, "hidden_size"):
            return int(lm_module.config.hidden_size)

        raise ValueError("Could not infer hidden size from LLaVA backbone.")

    def _infer_visual_hidden_size(self) -> int:
        vision_config = getattr(self.model.config, "vision_config", None)
        if vision_config is not None and hasattr(vision_config, "hidden_size"):
            return int(vision_config.hidden_size)

        vision_tower = self._get_vision_tower()
        if hasattr(vision_tower, "config") and hasattr(vision_tower.config, "hidden_size"):
            return int(vision_tower.config.hidden_size)

        raise ValueError("Could not infer visual hidden size.")

    def _estimate_num_visual_tokens(self) -> int:
        image_size = int(self.dataset_cfg["image_size"])

        patch_size = None
        if hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "patch_size"):
            patch_size = int(self.model.config.vision_config.patch_size)

        if patch_size is None:
            raise ValueError("Could not infer vision patch size from model config.")

        patches_per_side = image_size // patch_size
        num_patches = patches_per_side * patches_per_side

        strategy = self.model_cfg.get(
            "vision_feature_select_strategy",
            getattr(self.model.config, "vision_feature_select_strategy", "default"),
        )

        if strategy == "default":
            return num_patches
        if strategy == "full":
            return num_patches + 1

        raise ValueError(f"Unsupported vision_feature_select_strategy: {strategy}")

    def _build_token_selector(self) -> DynamicTokenSelector:
        return DynamicTokenSelector(
            visual_dim=self.visual_hidden_size,
            question_dim=self.hidden_size,
            shared_dim=int(self.token_cfg.get("shared_dim", 512)),
            scorer_hidden_dim=int(self.token_cfg.get("scorer_hidden_dim", 256)),
            budget_hidden_dim=int(self.token_cfg.get("budget_hidden_dim", 256)),
            dropout=float(self.token_cfg.get("dropout", 0.1)),
            min_keep_tokens=int(self.token_cfg.get("min_keep_tokens", 64)),
            max_keep_tokens=int(self.token_cfg.get("max_keep_tokens", self.num_visual_tokens)),
            num_visual_tokens=self.num_visual_tokens,
            train_selection_mode=str(self.token_cfg.get("train_selection_mode", "soft")),
            eval_selection_mode=str(self.token_cfg.get("eval_selection_mode", "hard")),
            scoring_mode=str(self.token_cfg.get("scoring_mode", "learned_only")),
            budget_strategy=str(self.token_cfg.get("budget_strategy", "learned")),
            fixed_keep_ratio=float(self.token_cfg.get("fixed_keep_ratio", 0.5)),
            cls_alpha=float(self.token_cfg.get("cls_alpha", 0.2)),
            soft_temperature=float(self.token_cfg.get("soft_temperature", 0.10)),
            question_type_emb_dim=int(self.token_cfg.get("question_type_emb_dim", 0)),
        )

    def _load_id_to_answer_if_available(self) -> Optional[Dict[int, str]]:
        answer_vocab_path = self.dataset_cfg.get("answer_vocab_path", None)
        if answer_vocab_path is None:
            return None

        with open(answer_vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "id_to_answer" in data:
            return {int(k): v for k, v in data["id_to_answer"].items()}

        if "answer_to_id" in data:
            return {int(idx): ans for ans, idx in data["answer_to_id"].items()}

        return {int(idx): ans for ans, idx in data.items()}

    def _infer_answer_vocab_size(self) -> int:
        cfg_vocab_size = self.model_cfg.get("answer_vocab_size", None)
        if cfg_vocab_size is not None:
            return int(cfg_vocab_size)

        if self.id_to_answer is None:
            raise ValueError(
                "answer_vocab_size is null and answer vocab file could not be loaded."
            )

        return len(self.id_to_answer)

    def _build_answer_head_if_needed(self) -> Optional[nn.Module]:
        if not self.use_answer_head:
            return None

        answer_head_type = self.model_cfg.get("answer_head_type", None)
        if answer_head_type != "mlp":
            raise ValueError(
                f"Currently only answer_head_type='mlp' is supported, got {answer_head_type}"
            )

        hidden_dim = int(self.model_cfg["answer_head_hidden_dim"])
        output_dim = self._infer_answer_vocab_size()
        dropout = float(self.model_cfg.get("answer_head_dropout", 0.1))
        train_dtype = self.model_cfg.get("answer_head_train_dtype", "float32")

        return AnswerHeadMLP(
            input_dim=self.hidden_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            train_dtype=train_dtype,
        )

    def _load_optional_answer_head_init(self) -> None:
        ckpt_path = self.model_cfg.get("init_answer_head_from", None)
        if not ckpt_path:
            return

        print(f"[Info] Loading answer head initialization from: {ckpt_path}", flush=True)

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "answer_head_state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint {ckpt_path} does not contain answer_head_state_dict."
            )

        missing, unexpected = self.answer_head.load_state_dict(
            ckpt["answer_head_state_dict"],
            strict=False,
        )

        if missing:
            print(f"[Warn] Missing answer head keys: {missing}", flush=True)
        if unexpected:
            print(f"[Warn] Unexpected answer head keys: {unexpected}", flush=True)

        self.answer_head.to(self._get_model_device())
        print("[Info] Answer head initialization loaded.", flush=True)

    def _build_conversations(self, questions: List[str]) -> List[List[Dict[str, Any]]]:
        conversations = []
        for question in questions:
            prompted = question.strip() + " Answer the question using a single word or phrase."
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompted},
                        ],
                    }
                ]
            )
        return conversations

    def _prepare_inputs(
        self,
        images: List[Any],
        questions: List[str],
    ) -> Dict[str, torch.Tensor]:
        conversations = self._build_conversations(questions)

        text_prompts = [
            self.processor.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conv in conversations
        ]

        inputs = self.processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        device = self._get_model_device()
        moved_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(device)
            else:
                moved_inputs[key] = value

        return moved_inputs

    def _compute_processor_input_lengths(
        self,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return attention_mask.sum(dim=1)

    def _compute_raw_question_lengths(
        self,
        questions: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        if not hasattr(self.processor, "tokenizer") or self.processor.tokenizer is None:
            raise ValueError("Processor tokenizer is required to compute raw question lengths.")

        tokenizer = self.processor.tokenizer
        encoded = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.model_cfg.get("question_max_length", 64)),
            add_special_tokens=False,
        )

        attention_mask = encoded["attention_mask"]
        raw_lengths = attention_mask.sum(dim=1).to(device)
        return raw_lengths

    def _gather_last_valid_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        last_indices = attention_mask.sum(dim=1) - 1
        last_indices = last_indices.clamp(min=0)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_indices, :]
        return pooled

    def _map_pred_ids_to_answers(self, pred_ids: torch.Tensor) -> Optional[List[str]]:
        if self.id_to_answer is None:
            return None
        return [self.id_to_answer.get(int(idx), "") for idx in pred_ids.detach().cpu().tolist()]

    def _run_vision_encoder(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run frozen CLIP vision encoder.

        Returns:
            visual_features: [B, N, Dv]
            final_attentions: [B, H, 1+N, 1+N]
        """
        vision_tower = self._get_vision_tower()

        try:
            vision_dtype = next(vision_tower.parameters()).dtype
        except StopIteration:
            vision_dtype = pixel_values.dtype

        pixel_values = pixel_values.to(dtype=vision_dtype)

        with torch.no_grad():
            vision_outputs = vision_tower(
                pixel_values=pixel_values,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = vision_outputs.hidden_states
        attentions = vision_outputs.attentions

        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("Vision tower did not return hidden_states.")
        if attentions is None or len(attentions) == 0:
            raise ValueError("Vision tower did not return attentions.")

        feature_layer = int(self.model_cfg.get("vision_feature_layer", -2))
        selected_hidden = hidden_states[feature_layer]
        final_attentions = attentions[feature_layer]  # same layer as features

        strategy = self.model_cfg.get(
            "vision_feature_select_strategy",
            getattr(self.model.config, "vision_feature_select_strategy", "default"),
        )

        if strategy == "default":
            visual_features = selected_hidden[:, 1:, :]
        elif strategy == "full":
            visual_features = selected_hidden
        else:
            raise ValueError(f"Unsupported vision_feature_select_strategy: {strategy}")

        return visual_features, final_attentions

    def _project_visual_features(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        projector = self._get_projector()

        try:
            projector_dtype = next(projector.parameters()).dtype
        except StopIteration:
            projector_dtype = visual_features.dtype

        if visual_features.dtype != projector_dtype:
            visual_features = visual_features.to(dtype=projector_dtype)

        return projector(visual_features)

    def _strip_image_tokens_and_embed_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_index = getattr(self.model.config, "image_token_index", None)
        if image_token_index is None:
            raise ValueError("LLaVA config does not expose image_token_index.")

        lm = self._get_language_model()
        input_embedding_layer = lm.get_input_embeddings()

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0
        if pad_token_id is None:
            pad_token_id = 0

        batch_text_ids = []
        max_text_len = 0

        for i in range(input_ids.size(0)):
            valid_ids = input_ids[i][attention_mask[i].bool()]
            valid_ids = valid_ids[valid_ids != image_token_index]

            if valid_ids.numel() == 0:
                valid_ids = torch.tensor(
                    [pad_token_id],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )

            batch_text_ids.append(valid_ids)
            max_text_len = max(max_text_len, int(valid_ids.numel()))

        padded_ids = []
        padded_masks = []

        for ids in batch_text_ids:
            cur_len = int(ids.numel())
            pad_len = max_text_len - cur_len

            if pad_len > 0:
                pad_ids = torch.full(
                    (pad_len,),
                    fill_value=pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                padded = torch.cat([ids, pad_ids], dim=0)
                mask = torch.cat(
                    [
                        torch.ones(cur_len, dtype=torch.long, device=ids.device),
                        torch.zeros(pad_len, dtype=torch.long, device=ids.device),
                    ],
                    dim=0,
                )
            else:
                padded = ids
                mask = torch.ones(cur_len, dtype=torch.long, device=ids.device)

            padded_ids.append(padded)
            padded_masks.append(mask)

        text_input_ids = torch.stack(padded_ids, dim=0)
        text_attention_mask = torch.stack(padded_masks, dim=0)
        text_embeds = input_embedding_layer(text_input_ids)

        return text_input_ids, text_attention_mask, text_embeds

    def _get_question_feature(
        self,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        strategy = str(self.token_cfg.get("question_feature_strategy", "mean"))

        if strategy == "mean":
            mask = text_attention_mask.float().unsqueeze(-1)
            if mask.dtype != text_embeds.dtype:
                mask = mask.to(dtype=text_embeds.dtype)
            summed = (text_embeds * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom

        if strategy == "last":
            last_idx = text_attention_mask.sum(dim=1) - 1
            last_idx = last_idx.clamp(min=0)
            batch_idx = torch.arange(text_embeds.size(0), device=text_embeds.device)
            return text_embeds[batch_idx, last_idx, :]

        raise ValueError(f"Unsupported question_feature_strategy={strategy}")

    def _infer_question_type_ids(
        self,
        questions: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Heuristic question-type classifier for Stage 4 adaptive budgeting.

        Type IDs:
            0 = yes/no question
            1 = attribute/object question
            2 = counting question
            3 = spatial/complex question

        This is intentionally simple and deterministic. It is not used as a
        label for answering; it only gives the budget controller a weak signal
        about expected visual reasoning difficulty.
        """
        yes_no_starts = (
            "is ", "are ", "was ", "were ",
            "do ", "does ", "did ",
            "can ", "could ", "will ", "would ",
            "has ", "have ", "had ",
            "is there", "are there",
        )

        counting_patterns = (
            "how many",
            "number of",
            "count ",
            "amount of",
        )

        spatial_patterns = (
            "where",
            "left",
            "right",
            "behind",
            "front",
            "in front",
            "on top",
            "under",
            "above",
            "below",
            "next to",
            "near",
            "between",
            "side",
            "position",
            "located",
        )

        type_ids = []

        for q in questions:
            q_norm = str(q).lower().strip()
            q_norm = " ".join(q_norm.split())

            if any(q_norm.startswith(pat) for pat in counting_patterns) or any(
                pat in q_norm for pat in counting_patterns
            ):
                q_type = 2
            elif any(q_norm.startswith(pat) for pat in yes_no_starts):
                q_type = 0
            elif any(pat in q_norm for pat in spatial_patterns):
                q_type = 3
            else:
                q_type = 1

            type_ids.append(q_type)

        return torch.tensor(type_ids, dtype=torch.long, device=device)

    def _build_question_type_budget_targets(
        self,
        question_type_ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build target keep ratios for Stage 4.

        Default mapping:
            type 0 yes/no             -> 0.38
            type 1 attribute/object   -> 0.48
            type 2 counting           -> 0.58
            type 3 spatial/complex    -> 0.62

        These can be overridden in the config using:
            training.question_type_target_ratios: [0.38, 0.48, 0.58, 0.62]
        """
        ratios = self.training_cfg.get(
            "question_type_target_ratios",
            [0.38, 0.48, 0.58, 0.62],
        )

        if len(ratios) != 4:
            raise ValueError(
                "training.question_type_target_ratios must contain exactly 4 values: "
                "[yes_no, attribute_object, counting, spatial_complex]"
            )

        ratio_tensor = torch.tensor(
            [float(x) for x in ratios],
            dtype=dtype,
            device=device,
        )

        return ratio_tensor[question_type_ids.to(device=device).long()]

    def _choose_selection_mode(self) -> str:
        if self.token_selector.training:
            return str(self.token_cfg.get("train_selection_mode", "soft"))
        return str(self.token_cfg.get("eval_selection_mode", "hard"))

    def _build_split_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        projected_visual: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build [prefix, visual, suffix] embeddings with visual tokens placed at
        the <image> placeholder position — the layout LLaVA-1.5 was trained with:
            [BOS, SYS, USER:, v1…vK, \\n, question, ASSISTANT:]
        Strips input_ids padding per-sample before splitting, so this is
        correct for both left- and right-padded batches.
        visual_attention_mask [B, K]: 0 for padded visual slots (variable-K batches).
        Returns (lm_inputs_embeds [B, L, D], lm_attention_mask [B, L]).
        """
        image_token_index = getattr(self.model.config, "image_token_index", None)
        if image_token_index is None:
            raise ValueError("LLaVA config does not expose image_token_index.")

        lm = self._get_language_model()
        embed_layer = lm.get_input_embeddings()
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = (tokenizer.pad_token_id if tokenizer is not None else 0) or 0

        batch_size = input_ids.size(0)
        K = projected_visual.size(1)
        device = projected_visual.device

        seqs_emb: list = []
        seqs_mask: list = []

        for i in range(batch_size):
            ids_i = input_ids[i][attention_mask[i].bool()]  # strip padding → valid tokens

            img_pos = (ids_i == image_token_index).nonzero(as_tuple=True)[0]
            if img_pos.numel() == 0:
                raise ValueError(f"No <image> token found in input_ids sample {i}.")
            img_start = int(img_pos[0])
            img_end = int(img_pos[-1]) + 1

            prefix_ids = ids_i[:img_start]   # [BOS, SYS, USER:]
            suffix_ids = ids_i[img_end:]      # [\n, question, ASST:]

            prefix_emb = embed_layer(prefix_ids.unsqueeze(0)).squeeze(0)   # [P, D]
            suffix_emb = embed_layer(suffix_ids.unsqueeze(0)).squeeze(0)   # [S, D]
            vis_emb = projected_visual[i].to(dtype=prefix_emb.dtype)        # [K, D]

            seq_emb = torch.cat([prefix_emb, vis_emb, suffix_emb], dim=0)

            if visual_attention_mask is not None:
                vis_mask = visual_attention_mask[i].to(device=device, dtype=torch.long)
            else:
                vis_mask = torch.ones(K, dtype=torch.long, device=device)

            prefix_mask = torch.ones(prefix_emb.size(0), dtype=torch.long, device=device)
            suffix_mask = torch.ones(suffix_emb.size(0), dtype=torch.long, device=device)
            seq_mask = torch.cat([prefix_mask, vis_mask, suffix_mask], dim=0)

            seqs_emb.append(seq_emb)
            seqs_mask.append(seq_mask)

        max_len = max(s.size(0) for s in seqs_emb)
        padded_embs: list = []
        padded_masks: list = []

        for emb, mask in zip(seqs_emb, seqs_mask):
            pad_len = max_len - emb.size(0)
            if pad_len > 0:
                pad_emb = embed_layer(
                    torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device)
                ).to(dtype=emb.dtype)
                emb = torch.cat([emb, pad_emb], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long, device=device)], dim=0)
            padded_embs.append(emb)
            padded_masks.append(mask)

        return torch.stack(padded_embs, dim=0), torch.stack(padded_masks, dim=0)

    def _build_dynamic_multimodal_inputs(
        self,
        model_inputs: Dict[str, torch.Tensor],
        questions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        pixel_values = model_inputs.get("pixel_values", None)
        input_ids = model_inputs.get("input_ids", None)
        attention_mask = model_inputs.get("attention_mask", None)

        if pixel_values is None:
            raise ValueError("pixel_values are required for dynamic visual token pruning.")
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required.")

        visual_features, final_attentions = self._run_vision_encoder(pixel_values)

        _, text_attention_mask, text_embeds = self._strip_image_tokens_and_embed_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        question_feature = self._get_question_feature(
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        )

        selection_mode = self._choose_selection_mode()

        question_type_ids = self._infer_question_type_ids(
            questions=questions,
            device=visual_features.device,
        )

        selector_out = self.token_selector(
            visual_features=visual_features,
            question_feature=question_feature,
            final_layer_attentions=final_attentions,
            selection_mode=selection_mode,
            question_type_ids=question_type_ids,
        )

        projected_image_features = self._project_visual_features(
            selector_out["selected_features"]
        )

        visual_attention_mask = selector_out["selected_attention_mask"].to(
            device=text_attention_mask.device,
            dtype=text_attention_mask.dtype,
        )

        lm_inputs_embeds, lm_attention_mask = self._build_split_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            projected_visual=projected_image_features,
            visual_attention_mask=visual_attention_mask,
        )

        token_stats = {
            "num_visual_tokens_before_selection": selector_out["num_tokens_before"],
            "num_visual_tokens_after_selection": selector_out["num_tokens_after"],
            "retention_ratio": selector_out["retention_ratio"],

            "soft_keep_ratio": selector_out["soft_keep_ratio"],
            "budget_gate": selector_out["budget_gate"],
            "budget_threshold": selector_out["budget_threshold"],

            "selected_token_indices": selector_out["selected_indices"],
            "selected_token_scores": selector_out["selected_scores"],

            "selection_mode": selector_out["selection_mode"],
            "question_type_ids": question_type_ids,
        }

        budget_strategy = str(self.training_cfg.get("budget_loss_type", "target"))
        keep_ratio = selector_out["soft_keep_ratio"]

        question_type_target_keep_ratio = self._build_question_type_budget_targets(
            question_type_ids=question_type_ids,
            dtype=keep_ratio.dtype,
            device=keep_ratio.device,
        )

        token_stats["question_type_target_keep_ratio"] = question_type_target_keep_ratio

        if budget_strategy == "target":
            target_keep_ratio = float(self.training_cfg.get("target_keep_ratio", 0.50))
            budget_loss = (keep_ratio.mean() - target_keep_ratio) ** 2

        elif budget_strategy == "upper_bound":
            max_target = float(self.training_cfg.get("budget_max_target", 0.60))
            budget_loss = F.relu(keep_ratio - max_target).mean()

        elif budget_strategy == "question_type_target":
            # Stage 4:
            # Per-sample budget supervision based on simple question type.
            # This works with batch_size=1 because every sample has its own target.
            budget_loss = F.mse_loss(
                keep_ratio,
                question_type_target_keep_ratio,
                reduction="mean",
            )

        elif budget_strategy == "none":
            budget_loss = torch.zeros((), dtype=keep_ratio.dtype, device=keep_ratio.device)

        else:
            raise ValueError(f"Unsupported budget_loss_type={budget_strategy}")

        entropy_loss = selector_out["score_entropy"].mean()

        # Encourage the learned budget to vary across samples.
        # Without this, the controller can learn a nearly constant budget.
        keep_ratio_std = keep_ratio.std(unbiased=False)
        budget_diversity_loss = -keep_ratio_std

        dynamic_losses = {
            "budget_loss": budget_loss,
            "entropy_loss": entropy_loss,
            "budget_diversity_loss": budget_diversity_loss,
            "question_type_target_keep_ratio": question_type_target_keep_ratio,
        }

        return lm_inputs_embeds, lm_attention_mask, token_stats, dynamic_losses

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        images = batch["images"]
        questions = batch["questions"]
        answer_labels = batch.get("answer_labels", None)

        model_inputs = self._prepare_inputs(images=images, questions=questions)

        device = self._get_model_device()

        processor_input_lengths = self._compute_processor_input_lengths(
            model_inputs.get("attention_mask", None)
        )

        raw_question_lengths = self._compute_raw_question_lengths(
            questions=questions,
            device=device,
        )

        lm_inputs_embeds, lm_attention_mask, token_stats, dynamic_losses = (
            self._build_dynamic_multimodal_inputs(
                model_inputs=model_inputs,
                questions=questions,
            )
        )

        language_model = self._get_language_model()

        # Stage 3 option:
        # If enabled during training, gradients flow through the frozen LLM
        # to the visual input embeddings. The LLM weights remain frozen because
        # requires_grad=False, but the selector/scorer can now receive answer-loss
        # gradients through lm_inputs_embeds.
        enable_selector_grad = bool(
            self.training_cfg.get("enable_selector_grad_through_lm", False)
        ) and self.training

        if enable_selector_grad:
            lm_outputs = language_model(
                inputs_embeds=lm_inputs_embeds,
                attention_mask=lm_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        else:
            with torch.no_grad():
                lm_outputs = language_model(
                    inputs_embeds=lm_inputs_embeds,
                    attention_mask=lm_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

        if lm_outputs.hidden_states is None:
            raise ValueError(
                "Language model did not return hidden states; required for answer head training."
            )

        last_hidden = lm_outputs.hidden_states[-1]

        pooled_features = self._gather_last_valid_hidden(
            hidden_states=last_hidden,
            attention_mask=lm_attention_mask,
        )

        logits = self.answer_head(pooled_features)

        ce_loss = None
        total_loss = None

        if answer_labels is not None:
            answer_labels = answer_labels.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            # Guard against an all-ignored batch. With batch_size=1 (and
            # filter_unknown_answers=False), a sample whose answer is not in the
            # vocab has answer_label=-1. CrossEntropyLoss(ignore_index=-1) then
            # averages over ZERO valid targets and returns NaN, which poisons every
            # trainable weight through backward. Such samples carry no answer signal,
            # but they MUST still contribute the budget loss (the gate depends on it),
            # so we set ce_loss=0 for them rather than dropping the sample.
            if (answer_labels != -1).any():
                ce_loss = loss_fn(logits, answer_labels)
            else:
                ce_loss = logits.new_zeros(())

            budget_weight = float(self.training_cfg.get("budget_loss_weight", 0.0))
            entropy_weight = float(self.training_cfg.get("entropy_loss_weight", 0.0))
            budget_diversity_weight = float(
                self.training_cfg.get("budget_diversity_weight", 0.0)
            )

            total_loss = (
                ce_loss
                + budget_weight * dynamic_losses["budget_loss"]
                + entropy_weight * dynamic_losses["entropy_loss"]
                + budget_diversity_weight * dynamic_losses["budget_diversity_loss"]
            )

        pred_ids = logits.argmax(dim=-1)
        pred_answers = self._map_pred_ids_to_answers(pred_ids)

        multimodal_seq_len = (
            raw_question_lengths + token_stats["num_visual_tokens_after_selection"]
        )

        return {
            "predictions": {
                "logits": logits,
                "pred_answer_ids": pred_ids,
                "pred_answers": pred_answers,
                "loss": total_loss,
                "ce_loss": ce_loss,
            },
            "token_stats": token_stats,
            "dynamic_losses": dynamic_losses,
            "analysis": {
                "question_ids": batch.get("question_ids"),
                "image_ids": batch.get("image_ids"),
                "raw_question_lengths": raw_question_lengths,
                "processor_input_lengths": processor_input_lengths,
                "multimodal_sequence_length": multimodal_seq_len,
            },
        }
