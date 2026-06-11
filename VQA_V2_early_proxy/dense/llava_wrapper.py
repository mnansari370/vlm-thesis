"""Dense baseline wrapper: frozen LLaVA-1.5, all 576 visual tokens, trainable answer head."""

import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

from .answer_head import AnswerHeadMLP
from .token_selector import DenseTokenSelector


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


class LlavaDenseVQAModel(nn.Module):
    """
    Dense LLaVA-style VQA model wrapper aligned with the current configs.

    Supported roles:
    1. Debug dense sanity check:
       - answer_mode = generation
       - use_answer_head = false
       - training.mode = eval_only

    2. Official dense baseline:
       - answer_mode = classification
       - use_answer_head = true
       - training.mode = train_answer_head

    Important metric design:
    - Actual model inputs still use the full processor/chat-template pipeline.
    - For thesis-side analytical efficiency reporting, we separately compute:
        raw_question_length = tokenizer length of the raw question only
      and define:
        multimodal_sequence_length = num_visual_tokens + raw_question_length
    - We also keep processor_input_length as a separate analysis field so the
      expanded prompt length is still observable.
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
        self.training_mode = self.training_cfg.get("mode", "eval_only")

        self.model = self._build_backbone()
        print(f"[Info] Backbone device after load: {self._get_model_device()}", flush=True)

        self.processor = self._build_processor()

        self.token_selector = DenseTokenSelector(
            keep_cls_token=bool(self.token_cfg.get("keep_cls_token", False))
        )

        self.hidden_size = self._infer_hidden_size()
        self.id_to_answer = self._load_id_to_answer_if_available()
        self.answer_head = self._build_answer_head_if_needed()

        self._freeze_requested_modules()
        self._move_trainable_modules_to_backbone_device()

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

        model.config.use_cache = self.training_mode != "train_answer_head"
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

    def _freeze_module(self, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_requested_modules(self) -> None:
        vision_module = getattr(self.model, "vision_tower", None)
        if vision_module is None and hasattr(self.model, "model"):
            vision_module = getattr(self.model.model, "vision_tower", None)

        projector_module = getattr(self.model, "multi_modal_projector", None)
        if projector_module is None and hasattr(self.model, "model"):
            projector_module = getattr(self.model.model, "multi_modal_projector", None)

        lm_module = getattr(self.model, "language_model", None)
        if lm_module is None and hasattr(self.model, "model"):
            lm_module = getattr(self.model.model, "language_model", None)

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
        if self.answer_head is not None:
            self.answer_head.to(device)

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            return int(self.model.config.text_config.hidden_size)

        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)

        lm_module = getattr(self.model, "language_model", None)
        if lm_module is not None and hasattr(lm_module.config, "hidden_size"):
            return int(lm_module.config.hidden_size)

        raise ValueError("Could not infer hidden size from LLaVA backbone.")

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

    def _build_conversations(self, questions: List[str]) -> List[List[Dict[str, Any]]]:
        conversations = []
        for question in questions:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
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

    def _build_dense_token_stats(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        num_visual_tokens = self._estimate_num_visual_tokens()

        dummy_features = torch.zeros(
            batch_size,
            num_visual_tokens,
            1,
            device=device,
        )

        selector_out = self.token_selector(dummy_features)

        return {
            "num_visual_tokens_before_selection": selector_out["num_tokens_before"],
            "num_visual_tokens_after_selection": selector_out["num_tokens_after"],
            "retention_ratio": selector_out["retention_ratio"],
            "selected_token_indices": selector_out["selected_indices"],
        }

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

    def _decode_generated_answers(
        self,
        generated_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> List[str]:
        prompt_len = input_ids.shape[1]
        answer_ids = generated_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [text.strip() for text in decoded]

    def _map_pred_ids_to_answers(self, pred_ids: torch.Tensor) -> Optional[List[str]]:
        if self.id_to_answer is None:
            return None
        return [self.id_to_answer.get(int(idx), "") for idx in pred_ids.detach().cpu().tolist()]

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

        token_stats = self._build_dense_token_stats(batch_size=len(images), device=device)

        if self.answer_mode == "generation" and not self.use_answer_head:
            gen_kwargs = {
                "max_new_tokens": int(self.model_cfg.get("generation_max_new_tokens", 8)),
                "do_sample": bool(self.model_cfg.get("do_sample", False)),
            }

            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs,
            )

            pred_answers = self._decode_generated_answers(
                generated_ids=generated_ids,
                input_ids=model_inputs["input_ids"],
            )

            multimodal_seq_len = raw_question_lengths + token_stats["num_visual_tokens_after_selection"]

            return {
                "predictions": {
                    "logits": None,
                    "pred_answer_ids": None,
                    "pred_answers": pred_answers,
                    "loss": None,
                },
                "token_stats": token_stats,
                "analysis": {
                    "question_ids": batch.get("question_ids"),
                    "image_ids": batch.get("image_ids"),
                    "raw_question_lengths": raw_question_lengths,
                    "processor_input_lengths": processor_input_lengths,
                    "multimodal_sequence_length": multimodal_seq_len,
                },
            }

        if self.answer_mode == "classification" and self.use_answer_head:
            with torch.no_grad():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            if outputs.hidden_states is None:
                raise ValueError(
                    "Backbone did not return hidden states; required for answer head training."
                )

            last_hidden = outputs.hidden_states[-1]
            attention_mask = model_inputs.get("attention_mask", None)
            if attention_mask is None:
                raise ValueError("attention_mask is required for answer-head pooling.")

            pooled_features = self._gather_last_valid_hidden(
                hidden_states=last_hidden,
                attention_mask=attention_mask,
            )

            logits = self.answer_head(pooled_features)

            loss = None
            if answer_labels is not None:
                answer_labels = answer_labels.to(logits.device)
                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fn(logits, answer_labels)

            pred_ids = logits.argmax(dim=-1)
            pred_answers = self._map_pred_ids_to_answers(pred_ids)

            multimodal_seq_len = raw_question_lengths + token_stats["num_visual_tokens_after_selection"]

            return {
                "predictions": {
                    "logits": logits,
                    "pred_answer_ids": pred_ids,
                    "pred_answers": pred_answers,
                    "loss": loss,
                },
                "token_stats": token_stats,
                "analysis": {
                    "question_ids": batch.get("question_ids"),
                    "image_ids": batch.get("image_ids"),
                    "raw_question_lengths": raw_question_lengths,
                    "processor_input_lengths": processor_input_lengths,
                    "multimodal_sequence_length": multimodal_seq_len,
                },
            }

        raise ValueError(
            "Unsupported model configuration. "
            "Expected either generation mode without answer head, "
            "or classification mode with answer head."
        )