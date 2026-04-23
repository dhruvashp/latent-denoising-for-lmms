"""
Training script for SGLD-Qwen-2.5-VL.

Dedicated pipeline because LLaVA's data preprocessing is tightly coupled to
CLIP + Vicuna (IMAGE_TOKEN_INDEX=-200, conversation templates, etc.).
This script uses Qwen's native processor and ChatML formatting.

Usage:
    deepspeed llava/train/train_qwen.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --data_path ./playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
        --image_folder ./playground/data \
        ...
"""

import os
import copy
import json
import math
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Monkey-patch: torch 2.4 + transformers 5.x compatibility.
# Transformers 5.x calls torch.load(..., weights_only=True) when resuming
# checkpoints, but torch 2.4's weights_only implementation cannot handle
# numpy objects in optimizer/scheduler .pt files (fixed in torch 2.6).
# Since we only load our own checkpoints, weights_only=False is safe.
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Monkey-patch for PyTorch 2.4 + DeepSpeed compatibility.
# DeepSpeed's count_used_parameters_in_backward calls _get_grad_fn_or_grad_acc
# which crashes when t.view_as(t).grad_fn is None (can happen with certain
# parameter initialization patterns in PyTorch 2.4).
# ---------------------------------------------------------------------------
import torch.autograd.graph as _autograd_graph
if hasattr(_autograd_graph, '_get_grad_fn_or_grad_acc'):
    _orig_get_grad_fn = _autograd_graph._get_grad_fn_or_grad_acc

    def _safe_get_grad_fn_or_grad_acc(t):
        v = t.view_as(t)
        if v.grad_fn is None:
            return None
        return v.grad_fn.next_functions[0][0]

    _autograd_graph._get_grad_fn_or_grad_acc = _safe_get_grad_fn_or_grad_acc

import transformers
from transformers import (
    Trainer,
    AutoProcessor,
)
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

from PIL import Image, ImageFile, UnidentifiedImageError

# Some datasets contain truncated images; let PIL load recoverable files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IGNORE_INDEX = -100

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------
@dataclass
class QwenModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    freeze_vision: bool = field(default=True, metadata={"help": "Freeze vision tower."})
    image_min_pixels: int = field(default=3136, metadata={"help": "Minimum pixels for Qwen processor."})
    image_max_pixels: int = field(default=451584, metadata={"help": "Maximum pixels (672*672=451584 → ~576 merged tokens)."})


@dataclass
class QwenDataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to LLaVA-format JSON."})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Root folder for images."})


@dataclass
class QwenTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=2048)
    group_by_modality_length: bool = field(default=False)

    # SGLD arguments (identical to llava/train/train.py L117-156)
    sgld_enable: bool = field(default=True, metadata={"help": "Enable SGLD auxiliary losses."})
    sgld_stage1_enable: bool = field(default=True)
    sgld_stage2_enable: bool = field(default=True)
    sgld_rho_noise: float = field(default=0.20)
    sgld_rho_mask: float = field(default=0.03)
    sgld_tau_s: float = field(default=0.07)
    sgld_sigma: float = field(default=1.0)
    sgld_tau_max: float = field(default=0.30)
    sgld_use_rec: bool = field(default=True)
    sgld_use_rel: bool = field(default=True)
    sgld_use_con: bool = field(default=True)
    sgld_lambda_rec: float = field(default=0.2)
    sgld_lambda_rel: float = field(default=0.05)
    sgld_lambda_con: float = field(default=0.05)
    sgld_tau_r: float = field(default=0.10)
    sgld_tau_c: float = field(default=0.07)
    sgld_teacher: str = field(default="vision_tower")
    sgld_saliency_type: str = field(default="l2_norm")
    sgld_decoder_lr: Optional[float] = field(default=None)
    sgld_pretrain_weights: Optional[str] = field(default=None)
    sgld_lambda_schedule: str = field(default="none")
    sgld_mode: str = field(default="scale")
    sgld_p_max: float = field(default=0.5)
    sgld_tau_bins: int = field(default=8)
    sgld_use_tau_embed: bool = field(default=True)
    sgld_schedule_type: str = field(default="whd")
    sgld_warmup_ratio: float = field(default=0.05)
    sgld_decay_ratio: float = field(default=0.20)
    sgld_student_layer: int = field(default=-1)


# ---------------------------------------------------------------------------
# Dataset: Converts LLaVA JSON → Qwen ChatML
# ---------------------------------------------------------------------------
class QwenSgldDataset(Dataset):
    """Load LLaVA-format data and convert to Qwen ChatML for processor."""

    def __init__(self, data_path: str, image_folder: str, processor: AutoProcessor,
                 max_length: int = 2048):
        super().__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length
        self._skipped_image_count = 0

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        """For group_by_modality_length sampler compatibility."""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _convert_to_qwen_messages(self, conversations, has_image):
        """Convert LLaVA conversation format to Qwen ChatML messages.

        LLaVA format: [{from: "human"/"gpt", value: "..."}]
        Qwen format:  [{role: "user"/"assistant", content: [{type: "image"} / {type: "text", text: "..."}]}]
        """
        messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            text = conv["value"]

            # Remove <image> placeholder tokens used by LLaVA
            text = text.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            text = text.strip()

            if role == "user" and has_image and len(messages) == 0:
                # First user message with image
                content = [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ]
            else:
                content = [{"type": "text", "text": text}]

            messages.append({"role": role, "content": content})

        return messages

    def __getitem__(self, i) -> Dict[str, Any]:
        max_retries = 20
        candidate_idx = i

        for retry_idx in range(max_retries):
            sample = self.list_data_dict[candidate_idx]
            conversations = sample["conversations"]
            has_image = "image" in sample

            pil_image = None
            if has_image:
                image_file = sample["image"]
                image_path = os.path.join(self.image_folder, image_file)
                try:
                    with Image.open(image_path) as img:
                        pil_image = img.convert("RGB")
                except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
                    self._skipped_image_count += 1
                    if self._skipped_image_count <= 20 or self._skipped_image_count % 100 == 0:
                        logging.warning(
                            "Skipping bad image at idx=%s, path=%s, reason=%s (skip_count=%s)",
                            candidate_idx, image_path, repr(exc), self._skipped_image_count,
                        )
                    candidate_idx = (candidate_idx + 1) % len(self.list_data_dict)
                    continue

            # Convert to Qwen ChatML format
            messages = self._convert_to_qwen_messages(conversations, has_image)

            # Apply chat template to get the full text
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

            # Process with Qwen processor
            if has_image and pil_image is not None:
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[text],
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

            input_ids = inputs["input_ids"].squeeze(0)

            # Build labels: mask everything except assistant responses
            labels = self._build_labels(input_ids, messages)

            result = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": inputs["attention_mask"].squeeze(0),
            }

            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

            return result

        raise RuntimeError(
            f"Failed to fetch valid sample after {max_retries} retries from idx {i}."
        )

    def _build_labels(self, input_ids, messages):
        """Create labels with IGNORE_INDEX for non-assistant tokens.

        Qwen ChatML format:
            <|im_start|>system\n...\n<|im_end|>\n
            <|im_start|>user\n...\n<|im_end|>\n
            <|im_start|>assistant\n...\n<|im_end|>\n
        We only want loss on assistant message content (including <|im_end|>).
        """
        labels = input_ids.clone()
        tokenizer = self.processor.tokenizer

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Find all <|im_start|> positions
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()

        # Determine which turns are assistant turns
        assistant_turn_indices = set()
        turn_idx = 0
        # System message is turn 0 (if present), then alternating user/assistant
        # But we need to match against the actual messages structure
        # In ChatML, there may be a system prompt prepended by the template
        has_system = True  # Qwen always adds system
        for msg_idx, msg in enumerate(messages):
            actual_turn = msg_idx + (1 if has_system else 0)
            if msg["role"] == "assistant":
                assistant_turn_indices.add(actual_turn)

        # Mask non-assistant turns
        for turn_idx, start_pos in enumerate(im_start_positions):
            # Find the corresponding <|im_end|>
            end_positions = (input_ids[start_pos:] == im_end_id).nonzero(as_tuple=True)[0]
            if len(end_positions) == 0:
                end_pos = len(input_ids) - 1
            else:
                end_pos = start_pos + end_positions[0].item()

            if turn_idx not in assistant_turn_indices:
                # Mask entire turn including <|im_start|>...<|im_end|> and trailing \n
                mask_end = min(end_pos + 2, len(input_ids))  # +2 for <|im_end|>\n
                labels[start_pos:mask_end] = IGNORE_INDEX
            else:
                # For assistant turns, mask only <|im_start|>assistant\n prefix
                # Find the newline after "assistant"
                # Typically: <|im_start|> + "assistant" tokens + \n
                # We mask from start to after the \n following "assistant"
                newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
                offset = 1  # skip <|im_start|>
                while start_pos + offset < end_pos:
                    if input_ids[start_pos + offset] == newline_id:
                        offset += 1  # include the newline in mask
                        break
                    offset += 1
                labels[start_pos:start_pos + offset] = IGNORE_INDEX

        return labels


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class QwenDataCollator:
    """Collate variable-length samples into a batch."""

    pad_token_id: int = 151643  # Qwen's pad token

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        attention_mask = [inst["attention_mask"] for inst in instances]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Concatenate pixel_values (flat patches from different images)
        pixel_values_list = [inst["pixel_values"] for inst in instances
                             if "pixel_values" in inst]
        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        # Stack image_grid_thw
        grid_thw_list = [inst["image_grid_thw"] for inst in instances
                         if "image_grid_thw" in inst]
        if grid_thw_list:
            # Each may be (1, 3) or (N, 3) for multi-image
            batch["image_grid_thw"] = torch.cat(
                [g.unsqueeze(0) if g.dim() == 1 else g for g in grid_thw_list], dim=0
            )

        return batch


# ---------------------------------------------------------------------------
# Trainer with SGLD support (mirrors LLaVATrainer)
# ---------------------------------------------------------------------------
class QwenSgldTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sgld_loss_lang_acc = 0.0
        self._sgld_loss_rec_acc = 0.0
        self._sgld_loss_rel_acc = 0.0
        self._sgld_loss_con_acc = 0.0
        self._sgld_num_corrupted_acc = 0
        self._sgld_acc_steps = 0

    def create_accelerator_and_postprocess(self):
        """Set sync_each_batch=True for DeepSpeed compatibility."""
        super().create_accelerator_and_postprocess()
        if self.is_deepspeed_enabled:
            self.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Pass SGLD step info and collect loss components."""
        sgld_enable = getattr(self.args, 'sgld_enable', False) or \
                      getattr(model.config, 'sgld_enable', False)

        if sgld_enable:
            inputs = dict(inputs)
            inputs['sgld_global_step'] = self.state.global_step
            inputs['sgld_max_steps'] = self.state.max_steps

        outputs = model(**inputs)

        if sgld_enable and hasattr(outputs, 'loss_lang') and outputs.loss_lang is not None:
            self._sgld_loss_lang_acc += outputs.loss_lang.detach().item()
            self._sgld_loss_rec_acc += (outputs.loss_rec.detach().item()
                                        if outputs.loss_rec is not None else 0.0)
            self._sgld_loss_rel_acc += (outputs.loss_rel.detach().item()
                                        if outputs.loss_rel is not None else 0.0)
            self._sgld_loss_con_acc += (outputs.loss_con.detach().item()
                                        if outputs.loss_con is not None else 0.0)
            self._sgld_num_corrupted_acc += (outputs.num_corrupted_tokens
                                             if outputs.num_corrupted_tokens is not None else 0)
            self._sgld_acc_steps += 1

        loss = outputs.loss if isinstance(outputs, dict) or hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """Add SGLD-specific metrics to logs."""
        if self._sgld_acc_steps > 0:
            n = self._sgld_acc_steps
            logs['loss_lang'] = self._sgld_loss_lang_acc / n
            logs['loss_rec'] = self._sgld_loss_rec_acc / n
            logs['loss_rel'] = self._sgld_loss_rel_acc / n
            logs['loss_con'] = self._sgld_loss_con_acc / n
            logs['num_corrupted_tokens'] = self._sgld_num_corrupted_acc / n

            if self.state.max_steps > 0:
                schedule_type = getattr(self.args, 'sgld_schedule_type', 'whd')
                progress = min(self.state.global_step / self.state.max_steps, 1.0)
                m = 1.0
                if schedule_type == 'whd':
                    warmup_r = getattr(self.args, 'sgld_warmup_ratio', 0.05)
                    decay_r = getattr(self.args, 'sgld_decay_ratio', 0.20)
                    hold_end = 1.0 - decay_r
                    if progress < warmup_r:
                        u = progress / max(warmup_r, 1e-8)
                        m = 0.5 * (1.0 - math.cos(math.pi * min(u, 1.0)))
                    elif progress < hold_end:
                        m = 1.0
                    else:
                        u = (progress - hold_end) / max(decay_r, 1e-8)
                        m = 0.5 * (1.0 + math.cos(math.pi * min(u, 1.0)))
                elif schedule_type == 'cosine':
                    m = 0.5 * (1.0 + math.cos(math.pi * progress))
                elif schedule_type == 'linear':
                    m = max(0.0, 1.0 - progress)
                logs['sgld_schedule_mult'] = m

            self._sgld_loss_lang_acc = 0.0
            self._sgld_loss_rec_acc = 0.0
            self._sgld_loss_rel_acc = 0.0
            self._sgld_loss_con_acc = 0.0
            self._sgld_num_corrupted_acc = 0
            self._sgld_acc_steps = 0

        super().log(logs, start_time)

    def create_optimizer(self):
        """Setup optimizer with separate param group for SGLD modules."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            sgld_parameters = [name for name, _ in opt_model.named_parameters()
                               if "sgld_" in name]
            sgld_decoder_lr = getattr(self.args, 'sgld_decoder_lr', None)

            # Base groups (excluding SGLD)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n not in sgld_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n not in sgld_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # SGLD param groups with optional custom LR
            if sgld_parameters:
                if sgld_decoder_lr is not None:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in sgld_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": sgld_decoder_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in sgld_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": sgld_decoder_lr,
                        },
                    ])
                else:
                    optimizer_grouped_parameters[0]["params"].extend([
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n in sgld_parameters and p.requires_grad)
                    ])
                    optimizer_grouped_parameters[1]["params"].extend([
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n in sgld_parameters and p.requires_grad)
                    ])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _get_train_sampler(self, *args, **kwargs):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if getattr(self.args, 'group_by_modality_length', False):
            from llava.train.llava_trainer import LengthGroupedSampler
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        return super()._get_train_sampler(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (QwenModelArguments, QwenDataArguments, QwenTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16
                     else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # ---- Load model ----
    from llava.model.language_model.sgld_qwen import SgldQwenForConditionalGeneration

    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    model = SgldQwenForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Sanitize generation config (avoid save-time validation errors)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        if getattr(model.generation_config, "do_sample", False) is False:
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = 1.0
            if hasattr(model.generation_config, "top_p"):
                model.generation_config.top_p = 1.0

    # ---- Load processor ----
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=model_args.image_min_pixels,
        max_pixels=model_args.image_max_pixels,
        padding_side="right",
    )

    # ---- Freeze vision tower ----
    if model_args.freeze_vision:
        rank0_print("Freezing vision tower...")
        model.model.visual.requires_grad_(False)

    # ---- Gradient checkpointing ----
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ---- Initialize SGLD modules ----
    if training_args.sgld_enable:
        rank0_print("Initializing SGLD modules (mask embedding + decoder)...")
        model.initialize_sgld_modules(training_args)

        # Load pretrained SGLD weights if specified
        if training_args.sgld_pretrain_weights:
            rank0_print(f"Loading pretrained SGLD weights from {training_args.sgld_pretrain_weights}")
            sgld_weights = torch.load(training_args.sgld_pretrain_weights, map_location='cpu')
            if 'sgld_mask_embed' in sgld_weights:
                model.model.sgld_mask_embed.data = sgld_weights['sgld_mask_embed']
            if 'sgld_decoder' in sgld_weights:
                model.model.sgld_decoder.load_state_dict(sgld_weights['sgld_decoder'])
            rank0_print("  - Pretrained SGLD weights loaded successfully")

        # Move to correct dtype (avoid .data= assignment, which breaks autograd
        # graph in PyTorch 2.4+ and causes DeepSpeed backward hook failures)
        target_dtype = compute_dtype if compute_dtype != torch.float32 else None
        if target_dtype is not None:
            model.model.sgld_decoder.to(dtype=target_dtype)
            model.model.sgld_mask_embed = nn.Parameter(
                model.model.sgld_mask_embed.detach().to(target_dtype)
            )
            if model.model.sgld_tau_embed is not None:
                model.model.sgld_tau_embed.to(dtype=target_dtype)

        # Enable gradients for SGLD parameters
        for p in model.model.sgld_decoder.parameters():
            p.requires_grad = True
        model.model.sgld_mask_embed.requires_grad = True
        if model.model.sgld_tau_embed is not None:
            for p in model.model.sgld_tau_embed.parameters():
                p.requires_grad = True

        rank0_print(f"  - SGLD rho_noise: {training_args.sgld_rho_noise}")
        rank0_print(f"  - SGLD rho_mask: {training_args.sgld_rho_mask}")
        rank0_print(f"  - SGLD lambda_rec: {training_args.sgld_lambda_rec}")
        rank0_print(f"  - SGLD lambda_rel: {training_args.sgld_lambda_rel}")
        rank0_print(f"  - SGLD lambda_con: {training_args.sgld_lambda_con}")
        rank0_print(f"  - SGLD saliency: {training_args.sgld_saliency_type}")
        rank0_print(f"  - SGLD use_rec={training_args.sgld_use_rec} use_rel={training_args.sgld_use_rel} use_con={training_args.sgld_use_con}")
        rank0_print(f"  - SGLD mode: {training_args.sgld_mode}")
        rank0_print(f"  - SGLD schedule_type: {training_args.sgld_schedule_type}")
        if training_args.sgld_schedule_type == 'whd':
            rank0_print(f"  - SGLD warmup_ratio: {training_args.sgld_warmup_ratio}")
            rank0_print(f"  - SGLD decay_ratio: {training_args.sgld_decay_ratio}")
        rank0_print(f"  - SGLD tau_embed: {training_args.sgld_use_tau_embed} (bins={training_args.sgld_tau_bins})")
        rank0_print(f"  - SGLD student_layer: {training_args.sgld_student_layer} "
                     f"({'last' if training_args.sgld_student_layer == -1 else f'layer {training_args.sgld_student_layer}'})")

    # ---- Dataset ----
    rank0_print(f"Loading dataset: {data_args.data_path}")
    train_dataset = QwenSgldDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        processor=processor,
        max_length=training_args.model_max_length,
    )
    rank0_print(f"  - Dataset size: {len(train_dataset)}")

    data_collator = QwenDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # ---- Trainer ----
    trainer = QwenSgldTrainer(
        model=model,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    # ---- Train ----
    explicit_resume = getattr(training_args, "resume_from_checkpoint", None)
    if explicit_resume:
        if not os.path.isdir(explicit_resume):
            raise ValueError(f"--resume_from_checkpoint path does not exist: {explicit_resume}")
        trainer.train(resume_from_checkpoint=explicit_resume)
    elif list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # Save final model (matches Qwen3-VL reference pattern)
    def safe_save_model_for_hf_trainer(trainer, output_dir):
        """Collects the state dict and dump to disk."""
        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(output_dir)
            return
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print("Training complete!")


if __name__ == "__main__":
    train()
