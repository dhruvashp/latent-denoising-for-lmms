#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    import deepspeed
    _has_deepspeed = True
except ImportError:
    _has_deepspeed = False

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass
class SGLDCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Extended output class that includes SGLD losses."""
    loss_lang: Optional[torch.FloatTensor] = None
    loss_rec: Optional[torch.FloatTensor] = None
    loss_rel: Optional[torch.FloatTensor] = None
    loss_con: Optional[torch.FloatTensor] = None
    num_corrupted_tokens: Optional[int] = None


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

    def forward(self, *args, **kwargs):
        capture_layer = getattr(self, '_sgld_capture_layer', None)
        if capture_layer is not None and capture_layer >= 0:
            # v5: request all hidden states to extract mid-layer features
            kwargs['output_hidden_states'] = True
            result = super().forward(*args, **kwargs)
            # Index L+1 because index 0 = embedding output, index L+1 = output of layer L
            self._sgld_captured_hidden = result.hidden_states[capture_layer + 1]
            # Free the full hidden_states tuple — only keep the one we captured
            result = BaseModelOutputWithPast(
                last_hidden_state=result.last_hidden_state,
                past_key_values=result.past_key_values,
                hidden_states=None,
                attentions=result.attentions,
            )
            return result
        return super().forward(*args, **kwargs)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        sgld_global_step: Optional[int] = None,
        sgld_max_steps: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        target_device = next(self.get_model().embed_tokens.parameters()).device

        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if labels is not None and labels.device != target_device:
            labels = labels.to(target_device)
        if images is not None and not isinstance(images, list) and images.device != target_device:
            images = images.to(target_device)

        # Determine if SGLD should be applied
        sgld_enable = getattr(self.config, 'sgld_enable', False)
        sgld_mode = getattr(self.config, 'sgld_mode', 'scale')
        sgld_apply = False
        if sgld_enable and sgld_mode != 'none' and self.training and images is not None:
            tune_mm_mlp_adapter = getattr(self.config, 'tune_mm_mlp_adapter', False)
            if tune_mm_mlp_adapter:
                sgld_apply = getattr(self.config, 'sgld_stage1_enable', True)
            else:
                sgld_apply = getattr(self.config, 'sgld_stage2_enable', True)

        # SGLD aux data
        teacher_y = None
        corr_mask = None
        num_image_patches = 0

        # Compute SGLD schedule multiplier
        schedule_mult = 1.0
        if sgld_global_step is not None and sgld_max_steps is not None and sgld_max_steps > 0:
            schedule_type = getattr(self.config, 'sgld_schedule_type', 'whd')
            progress = min(sgld_global_step / sgld_max_steps, 1.0)

            if schedule_type == 'whd':
                # v4 Warmup-Hold-Decay
                warmup_r = getattr(self.config, 'sgld_warmup_ratio', 0.05)
                decay_r = getattr(self.config, 'sgld_decay_ratio', 0.20)
                hold_end = 1.0 - decay_r
                if progress < warmup_r:
                    u = progress / max(warmup_r, 1e-8)
                    schedule_mult = 0.5 * (1.0 - math.cos(math.pi * min(u, 1.0)))
                elif progress < hold_end:
                    schedule_mult = 1.0
                else:
                    u = (progress - hold_end) / max(decay_r, 1e-8)
                    schedule_mult = 0.5 * (1.0 + math.cos(math.pi * min(u, 1.0)))
            elif schedule_type == 'cosine':
                # v2 backward compat
                schedule_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif schedule_type == 'linear':
                schedule_mult = max(0.0, 1.0 - progress)
            # schedule_type == 'none': stays 1.0

        if inputs_embeds is None:
            if sgld_apply and images is not None and not (isinstance(images, list) or (torch.is_tensor(images) and images.ndim == 5)):
                # SGLD path: encode with saliency-guided corruption
                corrupted_features, teacher_y, corr_mask = \
                    self.encode_images_with_sgld(
                        images, schedule_mult=schedule_mult,
                    )
                num_image_patches = corrupted_features.shape[1]

                # Monkey-patch encode_images to use our corrupted features
                original_encode = self.encode_images
                def encode_with_sgld(imgs):
                    return corrupted_features
                self.encode_images = encode_with_sgld

                (
                    input_ids, position_ids, attention_mask,
                    past_key_values, inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, image_sizes
                )
                self.encode_images = original_encode
            else:
                # Standard path
                (
                    input_ids, position_ids, attention_mask,
                    past_key_values, inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, image_sizes
                )

        # --- Call self.model() directly to get last hidden state ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # v5: set mid-layer capture flag if SGLD needs intermediate features
        sgld_student_layer = getattr(self.config, 'sgld_student_layer', -1)
        if sgld_apply and sgld_student_layer >= 0:
            self.model._sgld_capture_layer = sgld_student_layer
        else:
            self.model._sgld_capture_layer = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,  # LlavaLlamaModel.forward() overrides if capture needed
            return_dict=True,
        )

        # Clear capture flag
        self.model._sgld_capture_layer = None

        hidden_states = outputs[0]  # last hidden state: (B, seq_len, d_h)

        # Compute logits
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [F.linear(hidden_states, lm_head_slices[i])
                      for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute language loss
        loss_lang = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss_lang = loss_fct(shift_logits, shift_labels)

        # --- Compute SGLD losses ---
        zero = torch.tensor(0.0, device=target_device)
        loss_rec = zero
        loss_rel = zero
        loss_con = zero
        num_corrupted = 0

        if sgld_apply and teacher_y is not None and corr_mask is not None:
            # Gather vision token positions from prepare_inputs_labels_for_multimodal
            vision_starts = getattr(self, '_sgld_vision_token_starts', None)

            if vision_starts is not None and num_image_patches > 0:
                B = hidden_states.shape[0]
                S = num_image_patches

                # v5: use mid-layer features if sgld_student_layer >= 0, else final layer (v4)
                if sgld_student_layer >= 0:
                    feature_source = getattr(self.model, '_sgld_captured_hidden', None)
                    self.model._sgld_captured_hidden = None  # free reference
                    if feature_source is None:
                        feature_source = hidden_states  # fallback
                else:
                    feature_source = hidden_states

                # Build H_vis: (B, S, d_h) from feature_source.
                # If the sequence was truncated, only the first S_actual
                # vision tokens survive; compute losses on that slice.
                seq_len = feature_source.shape[1]
                h_vis_list = []
                valid_batch = []
                actual_S = S  # may be reduced if vision tokens got truncated
                for b in range(B):
                    start = vision_starts[b] if b < len(vision_starts) else None
                    if start is not None and start < seq_len:
                        S_avail = min(S, seq_len - start)
                        actual_S = min(actual_S, S_avail)
                        h_vis_list.append(feature_source[b, start:start + S_avail, :])
                        valid_batch.append(b)

                if h_vis_list:
                    # Clamp all entries to the same length (the minimum)
                    h_vis_list = [h[:actual_S] for h in h_vis_list]
                    H_vis = torch.stack(h_vis_list, dim=0)  # (B', actual_S, d_h)
                    # Slice teacher_y and corr_mask to match
                    t_y = teacher_y[valid_batch, :actual_S]
                    c_m = corr_mask[valid_batch, :actual_S]

                    loss_rec, loss_rel, loss_con, num_corrupted = \
                        self.compute_sgld_losses(H_vis, t_y, c_m)

            # Clean up stored state
            self._sgld_vision_token_starts = None

        # ZeRO-3 fix: on text-only batches (images=None) or when schedule_mult=0
        # (no corruption), SGLD modules are never called, but DeepSpeed's
        # prefetcher may have already gathered their params based on the
        # previous step's trace. Run zero-contribution dummy forwards so
        # DeepSpeed's normal hooks consume the prefetched params.
        if sgld_apply and num_corrupted == 0 and hasattr(self.get_model(), 'sgld_decoder'):
            decoder = self.get_model().sgld_decoder
            _dummy_in = torch.zeros(1, 1, hidden_states.shape[-1],
                                    device=target_device, dtype=hidden_states.dtype)
            _dummy_out = decoder(_dummy_in)
            loss_rec = loss_rec + _dummy_out.sum() * 0.0

            # Also consume tau_embed params for DeepSpeed ZeRO
            tau_embed = getattr(self.get_model(), 'sgld_tau_embed', None)
            if tau_embed is not None:
                _dummy_idx = torch.zeros(1, dtype=torch.long, device=target_device)
                _dummy_embed = tau_embed(_dummy_idx)
                loss_rec = loss_rec + _dummy_embed.sum() * 0.0

        # Combine losses
        if loss_lang is not None:
            total_loss = loss_lang.clone()
            if sgld_apply and num_corrupted > 0:
                cfg = self.config
                lam_rec = getattr(cfg, 'sgld_lambda_rec', 0.2) * schedule_mult
                lam_rel = getattr(cfg, 'sgld_lambda_rel', 0.05) * schedule_mult
                lam_con = getattr(cfg, 'sgld_lambda_con', 0.05) * schedule_mult

                if getattr(cfg, 'sgld_use_rec', True):
                    total_loss = total_loss + lam_rec * loss_rec
                if getattr(cfg, 'sgld_use_rel', True):
                    total_loss = total_loss + lam_rel * loss_rel
                if getattr(cfg, 'sgld_use_con', True):
                    total_loss = total_loss + lam_con * loss_con
        else:
            total_loss = None

        if return_dict is False:
            output = (logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return SGLDCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
            loss_lang=loss_lang,
            loss_rec=loss_rec,
            loss_rel=loss_rel,
            loss_con=loss_con,
            num_corrupted_tokens=num_corrupted,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
