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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import deepspeed
    _has_deepspeed = True
except ImportError:
    _has_deepspeed = False

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class SGLDDecoder(nn.Module):
    """Decoder head for SGLD: D: R^{d_h} -> R^{d_t}"""
    def __init__(self, hidden_size, teacher_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, teacher_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_sgld_modules(self, model_args):
        """Initialize SGLD modules: mask embedding and decoder head."""
        vision_hidden_size = self.get_vision_tower().hidden_size  # d_v = d_t (teacher dim)
        hidden_size = self.config.hidden_size  # d_h (LLM dim)

        # Learnable mask embedding e_mask in R^{d_h}
        self.sgld_mask_embed = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.sgld_mask_embed, std=0.02)

        # Decoder head D: R^{d_h} -> R^{d_t}
        self.sgld_decoder = SGLDDecoder(hidden_size, vision_hidden_size)

        # Tau conditioning embedding: bins 0..(K-1) for noised patches
        # Clean and masked patches get zero (not stored in table)
        tau_bins = getattr(model_args, 'sgld_tau_bins', 8)
        if getattr(model_args, 'sgld_use_tau_embed', True) and tau_bins > 0:
            self.sgld_tau_embed = nn.Embedding(tau_bins, hidden_size)
            nn.init.normal_(self.sgld_tau_embed.weight, std=0.02)
        else:
            self.sgld_tau_embed = None

        # Store all SGLD config
        self.config.sgld_enable = getattr(model_args, 'sgld_enable', True)
        self.config.sgld_stage1_enable = getattr(model_args, 'sgld_stage1_enable', True)
        self.config.sgld_stage2_enable = getattr(model_args, 'sgld_stage2_enable', True)
        # Corruption
        self.config.sgld_rho_noise = getattr(model_args, 'sgld_rho_noise', 0.20)
        self.config.sgld_rho_mask = getattr(model_args, 'sgld_rho_mask', 0.03)
        self.config.sgld_tau_s = getattr(model_args, 'sgld_tau_s', 0.07)
        self.config.sgld_sigma = getattr(model_args, 'sgld_sigma', 1.0)
        self.config.sgld_tau_max = getattr(model_args, 'sgld_tau_max', 0.30)
        # Loss toggles
        self.config.sgld_use_rec = getattr(model_args, 'sgld_use_rec', True)
        self.config.sgld_use_rel = getattr(model_args, 'sgld_use_rel', True)
        self.config.sgld_use_con = getattr(model_args, 'sgld_use_con', True)
        # Loss weights + temps
        self.config.sgld_lambda_rec = getattr(model_args, 'sgld_lambda_rec', 0.2)
        self.config.sgld_lambda_rel = getattr(model_args, 'sgld_lambda_rel', 0.05)
        self.config.sgld_lambda_con = getattr(model_args, 'sgld_lambda_con', 0.05)
        self.config.sgld_tau_r = getattr(model_args, 'sgld_tau_r', 0.10)
        self.config.sgld_tau_c = getattr(model_args, 'sgld_tau_c', 0.07)
        # Teacher / saliency
        self.config.sgld_teacher = getattr(model_args, 'sgld_teacher', 'vision_tower')
        self.config.sgld_saliency_type = getattr(model_args, 'sgld_saliency_type', 'cls_attn')
        # Lambda/corruption schedule
        self.config.sgld_lambda_schedule = getattr(model_args, 'sgld_lambda_schedule', 'none')
        # v3: mixing mode + tau embedding
        self.config.sgld_mode = getattr(model_args, 'sgld_mode', 'scale')
        self.config.sgld_p_max = getattr(model_args, 'sgld_p_max', 0.5)
        self.config.sgld_tau_bins = tau_bins
        self.config.sgld_use_tau_embed = getattr(model_args, 'sgld_use_tau_embed', True)
        # v4: schedule config
        self.config.sgld_schedule_type = getattr(model_args, 'sgld_schedule_type', 'whd')
        self.config.sgld_warmup_ratio = getattr(model_args, 'sgld_warmup_ratio', 0.05)
        self.config.sgld_decay_ratio = getattr(model_args, 'sgld_decay_ratio', 0.20)
        # v5: intermediate layer for student features
        self.config.sgld_student_layer = getattr(model_args, 'sgld_student_layer', -1)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_with_sgld(self, images, schedule_mult=1.0):
        """
        Encode images with SGLD saliency-guided corruption.

        Args:
            images: Input images tensor
            schedule_mult: Multiplier from schedule (1.0 = full corruption, 0.0 = no corruption).
                          Applied to rho_noise, rho_mask, and tau_max.

        Returns:
            corrupted_features: Projected features with corruption applied (B, S, d_h)
            teacher_y: Teacher targets - raw CLIP features (B, S, d_t), detached
            corr_mask: Boolean mask of all corrupted patches (B, S)
        """
        target_device = next(self.get_model().embed_tokens.parameters()).device
        if images.device != target_device:
            images = images.to(target_device)

        cfg = self.config
        saliency_type = getattr(cfg, 'sgld_saliency_type', 'cls_attn')
        vision_tower = self.get_model().get_vision_tower()

        # Get patch features + saliency from vision tower (frozen, no_grad)
        raw_features, saliency = vision_tower.forward_with_saliency(
            images, saliency_type=saliency_type
        )  # raw_features: (B, S, d_v), saliency: (B, S)

        if raw_features.device != target_device:
            raw_features = raw_features.to(target_device)
        if saliency.device != target_device:
            saliency = saliency.to(target_device)

        teacher_y = raw_features.detach()  # (B, S, d_t)

        # Project features to LLM dimension
        projected = self.get_model().mm_projector(raw_features)  # (B, S, d_h)
        if projected.device != target_device:
            projected = projected.to(target_device)

        B, S, d_h = projected.shape

        rho_noise = getattr(cfg, 'sgld_rho_noise', 0.20) * schedule_mult
        rho_mask = getattr(cfg, 'sgld_rho_mask', 0.03) * schedule_mult
        tau_s = getattr(cfg, 'sgld_tau_s', 0.07)
        sigma = getattr(cfg, 'sgld_sigma', 1.0)
        tau_max = getattr(cfg, 'sgld_tau_max', 0.30) * schedule_mult

        K_N = int(rho_noise * S)
        K_M = int(rho_mask * S)

        corrupted = projected.clone()
        mask_N = torch.zeros(B, S, dtype=torch.bool, device=target_device)
        mask_M = torch.zeros(B, S, dtype=torch.bool, device=target_device)

        use_random = (saliency_type == 'random')

        for b in range(B):
            s = saliency[b].float()  # (S,)

            # Sample noise set N (high-saliency preferential, or uniform if random)
            if K_N > 0:
                p_N = torch.ones(S, device=target_device) / S if use_random else F.softmax(s / tau_s, dim=0)
                idx_N = torch.multinomial(p_N, K_N, replacement=False)
                mask_N[b, idx_N] = True

                # Sample mask set M (low-saliency preferential, or uniform if random) from remaining
                if K_M > 0:
                    p_M = torch.ones(S, device=target_device) if use_random else F.softmax(-s / tau_s, dim=0)
                    p_M[idx_N] = 0.0
                    p_M_sum = p_M.sum()
                    if p_M_sum > 0:
                        p_M = p_M / p_M_sum
                        idx_M = torch.multinomial(p_M, min(K_M, (p_M > 0).sum().item()), replacement=False)
                        mask_M[b, idx_M] = True
            elif K_M > 0:
                p_M = torch.ones(S, device=target_device) / S if use_random else F.softmax(-s / tau_s, dim=0)
                idx_M = torch.multinomial(p_M, K_M, replacement=False)
                mask_M[b, idx_M] = True

        # Apply noise corruption: z_noised = (1-tau)*z + tau*eps
        if mask_N.any():
            noise_positions = mask_N.unsqueeze(-1).expand_as(corrupted)  # (B,S,d_h)
            tau = torch.rand(B, S, 1, device=target_device, dtype=projected.dtype) * tau_max
            eps = torch.randn_like(corrupted) * sigma
            noised_features = (1.0 - tau) * projected + tau * eps
            corrupted = torch.where(noise_positions, noised_features, corrupted)

            # Tau conditioning embedding for noised patches
            tau_embed_table = getattr(self.get_model(), 'sgld_tau_embed', None)
            if tau_embed_table is not None:
                tau_bins_K = getattr(cfg, 'sgld_tau_bins', 8)
                # Bin by tau_max_base (not scaled) so bins have consistent meaning
                tau_max_base = getattr(cfg, 'sgld_tau_max', 0.15)
                bin_ids = torch.floor(tau.squeeze(-1) / max(tau_max_base, 1e-8) * tau_bins_K).long()
                bin_ids = bin_ids.clamp(0, tau_bins_K - 1)  # (B, S)
                # Add embedding only to noised patches
                for b in range(B):
                    noised_idx = mask_N[b].nonzero(as_tuple=True)[0]
                    if noised_idx.numel() > 0:
                        embed_vals = tau_embed_table(bin_ids[b, noised_idx])  # (k, d_h)
                        corrupted[b, noised_idx] = corrupted[b, noised_idx] + embed_vals

        # Apply mask corruption: z_masked = e_mask
        if mask_M.any():
            mask_embed = self.get_model().sgld_mask_embed.to(
                device=target_device, dtype=projected.dtype
            )
            corrupted[mask_M] = mask_embed

        corr_mask = mask_N | mask_M  # (B, S)

        return corrupted, teacher_y, corr_mask

    def compute_sgld_losses(self, H_vis, teacher_y, corr_mask):
        """
        Compute SGLD auxiliary losses from LLM hidden states at vision positions.

        Args:
            H_vis: LLM hidden states at vision token positions (B, S, d_h)
            teacher_y: Teacher targets (B, S, d_t), detached
            corr_mask: Boolean mask of corrupted patches (B, S)

        Returns:
            loss_rec, loss_rel, loss_con: scalar tensors
            num_corrupted: int for logging
        """
        cfg = self.config
        device = H_vis.device
        decoder = self.get_model().sgld_decoder
        B, S, d_h = H_vis.shape

        # Decode all vision hidden states
        # Wrap with GatheredParameters for ZeRO-3: the decoder is a submodule
        # of self.model but called here after self.model() forward has exited,
        # so DeepSpeed's automatic param lifecycle doesn't track it properly.
        decoder_params = list(decoder.parameters())
        if _has_deepspeed and any(hasattr(p, 'ds_id') for p in decoder_params):
            with deepspeed.zero.GatheredParameters(decoder_params, modifier_rank=None):
                pred_y = decoder(H_vis)  # (B, S, d_t)
        else:
            pred_y = decoder(H_vis)  # (B, S, d_t)
        teacher_y = teacher_y.to(device=device, dtype=pred_y.dtype)
        corr_mask = corr_mask.to(device)

        num_corrupted = corr_mask.sum().item()
        zero = torch.tensor(0.0, device=device, dtype=pred_y.dtype)

        # --- Reconstruction loss (on corrupted patches) ---
        loss_rec = zero
        if getattr(cfg, 'sgld_use_rec', True) and num_corrupted > 0:
            pred_c = F.normalize(pred_y, dim=-1)
            tgt_c = F.normalize(teacher_y, dim=-1)
            rec_per_patch = ((pred_c - tgt_c) ** 2).sum(dim=-1)  # (B, S)
            loss_rec = rec_per_patch[corr_mask].mean()

        # --- Relational loss (per-image KL on corrupted patches) ---
        loss_rel = zero
        if getattr(cfg, 'sgld_use_rel', True) and num_corrupted > 0:
            tau_r = getattr(cfg, 'sgld_tau_r', 0.10)
            rel_losses = []
            for b in range(B):
                ids = corr_mask[b].nonzero(as_tuple=True)[0]
                k = ids.shape[0]
                if k < 2:
                    continue
                P = F.normalize(pred_y[b, ids], dim=-1)   # (k, d_t)
                T = F.normalize(teacher_y[b, ids], dim=-1) # (k, d_t)
                S_T = T @ T.t()  # (k, k)
                S_S = P @ P.t()  # (k, k)
                p = F.softmax(S_T / tau_r, dim=-1)
                logq = F.log_softmax(S_S / tau_r, dim=-1)
                kl = (p * (torch.log(p + 1e-8) - logq)).sum(dim=-1).mean()
                rel_losses.append(kl)
            if rel_losses:
                loss_rel = torch.stack(rel_losses).mean()

        # --- Contrastive loss (per-image InfoNCE on corrupted patches) ---
        loss_con = zero
        if getattr(cfg, 'sgld_use_con', True) and num_corrupted > 0:
            tau_c = getattr(cfg, 'sgld_tau_c', 0.07)
            con_losses = []
            for b in range(B):
                ids = corr_mask[b].nonzero(as_tuple=True)[0]
                k = ids.shape[0]
                if k < 2:
                    continue
                P = F.normalize(pred_y[b, ids], dim=-1)   # (k, d_t)
                T = F.normalize(teacher_y[b, ids], dim=-1) # (k, d_t)
                logits = (P @ T.t()) / tau_c  # (k, k)
                labels = torch.arange(k, device=device)
                con_losses.append(F.cross_entropy(logits, labels))
            if con_losses:
                loss_con = torch.stack(con_losses).mean()

        return loss_rec, loss_rel, loss_con, int(num_corrupted)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Ensure labels is on the same device as input_ids
        if labels is not None:
            labels = labels.to(input_ids.device)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        # Track vision token positions for SGLD loss computation
        vision_token_starts = []  # per batch item: start index or None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                vision_token_starts.append(None)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # Ensure concatenated input_ids are on the correct device (same as embedding layer)
            cat_input_ids = torch.cat(cur_input_ids_noim)
            embed_device = next(self.get_model().embed_tokens.parameters()).device
            if cat_input_ids.device != embed_device:
                cat_input_ids = cat_input_ids.to(embed_device)
            cur_input_embeds = self.get_model().embed_tokens(cat_input_ids)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            first_image_start = None
            running_len = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                running_len += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    if first_image_start is None:
                        first_image_start = running_len
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    running_len += cur_image_features.shape[0]

            vision_token_starts.append(first_image_start)

            # Use embed_device for consistency (the device of embed_tokens)
            cur_new_input_embeds = [x.to(embed_device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # Adjust vision_token_starts for left-padding if needed
        # and store on self for SGLD loss computation
        padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
        adjusted_positions = []
        for i, start in enumerate(vision_token_starts):
            if start is None:
                adjusted_positions.append(None)
            else:
                cur_len = new_input_embeds[i].shape[0] if i < len(new_input_embeds) else 0
                if padding_side == "left":
                    pad_offset = max_len - cur_len
                    adjusted_positions.append(start + pad_offset)
                else:
                    adjusted_positions.append(start)
        self._sgld_vision_token_starts = adjusted_positions

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
