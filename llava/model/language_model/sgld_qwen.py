"""
SGLD (Saliency-Guided Latent Denoising) for Qwen-2.5-VL.

Subclasses Qwen2_5_VLForConditionalGeneration to add SGLD auxiliary losses
during training. All SGLD logic mirrors the LLaVA implementation in
llava_llama.py / llava_arch.py, adapted for Qwen's architecture:
  - Teacher: post-merger features (3584-dim, out_hidden_size)
    NOTE: Qwen's pre-merger ViT features (1280-dim) have near-zero diversity
    (pairwise cosine sim ~0.99, effective rank ~20) because the final ViT
    layer encodes information in magnitude, not direction.  Post-merger
    features (3584-dim) have healthy diversity (cosine sim ~0.28, eff rank
    ~240) and serve as the teacher target.
  - Student dim: 3584 (Qwen LLM hidden_size)
  - Decoder: 3584 → 3584 → 3584
  - Mid-layer: 13/28 (analogous to 15/32 in LLaVA)
  - Saliency: L2-norm on post-merger features (no CLS token in Qwen's ViT)
  - Per-image corruption (variable resolution, not batched)
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import deepspeed
    _has_deepspeed = True
except ImportError:
    _has_deepspeed = False

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from llava.model.llava_arch import SGLDDecoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class SGLDQwenConfig(Qwen2_5_VLConfig):
    model_type = "sgld_qwen2_5_vl"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class SGLDQwenCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """Extended output that includes SGLD loss components."""
    loss_lang: Optional[torch.FloatTensor] = None
    loss_rec: Optional[torch.FloatTensor] = None
    loss_rel: Optional[torch.FloatTensor] = None
    loss_con: Optional[torch.FloatTensor] = None
    num_corrupted_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SgldQwenForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen-2.5-VL with SGLD auxiliary losses for robust vision training."""

    config_class = SGLDQwenConfig

    def __init__(self, config):
        super().__init__(config)
        # Parent creates self.model (Qwen2_5_VLModel) and self.lm_head

    # -----------------------------------------------------------------
    # SGLD module initialization
    # -----------------------------------------------------------------
    def initialize_sgld_modules(self, model_args):
        """Create SGLD decoder, mask embedding, and tau embedding.

        Mirrors LlavaMetaModel.initialize_sgld_modules (llava_arch.py L120-176).
        Modules are attached to self.model so DeepSpeed discovers them.
        """
        hidden_size = self.config.text_config.hidden_size            # 3584 (d_h)
        # Teacher dim = post-merger out_hidden_size = hidden_size (3584)
        teacher_dim = self.config.vision_config.out_hidden_size      # 3584

        # Learnable mask embedding  e_mask ∈ R^{d_h}
        self.model.sgld_mask_embed = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.model.sgld_mask_embed, std=0.02)

        # Decoder head  D: R^{d_h} → R^{d_t}  (3584 → 3584)
        self.model.sgld_decoder = SGLDDecoder(hidden_size, teacher_dim)

        # Tau conditioning embedding
        tau_bins = getattr(model_args, 'sgld_tau_bins', 8)
        if getattr(model_args, 'sgld_use_tau_embed', True) and tau_bins > 0:
            self.model.sgld_tau_embed = nn.Embedding(tau_bins, hidden_size)
            nn.init.normal_(self.model.sgld_tau_embed.weight, std=0.02)
        else:
            self.model.sgld_tau_embed = None

        # Store SGLD config (same attribute names as LLaVA for consistency)
        cfg = self.config
        cfg.sgld_enable = getattr(model_args, 'sgld_enable', True)
        cfg.sgld_stage1_enable = getattr(model_args, 'sgld_stage1_enable', True)
        cfg.sgld_stage2_enable = getattr(model_args, 'sgld_stage2_enable', True)
        cfg.sgld_rho_noise = getattr(model_args, 'sgld_rho_noise', 0.20)
        cfg.sgld_rho_mask = getattr(model_args, 'sgld_rho_mask', 0.03)
        cfg.sgld_tau_s = getattr(model_args, 'sgld_tau_s', 0.07)
        cfg.sgld_sigma = getattr(model_args, 'sgld_sigma', 1.0)
        cfg.sgld_tau_max = getattr(model_args, 'sgld_tau_max', 0.30)
        cfg.sgld_use_rec = getattr(model_args, 'sgld_use_rec', True)
        cfg.sgld_use_rel = getattr(model_args, 'sgld_use_rel', True)
        cfg.sgld_use_con = getattr(model_args, 'sgld_use_con', True)
        cfg.sgld_lambda_rec = getattr(model_args, 'sgld_lambda_rec', 0.2)
        cfg.sgld_lambda_rel = getattr(model_args, 'sgld_lambda_rel', 0.05)
        cfg.sgld_lambda_con = getattr(model_args, 'sgld_lambda_con', 0.05)
        cfg.sgld_tau_r = getattr(model_args, 'sgld_tau_r', 0.10)
        cfg.sgld_tau_c = getattr(model_args, 'sgld_tau_c', 0.07)
        cfg.sgld_teacher = getattr(model_args, 'sgld_teacher', 'vision_tower')
        cfg.sgld_saliency_type = getattr(model_args, 'sgld_saliency_type', 'l2_norm')
        cfg.sgld_lambda_schedule = getattr(model_args, 'sgld_lambda_schedule', 'none')
        cfg.sgld_mode = getattr(model_args, 'sgld_mode', 'scale')
        cfg.sgld_p_max = getattr(model_args, 'sgld_p_max', 0.5)
        cfg.sgld_tau_bins = tau_bins
        cfg.sgld_use_tau_embed = getattr(model_args, 'sgld_use_tau_embed', True)
        cfg.sgld_schedule_type = getattr(model_args, 'sgld_schedule_type', 'whd')
        cfg.sgld_warmup_ratio = getattr(model_args, 'sgld_warmup_ratio', 0.05)
        cfg.sgld_decay_ratio = getattr(model_args, 'sgld_decay_ratio', 0.20)
        cfg.sgld_student_layer = getattr(model_args, 'sgld_student_layer', -1)

    # -----------------------------------------------------------------
    # Vision forward with saliency (replaces clip_encoder.forward_with_saliency)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def visual_forward_with_saliency(self, pixel_values, image_grid_thw):
        """Run Qwen's ViT and extract post-merger features as teacher + saliency.

        Uses the native visual.forward() as a single module call for
        DeepSpeed ZeRO-3 compatibility (avoids manual block iteration
        which can break parameter lifecycle management).

        Teacher features are the post-merger outputs (3584-dim) rather than
        pre-merger ViT features (1280-dim), because Qwen's final ViT layer
        has near-zero feature diversity (cosine sim ~0.99, eff rank ~20).
        Post-merger features have healthy diversity (cosine sim ~0.28).

        Returns:
            merged_features  : (N_merged, out_hidden_size)  post-merger, for LLM input
            teacher_features : (N_merged, out_hidden_size)  post-merger (detached), teacher target
            saliency         : (N_merged,)                   L2-norm based patch saliency
        """
        visual = self.model.visual
        pixel_values = pixel_values.type(visual.dtype)

        # Single native module call — DeepSpeed handles parameter lifecycle
        vision_outputs = visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)

        # pooler_output: post-merger (N_merged, 3584), already reverse-indexed
        merged_features = vision_outputs.pooler_output

        # Teacher = post-merger features (same tensor, detached in encode_images_with_sgld)
        teacher_features = merged_features
        saliency = teacher_features.norm(dim=-1)  # (N_merged,) — raw L2 norms

        return merged_features, teacher_features, saliency

    # -----------------------------------------------------------------
    # SGLD corruption (replaces llava_arch.encode_images_with_sgld)
    # -----------------------------------------------------------------
    def encode_images_with_sgld(self, pixel_values, image_grid_thw, schedule_mult=1.0):
        """Encode images with saliency-guided corruption.

        Returns:
            corrupted_embeds : tuple of (N_i, d_h) tensors  (matches get_image_features format)
            teacher_y_list   : list of (N_i, d_t) tensors   (detached teacher targets)
            corr_mask_list   : list of (N_i,) bool tensors   (corruption masks)
        """
        merged, teacher, saliency = self.visual_forward_with_saliency(
            pixel_values, image_grid_thw
        )

        # Split by per-image token counts
        merge_size = self.model.visual.spatial_merge_size
        split_sizes = (image_grid_thw.prod(-1) // (merge_size ** 2)).tolist()
        merged_splits = torch.split(merged, split_sizes)
        teacher_splits = torch.split(teacher, split_sizes)
        saliency_splits = torch.split(saliency, split_sizes)

        cfg = self.config
        rho_noise_base = getattr(cfg, 'sgld_rho_noise', 0.10) * schedule_mult
        rho_mask_base = getattr(cfg, 'sgld_rho_mask', 0.02) * schedule_mult
        tau_s = getattr(cfg, 'sgld_tau_s', 0.07)
        sigma = getattr(cfg, 'sgld_sigma', 1.0)
        tau_max = getattr(cfg, 'sgld_tau_max', 0.15) * schedule_mult

        corrupted_list = []
        teacher_y_list = []
        corr_mask_list = []

        for img_idx in range(len(split_sizes)):
            features = merged_splits[img_idx]               # (S, d_h)
            t_y = teacher_splits[img_idx].detach()           # (S, d_t)
            sal_raw = saliency_splits[img_idx].float()        # (S,) raw L2 norms
            sal = sal_raw / sal_raw.sum()                     # L1-normalize to distribution
            S = features.shape[0]
            device = features.device
            dtype = features.dtype

            # Per-image dynamic corruption counts, matching LLaVA SGLD
            # (llava_arch.py L271-272).  The always-on corruption fix keeps
            # mask_embed/tau_embed in the deep gradient path even when K=0,
            # so uniform counts across ranks are no longer needed.
            K_N = int(rho_noise_base * S)
            K_M = int(rho_mask_base * S)

            corrupted = features.clone()
            mask_N = torch.zeros(S, dtype=torch.bool, device=device)
            mask_M = torch.zeros(S, dtype=torch.bool, device=device)

            # --- Saliency-guided sampling ---
            # Qwen uses L1-normalized L2 norms as saliency (no CLS token).
            # sal is already a valid distribution (sums to 1, proportional to
            # patch importance).  Use it directly as p_N (noise targets
            # high-saliency) and inverse-saliency for p_M (mask targets
            # low-saliency).  No outer softmax needed — unlike LLaVA where
            # CLS-attention saliency was passed through softmax(sal/tau_s).
            if K_N > 0:
                p_N = sal  # already L1-normalized distribution
                idx_N = torch.multinomial(p_N, K_N, replacement=False)
                mask_N[idx_N] = True

                if K_M > 0:
                    p_M = 1.0 / (sal + 1e-8)  # inverse saliency
                    p_M[idx_N] = 0.0
                    p_M_sum = p_M.sum()
                    if p_M_sum > 0:
                        p_M = p_M / p_M_sum
                        idx_M = torch.multinomial(
                            p_M, min(K_M, (p_M > 0).sum().item()), replacement=False
                        )
                        mask_M[idx_M] = True
            elif K_M > 0:
                p_M = 1.0 / (sal + 1e-8)  # inverse saliency
                p_M = p_M / p_M.sum()
                idx_M = torch.multinomial(p_M, K_M, replacement=False)
                mask_M[idx_M] = True

            # --- Noise corruption: z_noised = (1-τ)*z + τ*ε ---
            # Always compute noise and tau_embed, using float mask to
            # zero-out the contribution when K_N=0.  This keeps tau_embed
            # in the deep gradient path (through 28 transformer layers)
            # on every forward, preventing DeepSpeed IPG bucket de-sync
            # when K_N transitions from 0 → non-zero.
            tau = torch.rand(S, 1, device=device, dtype=dtype) * tau_max
            eps = torch.randn_like(corrupted) * sigma
            noised = (1.0 - tau) * features + tau * eps

            noise_weight = mask_N.float().unsqueeze(-1)  # (S, 1): 1 where noised, 0 elsewhere
            corrupted = corrupted * (1.0 - noise_weight) + noised * noise_weight

            # Tau conditioning embedding — always applied
            tau_embed_table = getattr(self.model, 'sgld_tau_embed', None)
            if tau_embed_table is not None:
                tau_bins_K = getattr(cfg, 'sgld_tau_bins', 8)
                tau_max_base = getattr(cfg, 'sgld_tau_max', 0.15)
                bin_ids = torch.floor(
                    tau.squeeze(-1) / max(tau_max_base, 1e-8) * tau_bins_K
                ).long().clamp(0, tau_bins_K - 1)
                embed_vals = tau_embed_table(
                    bin_ids.to(tau_embed_table.weight.device)
                ).to(device=device, dtype=dtype)           # (S, d_h)
                corrupted = corrupted + embed_vals * noise_weight

            # --- Mask corruption: z_masked = e_mask ---
            # Always apply mask_embed, using float mask to zero-out when
            # K_M=0.  Same rationale as noise above.
            mask_embed = self.model.sgld_mask_embed.to(
                device=device, dtype=dtype
            )
            mask_weight = mask_M.float().unsqueeze(-1)    # (S, 1): 1 where masked, 0 elsewhere
            corrupted = corrupted * (1.0 - mask_weight) + mask_embed.unsqueeze(0) * mask_weight

            corr_mask = mask_N | mask_M
            corrupted_list.append(corrupted)
            teacher_y_list.append(t_y)
            corr_mask_list.append(corr_mask)

        return tuple(corrupted_list), teacher_y_list, corr_mask_list

    # -----------------------------------------------------------------
    # SGLD loss computation (mirrors llava_arch.compute_sgld_losses)
    # -----------------------------------------------------------------
    def compute_sgld_losses(self, H_vis_list, teacher_y_list, corr_mask_list):
        """Compute reconstruction, relational, and contrastive SGLD losses.

        Args:
            H_vis_list    : list of (N_i, d_h) — LLM hidden states at vision positions
            teacher_y_list: list of (N_i, d_t) — teacher targets
            corr_mask_list: list of (N_i,) bool — corruption masks
        Returns:
            loss_rec, loss_rel, loss_con, num_corrupted
        """
        cfg = self.config
        device = H_vis_list[0].device if H_vis_list else torch.device('cuda')
        decoder = self.model.sgld_decoder
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Decode with DeepSpeed GatheredParameters if needed
        decoder_params = list(decoder.parameters())
        use_gathered = _has_deepspeed and any(hasattr(p, 'ds_id') for p in decoder_params)

        total_corrupted = 0
        rec_losses = []
        rel_losses = []
        con_losses = []

        for img_idx in range(len(H_vis_list)):
            h_vis = H_vis_list[img_idx]          # (N_i, d_h)
            t_y = teacher_y_list[img_idx]        # (N_i, d_t)
            c_m = corr_mask_list[img_idx]        # (N_i,)

            if h_vis.shape[0] == 0:
                continue

            # Trim to matching length
            min_len = min(h_vis.shape[0], t_y.shape[0], c_m.shape[0])
            h_vis = h_vis[:min_len]
            t_y = t_y[:min_len].to(device=device)
            c_m = c_m[:min_len].to(device)

            k = c_m.sum().item()
            if k == 0:
                continue
            total_corrupted += k

            # Decoder forward (ensure device + dtype alignment)
            decoder_device = next(decoder.parameters()).device
            decoder_dtype = next(decoder.parameters()).dtype
            h_vis_dec = h_vis.to(device=decoder_device, dtype=decoder_dtype)
            if use_gathered:
                with deepspeed.zero.GatheredParameters(decoder_params, modifier_rank=None):
                    pred_y = decoder(h_vis_dec)
            else:
                pred_y = decoder(h_vis_dec)
            pred_y = pred_y.to(device).float()
            t_y_f = t_y.float()

            # --- Reconstruction loss (MSE on normalized corrupted patches) ---
            if getattr(cfg, 'sgld_use_rec', True):
                pred_c = F.normalize(pred_y, dim=-1)
                tgt_c = F.normalize(t_y_f, dim=-1)
                rec_per_patch = ((pred_c - tgt_c) ** 2).sum(dim=-1)
                rec_losses.append(rec_per_patch[c_m].mean())

            # --- Relational loss (KL on similarity matrices) ---
            if getattr(cfg, 'sgld_use_rel', True):
                tau_r = getattr(cfg, 'sgld_tau_r', 0.10)
                ids = c_m.nonzero(as_tuple=True)[0]
                if ids.shape[0] >= 2:
                    P = F.normalize(pred_y[ids], dim=-1)
                    T = F.normalize(t_y_f[ids], dim=-1)
                    S_T = T @ T.t()
                    S_S = P @ P.t()
                    p = F.softmax(S_T / tau_r, dim=-1)
                    logq = F.log_softmax(S_S / tau_r, dim=-1)
                    kl = (p * (torch.log(p + 1e-8) - logq)).sum(dim=-1).mean()
                    rel_losses.append(kl)

            # --- Contrastive loss (InfoNCE) ---
            if getattr(cfg, 'sgld_use_con', True):
                tau_c = getattr(cfg, 'sgld_tau_c', 0.07)
                ids = c_m.nonzero(as_tuple=True)[0]
                if ids.shape[0] >= 2:
                    P = F.normalize(pred_y[ids], dim=-1)
                    T = F.normalize(t_y_f[ids], dim=-1)
                    logits = (P @ T.t()) / tau_c
                    labels = torch.arange(ids.shape[0], device=device)
                    con_losses.append(F.cross_entropy(logits, labels))

        loss_rec = torch.stack(rec_losses).mean() if rec_losses else zero
        loss_rel = torch.stack(rel_losses).mean() if rel_losses else zero
        loss_con = torch.stack(con_losses).mean() if con_losses else zero

        return loss_rec, loss_rel, loss_con, int(total_corrupted)

    # -----------------------------------------------------------------
    # Forward (overrides Qwen2_5_VLForConditionalGeneration.forward)
    # -----------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # SGLD args
        sgld_global_step: Optional[int] = None,
        sgld_max_steps: Optional[int] = None,
        **kwargs,
    ):
        # ---- 1. Determine if SGLD should apply ----
        sgld_enable = getattr(self.config, 'sgld_enable', False)
        sgld_mode = getattr(self.config, 'sgld_mode', 'scale')
        sgld_apply = (
            sgld_enable and sgld_mode != 'none'
            and self.training and pixel_values is not None
        )
        if sgld_apply:
            sgld_apply = getattr(self.config, 'sgld_stage2_enable', True)

        # ---- 2. WHD schedule multiplier (llava_llama.py L140-163) ----
        schedule_mult = 1.0
        if sgld_global_step is not None and sgld_max_steps is not None and sgld_max_steps > 0:
            schedule_type = getattr(self.config, 'sgld_schedule_type', 'whd')
            progress = min(sgld_global_step / sgld_max_steps, 1.0)
            if schedule_type == 'whd':
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
                schedule_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif schedule_type == 'linear':
                schedule_mult = max(0.0, 1.0 - progress)

        # ---- 3. SGLD corruption path ----
        teacher_y_list = None
        corr_mask_list = None
        original_get_image_features = None

        if sgld_apply and pixel_values is not None:
            corrupted_embeds, teacher_y_list, corr_mask_list = \
                self.encode_images_with_sgld(pixel_values, image_grid_thw, schedule_mult)

            # Monkey-patch get_image_features to return corrupted embeddings
            # Parent expects BaseModelOutputWithPooling with .pooler_output = tuple of per-image tensors
            original_get_image_features = self.model.get_image_features
            def patched_get_image_features(*args, **kwargs):
                from transformers.modeling_outputs import BaseModelOutputWithPooling
                return BaseModelOutputWithPooling(
                    pooler_output=corrupted_embeds,
                    last_hidden_state=None,
                )
            self.model.get_image_features = patched_get_image_features

        # ---- 4. Record vision token positions ----
        vision_positions = None
        if sgld_apply and input_ids is not None:
            image_token_id = self.config.image_token_id
            vision_positions = []
            for b in range(input_ids.shape[0]):
                pos = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
                vision_positions.append(pos)

        # ---- 5. Mid-layer capture ----
        sgld_student_layer = getattr(self.config, 'sgld_student_layer', -1)
        request_hidden_states = output_hidden_states
        if sgld_apply and sgld_student_layer >= 0:
            request_hidden_states = True

        # ---- 6. Parent model forward ----
        _output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        _output_hidden_states = request_hidden_states if request_hidden_states is not None else self.config.output_hidden_states

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=_output_attentions,
            output_hidden_states=_output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # ---- 7. Restore monkey-patch ----
        if original_get_image_features is not None:
            self.model.get_image_features = original_get_image_features

        hidden_states = outputs.last_hidden_state  # (B, seq_len, 3584)

        # ---- 8. Logits + language loss ----
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss_lang = None
        if labels is not None:
            loss_lang = self.loss_function(
                logits=logits, labels=labels,
                vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        # ---- 9. SGLD losses ----
        target_device = hidden_states.device
        zero = torch.tensor(0.0, device=target_device, dtype=hidden_states.dtype)
        loss_rec, loss_rel, loss_con = zero, zero, zero
        num_corrupted = 0

        if sgld_apply and teacher_y_list is not None and vision_positions is not None:
            # Select feature source: mid-layer or final
            if sgld_student_layer >= 0 and outputs.hidden_states is not None:
                feature_source = outputs.hidden_states[sgld_student_layer + 1]
            else:
                feature_source = hidden_states

            # Extract H_vis per image
            # Map batch items → images. For mix665k, typically 1 image per sample.
            merge_size = self.model.visual.spatial_merge_size
            per_image_counts = (image_grid_thw.prod(-1) // (merge_size ** 2)).tolist()

            H_vis_list = []
            matched_teacher = []
            matched_mask = []

            img_cursor = 0
            for b in range(feature_source.shape[0]):
                if b >= len(vision_positions) or vision_positions[b].numel() == 0:
                    continue

                pos = vision_positions[b]
                pos = pos[pos < feature_source.shape[1]]  # clamp to seq_len
                h_vis_b = feature_source[b, pos, :]         # (total_vis_tokens_in_b, d_h)

                # Split h_vis_b by images in this sample
                # Count how many images belong to this batch item
                offset = 0
                while img_cursor < len(per_image_counts):
                    n_tok = per_image_counts[img_cursor]
                    if offset + n_tok > h_vis_b.shape[0]:
                        break
                    H_vis_list.append(h_vis_b[offset:offset + n_tok])
                    matched_teacher.append(teacher_y_list[img_cursor])
                    matched_mask.append(corr_mask_list[img_cursor])
                    offset += n_tok
                    img_cursor += 1

            if H_vis_list:
                loss_rec, loss_rel, loss_con, num_corrupted = \
                    self.compute_sgld_losses(H_vis_list, matched_teacher, matched_mask)

        # ---- 10. Combine losses ----
        total_loss = None
        if loss_lang is not None:
            total_loss = loss_lang.clone()
            if sgld_apply and num_corrupted > 0:
                lam_rec = getattr(self.config, 'sgld_lambda_rec', 0.10) * schedule_mult
                lam_rel = getattr(self.config, 'sgld_lambda_rel', 0.025) * schedule_mult
                lam_con = getattr(self.config, 'sgld_lambda_con', 0.025) * schedule_mult
                if getattr(self.config, 'sgld_use_rec', True):
                    total_loss = total_loss + lam_rec * loss_rec
                if getattr(self.config, 'sgld_use_rel', True):
                    total_loss = total_loss + lam_rel * loss_rel
                if getattr(self.config, 'sgld_use_con', True):
                    total_loss = total_loss + lam_con * loss_con

        # ---- 10b. Dummy decoder forward for DeepSpeed ZeRO-2 consistency ----
        # During WHD warmup, schedule_mult is small → rho*schedule_mult*S < 1
        # → K_N=K_M=0 for smaller images.  Variable resolution means some
        # ranks may have num_corrupted>0 (decoder called) while others have
        # num_corrupted==0 (decoder skipped), creating an IPG bucket desync
        # (NCCL ALLREDUCE timeout).  A zero-contribution decoder forward
        # ensures its AccumulateGrad hooks fire on every rank every step.
        # (Analogous to llava_llama.py L306-311.)
        if sgld_apply and num_corrupted == 0 and total_loss is not None:
            decoder = self.model.sgld_decoder
            _dummy_in = torch.zeros(
                1, hidden_states.shape[-1],
                device=target_device, dtype=hidden_states.dtype,
            )
            _dummy_out = decoder(_dummy_in)
            total_loss = total_loss + _dummy_out.sum() * 0.0

        return SGLDQwenCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            loss_lang=loss_lang,
            loss_rec=loss_rec if sgld_apply else None,
            loss_rel=loss_rel if sgld_apply else None,
            loss_con=loss_con if sgld_apply else None,
            num_corrupted_tokens=num_corrupted if sgld_apply else None,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
AutoConfig.register("sgld_qwen2_5_vl", SGLDQwenConfig)
AutoModelForCausalLM.register(SGLDQwenConfig, SgldQwenForConditionalGeneration)
