import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig


class SigLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        # Force eager attention so we can extract attention weights for saliency.
        # SDPA (the default) fuses the attention computation and doesn't return weights.
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map, attn_implementation="eager"
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # SigLIP has no CLS token — all positions are patch tokens.
        # 'patch' and 'cls_patch' both return the full sequence.
        if self.select_feature in ('patch', 'cls_patch'):
            return image_features
        raise ValueError(f'Unexpected select feature: {self.select_feature}')

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @torch.no_grad()
    def forward_with_saliency(self, images, saliency_type='cls_attn'):
        """
        Forward pass that also returns per-patch saliency scores.

        SigLIP has no CLS token, so attention-based saliency uses the
        column-mean: for each patch, the average attention it *receives*
        from all other patches, averaged across heads.

        Returns:
            image_features: (B, S, d_v) patch features (same as forward())
            saliency: (B, S) per-patch saliency scores (non-negative)
        """
        images = images.to(device=self.device, dtype=self.dtype)

        if saliency_type == 'cls_attn':
            encoder = self.vision_tower.vision_model.encoder
            num_layers = len(encoder.layers)
            hs_count = num_layers + 1
            target_hs_idx = self.select_layer % hs_count
            target_layer_idx = max(target_hs_idx - 1, 0)

            # Use a hook to capture attention weights from the target layer,
            # since SigLIP's encoder layer discards them internally.
            captured_attn = {}
            def attn_hook(module, args, output):
                # SiglipAttention.forward returns (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    captured_attn['weights'] = output[1]

            target_attn = encoder.layers[target_layer_idx].self_attn
            handle = target_attn.register_forward_hook(attn_hook)

            try:
                image_forward_outs = self.vision_tower(
                    images, output_hidden_states=True
                )
            finally:
                handle.remove()

            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            saliency_attn = captured_attn.get('weights')
            if saliency_attn is not None:
                # Column-mean saliency: how much attention each patch receives
                attn_avg = saliency_attn.mean(dim=1)      # (B, seq, seq)
                saliency = attn_avg.mean(dim=-2)           # (B, seq)
            else:
                # Fallback if attention capture failed
                saliency = image_features.norm(dim=-1)
        else:
            # Fallback: use patch L2 norm as saliency
            image_forward_outs = self.vision_tower(
                images, output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            saliency = image_features.norm(dim=-1)  # (B, S)

        return image_features, saliency

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
