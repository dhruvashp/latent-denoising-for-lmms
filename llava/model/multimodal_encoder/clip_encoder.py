import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
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
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

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

        Extracts attention from the same encoder layer used for feature
        selection (self.select_layer) rather than always using the last layer.

        For models with a CLS token (position 0), saliency is the CLS-to-patch
        attention averaged over heads.  For models without a CLS token, saliency
        is the column-mean of the head-averaged attention matrix — i.e. how much
        attention each patch receives from all other patches.

        Returns:
            image_features: (B, S, d_v) patch features (same as forward())
            saliency: (B, S) per-patch saliency scores (non-negative, sums to ~1)
        """
        images = images.to(device=self.device, dtype=self.dtype)

        if saliency_type == 'cls_attn':
            encoder = self.vision_tower.vision_model.encoder
            embeddings = self.vision_tower.vision_model.embeddings(images)
            hidden_states = self.vision_tower.vision_model.pre_layrnorm(embeddings) \
                if hasattr(self.vision_tower.vision_model, 'pre_layrnorm') else embeddings

            num_layers = len(encoder.layers)
            # all_hidden_states will have num_layers + 1 entries (input + N
            # layer outputs), matching the HuggingFace hidden_states format so
            # that feature_select(out)[self.select_layer] picks the same tensor
            # as in the standard forward() path.
            hs_count = num_layers + 1
            target_hs_idx = self.select_layer % hs_count  # works for negative indices
            # Entry k (k >= 1) is produced by encoder.layers[k - 1].
            target_layer_idx = max(target_hs_idx - 1, 0)

            all_hidden_states = [hidden_states]
            saliency_attn = None

            for i, layer in enumerate(encoder.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=(i == target_layer_idx),
                )
                hidden_states = layer_outputs[0]
                all_hidden_states.append(hidden_states)
                if i == target_layer_idx:
                    saliency_attn = layer_outputs[1]  # (B, heads, seq, seq)

            # Build a namespace-like object so feature_select works
            class _Out:
                pass
            out = _Out()
            out.hidden_states = all_hidden_states

            image_features = self.feature_select(out).to(images.dtype)

            # Head-averaged attention: (B, seq, seq)
            attn_avg = saliency_attn.mean(dim=1)

            has_cls = (attn_avg.shape[-1] == self.num_patches + 1)
            if has_cls:
                # CLS at position 0 → use its attention over patch positions
                saliency = attn_avg[:, 0, 1:]  # (B, num_patches)
            else:
                # No CLS token: for each patch, average the attention it
                # *receives* from every other patch (column-mean).
                saliency = attn_avg.mean(dim=-2)  # (B, num_patches)
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



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
