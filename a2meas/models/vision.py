"""Frozen DINO encoding and inference settings."""

from __future__ import annotations

import os
from typing import Union, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def lam_dino_bf16_enabled() -> bool:
    """Autocast frozen DINO encode in bf16 (default on)."""
    return _env_bool("LAM_DINO_BF16", True)


def lam_dino_fast_penultimate_enabled() -> bool:
    """Skip final DINO block when ``latent_layer_to_use=-2`` (bit-exact vs hidden_states[-2])."""
    return _env_bool("LAM_DINO_FAST_PENULTIMATE", True)


def lam_dino_encode_microbatch() -> int:
    """Maximum DINO encoding batch size; zero disables microbatching."""
    raw = os.environ.get("LAM_DINO_ENCODE_MICROBATCH", "32").strip()
    if not raw or raw.lower() in {"0", "false", "none", "off"}:
        return 0
    value = int(raw)
    if value < 1:
        raise ValueError("LAM_DINO_ENCODE_MICROBATCH must be positive or 0/off")
    return value


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_latent_layers: int = 1,
        norm_layer_type: str = "l2",
        enable_norm: bool = False,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_latent_layers = max(int(num_latent_layers), 1)
        self.norm_layer_type = norm_layer_type
        self.enable_norm = enable_norm
        if enable_norm and norm_layer_type not in {"bn", "ln", "l2"}:
            raise ValueError(f"Invalid DINO normalization: {norm_layer_type}")
        config = AutoConfig.from_pretrained(self.model_id, local_files_only=True)
        model = AutoModel.from_config(config, dtype=torch.float32)
        model.eval()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.feature_dim = int(self.model.config.hidden_size)
        if self.norm_layer_type in ("bn", "ln"):
            if self.norm_layer_type == "bn":
                norm_builder = lambda: nn.SyncBatchNorm(self.feature_dim, affine=False)
            else:
                norm_builder = lambda: nn.LayerNorm(
                    self.feature_dim, elementwise_affine=False
                )
            self.latent_norms = nn.ModuleList(
                [norm_builder() for _ in range(self.num_latent_layers)]
            )
        else:
            self.latent_norms = None


    def train(self, mode=True):
        return super().train(False)

    def _apply_token_norm(
        self, tokens: torch.Tensor, *, norm_idx: int = 0
    ) -> torch.Tensor:
        if not self.enable_norm:
            return tokens
        if self.norm_layer_type == "bn":
            tokens_2d = tokens.reshape(-1, self.feature_dim)
            tokens_2d = self.latent_norms[norm_idx](tokens_2d)
            return tokens_2d.view(tokens.shape[0], tokens.shape[1], self.feature_dim)
        if self.norm_layer_type == "ln":
            return self.latent_norms[norm_idx](tokens)
        return F.normalize(tokens, p=2, dim=-1)

    def _encode_penultimate_fast(self, images: torch.Tensor) -> torch.Tensor:
        """Bit-exact vs ``hidden_states[-2]``: run all but the last transformer block."""
        model = self.model
        pixel_values = images.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(pixel_values, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(pixel_values)
        for layer_module in model.layer[:-1]:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        remove_cls: bool = True,
        n: Union[int, Sequence] = -1,
    ) -> torch.Tensor:
        if images.dim() != 4:
            B, T, C, H, W = images.shape
            images = images.reshape(
                -1, images.shape[-3], images.shape[-2], images.shape[-1]
            )
        else:
            B, C, H, W = images.shape
            T = 1
        use_dino_bf16 = lam_dino_bf16_enabled() and images.is_cuda
        if isinstance(n, int) and n == -2 and lam_dino_fast_penultimate_enabled():
            if use_dino_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    layer_tokens = self._encode_penultimate_fast(images)
            else:
                layer_tokens = self._encode_penultimate_fast(images)
            if remove_cls:
                layer_tokens = layer_tokens[:, 5:, :]
            layer_tokens = self._apply_token_norm(layer_tokens, norm_idx=0)
            return layer_tokens.reshape(B, T, -1, self.feature_dim).detach()
        need_all_layers = not (isinstance(n, int) and n == -1)
        if use_dino_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    pixel_values=images, output_hidden_states=need_all_layers
                )
        else:
            outputs = self.model(
                pixel_values=images, output_hidden_states=need_all_layers
            )
        if not need_all_layers:
            last = outputs.last_hidden_state
            tokens = last[:, 5:, :] if remove_cls else last
            tokens = self._apply_token_norm(tokens, norm_idx=0)
            return tokens.reshape(B, T, -1, self.feature_dim).detach()
        hidden_states = outputs.hidden_states
        if isinstance(n, int):
            list_n = [n]
        else:
            list_n = n
        assert len(list_n) <= self.num_latent_layers, (
            f"DINOv3Encoder expected at most {self.num_latent_layers} normalization layers, but received {len(list_n)} feature layers. Ensure this matches len(latent_layer_to_use)."
        )
        features = []
        for idx, i in enumerate(list_n):
            layer_tokens = hidden_states[i]
            if remove_cls:
                layer_tokens = layer_tokens[:, 5:, :]
            layer_tokens = self._apply_token_norm(layer_tokens, norm_idx=idx)
            features.append(layer_tokens.reshape(B, T, -1, self.feature_dim))
        return (
            features[0].detach()
            if isinstance(n, int)
            else [f.detach() for f in features]
        )


def build_vision_encoder(
    model_id, num_latent_layers=1, norm_layer_type="l2", enable_norm=False
):
    encoder = DINOv3Encoder(
        model_id=model_id,
        num_latent_layers=num_latent_layers,
        norm_layer_type=norm_layer_type,
        enable_norm=enable_norm,
    )
    return (encoder, encoder.feature_dim)
