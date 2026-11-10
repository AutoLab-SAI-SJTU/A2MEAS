"""LaWAM vision features and future decoder used by policy inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn
from a2meas.models.world_layers import LAMDecoder, MultiViewLAMDecoder
from a2meas.models.vision import build_vision_encoder, lam_dino_encode_microbatch

LAM_PRIMARY_CAMERA_KEY = "observation.rgb.zed_link_camera_0"


class LatentWorldModel(nn.Module):
    def __init__(self, vision_model_id, *, camera_order, dim, num_heads,
                 ffn_expansion_factor, dec_layers, code_dim, num_queries,
                 num_embodiments, image_hw, patch_size, latent_layer_to_use,
                 norm_latents, norm_latents_type, decoder_multi_view, decoder_last_ln):
        super().__init__()
        if image_hw != [256, 256] or patch_size != 16:
            raise ValueError("LaWAM requires 256x256 images and 16x16 patches")
        if len(set(camera_order)) != len(camera_order) or camera_order[0] != LAM_PRIMARY_CAMERA_KEY:
            raise ValueError("Invalid LaWAM camera order")
        self.multi_view_camera_keys = camera_order[1:]
        self.num_view_streams = len(camera_order)
        self.decoder_multi_view = decoder_multi_view
        self.code_dim = code_dim
        self.latent_layer_to_use = latent_layer_to_use
        num_latent_layers = len(latent_layer_to_use) if isinstance(latent_layer_to_use, list) else 1
        self.vision_encoder, self.input_dim = build_vision_encoder(
            vision_model_id, num_latent_layers=num_latent_layers,
            norm_layer_type=norm_latents_type, enable_norm=norm_latents,
        )
        decoder = MultiViewLAMDecoder if decoder_multi_view else LAMDecoder
        view_kwargs = {"num_views": self.num_view_streams} if decoder_multi_view else {}
        self.decoder = decoder(
            context_dim=dim, input_dim=self.input_dim, num_queries=num_queries,
            num_layers=dec_layers, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
            num_embodiments=num_embodiments, code_dim=code_dim, grid_hw=(16, 16),
            last_ln=decoder_last_ln, **view_kwargs,
        )

    def _encode_vision_microbatched(self, videos: torch.Tensor, *, n: Any) -> Any:
        """Chunk frozen vision encode along batch dim (result-preserving concat)."""
        microbatch = lam_dino_encode_microbatch()
        batch_size = int(videos.shape[0])
        if microbatch <= 0 or batch_size <= microbatch:
            return self.vision_encoder.encode(videos, n=n)
        chunks: list[Any] = []
        for start in range(0, batch_size, microbatch):
            chunk = videos[start : start + microbatch]
            chunks.append(self.vision_encoder.encode(chunk, n=n))
        if isinstance(chunks[0], (list, tuple)):
            return [
                torch.cat([part[i] for part in chunks], dim=0)
                for i in range(len(chunks[0]))
            ]
        return torch.cat(chunks, dim=0)

    def _extract_multi_view_feature_grid(
        self, primary_videos: torch.Tensor, multi_view_videos: torch.Tensor, *, n: Any
    ) -> torch.Tensor:
        """Return primary+aux DINO features as ``[B,T,V,K,D]``."""
        primary = self.extract_vision_features(primary_videos, n=n)
        mv = multi_view_videos
        streams = [primary]
        for view_idx in range(int(mv.shape[2])):
            feats = self._encode_vision_microbatched(mv[:, :, view_idx], n=n)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            streams.append(feats)
        return torch.stack(streams, dim=2)

    @torch.no_grad()
    def extract_vision_features(
        self, videos: torch.Tensor, *, n: Optional[Any] = -2
    ) -> torch.Tensor:
        """Encode primary-camera clips only → [B,T,K,D] (Stage-2 h_t / perceptual)."""
        n_used = n if n is not None else self.latent_layer_to_use
        feats = self._encode_vision_microbatched(videos, n=n_used)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    @torch.no_grad()
    def extract_multi_view_features(
        self,
        videos: torch.Tensor,
        multi_view_videos: torch.Tensor,
        *,
        n: Optional[Any] = None,
    ) -> torch.Tensor:
        """Encode primary+aux cameras once and keep view axis ``[B,T,V,K,D]``."""
        n_used = self.latent_layer_to_use if n is None else n
        mv = multi_view_videos
        return self._extract_multi_view_feature_grid(videos, mv, n=n_used)


def build_latent_action_model(config_path, vision_model_id, camera_order):
    architecture = json.loads(Path(config_path).read_text())
    if architecture["camera_order"] != camera_order:
        raise ValueError("LaWAM architecture camera order differs from the policy checkpoint")
    return LatentWorldModel(vision_model_id, **architecture)
