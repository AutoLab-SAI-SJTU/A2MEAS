# Derived from LeRobot / OpenPI, Apache-2.0. See README.md.
"""Shared LaWAM query, visual-prefix and alternate-KV conditioning."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from a2meas.models.action_expert import _key_rotary_position_embeddings, _apply_rotary_key_only


def lawam_inject_mode_is_alternate_kv(inject_mode: str) -> bool:
    return inject_mode == "alternate_kv"


def compute_alternate_kv_rope_position_ids(
    *, prefix_pad_masks: torch.Tensor, suffix_pad_masks: torch.Tensor, n_cond: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """LaWAM-key and suffix-query RoPE ids for ``alternate_kv`` (shared axis).

    VLM prefix ``0..L-1``, cond ``L..L+n_cond-1``, suffix ``L+n_cond…``.
    Relative Δ(Q_i, cond_j) = ``n_cond+i-j`` (independent of prompt length ``L``).
    Even (VLM-KV) layers reuse the same suffix ids, leaving a length-``n_cond`` gap
    between the last VLM key and the first suffix query.
    """
    if int(n_cond) < 1:
        raise ValueError(f"n_cond must be >= 1, got {n_cond}")
    device = suffix_pad_masks.device
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    cond_arange = torch.arange(int(n_cond), device=device)[None, :]
    lawam_position_ids = prefix_offsets + cond_arange
    suffix_position_ids = prefix_offsets + int(n_cond) + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return (lawam_position_ids, suffix_position_ids)


def lawam_layer_uses_vlm_kv(
    layer_idx: int,
    *,
    alternate_first: str = "vlm",
    alternate_period: int = 1,
) -> bool:
    phase = (layer_idx // alternate_period) % 2
    return phase == (0 if alternate_first == "vlm" else 1)


def lawam_cond_kv_layer_indices(
    num_layers: int, *, alternate_first: str = "vlm", alternate_period: int = 1
) -> list[int]:
    """Expert layer indices that attend to LaWAM cond KV (not VLM prefix KV)."""
    indices = [
        layer_idx
        for layer_idx in range(int(num_layers))
        if not lawam_layer_uses_vlm_kv(
            layer_idx, alternate_first=alternate_first, alternate_period=alternate_period
        )
    ]
    if not indices:
        raise ValueError(
            f"LaWAMKVBank schedule assigns zero cond-KV layers (num_layers={num_layers}, alternate_first={alternate_first!r}, alternate_period={alternate_period})"
        )
    return indices


def encode_lawam_visual_cond(
    prefix_adapter: LaWAMPrefixAdapter,
    *,
    h_t: torch.Tensor,
    uhat: torch.Tensor,
    prefix_mode: str,
    inject_current: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build expert-dim cond tokens from DINO streams (shared by append + alternate_kv)."""

    def _encode_visual_stream(vision_tokens: torch.Tensor, stream: str) -> torch.Tensor:
        if prefix_mode == "raw":
            return prefix_adapter.forward_raw(vision_tokens, stream=stream)
        return prefix_adapter(vision_tokens, stream=stream)

    future_tokens = _encode_visual_stream(uhat, "future")
    curr_len = 0
    if inject_current:
        ht_for_prefix = h_t.detach().to(device=device, dtype=dtype)
        current_tokens = _encode_visual_stream(ht_for_prefix, "current")
        cond_embs = torch.cat([current_tokens, future_tokens], dim=1)
        curr_len = int(current_tokens.shape[1])
    else:
        cond_embs = future_tokens
    cond_pad_mask = torch.ones(cond_embs.shape[:2], dtype=torch.bool, device=cond_embs.device)
    return (cond_embs, cond_pad_mask, curr_len)


def encode_lawam_multi_view_visual_cond(
    prefix_adapter: LaWAMPrefixAdapter,
    *,
    h_t: torch.Tensor,
    uhat: torch.Tensor,
    prefix_mode: str,
    inject_current: bool,
    camera_embedding: torch.Tensor,
    current_valid: torch.Tensor,
    future_valid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    token_budget_mode: str = "per_view",
    vision_camera_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build ``current_views -> future_views`` cond tokens with shared weights."""
    if h_t.dim() != 4 or uhat.dim() != 4 or h_t.shape != uhat.shape:
        raise ValueError(
            f"Multi-view LaWAM cond expects matching [B,V,K,D] tensors, got h_t={tuple(h_t.shape)} uhat={tuple(uhat.shape)}"
        )
    bsz, num_views = (int(h_t.shape[0]), int(h_t.shape[1]))
    if camera_embedding.shape != (num_views, prefix_adapter.output_dim):
        raise ValueError(
            f"camera_embedding shape mismatch: got {tuple(camera_embedding.shape)}, expected {(num_views, prefix_adapter.output_dim)}"
        )
    expected_valid = (bsz, num_views)
    if current_valid.shape != expected_valid or future_valid.shape != expected_valid:
        raise ValueError(
            f"LaWAM view-valid masks must be [B,V]: current={tuple(current_valid.shape)} future={tuple(future_valid.shape)} expected={expected_valid}"
        )
    budget_mode = str(token_budget_mode).strip().lower()
    if budget_mode not in {"per_view", "shared_total"}:
        raise ValueError(f"Unknown multi-view token budget mode: {token_budget_mode!r}")

    def _encode_views(
        features: torch.Tensor, stream: str, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if budget_mode == "shared_total":
            if prefix_mode != "compressed":
                raise ValueError("shared_total multi-view budget requires compressed prefix mode")
            if vision_camera_embedding is None:
                raise ValueError("shared_total multi-view budget requires vision camera embedding")
            expected_camera = (num_views, prefix_adapter.vision_dim)
            if tuple(vision_camera_embedding.shape) != expected_camera:
                raise ValueError(
                    f"vision camera embedding shape mismatch: got {tuple(vision_camera_embedding.shape)}, expected {expected_camera}"
                )
            camera = vision_camera_embedding.to(device=features.device, dtype=features.dtype)
            tagged = features + camera[None, :, None, :]
            flattened = tagged.reshape(
                bsz, num_views * int(features.shape[2]), int(features.shape[3])
            )
            key_padding_mask = (~valid.to(device=flattened.device, dtype=torch.bool))[:, :, None]
            key_padding_mask = key_padding_mask.expand(-1, -1, int(features.shape[2])).reshape(
                bsz, -1
            )
            any_valid = valid.to(device=flattened.device, dtype=torch.bool).any(dim=1)
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[~any_valid, 0] = False
            encoded = prefix_adapter(flattened, stream=stream, key_padding_mask=key_padding_mask)
            encoded = encoded.masked_fill(~any_valid[:, None, None], 0)
            output_valid = any_valid[:, None].expand(-1, int(encoded.shape[1]))
            return (encoded, output_valid)
        flattened = features.reshape(
            bsz * num_views, int(features.shape[2]), int(features.shape[3])
        )
        if prefix_mode == "raw":
            encoded = prefix_adapter.forward_raw(flattened, stream=stream)
        else:
            encoded = prefix_adapter(flattened, stream=stream)
        encoded = encoded.reshape(bsz, num_views, int(encoded.shape[1]), int(encoded.shape[2]))
        camera = camera_embedding.to(device=encoded.device, dtype=encoded.dtype)
        encoded = encoded + camera[None, :, None, :]
        tokens_per_view = int(encoded.shape[2])
        encoded = encoded.reshape(bsz, num_views * tokens_per_view, -1)
        output_valid = valid.to(device=encoded.device, dtype=torch.bool)
        output_valid = output_valid[:, :, None].expand(-1, -1, tokens_per_view).reshape(bsz, -1)
        return (encoded, output_valid)

    future_tokens, future_pad = _encode_views(uhat, "future", future_valid)
    curr_len = 0
    if inject_current:
        current_tokens, current_pad = _encode_views(
            h_t.detach().to(device=device, dtype=dtype), "current", current_valid
        )
        cond_embs = torch.cat([current_tokens, future_tokens], dim=1)
        cond_pad_mask = torch.cat([current_pad, future_pad], dim=1)
        curr_len = int(current_tokens.shape[1])
    else:
        cond_embs = future_tokens
        cond_pad_mask = future_pad
    return (cond_embs, cond_pad_mask, curr_len)


class LaWAMKVBank(nn.Module):
    """K/V projections for expert layers that attend to LaWAM cond tokens.

    Allocates Linears only for the alternate schedule's cond-KV layers
    (default: odd layers). VLM-KV layers are not given unused parameters.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_dim: int,
        num_kv_heads: int,
        head_dim: int,
        alternate_first: str = "vlm",
        alternate_period: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.layer_indices = lawam_cond_kv_layer_indices(
            self.num_layers,
            alternate_first=alternate_first,
            alternate_period=alternate_period,
        )
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_projs = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_dim, kv_dim, bias=False) for i in self.layer_indices}
        )
        self.v_projs = nn.ModuleDict(
            {str(i): nn.Linear(self.hidden_dim, kv_dim, bias=False) for i in self.layer_indices}
        )

    def build_kv_for_layer(
        self,
        layer_idx: int,
        cond_embs: torch.Tensor,
        *,
        rotary_model: nn.Module,
        position_ids: torch.Tensor,
        apply_rotary_key_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prefix_key/value shaped for ``_run_suffix_expert_layer``."""
        key = str(layer_idx)
        k_proj = self.k_projs[key]
        v_proj = self.v_projs[key]
        param_dtype = k_proj.weight.dtype
        cond = cond_embs.to(device=k_proj.weight.device, dtype=param_dtype)
        bsz, seq_len, _ = cond.shape
        k = k_proj(cond)
        v = v_proj(cond)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = apply_rotary_key_fn(rotary_model, k, position_ids)
        return k, v

    def build_kv_for_layers(
        self,
        cond_embs: torch.Tensor,
        *,
        rotary_model: nn.Module,
        prefix_pad_masks: torch.Tensor,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Build immutable LaWAM K/V once for one observation's denoising passes."""
        position_ids = prefix_pad_masks.sum(dim=-1)[:, None] + torch.arange(
            cond_embs.shape[1], device=prefix_pad_masks.device
        )[None, :]
        rope = {}

        def apply_rotary_key_fn(model, keys, positions):
            key = (keys.device, keys.dtype)
            if key not in rope:
                rope[key] = _key_rotary_position_embeddings(model, keys, positions)
            return _apply_rotary_key_only(model, keys, positions, position_embeddings=rope[key])

        return {
            int(layer_idx): self.build_kv_for_layer(
                int(layer_idx),
                cond_embs,
                rotary_model=rotary_model,
                position_ids=position_ids,
                apply_rotary_key_fn=apply_rotary_key_fn,
            )
            for layer_idx in self.layer_indices
        }


class LaWAMPrefixAdapter(nn.Module):
    """Compress DINO tokens [B, K, Dv] into N expert-prefix tokens.

    Shared QFormer weights for current ``h_t`` and future ``uhat_T``; stream identity
    is marked by separate type embeddings (open-source DiT cats both streams).
    """

    def __init__(
        self,
        *,
        vision_dim: int,
        output_dim: int,
        num_tokens: int = 16,
        num_heads: int = 8,
        need_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.vision_dim = int(vision_dim)
        self.output_dim = int(output_dim)
        self.need_weights = bool(need_weights)
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, self.vision_dim) * 0.02)
        heads = int(num_heads)
        self.norm_q = nn.LayerNorm(self.vision_dim)
        self.norm_kv = nn.LayerNorm(self.vision_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.vision_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.vision_dim * 4),
            nn.GELU(),
            nn.Linear(self.vision_dim * 4, self.vision_dim),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        # ``token_type_embedding`` kept as the future-stream marker for ckpt compatibility.
        self.token_type_embedding = nn.Parameter(torch.zeros(self.output_dim))
        self.current_token_type_embedding = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        nn.init.normal_(self.current_token_type_embedding, std=0.02)

    def _stream_type_embedding(self, stream):
        return {"current": self.current_token_type_embedding, "future": self.token_type_embedding}[
            stream
        ]

    def forward(
        self,
        vision_tokens: torch.Tensor,
        *,
        stream: str = "future",
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress vision_tokens: [B, K, vision_dim] -> [B, num_tokens, output_dim].

        ``stream``: ``\"current\"`` for ``h_t``, ``\"future\"`` for ``uhat_T`` / ``h_t1*``.
        """
        type_emb = self._stream_type_embedding(stream)
        param_dtype = self.queries.dtype
        vision_tokens = vision_tokens.to(device=self.queries.device, dtype=param_dtype)
        bsz = int(vision_tokens.shape[0])
        q = self.queries.expand(bsz, -1, -1)
        # Separate K/V normalization preserves attention kernel selection and numerical output.
        attn_out, _ = self.attn(
            self.norm_q(q),
            self.norm_kv(vision_tokens),
            self.norm_kv(vision_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=self.need_weights,
        )
        q = q + attn_out
        q = q + self.ffn(q)
        tokens = self.projector(q)
        return tokens + type_emb[None, None, :]

    def forward_raw(self, vision_tokens: torch.Tensor, *, stream: str = "future") -> torch.Tensor:
        """Project every DINO token without length compression: [B, K, Dv] -> [B, K, output_dim]."""
        type_emb = self._stream_type_embedding(stream)
        param_dtype = self.queries.dtype
        vision_tokens = vision_tokens.to(device=self.queries.device, dtype=param_dtype)
        tokens = self.projector(vision_tokens)
        return tokens + type_emb[None, None, :]


def prepare_lawam_rgb_batch(images: torch.Tensor, *, image_resolution: int = 256) -> torch.Tensor:
    """Convert policy RGB in [-1, 1] to LAM-ready [B,3,H,W] in [0, 1]."""
    if images.dim() == 5:
        raise ValueError(
            "prepare_lawam_rgb_batch expects a single-frame tensor [B,C,H,W] or [B,H,W,C]"
        )
    img = images
    if img.dim() != 4:
        raise ValueError(f"Expected 4D image batch, got {tuple(img.shape)}")
    if img.shape[1] == 3:
        pass
    elif img.shape[-1] == 3:
        img = img.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Cannot infer channel layout from shape {tuple(img.shape)}")
    img = (img.to(dtype=torch.float32) + 1.0) * 0.5
    img = img.clamp(0.0, 1.0)
    if img.shape[-2:] != (image_resolution, image_resolution):
        img = F.interpolate(
            img, size=(image_resolution, image_resolution), mode="bilinear", align_corners=False
        )
    return img.contiguous()


class VLMToLAMQFormer(nn.Module):
    """Refine VLM LA-query hidden states into one latent action embedding."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, int(lam_code_dim)) * 0.02)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=int(lam_code_dim),
                    kdim=int(vlm_hidden_dim),
                    vdim=int(vlm_hidden_dim),
                    num_heads=int(num_heads),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_qs = nn.ModuleList(
            [nn.LayerNorm(int(lam_code_dim)) for _ in range(int(num_layers))]
        )
        self.norm_kvs = nn.ModuleList(
            [nn.LayerNorm(int(vlm_hidden_dim)) for _ in range(int(num_layers))]
        )
        hidden_dim = int(int(lam_code_dim) * float(ffn_expansion_factor))
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("0", nn.LayerNorm(int(lam_code_dim))),
                            ("1", nn.Linear(int(lam_code_dim), hidden_dim)),
                            ("2", nn.GELU()),
                            ("4", nn.Linear(hidden_dim, int(lam_code_dim))),
                        ]
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, Q, vlm_hidden] -> [B, 1, code_dim]."""
        # Keep compute in module param dtype (usually fp32) even when VLM hiddens are bf16.
        param_dtype = self.query.dtype
        context = context.to(device=self.query.device, dtype=param_dtype)
        bsz = int(context.shape[0])
        queries = self.query.expand(bsz, -1, -1)
        for norm_q, norm_kv, xattn, ffn in zip(
            self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns
        ):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        return self.final_norm(queries)


def decode_future_tokens(
    lam: nn.Module, h_t: torch.Tensor, pred_action_emb: torch.Tensor
) -> torch.Tensor:
    decoder_input = h_t.unsqueeze(1) if lam.decoder_multi_view else h_t
    return lam.decoder(decoder_input, pred_action_emb)[:, 0]
