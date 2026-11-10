# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Derived from Transformers and Anyverse-VLA; Apache-2.0. See README.md.
"""RoboTwin packed video compression and a backbone without a language head."""

from __future__ import annotations

import os
from typing import Callable, Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import auto_docstring, TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionMLP,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionPatchMerger,
    apply_rotary_pos_emb_vision,
    repeat_kv,
)
from a2meas.models import qwen3_vl as shared
from a2meas.models.qwen3_vl import Qwen3VLTextModel
from a2meas.models.temporal_memory import build_zero_anchored_temporal_embedding


def _lerobot_metadata_fastpath_enabled() -> bool:
    """Reuse exact integer/grid metadata by default."""
    return os.environ.get("LEROBOT_QWEN_METADATA_FASTPATH", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** (-0.5)
        self.config = config
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            assume_equal_segments = os.environ.get(
                "QWEN_VISION_SDPA_ASSUME_EQUAL_SEGMENTS", "0"
            ).lower() in ("1", "true", "yes", "on")
            lengths_list = None if assume_equal_segments else lengths.tolist()
            use_batched_sdpa_segments = (
                os.environ.get("QWEN_VISION_SDPA_BATCH_SEGMENTS", "0").lower()
                in ("1", "true", "yes", "on")
                and self.config._attn_implementation == "sdpa"
                and (
                    assume_equal_segments
                    or (
                        len(lengths_list) > 1
                        and len(set((int(length) for length in lengths_list))) == 1
                    )
                )
            )
            if use_batched_sdpa_segments:
                if assume_equal_segments:
                    num_segments = int(lengths.shape[0])
                    if num_segments < 1 or seq_length % num_segments != 0:
                        raise ValueError(
                            "QWEN_VISION_SDPA_ASSUME_EQUAL_SEGMENTS requires evenly divisible packed segments"
                        )
                    segment_length = seq_length // num_segments
                else:
                    num_segments = len(lengths_list)
                    segment_length = int(lengths_list[0])
                q = (
                    query_states.reshape(1, self.num_heads, num_segments, segment_length, -1)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(num_segments, self.num_heads, segment_length, -1)
                    .contiguous()
                )
                k = (
                    key_states.reshape(1, self.num_heads, num_segments, segment_length, -1)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(num_segments, self.num_heads, segment_length, -1)
                    .contiguous()
                )
                v = (
                    value_states.reshape(1, self.num_heads, num_segments, segment_length, -1)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(num_segments, self.num_heads, segment_length, -1)
                    .contiguous()
                )
                attn_output, _ = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0,
                    is_causal=False,
                    **kwargs,
                )
                attn_output = attn_output.reshape(
                    1, num_segments * segment_length, self.num_heads, -1
                )
            else:
                if lengths_list is None:
                    lengths_list = lengths.tolist()
                splits = [
                    torch.split(tensor, lengths_list, dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                attn_outputs = [
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        if _lerobot_metadata_fastpath_enabled() and grid_thw.device.type == "cpu":
            grid_key = tuple((tuple(row) for row in grid_thw.tolist()))
            max_hw = max((max(h, w) for _, h, w in grid_key))
            freq_table = self.rotary_pos_emb(max_hw)
            cache_key = (grid_key, merge_size, freq_table.device)
            cache = getattr(self, "_lerobot_rot_pos_index_cache", None)
            if cache is None:
                cache = self._lerobot_rot_pos_index_cache = {}
            pos_ids = cache.get(cache_key)
            if pos_ids is None:
                coordinates = []
                for num_frames, height, width in grid_key:
                    merged_h, merged_w = (height // merge_size, width // merge_size)
                    block_rows = torch.arange(merged_h, device=freq_table.device)
                    block_cols = torch.arange(merged_w, device=freq_table.device)
                    intra = torch.arange(merge_size, device=freq_table.device)
                    rows = block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]
                    cols = block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]
                    rows = rows.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                    cols = cols.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                    coords = torch.stack((rows, cols), dim=-1)
                    coordinates.append(coords.repeat(num_frames, 1) if num_frames > 1 else coords)
                pos_ids = torch.cat(coordinates)
                if len(cache) >= 16:
                    cache.clear()
                cache[cache_key] = pos_ids
            return freq_table[pos_ids].flatten(1)
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = (height // merge_size, width // merge_size)
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = (grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2])
        use_cache = _lerobot_metadata_fastpath_enabled()
        pos_index_cache = None
        cache_key = None
        cached_indices = None
        if use_cache:
            grid_key = tuple((tuple((int(value) for value in row)) for row in grid_thw.tolist()))
            grid_ts, grid_hs, grid_ws = zip(*grid_key)
            cache_key = (
                grid_key,
                self.pos_embed.weight.device,
                self.pos_embed.weight.dtype,
            )
            pos_index_cache = getattr(self, "_lerobot_pos_interpolate_index_cache", None)
            if pos_index_cache is None:
                pos_index_cache = {}
                self._lerobot_pos_interpolate_index_cache = pos_index_cache
            cached_indices = pos_index_cache.get(cache_key)
        if cached_indices is None:
            idx_list = [[] for _ in range(4)]
            weight_list = [[] for _ in range(4)]
            for _, h, w in zip(grid_ts, grid_hs, grid_ws):
                h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
                w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
                h_idxs_floor = h_idxs.int()
                w_idxs_floor = w_idxs.int()
                h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
                w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
                dh = h_idxs - h_idxs_floor
                dw = w_idxs - w_idxs_floor
                base_h = h_idxs_floor * self.num_grid_per_side
                base_h_ceil = h_idxs_ceil * self.num_grid_per_side
                indices = [
                    (base_h[None].T + w_idxs_floor[None]).flatten(),
                    (base_h[None].T + w_idxs_ceil[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
                ]
                weights = [
                    ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                    ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(),
                    (dh[None].T * dw[None]).flatten(),
                ]
                for i in range(4):
                    idx_list[i].extend(indices[i].tolist())
                    weight_list[i].extend(weights[i].tolist())
            idx_tensor = torch.tensor(
                idx_list, dtype=torch.long, device=self.pos_embed.weight.device
            )
            weight_tensor = torch.tensor(
                weight_list,
                dtype=self.pos_embed.weight.dtype,
                device=self.pos_embed.weight.device,
            )
            if pos_index_cache is not None:
                if len(pos_index_cache) >= 16:
                    pos_index_cache.clear()
                pos_index_cache[cache_key] = (idx_tensor, weight_tensor)
        else:
            idx_tensor, weight_tensor = cached_indices
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def _mem_enabled(self, grid_thw: torch.Tensor) -> bool:
        return bool(getattr(self.config, "mem_enable_short_term_memory", False)) and bool(
            grid_thw is not None and grid_thw.numel() > 0 and (grid_thw[:, 0] > 1).any().item()
        )

    def _build_position_state(
        self, grid_thw: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = rotary_pos_emb.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        position_state = (position_embeddings, cu_seqlens)
        return position_state

    def _split_segments(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[torch.Tensor]:
        segments = []
        offset = 0
        for num_frames, height, width in grid_thw.tolist():
            tokens_per_frame = int(height * width)
            segment_len = int(num_frames * tokens_per_frame)
            segments.append(
                hidden_states[offset : offset + segment_len].view(
                    int(num_frames), tokens_per_frame, -1
                )
            )
            offset += segment_len
        return segments

    def _flatten_segments(self, segments: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([segment.reshape(-1, segment.shape[-1]) for segment in segments], dim=0)

    def _add_mem_temporal_bias(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        segments = []
        for segment in self._split_segments(hidden_states, grid_thw):
            if segment.shape[0] > 1:
                temporal_bias = build_zero_anchored_temporal_embedding(
                    segment.shape[0],
                    segment.shape[-1],
                    device=segment.device,
                    dtype=segment.dtype,
                ).view(segment.shape[0], 1, segment.shape[-1])
                segment = segment + temporal_bias
            segments.append(segment)
        return self._flatten_segments(segments)

    def _compress_segments_to_current_frame(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_segments = []
        current_grid = []
        for segment, (_, height, width) in zip(
            self._split_segments(hidden_states, grid_thw), grid_thw.tolist()
        ):
            current_segments.append(segment[-1])
            current_grid.append([1, int(height), int(width)])
        return (torch.cat(current_segments, dim=0), grid_thw.new_tensor(current_grid))

    @staticmethod
    def _current_frame_grid(grid_thw: torch.Tensor) -> torch.Tensor:
        grid = grid_thw.clone()
        grid[:, 0] = 1
        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        current_grid_thw = grid_thw
        mem_enabled = self._mem_enabled(current_grid_thw)
        if mem_enabled:
            hidden_states = self._add_mem_temporal_bias(hidden_states, current_grid_thw)
        position_embeddings, cu_seqlens = self._build_position_state(current_grid_thw)
        if os.environ.get("QWEN_VISION_SDPA_ASSUME_EQUAL_SEGMENTS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            if segment_lengths.numel() > 1 and (
                not bool(torch.all(segment_lengths == segment_lengths[0]).item())
            ):
                raise ValueError(
                    "QWEN_VISION_SDPA_ASSUME_EQUAL_SEGMENTS was enabled, but packed vision segments differ"
                )
        drop_history_last_n_layers = max(
            0, int(getattr(self.config, "mem_drop_history_last_n_layers", 2))
        )
        drop_start_layer = (
            max(0, len(self.blocks) - drop_history_last_n_layers)
            if drop_history_last_n_layers > 0
            else -1
        )
        history_dropped = False
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            if (
                mem_enabled
                and (not history_dropped)
                and (drop_history_last_n_layers > 0)
                and (layer_num == drop_start_layer)
            ):
                hidden_states, current_grid_thw = (
                    self._compress_segments_to_current_frame(hidden_states, current_grid_thw)
                )
                position_embeddings, cu_seqlens = self._build_position_state(current_grid_thw)
                history_dropped = True
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_source = hidden_states
                if mem_enabled:
                    if history_dropped:
                        deepstack_source = hidden_states
                    else:
                        deepstack_source, _ = self._compress_segments_to_current_frame(
                            deepstack_source, current_grid_thw
                        )
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](deepstack_source)
                deepstack_feature_lists.append(deepstack_feature)
        if mem_enabled and (not history_dropped):
            hidden_states, current_grid_thw = self._compress_segments_to_current_frame(
                hidden_states, current_grid_thw
            )
        hidden_states = self.merger(hidden_states)
        return (hidden_states, deepstack_feature_lists)


@auto_docstring
class Qwen3VLModel(shared.Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def _get_video_only_rope_index_fast(
        self,
        input_ids: torch.LongTensor,
        video_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Build the video-only Qwen RoPE metadata without many tiny Torch CPU ops.

        Returns ``None`` for other multimodal layouts so the generic
        implementation below remains the source of truth for those cases.
        """
        if (
            input_ids.device.type != "cpu"
            or video_grid_thw.device.type != "cpu"
            or attention_mask.device.type != "cpu"
            or (input_ids.ndim != 2)
            or (attention_mask.shape != input_ids.shape)
            or (video_grid_thw.ndim != 2)
            or (video_grid_thw.shape[1] != 3)
        ):
            return None
        import numpy as np

        input_array = input_ids.detach().contiguous().numpy()
        mask_array = attention_mask.detach().contiguous().numpy()
        grid_array = video_grid_thw.detach().contiguous().numpy()
        batch_size, padded_length = input_array.shape
        position_array = np.ones((3, batch_size, padded_length), dtype=input_array.dtype)
        delta_array = np.empty((batch_size, 1), dtype=input_array.dtype)
        merge_size = int(self.config.vision_config.spatial_merge_size)
        vision_start_id = int(self.config.vision_start_token_id)
        video_token_id = int(self.config.video_token_id)
        image_token_id = int(self.config.image_token_id)
        grid_index = 0
        for batch_index in range(batch_size):
            valid_indices = np.flatnonzero(mask_array[batch_index] == 1)
            valid_length = int(valid_indices.size)
            if valid_length == 0:
                return None
            if not np.array_equal(
                valid_indices, np.arange(valid_length, dtype=valid_indices.dtype)
            ):
                return None
            input_tokens = input_array[batch_index, :valid_length]
            vision_starts = np.flatnonzero(input_tokens[:-1] == vision_start_id)
            vision_token_positions = vision_starts + 1
            vision_tokens = input_tokens[vision_token_positions]
            if np.any(vision_tokens == image_token_id) or np.any(vision_tokens != video_token_id):
                return None
            row_positions = position_array[:, batch_index, :valid_length]
            source_offset = 0
            position_offset = 0
            expected_video_tokens = 0
            for video_position in vision_token_positions.tolist():
                if grid_index >= len(grid_array):
                    return None
                grid_t, grid_h, grid_w = (int(value) for value in grid_array[grid_index])
                grid_index += 1
                llm_t = grid_t
                llm_h = grid_h // merge_size
                llm_w = grid_w // merge_size
                visual_length = llm_t * llm_h * llm_w
                expected_video_tokens += visual_length
                video_position = int(video_position)
                visual_end = video_position + visual_length
                if (
                    llm_t < 1
                    or llm_h < 1
                    or llm_w < 1
                    or (visual_end > valid_length)
                    or (not np.all(input_tokens[video_position:visual_end] == video_token_id))
                ):
                    return None
                text_length = video_position - source_offset
                if text_length < 0:
                    return None
                if text_length:
                    text_positions = np.arange(
                        position_offset, position_offset + text_length, dtype=input_array.dtype
                    )
                    row_positions[:, source_offset:video_position] = text_positions
                    position_offset += text_length
                row_positions[0, video_position:visual_end] = (
                    np.repeat(np.arange(llm_t, dtype=input_array.dtype), llm_h * llm_w)
                    + position_offset
                )
                row_positions[1, video_position:visual_end] = (
                    np.tile(np.repeat(np.arange(llm_h, dtype=input_array.dtype), llm_w), llm_t)
                    + position_offset
                )
                row_positions[2, video_position:visual_end] = (
                    np.tile(np.arange(llm_w, dtype=input_array.dtype), llm_t * llm_h)
                    + position_offset
                )
                position_offset += max(llm_t, llm_h, llm_w)
                source_offset = visual_end
            if np.count_nonzero(input_tokens == video_token_id) != expected_video_tokens:
                return None
            if source_offset < valid_length:
                tail_length = valid_length - source_offset
                tail_positions = np.arange(
                    position_offset, position_offset + tail_length, dtype=input_array.dtype
                )
                row_positions[:, source_offset:valid_length] = tail_positions
            delta_array[batch_index, 0] = int(row_positions.max()) + 1 - padded_length
        if grid_index != len(grid_array):
            return None
        return (torch.from_numpy(position_array), torch.from_numpy(delta_array))

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        compress_video_tokens = bool(getattr(self.config, "mem_compress_video_tokens", False))
        video_embeds, deepstack_video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        if compress_video_tokens:
            effective_video_grid_thw = self.visual._current_frame_grid(video_grid_thw)
            split_sizes = (
                effective_video_grid_thw[:, 1]
                * effective_video_grid_thw[:, 2]
                // self.visual.spatial_merge_size**2
            ).tolist()
        else:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return (video_embeds, deepstack_video_embeds)

    def get_rope_index(
        self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None
    ):
        if video_grid_thw is not None:
            if getattr(self.config, "mem_compress_video_tokens", False):
                video_grid_thw = self.visual._current_frame_grid(video_grid_thw)
            else:
                video_grid_thw = torch.repeat_interleave(
                    video_grid_thw, video_grid_thw[:, 0], dim=0
                ).clone()
                video_grid_thw[:, 0] = 1
        if (
            input_ids is not None
            and image_grid_thw is None
            and video_grid_thw is not None
            and attention_mask is not None
            and _lerobot_metadata_fastpath_enabled()
        ):
            fast = self._get_video_only_rope_index_fast(input_ids, video_grid_thw, attention_mask)
            if fast is not None:
                return fast
        return super().get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)


class RynnBrainBackbone(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(
            pixel_values_videos,
            video_grid_thw,
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(self, **kwargs):
        return self.model(**kwargs)
