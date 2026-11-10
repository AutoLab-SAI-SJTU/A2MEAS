"""Six-frame visual memory: summarize at block four, fuse at block six."""

import math
import torch
import torch.nn.functional as F
from torch import nn


def build_zero_anchored_temporal_embedding(
    num_frames: int, hidden_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if num_frames <= 1:
        return torch.zeros((num_frames, hidden_size), device=device, dtype=dtype)
    half_dim = hidden_size // 2
    if half_dim == 0:
        return torch.zeros((num_frames, hidden_size), device=device, dtype=dtype)
    positions = torch.arange(-(num_frames - 1), 1, device=device, dtype=torch.float32)
    scale = -math.log(10000.0) / max(half_dim - 1, 1)
    inv_freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * scale)
    angles = positions[:, None] * inv_freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if hidden_size % 2 == 1:
        emb = F.pad(emb, (0, 1))
    emb[-1].zero_()
    return emb.to(dtype=dtype)


class Qwen3VLTemporalMemory(nn.Module):
    """Compress the two adjacent history changes and inject them into current tokens."""

    num_queries = 1

    def __init__(self, hidden_size: int, *, retain_motion_head=True) -> None:
        super().__init__()
        attention_dim = 256
        self.pool_size = 4
        self.num_heads = 4
        self.head_dim = attention_dim // self.num_heads

        self.summary_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.summary_in = nn.Linear(hidden_size, attention_dim)
        self.summary_queries = nn.Parameter(torch.empty(1, self.num_queries, attention_dim))
        self.summary_attention = nn.MultiheadAttention(attention_dim, self.num_heads, batch_first=True)
        self.delta_proj = nn.Linear(attention_dim, attention_dim, bias=False)
        self.scale_embedding = nn.Parameter(torch.empty(2, 1, attention_dim))
        self.temporal_norm = nn.LayerNorm(attention_dim, eps=1e-6)
        if retain_motion_head:
            self.motion_head = nn.Linear(attention_dim, 14)

        self.current_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.memory_norm = nn.LayerNorm(attention_dim, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, attention_dim)
        self.k_proj = nn.Linear(attention_dim, attention_dim)
        self.v_proj = nn.Linear(attention_dim, attention_dim)
        self.gate = nn.Linear(hidden_size, self.num_heads)
        self.out_proj = nn.Linear(attention_dim, hidden_size, bias=False)

        nn.init.normal_(self.summary_queries, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.out_proj.weight)
        if retain_motion_head:
            self.motion_head.requires_grad_(False)

    def _spatial_pool(
        self,
        frames: torch.Tensor,
        height: int,
        width: int,
        merge_size: int,
    ) -> torch.Tensor:
        batch_size, num_frames, _, hidden_size = frames.shape
        maps = (
            frames.reshape(
                batch_size * num_frames,
                height // merge_size,
                width // merge_size,
                merge_size,
                merge_size,
                hidden_size,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size * num_frames, height, width, hidden_size)
        )
        pooled = (
            F.adaptive_avg_pool2d(maps.permute(0, 3, 1, 2), (self.pool_size, self.pool_size))
            .flatten(2)
            .transpose(1, 2)
        )
        return pooled.reshape(batch_size, num_frames, self.pool_size**2, hidden_size)

    def summarize(
        self,
        frames: torch.Tensor,
        *,
        height: int,
        width: int,
        merge_size: int,
    ) -> torch.Tensor:
        pooled = self._spatial_pool(frames, height, width, merge_size)
        # Outside autocast, pooling output must match the normalization dtype.
        if not torch.is_autocast_enabled(pooled.device.type):
            pooled = pooled.to(self.summary_norm.weight.dtype)
        batch_size, num_frames, tokens_per_frame, hidden_size = pooled.shape
        source = self.summary_in(
            self.summary_norm(pooled.reshape(batch_size * num_frames, tokens_per_frame, hidden_size))
        )
        queries = self.summary_queries.expand(batch_size * num_frames, -1, -1)
        summaries, _ = self.summary_attention(queries, source, source, need_weights=False)
        summaries = summaries.reshape(batch_size, num_frames, self.num_queries, -1)
        memory = self.temporal_norm(
            self.delta_proj(summaries[:, 1:] - summaries[:, :-1]) + self.scale_embedding
        ).squeeze(2)
        return memory

    def fuse(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, current_len, _ = current.shape
        memory_len = memory.shape[1]
        current_input, memory_input = current, memory
        if not torch.is_autocast_enabled(current.device.type):
            current_input = current.to(self.current_norm.weight.dtype)
            memory_input = memory.to(self.memory_norm.weight.dtype)
        normed_current = self.current_norm(current_input)
        normed_memory = self.memory_norm(memory_input)
        query = self.q_proj(normed_current).view(batch_size, current_len, self.num_heads, self.head_dim)
        key = self.k_proj(normed_memory).view(batch_size, memory_len, self.num_heads, self.head_dim)
        value = self.v_proj(normed_memory).view(batch_size, memory_len, self.num_heads, self.head_dim)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        gate_logits = self.gate(normed_current)
        gate = torch.sigmoid(gate_logits).transpose(1, 2).unsqueeze(-1)
        gated_readout = (gate * attention).transpose(1, 2).reshape(batch_size, current_len, -1)
        update = self.out_proj(gated_readout)

        return current + update.to(current.dtype)

    def forward_vision(
        self,
        vision,
        pixels: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        video_frame_age_seconds: torch.Tensor | None = None,
    ):
        rows = grid_thw.tolist()
        if not rows or rows[0][0] != 3 or any(row != rows[0] for row in rows):
            raise ValueError("temporal memory requires six frames per camera and a shared spatial grid")
        _, height, width = rows[0]
        batch_size, tokens = len(rows), height * width
        current_grid = grid_thw.clone()
        current_grid[:, 0] = 1
        deepstack = []
        hidden = vision.patch_embed(pixels) + vision.fast_pos_embed_interpolate(grid_thw)
        frames = hidden.reshape(batch_size, 3, tokens, -1)
        bias = build_zero_anchored_temporal_embedding(3, frames.shape[-1], hidden.device, hidden.dtype)
        hidden = (frames + bias[None, :, None]).reshape_as(hidden)
        frame_age_enabled = bool(getattr(vision.config, "mem_frame_age_embedding_enable", False))
        if frame_age_enabled:
            if video_frame_age_seconds is None:
                raise ValueError("STM frame-age embedding is enabled but video_frame_age_seconds is missing")
            hidden = vision._add_mem_frame_age_bias(hidden, grid_thw, video_frame_age_seconds)
        elif video_frame_age_seconds is not None:
            raise ValueError("video_frame_age_seconds was provided while STM frame-age embedding is disabled")
        position_embeddings, cu_seqlens = vision._build_position_state(grid_thw)
        for index, block in enumerate(vision.blocks):
            hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if index == 3:
                frames = hidden.reshape(batch_size, 3, tokens, -1)
                memory = self.summarize(
                    frames, height=height, width=width, merge_size=vision.spatial_merge_size
                )
                hidden = frames[:, -1].reshape(-1, frames.shape[-1])
                position_embeddings, cu_seqlens = vision._build_position_state(current_grid)
            if index == 5:
                hidden = self.fuse(hidden.reshape(batch_size, tokens, -1), memory).reshape_as(hidden)
            if index in vision.deepstack_visual_indexes:
                current = (
                    hidden
                    if index >= 3
                    else hidden.reshape(batch_size, 3, tokens, -1)[:, -1].reshape(-1, hidden.shape[-1])
                )
                merger = vision.deepstack_merger_list[vision.deepstack_visual_indexes.index(index)]
                deepstack.append(merger(current))
        return vision.merger(hidden), deepstack
