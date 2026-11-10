# Derived from Hugging Face Transformers 4.57.1, licensed under Apache-2.0.
# Copyright 2024 Google Inc. and HuggingFace Inc.; 2025 The Qwen Team.
# See licenses/transformers.txt and README.md.

"""RynnBrain vision memory and pre-normalization outputs for policy inference."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs
from transformers.models.qwen3_vl import modeling_qwen3_vl as upstream
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionMLP,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionPatchMerger,
    Qwen3VLPreTrainedModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

def backbone_from_config(model_class, directory, *, dtype):
    """Build the backbone skeleton; the complete policy checkpoint supplies its weights."""
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with no_init_weights():
        model = model_class._from_config(config, dtype=dtype)
    model.tie_weights()
    return model






class Qwen3VLFrameAgeEmbedding(nn.Module):
    """Learn an additive visual bias from the raw-frame ages in one temporal patch."""

    def __init__(
        self,
        *,
        temporal_patch_size: int,
        embedding_dim: int,
        hidden_size: int,
        max_seconds: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.temporal_patch_size = int(temporal_patch_size)
        self.max_seconds = float(max_seconds)
        self.in_proj = nn.Linear(
            self.temporal_patch_size, int(embedding_dim), bias=False, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            int(embedding_dim), int(hidden_size), bias=False, device=device, dtype=dtype
        )
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, frame_age_seconds: torch.Tensor) -> torch.Tensor:
        age = frame_age_seconds.to(device=self.in_proj.weight.device, dtype=torch.float32)
        age = age.clamp(min=0.0, max=self.max_seconds)
        age = torch.log1p(age) / math.log1p(self.max_seconds)
        hidden = F.silu(F.linear(age.to(dtype=self.in_proj.weight.dtype), self.in_proj.weight))
        return F.linear(hidden, self.out_proj.weight)


class Qwen3VLVisionAttention(upstream.Qwen3VLVisionAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        (query_states, key_states, value_states) = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            (attn_output, _) = attention_interface(
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
                for (q, k, v) in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    forward = upstream.Qwen3VLVisionBlock.forward

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-06)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)


@dataclass
class Qwen3VLModelOutputWithPast(ModelOutput):
    """Multimodal hidden states, attention caches and rotary position metadata."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    pre_norm_last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


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
        self.temporal_memory = None
        if bool(getattr(config, "mem_frame_age_embedding_enable", False)):
            self.configure_mem_frame_age_embedding(
                enabled=True,
                max_seconds=float(getattr(config, "mem_frame_age_max_seconds", 300.0)),
                embedding_dim=int(getattr(config, "mem_frame_age_embedding_dim", 64)),
            )

    def configure_mem_frame_age_embedding(
        self, *, enabled: bool, max_seconds: float, embedding_dim: int
    ) -> None:
        """Configure optional STM age parameters after the base VLM is loaded."""
        self.config.mem_frame_age_embedding_enable = bool(enabled)
        self.config.mem_frame_age_max_seconds = float(max_seconds)
        self.config.mem_frame_age_embedding_dim = int(embedding_dim)
        if not enabled:
            if hasattr(self, "mem_frame_age_encoder"):
                del self.mem_frame_age_encoder
            return
        if max_seconds <= 0 or embedding_dim < 1:
            raise ValueError("STM frame-age max_seconds and embedding_dim must be positive")
        existing = getattr(self, "mem_frame_age_encoder", None)
        if existing is not None:
            if (
                int(existing.temporal_patch_size) != int(self.patch_embed.temporal_patch_size)
                or int(existing.in_proj.out_features) != int(embedding_dim)
                or (not math.isclose(float(existing.max_seconds), float(max_seconds)))
            ):
                raise ValueError("Existing STM frame-age encoder does not match the requested configuration")
            return
        reference = self.patch_embed.proj.weight
        self.mem_frame_age_encoder = Qwen3VLFrameAgeEmbedding(
            temporal_patch_size=int(self.patch_embed.temporal_patch_size),
            embedding_dim=int(embedding_dim),
            hidden_size=int(self.config.hidden_size),
            max_seconds=float(max_seconds),
            device=reference.device,
            dtype=reference.dtype,
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            (merged_h, merged_w) = (height // merge_size, width // merge_size)
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
        (grid_ts, grid_hs, grid_ws) = (grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2])
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
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for (h, w) in zip(grid_hs, grid_ws)])
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

    def _build_position_state(
        self, grid_thw: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        (seq_len, _) = rotary_pos_emb.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return position_embeddings, cu_seqlens

    def _add_mem_frame_age_bias(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, video_frame_age_seconds: torch.Tensor
    ) -> torch.Tensor:
        encoder = getattr(self, "mem_frame_age_encoder", None)
        if encoder is None:
            raise RuntimeError("STM frame-age input was provided but the frame-age encoder is not configured")
        if video_frame_age_seconds.ndim != 2:
            raise ValueError(
                f"video_frame_age_seconds must have shape [num_videos, raw_num_frames], got {tuple(video_frame_age_seconds.shape)}"
            )
        if int(video_frame_age_seconds.shape[0]) != int(grid_thw.shape[0]):
            raise ValueError(
                f"video_frame_age_seconds/video_grid_thw row mismatch: {video_frame_age_seconds.shape[0]} vs {grid_thw.shape[0]}"
            )
        temporal_patch_size = int(encoder.temporal_patch_size)
        age_bias_segments = []
        for age_row, (num_patches, height, width) in zip(
            video_frame_age_seconds, grid_thw.tolist(), strict=True
        ):
            expected_raw_frames = int(num_patches) * temporal_patch_size
            if int(age_row.numel()) > expected_raw_frames:
                raise ValueError(
                    f"Too many raw frame ages for the packed video grid: ages={age_row.numel()} expected<={expected_raw_frames}"
                )
            if int(age_row.numel()) < expected_raw_frames:
                if age_row.numel() == 0:
                    raise ValueError("At least one raw frame age is required per packed video")
                age_row = torch.cat(
                    [age_row, age_row[-1:].expand(expected_raw_frames - int(age_row.numel()))], dim=0
                )
            patch_ages = age_row.reshape(int(num_patches), temporal_patch_size)
            patch_bias = encoder(patch_ages).to(device=hidden_states.device, dtype=hidden_states.dtype)
            tokens_per_patch = int(height) * int(width)
            age_bias_segments.append(
                patch_bias[:, None, :].expand(-1, tokens_per_patch, -1).reshape(-1, hidden_states.shape[-1])
            )
        age_bias = torch.cat(age_bias_segments, dim=0)
        if age_bias.shape != hidden_states.shape:
            raise ValueError(
                f"STM frame-age bias shape {tuple(age_bias.shape)} != hidden state shape {tuple(hidden_states.shape)}"
            )
        return hidden_states + age_bias

    def forward(self, hidden_states, grid_thw, *, video_frame_age_seconds=None):
        return self.temporal_memory.forward_vision(
            self, hidden_states, grid_thw, video_frame_age_seconds=video_frame_age_seconds
        )


class Qwen3VLTextModel(upstream.Qwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_pre_norm_last_hidden_state_only: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and (not torch.jit.is_tracing()):
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
        pre_norm_hidden_states = hidden_states if output_pre_norm_last_hidden_state_only else None
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=(pre_norm_hidden_states,) if pre_norm_hidden_states is not None else None,
        )


class Qwen3VLModel(upstream.Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        # Construct the local memory-aware children without the upstream factory.
        upstream.Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multimodal rotary position indices from video timestamps."""
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            (image_index, video_index) = (0, 0)
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                (image_nums, video_nums) = (0, 0)
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                (remain_images, remain_videos) = (image_nums, video_nums)
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        (t, h, w) = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        (t, h, w) = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    (llm_grid_t, llm_grid_h, llm_grid_w) = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    t_index = (
                        torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    )
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
                )
            return (position_ids, mrope_position_deltas)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_frame_age_seconds: Optional[torch.FloatTensor] = None,
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds, deepstack_video_embeds = self.visual(
            pixel_values_videos,
            grid_thw=video_grid_thw,
            video_frame_age_seconds=video_frame_age_seconds,
        )
        split_sizes = (
            video_grid_thw[:, 1] * video_grid_thw[:, 2] // self.visual.spatial_merge_size**2
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return (video_embeds, deepstack_video_embeds)

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_frame_age_seconds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_last_hidden_state_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        """
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        video_frame_age_seconds (`torch.FloatTensor` of shape `(num_videos, raw_num_frames)`, *optional*):
            Causal elapsed time in seconds for every raw frame in each packed STM video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_mask = None
        video_mask = None
        if pixel_values is not None:
            (image_embeds, deepstack_image_embeds) = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            (image_mask, _) = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            (video_embeds, deepstack_video_embeds) = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_frame_age_seconds=video_frame_age_seconds,
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            (_, video_mask) = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(
                    img_embed.device
                )
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                input_ids is not None
                and input_ids.shape[1] != 1
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                cache_position is not None
                and cache_position[0] == 0
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                (position_ids, rope_deltas) = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor
                )
                self.rope_deltas = rope_deltas
            else:
                (batch_size, seq_length, _) = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        output_hidden_states = bool(kwargs.get("output_hidden_states", False))
        if output_last_hidden_state_only:
            kwargs["output_hidden_states"] = False
            kwargs["output_pre_norm_last_hidden_state_only"] = True
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        pre_norm_last_hidden_state = None
        hidden_states = None
        if output_last_hidden_state_only:
            pre_norm_last_hidden_state = outputs.hidden_states[-1]
        elif output_hidden_states:
            hidden_states = outputs.hidden_states
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            pre_norm_last_hidden_state=pre_norm_last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            rope_deltas=self.rope_deltas,
        )


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    """Multimodal hidden states, attention caches and rotary position metadata."""

    logits: Optional[torch.FloatTensor] = None
    pre_norm_last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3VLForConditionalGeneration(upstream.Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        # Construct the local memory-aware children without the upstream factory.
        upstream.Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_frame_age_seconds: Optional[torch.FloatTensor] = None,
    ):
        return self.model.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            video_frame_age_seconds=video_frame_age_seconds,
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_frame_age_seconds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor, None] = 0,
        output_last_hidden_state_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_frame_age_seconds=video_frame_age_seconds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_last_hidden_state_only=output_last_hidden_state_only,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = None
        if logits_to_keep is not None:
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            raise ValueError("labels are unsupported during evaluation")
        return Qwen3VLCausalLMOutputWithPast(
            logits=logits,
            pre_norm_last_hidden_state=outputs.pre_norm_last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            rope_deltas=outputs.rope_deltas,
        )
