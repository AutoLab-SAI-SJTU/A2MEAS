# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Derived from LeRobot / OpenPI, Apache-2.0. See README.md.
"""Shared A2WLA action-expert dimensions and inference operations."""

import math
from dataclasses import dataclass
import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.models.gemma.modeling_gemma import rotate_half, repeat_kv
from a2meas.models import gemma as modeling_gemma
from torch.nn import functional as F


def clone_prefix_kv_cache(past_key_values):
    """Give each denoise step an owned cache; keep the prefilled snapshot unchanged."""
    return DynamicCache(
        ddp_cache_data=[(layer.keys.clone(), layer.values.clone()) for layer in past_key_values.layers]
    )


@dataclass
class ActionExpertConfig:
    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256


def make_att_2d_masks(pad_masks, att_masks, attention_mode="causal"):
    if attention_mode not in ("causal", "block"):
        raise ValueError(f"attention_mode must be 'causal' or 'block', got '{attention_mode}'")
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    if attention_mode == "block":
        return pad_2d_masks
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def build_suffix_cross_attn_mask_2d(
    *, suffix_pad_masks, suffix_att_masks, prefix_pad_masks, suffix_attention_mode
):
    batch_size = int(suffix_pad_masks.shape[0])
    suffix_len = int(suffix_pad_masks.shape[1])
    prefix_len = int(prefix_pad_masks.shape[1])
    prefix_mask = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_mask = make_att_2d_masks(
        suffix_pad_masks, suffix_att_masks, attention_mode=suffix_attention_mode
    )
    return torch.cat([prefix_mask, suffix_mask], dim=2)


def _apply_rotary_query_key(
    rotary_model, query_state: torch.Tensor, key_state: torch.Tensor, position_ids: torch.Tensor
):
    dummy_tensor = torch.zeros(
        query_state.shape[0],
        query_state.shape[2],
        query_state.shape[-1],
        device=query_state.device,
        dtype=query_state.dtype,
    )
    cos, sin = rotary_model.rotary_emb(dummy_tensor, position_ids)
    return modeling_gemma.apply_rotary_pos_emb(query_state, key_state, cos, sin, unsqueeze_dim=1)


def _key_rotary_position_embeddings(
    rotary_model, key_state: torch.Tensor, position_ids: torch.Tensor
):
    dummy_tensor = torch.zeros(
        key_state.shape[0],
        key_state.shape[2],
        key_state.shape[-1],
        device=key_state.device,
        dtype=key_state.dtype,
    )
    return rotary_model.rotary_emb(dummy_tensor, position_ids)


def _apply_rotary_key_only(
    rotary_model, key_state: torch.Tensor, position_ids: torch.Tensor, *, position_embeddings=None
):
    if position_embeddings is None:
        position_embeddings = _key_rotary_position_embeddings(rotary_model, key_state, position_ids)
    cos, sin = position_embeddings
    cos, sin = (cos.unsqueeze(1), sin.unsqueeze(1))
    return key_state * cos + rotate_half(key_state) * sin


def _run_suffix_expert_layer(
    layer,
    rotary_model,
    suffix_hidden: torch.Tensor,
    suffix_attention_mask: torch.Tensor,
    suffix_position_ids: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    adarms_cond: torch.Tensor | None,
    *,
    attention_implementation: str = "eager",
) -> torch.Tensor:
    hidden_states, gate = layer.input_layernorm(suffix_hidden, cond=adarms_cond)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_state, key_state = _apply_rotary_query_key(
        rotary_model, query_state, key_state, suffix_position_ids
    )
    if prefix_key.dtype != key_state.dtype:
        prefix_key = prefix_key.to(dtype=key_state.dtype)
    if prefix_value.dtype != value_state.dtype:
        prefix_value = prefix_value.to(dtype=value_state.dtype)
    full_key_state = torch.cat([prefix_key, key_state], dim=2)
    full_value_state = torch.cat([prefix_value, value_state], dim=2)
    if attention_implementation == "sdpa":
        key_states = repeat_kv(full_key_state, layer.self_attn.num_key_value_groups)
        value_states = repeat_kv(full_value_state, layer.self_attn.num_key_value_groups)
        sdpa_mask = suffix_attention_mask.to(dtype=query_state.dtype)
        att_output = F.scaled_dot_product_attention(
            query_state,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            scale=layer.self_attn.scaling,
        )
        att_output = att_output.transpose(1, 2).contiguous()
    else:
        att_output, _ = modeling_gemma.eager_attention_forward(
            layer.self_attn,
            query_state,
            full_key_state,
            full_value_state,
            suffix_attention_mask,
            layer.self_attn.scaling,
        )
    att_output = att_output.reshape(suffix_hidden.shape[0], suffix_hidden.shape[1], -1)
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output)
    out_emb = modeling_gemma._gated_residual(suffix_hidden, out_emb, gate)
    after_first_residual = out_emb
    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    out_emb = layer.mlp(out_emb)
    return modeling_gemma._gated_residual(after_first_residual, out_emb, gate)


def build_full_att_2d_masks(
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    *,
    suffix_attention_mode: str,
) -> torch.Tensor:
    """Build the full [prefix+suffix]x[prefix+suffix] attention mask with configurable suffix mode."""
    prefix_att_2d_masks = make_att_2d_masks(
        prefix_pad_masks, prefix_att_masks, attention_mode="causal"
    )
    suffix_att_2d_masks = make_att_2d_masks(
        suffix_pad_masks, suffix_att_masks, attention_mode=suffix_attention_mode
    )
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    prefix_to_suffix_masks = torch.zeros(
        (batch_size, prefix_len, suffix_len), dtype=torch.bool, device=prefix_pad_masks.device
    )
    suffix_to_prefix_masks = suffix_pad_masks[:, :, None].expand(
        batch_size, suffix_len, prefix_len
    ) & prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    top_masks = torch.cat([prefix_att_2d_masks, prefix_to_suffix_masks], dim=2)
    bottom_masks = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)
    return torch.cat([top_masks, bottom_masks], dim=1)


def append_prefix_tokens(
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    tokens: torch.Tensor | None,
    token_pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate conditioning tokens onto the expert prefix."""
    if tokens is None:
        return (prefix_embs, prefix_pad_masks, prefix_att_masks)
    tokens = tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
    if token_pad_mask is None:
        pad = torch.ones(
            tokens.shape[:2], dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device
        )
    else:
        if tuple(token_pad_mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(
                f"token_pad_mask shape must match prefix tokens: mask={tuple(token_pad_mask.shape)} tokens={tuple(tokens.shape[:2])}"
            )
        pad = token_pad_mask.to(device=prefix_pad_masks.device, dtype=prefix_pad_masks.dtype)
    att = torch.zeros(
        tokens.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_att_masks.device
    )
    prefix_embs = torch.cat([prefix_embs, tokens], dim=1)
    prefix_pad_masks = torch.cat([prefix_pad_masks, pad], dim=1)
    prefix_att_masks = torch.cat([prefix_att_masks, att], dim=1)
    return (prefix_embs, prefix_pad_masks, prefix_att_masks)


class ActionExpertMixin:
    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        (prefix_embs, suffix_embs) = (inputs_embeds[0], inputs_embeds[1])
        expert_dtype = self.gemma_expert.model.layers[0].input_layernorm.dense.weight.dtype
        if adarms_cond is not None:
            if adarms_cond[0] is not None:
                adarms_cond[0] = adarms_cond[0].to(expert_dtype)
            if adarms_cond[1] is not None:
                adarms_cond[1] = adarms_cond[1].to(expert_dtype)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if suffix_embs is None:
                combined_embs = prefix_embs.to(expert_dtype)
                cond_index = 0
            elif prefix_embs is None:
                combined_embs = suffix_embs.to(expert_dtype)
                cond_index = 1
            else:
                combined_embs = torch.cat(
                    [prefix_embs.to(expert_dtype), suffix_embs.to(expert_dtype)], dim=1
                )
                cond_index = 1
            expert_out = self.gemma_expert.model(
                inputs_embeds=combined_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[cond_index] if adarms_cond is not None else None,
            )
            hidden = expert_out.last_hidden_state
            if suffix_embs is None:
                return [hidden, None], expert_out.past_key_values
            if prefix_embs is None:
                return [None, hidden], expert_out.past_key_values
            return [hidden[:, : prefix_embs.shape[1]], hidden[:, prefix_embs.shape[1] :]], None


def build_inject_sequence(base_inputs_embeds, base_pad_masks, *, la_query=None):
    """Append learned query tokens after the observed vision/text prefix."""
    prefix_embs = base_inputs_embeds
    prefix_pad_masks = base_pad_masks
    la_len = 0
    if la_query is not None:
        la_embs = la_query.unsqueeze(0).expand(prefix_embs.shape[0], -1, -1).to(
            device=prefix_embs.device, dtype=prefix_embs.dtype
        )
        la_len = int(la_embs.shape[1])
        la_pad = torch.ones(
            prefix_embs.shape[0], la_len,
            dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device,
        )
        prefix_embs = torch.cat([prefix_embs, la_embs], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, la_pad], dim=1)
    return prefix_embs, prefix_pad_masks, int(prefix_embs.shape[1]), la_len


@dataclass
class RynnBrainInjectForwardResult:
    full_hidden_pre_norm: Tensor
    base_len: int
    stage_base_len: int
    la_len: int = 0


def action_prefix_hidden_from_inject_result(forward_result):
    return forward_result.full_hidden_pre_norm[:, : forward_result.base_len]


def extract_la_hidden_from_inject_result(
    forward_result: RynnBrainInjectForwardResult,
) -> Tensor | None:
    """Slice LA-query hiddens: they sit right after vision+task."""
    la_len = int(forward_result.la_len)
    if la_len <= 0:
        return None
    start = int(forward_result.stage_base_len)
    end = start + la_len
    return forward_result.full_hidden_pre_norm[:, start:end, :]


def extend_visual_pos_masks(
    visual_pos_masks: Tensor | None, batch_size: int, num_aux_tokens: int, *, device: torch.device
) -> Tensor | None:
    if visual_pos_masks is None or num_aux_tokens <= 0:
        return visual_pos_masks
    aux_false = torch.zeros(batch_size, num_aux_tokens, dtype=torch.bool, device=device)
    return torch.cat([visual_pos_masks.to(device=device), aux_false], dim=1)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
