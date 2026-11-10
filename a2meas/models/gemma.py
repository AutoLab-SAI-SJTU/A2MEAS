# Derived from Hugging Face Transformers 4.57.1, licensed under Apache-2.0.
# Copyright 2024 Google Inc. and HuggingFace Inc.; 2025 The Qwen Team.
# See licenses/transformers.txt and README.md.

"""Gemma action expert with the released AdaRMS, residual and cache behavior."""

from typing import Optional
import torch
from torch import nn
from transformers import GemmaConfig as BaseGemmaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
from transformers.models.gemma import modeling_gemma as upstream
from transformers.models.gemma.modeling_gemma import (
    GemmaMLP,
    GemmaRotaryEmbedding,
    eager_attention_forward,
    apply_rotary_pos_emb,
)



class GemmaConfig(BaseGemmaConfig):
    def __init__(self, *, use_adarms=False, adarms_cond_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.use_adarms = use_adarms
        self.adarms_cond_dim = self.hidden_size if use_adarms and adarms_cond_dim is None else adarms_cond_dim


class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed_inputs = self._norm(x)
        if self.dense is not None and cond is None:
            return (normed_inputs.to(dtype), None)
        if self.dense is None:
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return (normed_inputs.to(dtype), None)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        (scale, shift, gate) = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
        return (normed_inputs.to(dtype), gate.to(dtype))

    def extra_repr(self):
        if self.dense is not None:
            repr_str = f"(adaptive), eps={self.eps}, cond_dim={self.cond_dim}"
        else:
            repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        return repr_str


def _gated_residual(x, y, gate):
    """
    Applies gated residual connection with optional gate parameter.

    Args:
        x: Input tensor (residual)
        y: Output tensor to be added
        gate: Optional gate tensor to modulate the addition

    Returns:
        x + y if gate is None, otherwise x + y * gate
    """
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


class GemmaAttention(upstream.GemmaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                (key_states, value_states) = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
                value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)
        (attn_output, attn_weights) = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        (hidden_states, gate) = self.input_layernorm(hidden_states, adarms_cond)
        (hidden_states, self_attn_weights) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = _gated_residual(residual, hidden_states, gate)
        residual = hidden_states
        (hidden_states, gate) = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class GemmaPreTrainedModel(PreTrainedModel):
    config_class = GemmaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GemmaRMSNorm):
            if hasattr(module, "weight"):
                module.weight.data.fill_(1.0)


class GemmaModel(GemmaPreTrainedModel):

    def __init__(self, config: GemmaConfig, *, embed_tokens=True):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) if embed_tokens else None
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.rotary_emb = GemmaRotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """
        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
            Condition for ADARMS.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        (hidden_states, _) = self.norm(hidden_states, adarms_cond)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )




class GemmaExpert(GemmaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaModel(config, embed_tokens=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
