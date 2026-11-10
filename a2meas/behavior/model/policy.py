#!/usr/bin/env python
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_model
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Qwen3VLProcessor
from transformers.cache_utils import DynamicCache

from a2meas.models.action_expert import (
    create_sinusoidal_pos_embedding,
    ActionExpertConfig, ActionExpertMixin, make_att_2d_masks, pad_vector,
    build_suffix_cross_attn_mask_2d, build_full_att_2d_masks, append_prefix_tokens,
    _run_suffix_expert_layer,
    clone_prefix_kv_cache,
)
from a2meas.models.qwen3_vl import backbone_from_config
from a2meas.models.gemma import GemmaExpert, GemmaConfig as ExpertGemmaConfig
from a2meas.models.qwen3_vl import Qwen3VLForConditionalGeneration
from a2meas.models.temporal_memory import Qwen3VLTemporalMemory
from a2meas.schema import ACTION, OBS_STATE, OPENPI_ATTENTION_MASK_VALUE, SHORT_TERM_MEMORY_FRAME_AGE_SECONDS
from .config import (
    A2WLAConfig,
    _resolve_action_head_indices,
    _resolve_behavior_b1k_semantic_action_head_groups,
)
from .conditioning import (
    GLOBAL_TASK_ID,
    GLOBAL_TASK_TEXT,
    GlobalTaskConditioner,
    GlobalTaskInputs,
    validate_task_mapping,
)
from .inputs import RynnBrainInputMixin
from a2meas.models.world_conditioning import compute_alternate_kv_rope_position_ids
from .world_conditioning import WorldModelMixin


_ACTION_INDEX_TENSOR_CACHE: dict[tuple[tuple[int, ...], str, int | None], Tensor] = {}


class A2WLABackbone(ActionExpertMixin, nn.Module):

    def __init__(
        self,
        action_expert_config,
        rynnbrain_path,
        precision="bfloat16",
        enable_temporal_memory: bool = False,
        temporal_memory_pool_size: int = 4,
        short_term_memory_frame_age_embedding_enable: bool = False,
        short_term_memory_frame_age_max_seconds: float = 300.0,
        short_term_memory_frame_age_embedding_dim: int = 64,
    ):
        super().__init__()
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
        self.rynnbrain = backbone_from_config(
            Qwen3VLForConditionalGeneration, rynnbrain_path, dtype=dtype
        )
        vision_model = self.rynnbrain.model.visual
        if enable_temporal_memory:
            reference = vision_model.patch_embed.proj.weight
            vision_model.temporal_memory = Qwen3VLTemporalMemory(
                vision_model.config.hidden_size
            ).to(device=reference.device, dtype=reference.dtype)
            vision_model.temporal_memory.pool_size = temporal_memory_pool_size
        self.rynnbrain_processor = Qwen3VLProcessor.from_pretrained(rynnbrain_path, local_files_only=True)
        vision_model.configure_mem_frame_age_embedding(
            enabled=short_term_memory_frame_age_embedding_enable,
            max_seconds=short_term_memory_frame_age_max_seconds,
            embedding_dim=short_term_memory_frame_age_embedding_dim,
        )
        action_expert_config_hf = ExpertGemmaConfig(
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=dtype,
            use_adarms=True,
            adarms_cond_dim=action_expert_config.width,
            attn_implementation="eager",
        )
        self.gemma_expert = GemmaExpert(config=action_expert_config_hf)
        self.gemma_expert = self.gemma_expert.to(dtype=dtype)
        torch.set_float32_matmul_precision("high")
        compile_mode = os.environ.get("A2MEAS_COMPILE_MODE", "max-autotune-no-cudagraphs")
        for block in vision_model.blocks:
            # Vision attention splits variable-length inputs on the CPU.
            block.mlp.forward = torch.compile(block.mlp.forward, mode=compile_mode, dynamic=True)
        for layer in self.rynnbrain.model.language_model.layers:
            layer.forward = torch.compile(layer.forward, mode=compile_mode, dynamic=True)
        for layer in self.gemma_expert.model.layers:
            # KV cache updates are stateful; compile only the stable feed-forward path.
            layer.mlp.forward = torch.compile(layer.mlp.forward, mode=compile_mode, dynamic=True)
        hidden_size = self.rynnbrain.config.text_config.hidden_size
        self.align_proj = nn.Linear(hidden_size, action_expert_config.width, dtype=dtype)
        self.num_expert_layers = action_expert_config.depth


    def get_text_prefix_embs(self, raw_texts: list[str], device: torch.device | str):
        if len(raw_texts) == 0:
            raise ValueError("raw_texts must not be empty")
        tokenizer = self.rynnbrain_processor.tokenizer
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": text}]}] for text in raw_texts
        ]
        prompts = self.rynnbrain_processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        tokenizer_kwargs = {"padding": True, "return_tensors": "pt", "return_token_type_ids": False}
        single_prompt = prompts[0]
        bos_token = tokenizer.bos_token
        if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
            tokenizer_kwargs["add_special_tokens"] = False
        text_inputs = tokenizer(prompts, **tokenizer_kwargs)
        model_inputs = {
            "input_ids": text_inputs["input_ids"].to(device=device),
            "attention_mask": text_inputs["attention_mask"].to(device=device),
        }
        outputs = self.rynnbrain(
            **model_inputs,
            output_hidden_states=False,
            output_last_hidden_state_only=True,
            use_cache=False,
            logits_to_keep=None,
        )
        prefix_hidden = outputs.pre_norm_last_hidden_state
        prefix_hidden = self.align_proj(
            prefix_hidden.to(device=self.align_proj.weight.device, dtype=self.align_proj.weight.dtype)
        )
        return (
            prefix_hidden,
            model_inputs["attention_mask"].to(device=prefix_hidden.device, dtype=torch.bool),
        )


@dataclass
class FlowInputs:
    noise: Tensor
    prefix_embs: Tensor | None
    prefix_pad_masks: Tensor
    prefix_att_masks: Tensor | None
    past_key_values: DynamicCache | None
    global_task_cond: Tensor | None
    lawam_kv_by_layer: dict[int, tuple[Tensor, Tensor]] | None
    lawam_cond_pad_mask: Tensor | None


class A2WLAModel(WorldModelMixin, RynnBrainInputMixin, nn.Module):
    "Core A2WLA PyTorch model."

    def __init__(self, config: A2WLAConfig):
        super().__init__()
        self.config = config
        action_expert_config = ActionExpertConfig()
        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"BEHAVIOR image inputs require a square resolution, got: {config.image_resolution}"
            )
        self.paligemma_with_expert = A2WLABackbone(
            action_expert_config,
            rynnbrain_path=config.rynnbrain_path,
            precision=config.dtype,
            enable_temporal_memory=config.enable_temporal_memory,
            temporal_memory_pool_size=config.temporal_memory_pool_size,
            short_term_memory_frame_age_embedding_enable=config.short_term_memory_frame_age_embedding_enable,
            short_term_memory_frame_age_max_seconds=config.short_term_memory_frame_age_max_seconds,
            short_term_memory_frame_age_embedding_dim=config.short_term_memory_frame_age_embedding_dim,
        )
        self.lawam_lam = None
        self.lawam_kv_bank = None
        self._lawam_cond_embs: Tensor | None = None
        self._lawam_cond_pad_mask: Tensor | None = None
        self.act_query = None
        self.vlm_to_lam = None
        self.lawam_prefix_adapter = None
        self.lawam_camera_embedding = None
        self.lawam_vision_camera_embedding = None
        self._lawam_teacher_done: torch.cuda.Event | None = None
        self._lawam_aux_done: torch.cuda.Event | None = None
        if config.enable_lawam_stage2:
            self._init_lawam_stage2(
                action_expert_config=action_expert_config, expert_width=int(action_expert_config.width)
            )
        self.trajectory_2d_tokens: nn.Parameter | None = None
        if config.enable_trajectory_2d_aux:
            with torch.random.fork_rng(devices=[]):
                vlm_width = self._rynnbrain_hidden_size()
                self.trajectory_2d_tokens = nn.Parameter(
                    torch.randn(config.trajectory_2d_num_points, vlm_width) * 0.02
                )
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = (
            nn.Linear(action_expert_config.width, config.max_action_dim)
            if not config.enable_behavior_b1k_semantic_action_heads and not config.enable_split_action_heads
            else None
        )
        action_dim = int(config.output_features[ACTION].shape[0])
        (joint_action_indices, gripper_action_indices) = _resolve_action_head_indices(
            config.gripper_action_indices, action_dim
        )
        self._joint_action_indices = tuple(joint_action_indices)
        self._gripper_action_indices = tuple(gripper_action_indices)
        self._behavior_b1k_action_head_groups = (
            _resolve_behavior_b1k_semantic_action_head_groups(action_dim)
            if config.enable_behavior_b1k_semantic_action_heads
            else tuple()
        )
        self.behavior_b1k_action_out_projs = (
            nn.ModuleDict(
                {
                    group_name: nn.Linear(action_expert_config.width, len(group_indices))
                    for (group_name, group_indices) in self._behavior_b1k_action_head_groups
                }
            )
            if self._behavior_b1k_action_head_groups
            else None
        )
        self.joint_action_out_proj = (
            nn.Linear(action_expert_config.width, len(self._joint_action_indices))
            if config.enable_split_action_heads and not config.enable_behavior_b1k_semantic_action_heads
            else None
        )
        self.gripper_action_out_proj = (
            nn.Linear(action_expert_config.width, len(self._gripper_action_indices))
            if config.enable_split_action_heads and not config.enable_behavior_b1k_semantic_action_heads
            else None
        )
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_in_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.state_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_prefix_token = nn.Parameter(torch.zeros(action_expert_config.width))
        self.global_task_conditioner = (
            GlobalTaskConditioner(
                language_dim=action_expert_config.width,
                action_dim=action_expert_config.width,
                hidden_dim=config.global_task_condition_hidden_dim,
                num_tasks=config.global_task_num_tasks,
                mode=config.global_task_conditioning_mode,
            )
            if config.enable_global_task_conditioning
            else None
        )

    def _project_action(self, suffix_out: Tensor) -> Tensor:
        if self.behavior_b1k_action_out_projs is not None:
            action_out = torch.zeros(
                *suffix_out.shape[:-1],
                self.config.max_action_dim,
                device=suffix_out.device,
                dtype=suffix_out.dtype,
            )
            output_dtype = suffix_out.dtype
            for group_name, group_indices in self._behavior_b1k_action_head_groups:
                group_out = self.behavior_b1k_action_out_projs[group_name](suffix_out)
                output_dtype = torch.promote_types(output_dtype, group_out.dtype)
                if action_out.dtype != output_dtype:
                    action_out = action_out.to(dtype=output_dtype)
                action_out[..., list(group_indices)] = group_out.to(dtype=output_dtype)
            return self._zero_unused_action_dims(action_out)
        if not self.config.enable_split_action_heads:
            return self.action_out_proj(suffix_out)
        joint_out = self.joint_action_out_proj(suffix_out)
        gripper_out = self.gripper_action_out_proj(suffix_out)
        output_dtype = torch.promote_types(joint_out.dtype, gripper_out.dtype)
        action_out = torch.zeros(
            *suffix_out.shape[:-1], self.config.max_action_dim, device=suffix_out.device, dtype=output_dtype
        )
        action_out[..., list(self._joint_action_indices)] = joint_out
        action_out[..., list(self._gripper_action_indices)] = gripper_out
        return self._zero_unused_action_dims(action_out)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


    def _effective_action_dim(self) -> int:
        return int(self.config.output_features[ACTION].shape[0])

    def _zero_unused_action_dims(self, tensor: Tensor) -> Tensor:
        if not self.config.enable_split_action_heads:
            return tensor
        action_dim = self._effective_action_dim()
        if tensor.shape[-1] <= action_dim:
            return tensor
        preserved = tensor[..., :action_dim]
        padded = torch.zeros_like(tensor[..., action_dim:])
        return torch.cat([preserved, padded], dim=-1)

    def sample_noise(self, shape, device):
        noise = torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
        return self._zero_unused_action_dims(noise)

    def _project_state_features(self, state_tensor: Tensor) -> Tensor:
        x = F.silu(self.state_in_proj(state_tensor))
        x = F.silu(self.state_mlp_in(x))
        return F.silu(self.state_mlp_out(x))

    def _encode_state_history_tokens(self, state_history: Tensor) -> Tensor:
        if state_history.ndim == 1:
            state_history = state_history[None, None, :]
        elif state_history.ndim == 2:
            state_history = state_history[:, None, :]
        elif state_history.ndim != 3:
            raise ValueError(f"Unsupported state_history ndim: {state_history.ndim}")
        state_history = state_history.to(device=self.state_in_proj.weight.device, dtype=torch.float32)
        state_history = pad_vector(state_history, self.config.max_state_dim)
        batch_size, history_len, _ = state_history.shape
        flat_state = state_history.reshape(batch_size * history_len, self.config.max_state_dim)
        state_tokens = self._project_state_features(flat_state)
        return state_tokens.reshape(batch_size, history_len, -1)


    def _encode_state_history_as_prefix_tokens(self, state_history: Tensor | None) -> Tensor | None:
        if not self.config.enable_state_in_prefix_tokens or state_history is None:
            return None
        state_tokens = self._encode_state_history_tokens(state_history)
        return state_tokens + self.state_prefix_token[None, None, :].to(
            device=state_tokens.device, dtype=state_tokens.dtype
        )


    def _append_state_tokens_to_prefix(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state_history):
        return append_prefix_tokens(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            self._encode_state_history_as_prefix_tokens(state_history),
        )

    def embed_suffix(self, noisy_actions, timestep, global_task_cond: Tensor | None = None):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        adarms_cond = time_emb
        if global_task_cond is not None:
            if global_task_cond.shape != time_emb.shape:
                raise ValueError("Global task condition must match time embedding [B,D]")
            adarms_cond = adarms_cond + global_task_cond.to(device=time_emb.device, dtype=time_emb.dtype)
        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_masks = torch.zeros(self.config.chunk_size, dtype=action_emb.dtype, device=action_emb.device)
        att_masks[0] = 1
        att_masks = att_masks[None, :].expand(bsize, -1)
        return (action_emb, pad_masks, att_masks, adarms_cond)

    def _cast_embs_to_model_dtype(self, *embs: Tensor) -> tuple[Tensor, ...]:
        """Cast prefix/suffix embeddings to the active backbone compute dtype."""
        need_bf16 = self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16
        if not need_bf16:
            return embs
        return tuple((emb.to(dtype=torch.bfloat16) for emb in embs))

    def _encode_global_task_condition(
        self, inputs: GlobalTaskInputs | None
    ) -> Tensor | None:
        conditioner = self.global_task_conditioner
        if conditioner is None:
            return None
        if inputs is None:
            raise ValueError("Global task conditioning requires the global-task preprocessor inputs")
        language = mask = None
        if conditioner.use_language:
            texts = inputs.get("texts")
            if not texts or any((not isinstance(text, str) or not text.strip() for text in texts)):
                raise ValueError("Global task conditioning requires goal-only text")
            (language, mask) = self.paligemma_with_expert.get_text_prefix_embs(
                texts, device=self.action_in_proj.weight.device
            )
        return conditioner(language, mask, inputs.get("ids"))

    def _forward_suffix_alternate_lawam_kv(
        self,
        *,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
        adarms_cond: Tensor | None,
        lawam_kv_by_layer: dict[int, tuple[Tensor, Tensor]],
        lawam_cond_pad_mask: Tensor,
        hidden_residual=None,
    ) -> Tensor:
        """18-layer suffix loop: alternate VLM prefix KV (from cache) vs LaWAM cond KV."""
        expert = self.paligemma_with_expert
        gemma_model = expert.gemma_expert.model
        expert_dtype = gemma_model.layers[0].input_layernorm.dense.weight.dtype
        (suffix_embs,) = self._cast_embs_to_model_dtype(suffix_embs)
        suffix_hidden = suffix_embs.to(dtype=expert_dtype)
        if adarms_cond is not None:
            adarms_cond = adarms_cond.to(dtype=expert_dtype)
        suffix_attention_mode = self.config.suffix_attention_mask
        vlm_att_2d = build_suffix_cross_attn_mask_2d(
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            prefix_pad_masks=prefix_pad_masks,
            suffix_attention_mode=suffix_attention_mode,
        )
        lawam_att_2d = build_suffix_cross_attn_mask_2d(
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            prefix_pad_masks=lawam_cond_pad_mask,
            suffix_attention_mode=suffix_attention_mode,
        )
        vlm_attention_mask = self._prepare_attention_masks_4d(vlm_att_2d)
        lawam_attention_mask = self._prepare_attention_masks_4d(lawam_att_2d)
        n_cond = int(lawam_cond_pad_mask.shape[1])
        (_, suffix_position_ids) = compute_alternate_kv_rope_position_ids(
            prefix_pad_masks=prefix_pad_masks, suffix_pad_masks=suffix_pad_masks, n_cond=n_cond
        )
        num_layers = int(expert.num_expert_layers)
        rotary_model = gemma_model
        for expert_idx, layer in enumerate(gemma_model.layers[:num_layers]):
            if self._lawam_layer_uses_vlm_kv(expert_idx):
                cached = past_key_values.layers[expert_idx]
                prefix_key = cached.keys.to(dtype=expert_dtype)
                prefix_value = cached.values.to(dtype=expert_dtype)
                layer_attention_mask = vlm_attention_mask
            else:
                (prefix_key, prefix_value) = lawam_kv_by_layer[expert_idx]
                layer_attention_mask = lawam_attention_mask
            suffix_hidden = _run_suffix_expert_layer(
                layer,
                rotary_model,
                suffix_hidden,
                layer_attention_mask,
                suffix_position_ids,
                prefix_key,
                prefix_value,
                adarms_cond,
            )
            if expert_idx == 8 and hidden_residual is not None:
                suffix_hidden = hidden_residual(suffix_hidden)
        (suffix_hidden, _) = gemma_model.norm(suffix_hidden, cond=adarms_cond)
        return suffix_hidden

    def _use_prefix_kv_cache_for_inference(self) -> bool:
        return self.config.enable_rynnbrain_prefix_kv_cache or self._lawam_alternate_kv_enabled()

    def _prefill_expert_prefix_kv(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor
    ):
        """Run expert on prefix only (no AdaRMS) and return past_key_values."""
        (prefix_embs,) = self._cast_embs_to_model_dtype(prefix_embs)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        expert = self.paligemma_with_expert
        gemma_model = expert.gemma_expert.model
        if self._lawam_alternate_kv_enabled():
            vlm_layers = [i for i in range(int(expert.num_expert_layers)) if self._lawam_layer_uses_vlm_kv(i)]
            if vlm_layers and vlm_layers[-1] + 1 < len(gemma_model.layers):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    expert_dtype = gemma_model.layers[0].input_layernorm.dense.weight.dtype
                    hidden_states = prefix_embs.to(expert_dtype)
                    if gemma_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                        hidden_states = hidden_states.to(torch.bfloat16)
                    cache = DynamicCache()
                    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                    position_embeddings = gemma_model.rotary_emb(hidden_states, prefix_position_ids)
                    for layer in gemma_model.layers[: vlm_layers[-1] + 1]:
                        hidden_states = layer(
                            hidden_states,
                            attention_mask=prefix_att_2d_masks_4d,
                            position_ids=prefix_position_ids,
                            past_key_value=cache,
                            output_attentions=False,
                            use_cache=True,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            adarms_cond=None,
                        )[0]
                return cache
        (_, past_key_values) = expert(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        return past_key_values

    @torch.no_grad()
    def prepare_flow(
        self,
        images,
        img_pad_masks,
        *,
        raw_texts: list[str] | None = None,
        state_history: Tensor | None = None,
        short_term_memory_frame_age_seconds: Tensor | None = None,
        noise=None,
        global_task_inputs: GlobalTaskInputs | None = None,
    ) -> FlowInputs:
        bsize = images[0].shape[0]
        device = images[0].device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = self.sample_noise(actions_shape, device).to(dtype=torch.float32)
        else:
            x_t = self._zero_unused_action_dims(noise.to(device=device, dtype=torch.float32))
        global_task_cond = self._encode_global_task_condition(global_task_inputs)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_pad_masks=img_pad_masks,
            raw_texts=raw_texts,
            state_history=state_history,
            short_term_memory_frame_age_seconds=short_term_memory_frame_age_seconds,
        )
        use_prefix_cache = self._use_prefix_kv_cache_for_inference()
        past_key_values = None
        if use_prefix_cache:
            past_key_values = self._prefill_expert_prefix_kv(prefix_embs, prefix_pad_masks, prefix_att_masks)
        return FlowInputs(
            noise=x_t,
            prefix_embs=prefix_embs if not use_prefix_cache else None,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks if not use_prefix_cache else None,
            past_key_values=past_key_values,
            global_task_cond=global_task_cond,
            lawam_kv_by_layer=(self.lawam_kv_bank.build_kv_for_layers(
                self._lawam_cond_embs, prefix_pad_masks=prefix_pad_masks,
                rotary_model=self.paligemma_with_expert.gemma_expert.model,
            ) if self._lawam_alternate_kv_enabled() else None),
            lawam_cond_pad_mask=self._lawam_cond_pad_mask,
        )

    @torch.no_grad()
    def integrate_flow(self, flow: FlowInputs, *, num_steps=None, hidden_residual=None) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        x_t = flow.noise
        bsize, device = x_t.shape[0], x_t.device
        dt = -1.0 / num_steps
        for step in range(num_steps):
            step_time = 1.0 + step * dt
            time_tensor = torch.tensor(step_time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                prefix_embs=flow.prefix_embs,
                prefix_pad_masks=flow.prefix_pad_masks,
                prefix_att_masks=flow.prefix_att_masks,
                past_key_values=flow.past_key_values,
                x_t=x_t,
                timestep=time_tensor,
                global_task_cond=flow.global_task_cond,
                lawam_kv_by_layer=flow.lawam_kv_by_layer,
                lawam_cond_pad_mask=flow.lawam_cond_pad_mask,
                hidden_residual=hidden_residual,
            )
            x_t = self._zero_unused_action_dims(x_t + dt * v_t)
        return x_t

    @torch.no_grad()
    def sample_actions(self, images, img_pad_masks, *, raw_texts=None, state_history=None,
                       short_term_memory_frame_age_seconds=None, noise=None, num_steps=None,
                       global_task_inputs=None) -> Tensor:
        flow = self.prepare_flow(
            images, img_pad_masks, raw_texts=raw_texts, state_history=state_history,
            short_term_memory_frame_age_seconds=short_term_memory_frame_age_seconds,
            noise=noise, global_task_inputs=global_task_inputs,
        )
        return self.integrate_flow(flow, num_steps=num_steps)

    def denoise_step(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        x_t,
        timestep,
        global_task_cond: Tensor | None = None,
        *,
        lawam_kv_by_layer: dict[int, tuple[Tensor, Tensor]] | None,
        lawam_cond_pad_mask: Tensor | None,
        hidden_residual=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        x_t = self._zero_unused_action_dims(x_t)
        (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond) = self.embed_suffix(
            x_t, timestep, global_task_cond=global_task_cond
        )
        if self._lawam_alternate_kv_enabled():
            suffix_out = self._forward_suffix_alternate_lawam_kv(
                suffix_embs=suffix_embs,
                suffix_pad_masks=suffix_pad_masks,
                suffix_att_masks=suffix_att_masks,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                adarms_cond=adarms_cond,
                lawam_kv_by_layer=lawam_kv_by_layer,
                lawam_cond_pad_mask=lawam_cond_pad_mask,
                hidden_residual=hidden_residual,
            )
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            action_out = self._project_action(suffix_out)
            return action_out
        if past_key_values is None:
            if prefix_embs is None or prefix_att_masks is None:
                raise ValueError(
                    "prefix_embs and prefix_att_masks are required when prefix cache is disabled"
                )
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_2d_masks = build_full_att_2d_masks(
                prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks,
                suffix_attention_mode=self.config.suffix_attention_mask,
            )
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            (outputs_embeds, _) = self.paligemma_with_expert(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        else:
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_attention_mode = self.config.suffix_attention_mask
            suffix_att_2d_masks = make_att_2d_masks(
                suffix_pad_masks, suffix_att_masks, attention_mode=suffix_attention_mode
            )
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            past_key_values = clone_prefix_kv_cache(past_key_values)
            (outputs_embeds, _) = self.paligemma_with_expert(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        action_out = self._project_action(suffix_out)
        return action_out


class A2WLAPolicy(nn.Module):
    """BEHAVIOR action prediction from a A2WLA checkpoint."""

    config_class = A2WLAConfig

    name = "pi05"

    def __init__(self, config: A2WLAConfig):
        super().__init__()
        config.validate_features()
        self.config = config
        self.model = A2WLAModel(config).to(config.device)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, checkpoint_path, config):
        directory = Path(pretrained_name_or_path)
        source = json.loads((directory / "config.json").read_text())
        if source.get("enable_global_task_conditioning", False) != config.enable_global_task_conditioning:
            raise ValueError("Global task conditioning must match the checkpoint")
        if config.enable_global_task_conditioning:
            for key in (
                "global_task_conditioning_mode",
                "global_task_conditioning_injection",
                "global_task_num_tasks",
                "global_task_condition_hidden_dim",
            ):
                if source.get(key, getattr(A2WLAConfig, key)) != getattr(config, key):
                    raise ValueError(f"Global task architecture differs from checkpoint: {key}")
            if config.global_task_conditioning_mode != "language":
                saved_mapping = validate_task_mapping(
                    source["global_task_text_to_id"], config.global_task_num_tasks
                )
                if saved_mapping != config.global_task_text_to_id:
                    raise ValueError("Refusing to change the checkpoint's global task IDs")
        model = cls(config)
        load_model(model, checkpoint_path, strict=True, device="cpu")
        return model.eval()

    def _preprocess_images(
        self, batch: dict[str, Tensor]
    ) -> list[Tensor]:
        """Map six-frame RGB inputs from [0, 1] to the backbone's [-1, 1] range."""
        return [batch[key] * 2.0 - 1.0 for key in self.config.image_features]

    def _global_task_inputs(self, batch):
        if not self.config.enable_global_task_conditioning:
            return None
        inputs = {}
        if self.config.global_task_conditioning_mode != "task_id":
            inputs["texts"] = batch[GLOBAL_TASK_TEXT]
        if self.config.global_task_conditioning_mode != "language":
            inputs["ids"] = batch[GLOBAL_TASK_ID]
        return inputs

    def _prepare_state_history_for_prefix(
        self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        if not self.config.enable_state_in_prefix_tokens:
            return None
        return batch[OBS_STATE][:, -1:, :].to(device=device, dtype=dtype)

    def _prepare_image_pad_masks(self, batch):
        if self.config.enable_temporal_memory:
            return [batch[f"{key}_is_pad"].to(dtype=torch.bool) for key in self.config.image_features]
        return [None] * len(self.config.image_features)

    @torch.no_grad()
    def prepare_flow(self, batch, *, noise=None) -> FlowInputs:
        images = self._preprocess_images(batch)
        return self.model.prepare_flow(
            images,
            self._prepare_image_pad_masks(batch),
            raw_texts=batch["task"],
            global_task_inputs=self._global_task_inputs(batch),
            state_history=self._prepare_state_history_for_prefix(batch, images[0].device, torch.float32),
            short_term_memory_frame_age_seconds=(
                batch[SHORT_TERM_MEMORY_FRAME_AGE_SECONDS]
                if self.config.short_term_memory_frame_age_embedding_enable else None
            ),
            noise=noise,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, noise=None, num_steps=None):
        flow = self.prepare_flow(batch, noise=noise)
        actions = self.model.integrate_flow(flow, num_steps=num_steps)
        return actions[:, :, : self.config.output_features[ACTION].shape[0]]
