from __future__ import annotations
from a2meas.robotwin.model.prefix import run_rynnbrain_inject_full_forward
import torch
from a2meas.models.action_expert import action_prefix_hidden_from_inject_result
import os
from pathlib import Path
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING
from a2meas.models.action_expert import create_sinusoidal_pos_embedding, ActionExpertConfig, ActionExpertMixin, make_att_2d_masks, pad_vector, build_suffix_cross_attn_mask_2d, append_prefix_tokens, _run_suffix_expert_layer
from a2meas.models.qwen3_vl import backbone_from_config
from a2meas.models.gemma import GemmaExpert
from a2meas.robotwin.model.qwen3_vl import RynnBrainBackbone
from transformers import Qwen3VLProcessor
from a2meas.robotwin.model.config import A2WLAConfig
from a2meas.schema import ACTION, OBS_STATE, OPENPI_ATTENTION_MASK_VALUE
_RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES = 4096
_RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS = 1 << 20

def resize_with_pad_torch(images: Tensor, height: int, width: int) -> Tensor:
    """Resize the canonical float32 NHWC policy images, then pad with -1."""
    images = images.permute(0, 3, 1, 2)
    current_height, current_width = images.shape[-2:]
    ratio = max(current_width / width, current_height / height)
    resized_height, resized_width = (int(current_height / ratio), int(current_width / ratio))
    images = F.interpolate(images, size=(resized_height, resized_width), mode='bilinear', align_corners=False)
    images = images.clamp(-1.0, 1.0)
    top, extra_height = divmod(height - resized_height, 2)
    left, extra_width = divmod(width - resized_width, 2)
    images = F.pad(images, (left, left + extra_width, top, top + extra_height), value=-1.0)
    return images.permute(0, 2, 3, 1)

class A2WLABackbone(ActionExpertMixin, nn.Module):

    def __init__(self, action_expert_config, rynnbrain_path='', precision='bfloat16'):
        super().__init__()
        dtype = torch.bfloat16 if precision == 'bfloat16' else torch.float32
        self.rynnbrain = backbone_from_config(RynnBrainBackbone, rynnbrain_path, dtype=dtype)
        self.rynnbrain_processor = Qwen3VLProcessor.from_pretrained(rynnbrain_path, local_files_only=True)
        self.rynnbrain.config.mem_compress_video_tokens = True
        vision_config = self.rynnbrain.config.vision_config
        vision_config.mem_enable_short_term_memory = True
        vision_config.mem_drop_history_last_n_layers = 4
        action_expert_config_hf = CONFIG_MAPPING['gemma'](head_dim=action_expert_config.head_dim, hidden_size=action_expert_config.width, intermediate_size=action_expert_config.mlp_dim, num_attention_heads=action_expert_config.num_heads, num_hidden_layers=action_expert_config.depth, num_key_value_heads=action_expert_config.num_kv_heads, vocab_size=257152, hidden_activation='gelu_pytorch_tanh', torch_dtype=dtype, use_adarms=True, adarms_cond_dim=action_expert_config.width)
        self.gemma_expert = GemmaExpert(config=action_expert_config_hf)
        self.gemma_expert = self.gemma_expert.to(dtype=dtype)
        hidden_size = self.rynnbrain.config.text_config.hidden_size
        self.align_proj = nn.Linear(hidden_size, action_expert_config.width, dtype=dtype)
        self.num_expert_layers = action_expert_config.depth


from a2meas.models.world_conditioning import compute_alternate_kv_rope_position_ids
from a2meas.robotwin.model.world_conditioning import WorldModelMixin

class A2WLAModel(WorldModelMixin, nn.Module):

    def __init__(self, config: A2WLAConfig):
        super().__init__()
        self.config = config
        self._rynnbrain_video_token_cache = {}
        self._rynnbrain_video_token_cache_tokens = 0
        action_expert_config = ActionExpertConfig()
        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(f'PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}')
        self.paligemma_with_expert = A2WLABackbone(action_expert_config, rynnbrain_path=config.rynnbrain_path, precision=config.dtype)
        self.lawam_lam = None
        self._lawam_vision_stream = None
        self.lawam_kv_bank = None
        self._lawam_cond_embs: Tensor | None = None
        self._lawam_cond_pad_mask: Tensor | None = None
        self.act_query = None
        self.vlm_to_lam = None
        self.lawam_prefix_adapter = None
        self.lawam_camera_embedding = None
        self.lawam_vision_camera_embedding = None
        self._init_lawam_stage2(action_expert_config=action_expert_config, expert_width=int(action_expert_config.width))
        self.trajectory_queries = nn.Parameter(torch.empty(config.trajectory_query_count, self._rynnbrain_hidden_size()))
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self._joint_action_indices = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
        self._gripper_action_indices = (6, 13)
        self.joint_action_out_proj = nn.Linear(action_expert_config.width, len(self._joint_action_indices))
        self.gripper_action_out_proj = nn.Linear(action_expert_config.width, len(self._gripper_action_indices))
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_in_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.state_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.state_prefix_token = nn.Parameter(torch.zeros(action_expert_config.width))
        self.flow_source_in_proj = nn.Linear(action_expert_config.width * 1, action_expert_config.width)
        self.flow_source_out_proj = nn.Linear(action_expert_config.width, config.chunk_size * config.max_action_dim)

    def _project_action(self, suffix_out: Tensor) -> Tensor:
        joint_out = self.joint_action_out_proj(suffix_out)
        gripper_out = self.gripper_action_out_proj(suffix_out)
        output_dtype = torch.promote_types(joint_out.dtype, gripper_out.dtype)
        action_out = torch.zeros(*suffix_out.shape[:-1], self.config.max_action_dim, device=suffix_out.device, dtype=output_dtype)
        action_out[..., list(self._joint_action_indices)] = joint_out
        action_out[..., list(self._gripper_action_indices)] = gripper_out
        return self._zero_unused_action_dims(action_out)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _effective_action_dim(self) -> int:
        return int(self.config.output_features[ACTION].shape[0])

    def _zero_unused_action_dims(self, tensor: Tensor) -> Tensor:
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

    def _encode_state_history_tokens(self, state_history: Tensor | None) -> Tensor | None:
        if state_history is None:
            return None
        if state_history.ndim == 1:
            state_history = state_history[None, None, :]
        elif state_history.ndim == 2:
            state_history = state_history[:, None, :]
        elif state_history.ndim != 3:
            raise ValueError(f'Unsupported state_history ndim: {state_history.ndim}')
        state_history = state_history.to(device=self.state_in_proj.weight.device, dtype=torch.float32)
        state_history = pad_vector(state_history, self.config.max_state_dim)
        batch_size, history_len, _ = state_history.shape
        flat_state = state_history.reshape(batch_size * history_len, self.config.max_state_dim)
        state_tokens = self._project_state_features(flat_state)
        return state_tokens.reshape(batch_size, history_len, -1)

    def _build_state_history_flow_source(self, device: torch.device, dtype: torch.dtype, state_history: Tensor | None=None) -> Tensor:
        state_tokens = self._encode_state_history_tokens(state_history)
        if state_tokens is None:
            raise ValueError('state_history is required when flow_source uses state history')
        flat_tokens = state_tokens.reshape(state_tokens.shape[0], -1)
        flow_source = self.flow_source_out_proj(F.silu(self.flow_source_in_proj(flat_tokens)))
        flow_source = flow_source.reshape(state_tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)
        return self._zero_unused_action_dims(flow_source).to(device=device, dtype=dtype)

    def _build_flow_source(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, state_history: Tensor | None=None) -> Tensor:
        gaussian_source = self.sample_noise(shape, device).to(dtype=dtype)
        state_source = self._build_state_history_flow_source(device, dtype, state_history=state_history)
        blend_alpha = float(self.config.flow_source_blend_alpha)
        flow_source = (1.0 - blend_alpha) * gaussian_source + blend_alpha * state_source
        return self._zero_unused_action_dims(flow_source).to(dtype=dtype)

    def get_rynnbrain_processor_inputs(self, images, img_pad_masks=None, raw_texts: list[str] | None=None):
        """Build the released two-frame video and text inputs."""
        device = next(self.parameters()).device
        num_frames = images[0].shape[1]
        def process_inputs(images_inner, img_pad_masks_inner, raw_texts_inner):
            bsize = images_inner[0].shape[0]
            cpu_img_pad_masks = [pad_mask.detach().to(device='cpu', non_blocking=False) if isinstance(pad_mask, torch.Tensor) and pad_mask.device.type != 'cpu' else pad_mask for pad_mask in img_pad_masks_inner]
            raw_texts_local = list(raw_texts_inner)
            if len(raw_texts_local) != bsize:
                raise ValueError(f'Expected {bsize} prompts, got {len(raw_texts_local)}')
            conversations = []
            batch_videos = []

            def _is_padding_frame(img_tensor: torch.Tensor) -> bool:
                return bool(torch.all(img_tensor == -1).item())

            def _is_explicitly_padded(pad_tensor: torch.Tensor | None, sample_idx: int, frame_idx: int | None=None) -> bool | None:
                if pad_tensor is None:
                    return None
                if pad_tensor.ndim == 1:
                    return bool(pad_tensor[sample_idx].item())
                if frame_idx is not None and pad_tensor.ndim >= 2:
                    return bool(pad_tensor[sample_idx, frame_idx].item())
                return None

            def _should_skip_frame(img_tensor: torch.Tensor, pad_tensor: torch.Tensor | None, sample_idx: int, frame_idx: int | None=None) -> bool:
                explicit_pad = _is_explicitly_padded(pad_tensor, sample_idx, frame_idx)
                if explicit_pad is not None:
                    return explicit_pad
                return _is_padding_frame(img_tensor)

            def _to_preprocessed_video_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
                return img_tensor.detach().contiguous()

            for b_idx in range(bsize):
                content = []
                sample_videos = []
                for cam_imgs, cam_pad_mask in zip(images_inner, cpu_img_pad_masks, strict=True):
                    prepared_frames: list[torch.Tensor | None] = []
                    for frame_idx in range(num_frames):
                        img_tensor = cam_imgs[b_idx, frame_idx]
                        if _should_skip_frame(img_tensor, cam_pad_mask, b_idx, frame_idx):
                            prepared_frames.append(None)
                            continue
                        prepared_frames.append(_to_preprocessed_video_tensor(img_tensor))
                    first_valid_frame = next((frame for frame in prepared_frames if frame is not None), None)
                    if first_valid_frame is not None:
                        frames = []
                        last_valid_frame = first_valid_frame
                        for frame in prepared_frames:
                            if frame is None:
                                frames.append(last_valid_frame.clone())
                            else:
                                last_valid_frame = frame
                                frames.append(frame)
                        video_tensor = torch.stack(frames, dim=0)
                        content.append({'type': 'video', 'video': video_tensor})
                        sample_videos.append(video_tensor)
                content.append({'type': 'text', 'text': raw_texts_local[b_idx]})
                conversations.append([{'role': 'user', 'content': content}])
                batch_videos.append(sample_videos)
            processor = self.paligemma_with_expert.rynnbrain_processor
            prompts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            has_videos = any((sample_videos for sample_videos in batch_videos))
            if has_videos:
                video_inputs = processor.video_processor(videos=batch_videos, do_sample_frames=False, do_convert_rgb=False, do_rescale=False, do_normalize=False, return_tensors='pt')
                merge_length = processor.video_processor.merge_size ** 2
                video_grid_thw = video_inputs['video_grid_thw']
                prompts_list = prompts if isinstance(prompts, list) else [prompts]
                grid_keys = []
                grid_offset = 0
                for sample_videos in batch_videos:
                    next_grid_offset = grid_offset + len(sample_videos)
                    sample_grid = video_grid_thw[grid_offset:next_grid_offset]
                    grid_keys.append(tuple((tuple((int(value) for value in row)) for row in sample_grid.tolist())))
                    grid_offset = next_grid_offset
                if grid_offset != len(video_grid_thw):
                    raise ValueError(f'RynnBrain video grid count mismatch: consumed={grid_offset}, available={len(video_grid_thw)}')
                cache_enabled = processor.tokenizer.padding_side == 'right'
                token_cache = None
                cache_keys = []
                cached_rows = []
                if cache_enabled:
                    token_cache = self._rynnbrain_video_token_cache
                    cache_keys = [(prompt, grid_key) for prompt, grid_key in zip(prompts_list, grid_keys, strict=True)]
                    cached_rows = [token_cache.get(cache_key) for cache_key in cache_keys]
                cache_hit = cache_enabled and all((cached is not None for cached in cached_rows))
                if not cache_hit:
                    processed_prompts = prompts_list.copy()
                    grid_index = 0
                    for prompt_index in range(len(processed_prompts)):
                        while processor.video_token in processed_prompts[prompt_index]:
                            grid_height = int(video_grid_thw[grid_index][1].item())
                            grid_width = int(video_grid_thw[grid_index][2].item())
                            frame_seqlen = grid_height * grid_width // merge_length
                            video_placeholder = processor.vision_start_token + '<|placeholder|>' * frame_seqlen + processor.vision_end_token
                            wrapped_video_token = f'{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}'
                            replace_token = wrapped_video_token if wrapped_video_token in processed_prompts[prompt_index] else processor.video_token
                            processed_prompts[prompt_index] = processed_prompts[prompt_index].replace(replace_token, video_placeholder, 1)
                            grid_index += 1
                        processed_prompts[prompt_index] = processed_prompts[prompt_index].replace('<|placeholder|>', processor.video_token)
                    if grid_index != len(video_grid_thw):
                        raise ValueError(f'RynnBrain prompt/grid mismatch: consumed={grid_index}, available={len(video_grid_thw)}')
                    tokenizer_kwargs = {'padding': True, 'return_tensors': 'pt', 'return_token_type_ids': False}
                    bos_token = processor.tokenizer.bos_token
                    single_prompt = processed_prompts[0] if processed_prompts else None
                    if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
                        tokenizer_kwargs['add_special_tokens'] = False
                    text_inputs = processor.tokenizer(processed_prompts, **tokenizer_kwargs)
                    processor._check_special_mm_tokens(processed_prompts, text_inputs, modalities=['video'])
                    position_ids, _ = self.paligemma_with_expert.rynnbrain.model.get_rope_index(text_inputs['input_ids'], None, video_grid_thw, attention_mask=text_inputs['attention_mask'])
                    rynnbrain_inputs = {**text_inputs, **video_inputs, 'position_ids': position_ids}
                    if cache_enabled:
                        staged_rows = {}
                        for row_index, cache_key in enumerate(cache_keys):
                            valid_mask = text_inputs['attention_mask'][row_index].bool()
                            staged_rows[cache_key] = (text_inputs['input_ids'][row_index, valid_mask].contiguous(), position_ids[:, row_index, valid_mask].contiguous())
                        staged_tokens = sum((row[0].numel() for row in staged_rows.values()))
                        if len(staged_rows) <= _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES and staged_tokens <= _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS:
                            cached_tokens = self._rynnbrain_video_token_cache_tokens
                            replaced_tokens = sum((token_cache[key][0].numel() for key in staged_rows if key in token_cache))
                            projected_tokens = cached_tokens - replaced_tokens + staged_tokens
                            projected_entries = len(set(token_cache).union(staged_rows))
                            if projected_entries > _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES or projected_tokens > _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS:
                                token_cache.clear()
                                projected_tokens = staged_tokens
                            token_cache.update(staged_rows)
                            self._rynnbrain_video_token_cache_tokens = projected_tokens
                else:
                    token_rows = [cached[0] for cached in cached_rows]
                    position_rows = [cached[1] for cached in cached_rows]
                    max_valid_tokens = max((row.shape[0] for row in token_rows))
                    padded_length = max_valid_tokens
                    pad_token_id = processor.tokenizer.pad_token_id
                    if pad_token_id is None:
                        raise ValueError('RynnBrain tokenizer requires a pad token for batched video inputs')
                    input_ids = torch.full((bsize, padded_length), pad_token_id, dtype=token_rows[0].dtype)
                    attention_mask = torch.zeros_like(input_ids)
                    position_ids = torch.ones((3, bsize, padded_length), dtype=position_rows[0].dtype)
                    for row_index, (token_row, position_row) in enumerate(zip(token_rows, position_rows, strict=True)):
                        valid_length = token_row.shape[0]
                        input_ids[row_index, :valid_length] = token_row
                        attention_mask[row_index, :valid_length] = 1
                        position_ids[:, row_index, :valid_length] = position_row
                    rynnbrain_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, **video_inputs, 'position_ids': position_ids}
            else:
                processor_kwargs = {'text': prompts, 'padding': True, 'return_tensors': 'pt'}
                single_prompt = prompts[0] if isinstance(prompts, list) and prompts else prompts
                bos_token = processor.tokenizer.bos_token
                if bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(bos_token):
                    processor_kwargs['add_special_tokens'] = False
                rynnbrain_inputs = processor(**processor_kwargs)
            if 'position_ids' not in rynnbrain_inputs:
                position_ids, _ = self.paligemma_with_expert.rynnbrain.model.get_rope_index(rynnbrain_inputs.get('input_ids'), rynnbrain_inputs.get('image_grid_thw'), rynnbrain_inputs.get('video_grid_thw'), attention_mask=rynnbrain_inputs.get('attention_mask'))
                rynnbrain_inputs['position_ids'] = position_ids
            cpu_metadata_keys = {'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts'}
            return {k: (v.contiguous() if k in cpu_metadata_keys else v.contiguous().to(device)) if isinstance(v, torch.Tensor) else v for k, v in rynnbrain_inputs.items()}
        normalized_img_pad_masks = list(img_pad_masks) if img_pad_masks is not None else [None] * len(images)
        rynnbrain_inputs = process_inputs(images, normalized_img_pad_masks, raw_texts)
        pad = rynnbrain_inputs['attention_mask'].bool()
        return rynnbrain_inputs, pad, torch.zeros_like(pad, dtype=torch.bool)

    def _encode_state_history_as_prefix_tokens(self, state_history: Tensor | None) -> Tensor | None:
        if state_history is None:
            return None
        state_tokens = self._encode_state_history_tokens(state_history)
        return state_tokens + self.state_prefix_token[None, None, :].to(device=state_tokens.device, dtype=state_tokens.dtype)

    def _append_state_tokens_to_prefix(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state_history):
        return append_prefix_tokens(prefix_embs, prefix_pad_masks, prefix_att_masks, self._encode_state_history_as_prefix_tokens(state_history))

    def embed_suffix(self, noisy_actions, timestep):
        time_emb = create_sinusoidal_pos_embedding(timestep, self.action_in_proj.out_features, min_period=self.config.min_period, max_period=self.config.max_period, device=timestep.device)
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        adarms_cond = time_emb
        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_masks = torch.zeros(self.config.chunk_size, dtype=action_emb.dtype, device=action_emb.device)
        att_masks[0] = 1
        att_masks = att_masks[None, :].expand(bsize, -1)
        return (action_emb, pad_masks, att_masks, adarms_cond)

    def _cast_embs_to_model_dtype(self, *embs: Tensor) -> tuple[Tensor, ...]:
        need_bf16 = self.paligemma_with_expert.rynnbrain.dtype == torch.bfloat16
        if not need_bf16:
            return embs
        return tuple((emb.to(dtype=torch.bfloat16) for emb in embs))

    @staticmethod
    def _get_expert_prefix_kv_at_layer(past_key_values, layer_idx: int):
        layer = past_key_values.layers[layer_idx]
        return (layer.keys, layer.values)

    def _forward_suffix_alternate_lawam_kv(self, *, suffix_embs: Tensor, suffix_pad_masks: Tensor, suffix_att_masks: Tensor, prefix_pad_masks: Tensor, past_key_values, adarms_cond: Tensor | None, lawam_kv_by_layer: dict[int, tuple[Tensor, Tensor]], lawam_cond_pad_mask: Tensor) -> Tensor:
        expert = self.paligemma_with_expert
        gemma_model = expert.gemma_expert.model
        gemma_model.config._attn_implementation = self._action_expert_attention_backend()
        expert_dtype = gemma_model.layers[0].input_layernorm.dense.weight.dtype
        suffix_embs, = self._cast_embs_to_model_dtype(suffix_embs)
        suffix_hidden = suffix_embs.to(dtype=expert_dtype)
        if adarms_cond is not None:
            adarms_cond = adarms_cond.to(dtype=expert_dtype)
        vlm_att_2d = build_suffix_cross_attn_mask_2d(suffix_pad_masks=suffix_pad_masks, suffix_att_masks=suffix_att_masks, prefix_pad_masks=prefix_pad_masks, suffix_attention_mode='block')
        lawam_att_2d = build_suffix_cross_attn_mask_2d(suffix_pad_masks=suffix_pad_masks, suffix_att_masks=suffix_att_masks, prefix_pad_masks=lawam_cond_pad_mask, suffix_attention_mode='block')
        vlm_attention_mask = self._prepare_attention_masks_4d(vlm_att_2d)
        lawam_attention_mask = self._prepare_attention_masks_4d(lawam_att_2d)
        n_cond = int(lawam_cond_pad_mask.shape[1])
        _, suffix_position_ids = compute_alternate_kv_rope_position_ids(prefix_pad_masks=prefix_pad_masks, suffix_pad_masks=suffix_pad_masks, n_cond=n_cond)
        num_layers = int(expert.num_expert_layers)
        rotary_model = gemma_model
        attention_implementation = self._action_expert_attention_backend()
        for expert_idx, layer in enumerate(gemma_model.layers[:num_layers]):
            if self._lawam_layer_uses_vlm_kv(expert_idx):
                prefix_key, prefix_value = self._get_expert_prefix_kv_at_layer(past_key_values, expert_idx)
                prefix_key = prefix_key.to(dtype=expert_dtype)
                prefix_value = prefix_value.to(dtype=expert_dtype)
                layer_attention_mask = vlm_attention_mask
            else:
                prefix_key, prefix_value = lawam_kv_by_layer[expert_idx]
                layer_attention_mask = lawam_attention_mask
            suffix_hidden = _run_suffix_expert_layer(layer, rotary_model, suffix_hidden, layer_attention_mask, suffix_position_ids, prefix_key, prefix_value, adarms_cond, attention_implementation=attention_implementation)
        suffix_hidden, _ = gemma_model.norm(suffix_hidden, cond=adarms_cond)
        return suffix_hidden

    @staticmethod
    def _action_expert_attention_backend() -> str:
        backend = os.environ.get('LEROBOT_ACTION_EXPERT_ATTN_IMPLEMENTATION', 'eager').strip().lower()
        if backend not in {'eager', 'sdpa'}:
            raise ValueError(f'LEROBOT_ACTION_EXPERT_ATTN_IMPLEMENTATION must be eager or sdpa, got {backend!r}')
        return backend

    def _prefill_expert_prefix_kv(self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor):
        prefix_embs, = self._cast_embs_to_model_dtype(prefix_embs)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        expert = self.paligemma_with_expert
        gemma_model = expert.gemma_expert.model
        gemma_model.config._attn_implementation = self._action_expert_attention_backend()
        from transformers.cache_utils import DynamicCache
        with torch.autocast(device_type=prefix_embs.device.type, dtype=torch.bfloat16,
                            enabled=prefix_embs.is_cuda):
            expert_dtype = gemma_model.layers[0].input_layernorm.dense.weight.dtype
            hidden_states = prefix_embs.to(expert_dtype)
            if gemma_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
            cache = DynamicCache()
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            position_embeddings = gemma_model.rotary_emb(hidden_states, prefix_position_ids)
            for layer in gemma_model.layers[:-1]:
                hidden_states = layer(hidden_states, attention_mask=prefix_att_2d_masks_4d,
                                      position_ids=prefix_position_ids, past_key_value=cache,
                                      output_attentions=False, use_cache=True, cache_position=cache_position,
                                      position_embeddings=position_embeddings, adarms_cond=None)[0]
        return cache

    def sample_actions(self, images, img_pad_masks, raw_texts: list[str] | None=None, state_history: Tensor | None=None, flow_source_state_history: Tensor | None=None, noise=None, num_steps=None) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = images[0].shape[0]
        device = next(self.parameters()).device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = self._build_flow_source(actions_shape, device, torch.float32, state_history=flow_source_state_history)
        else:
            x_t = self._zero_unused_action_dims(noise.to(device=device, dtype=torch.float32))
        try:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_lawam_prefix(
                images, img_pad_masks=img_pad_masks, raw_texts=raw_texts, state_history=state_history
            )
            past_key_values = self._prefill_expert_prefix_kv(prefix_embs, prefix_pad_masks, prefix_att_masks)
            lawam_kv_by_layer = self.lawam_kv_bank.build_kv_for_layers(
                self._lawam_cond_embs, prefix_pad_masks=prefix_pad_masks,
                rotary_model=self.paligemma_with_expert.gemma_expert.model,
            )
            dt = -1.0 / num_steps
            timesteps = torch.tensor([1.0 + step * dt for step in range(num_steps)], dtype=torch.float32, device=device)
            for step in range(num_steps):
                time_tensor = timesteps[step].expand(bsize)
                v_t = self.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=time_tensor, lawam_kv_by_layer=lawam_kv_by_layer)
                x_t = self._zero_unused_action_dims(x_t + dt * v_t)
            return x_t
        finally:
            self._lawam_cond_embs = None
            self._lawam_cond_pad_mask = None

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, lawam_kv_by_layer):
        x_t = self._zero_unused_action_dims(x_t)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = self._action_expert_attention_backend()
        suffix_out = self._forward_suffix_alternate_lawam_kv(suffix_embs=suffix_embs, suffix_pad_masks=suffix_pad_masks, suffix_att_masks=suffix_att_masks, prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, adarms_cond=adarms_cond, lawam_kv_by_layer=lawam_kv_by_layer, lawam_cond_pad_mask=self._lawam_cond_pad_mask)
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        action_out = self._project_action(suffix_out)
        return action_out

    def _rynnbrain_hidden_size(self) -> int:
        return self.paligemma_with_expert.rynnbrain.config.text_config.hidden_size

    def _run_feature_inject(self, inputs, base_pad, *, la_query=None):
        rb = self.paligemma_with_expert.rynnbrain
        device = next(rb.parameters()).device
        result = run_rynnbrain_inject_full_forward(rb, inputs, base_pad, la_query=la_query)
        hidden = action_prefix_hidden_from_inject_result(result)
        pad = base_pad
        if result.la_len:
            pad = torch.cat([pad, torch.ones(pad.shape[0], result.la_len, dtype=torch.bool, device=device)], dim=1)
        return (hidden, pad, torch.zeros_like(pad, dtype=torch.bool), result)

class A2WLAPolicy(nn.Module):
    config_class = A2WLAConfig
    name = 'pi05'

    def _preprocess_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        images = []
        device = next(self.parameters()).device
        for key in self.config.image_features:
            image = batch[key].to(device=device, dtype=torch.float32)
            batch_size, frames, channels, height, width = image.shape
            if frames != 2:
                raise ValueError('RoboTwin requires exactly two image frames')
            image = image.permute(0, 1, 3, 4, 2).reshape(batch_size * frames, height, width, channels)
            if (height, width) != self.config.image_resolution:
                image = resize_with_pad_torch(image, *self.config.image_resolution)
            image = (image * 2.0 - 1.0).permute(0, 3, 1, 2)
            images.append(image.reshape(batch_size, frames, *image.shape[1:]))
        return images

    def _prepare_state_history_for_prefix(self, batch: dict[str, Tensor], device: torch.device, dtype: torch.dtype) -> Tensor | None:
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim == 3:
            state = state[:, -1:, :]
        return state

    def _prepare_state_history_for_flow_source(self, batch, device, dtype):
        state = batch[OBS_STATE].to(device=device, dtype=dtype)
        if state.ndim == 1:
            return state[None, None, :]
        if state.ndim == 2:
            return state[:, None, :]
        return state[:, -1:, :]

    def _prepare_image_pad_masks(self, batch):
        return [None] * len(self.config.image_features)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        images = self._preprocess_images(batch)
        img_pad_masks = self._prepare_image_pad_masks(batch)
        device = next(self.parameters()).device
        raw_texts = batch['task']
        state_history = self._prepare_state_history_for_prefix(batch, device, torch.float32)
        flow_source_state_history = self._prepare_state_history_for_flow_source(batch, device, torch.float32)
        actions = self.model.sample_actions(images, img_pad_masks, raw_texts=raw_texts, state_history=state_history, flow_source_state_history=flow_source_state_history)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    def __init__(self, config):
        super().__init__()
        config.validate_features()
        self.config = config
        self.model = A2WLAModel(config)
        self.to(config.device)

    @classmethod
    def from_pretrained(cls, checkpoint, *, config):
        from a2meas.robotwin.weights import load_inference_weights
        from transformers.modeling_utils import no_init_weights
        with torch.random.fork_rng(devices=[]), no_init_weights():
            model = cls(config)
        load_inference_weights(model, Path(checkpoint) / 'model.safetensors')
        return model.eval().requires_grad_(False)
