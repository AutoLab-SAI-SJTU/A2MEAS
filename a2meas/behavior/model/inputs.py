#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


"""RynnBrain image transforms, multimodal inputs and latent-action query features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from a2meas.models.action_expert import (
    RynnBrainInjectForwardResult, extend_visual_pos_masks, action_prefix_hidden_from_inject_result,
    build_inject_sequence,
)


_RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES = 4096

_RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS = 1 << 20


def build_base_inputs_embeds_from_processor_inputs(
    rynnbrain_model: nn.Module, rynnbrain_inputs: dict[str, Tensor]
) -> tuple[Tensor, Tensor | None, list[Tensor] | None]:
    """Build VLM ``inputs_embeds`` (vision scattered) without running the language model."""
    model = rynnbrain_model.model
    input_ids = rynnbrain_inputs["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = None
    video_mask = None
    visual_pos_masks = None
    deepstack_visual_embeds = None
    pixel_values = rynnbrain_inputs.get("pixel_values")
    if pixel_values is not None:
        image_grid_thw = rynnbrain_inputs.get("image_grid_thw")
        image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    pixel_values_videos = rynnbrain_inputs.get("pixel_values_videos")
    if pixel_values_videos is not None:
        video_grid_thw = rynnbrain_inputs.get("video_grid_thw")
        video_embeds, deepstack_video_embeds = model.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            video_frame_age_seconds=rynnbrain_inputs.get("video_frame_age_seconds"),
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=True):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(
                img_embed.device
            )
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = deepstack_video_embeds
    return (inputs_embeds, visual_pos_masks, deepstack_visual_embeds)


def compute_base_mrope_position_ids(
    rynnbrain_model: nn.Module, rynnbrain_inputs: dict[str, Tensor]
) -> Tensor:
    """Compute mRoPE position ids for the base (processor) sequence."""
    model = rynnbrain_model.model
    attention_mask = rynnbrain_inputs.get("attention_mask")
    position_ids, _ = model.get_rope_index(
        rynnbrain_inputs["input_ids"],
        rynnbrain_inputs.get("image_grid_thw"),
        rynnbrain_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    return position_ids


def extend_mrope_position_ids_for_aux_tokens(
    base_position_ids: Tensor, base_pad_masks: Tensor, num_aux_tokens: int
) -> Tensor:
    """Append sequential text-style mRoPE positions after the base sequence."""
    if num_aux_tokens <= 0:
        return base_position_ids
    if base_position_ids.ndim != 3:
        raise ValueError(f"Expected base_position_ids [3, B, L], got {tuple(base_position_ids.shape)}")
    batch_size = base_position_ids.shape[1]
    device = base_position_ids.device
    dtype = base_position_ids.dtype
    valid = base_pad_masks.bool()
    aux_chunks: list[Tensor] = []
    for batch_idx in range(batch_size):
        if valid[batch_idx].any():
            pos_valid = base_position_ids[:, batch_idx, valid[batch_idx]]
            start = int(pos_valid.max().item()) + 1
        else:
            start = 0
        aux_pos = torch.arange(num_aux_tokens, device=device, dtype=dtype).view(1, -1).expand(3, -1) + start
        aux_chunks.append(aux_pos)
    aux_position_ids = torch.stack(aux_chunks, dim=1)
    return torch.cat([base_position_ids, aux_position_ids], dim=-1)


def forward_rynnbrain_inject_sequence(
    rynnbrain_model: nn.Module,
    *,
    inputs_embeds: Tensor,
    pad_masks: Tensor,
    base_position_ids: Tensor,
    visual_pos_masks: Tensor | None,
    deepstack_visual_embeds: list[Tensor] | None,
) -> Tensor:
    """Full RynnBrain language-model forward; returns pre-norm token hidden states [B, L, H]."""
    model = rynnbrain_model.model
    (batch_size, seq_len, _) = inputs_embeds.shape
    device = inputs_embeds.device
    base_len = base_position_ids.shape[-1]
    num_aux = seq_len - base_len
    position_ids = extend_mrope_position_ids_for_aux_tokens(
        base_position_ids, pad_masks[:, :base_len], num_aux
    )
    visual_pos_masks = extend_visual_pos_masks(visual_pos_masks, batch_size, num_aux, device=device)
    attention_mask = pad_masks.to(dtype=torch.long)
    outputs = model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=False,
        output_pre_norm_last_hidden_state_only=True,
    )
    if not outputs.hidden_states:
        raise RuntimeError("Expected pre-norm hidden states from RynnBrain inject forward")
    return outputs.hidden_states[-1]


def run_rynnbrain_inject_full_forward(
    rynnbrain_model, rynnbrain_inputs, base_pad_masks, *, la_query=None
):
    base_embs, visual_masks, deepstack = build_base_inputs_embeds_from_processor_inputs(
        rynnbrain_model, rynnbrain_inputs
    )
    stage_base_len = int(base_embs.shape[1])
    inputs_embeds, pad_masks, base_len, la_len = build_inject_sequence(
        base_embs, base_pad_masks, la_query=la_query
    )
    position_ids = compute_base_mrope_position_ids(rynnbrain_model, rynnbrain_inputs)
    hidden = forward_rynnbrain_inject_sequence(
        rynnbrain_model,
        inputs_embeds=inputs_embeds,
        pad_masks=pad_masks,
        base_position_ids=position_ids,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    return RynnBrainInjectForwardResult(hidden, base_len, stage_base_len, la_len)


class RynnBrainInputMixin:

    def _rynnbrain_hidden_size(self) -> int:
        return int(self.paligemma_with_expert.rynnbrain.config.text_config.hidden_size)

    def _prepare_rynnbrain_inputs(
        self,
        images,
        img_pad_masks=None,
        raw_texts: list[str] | None = None,
        short_term_memory_frame_age_seconds: Tensor | None = None,
    ) -> tuple[dict, Tensor]:
        """Pack RynnBrain visual history, task text and optional model query tokens."""
        num_frames = images[0].shape[1]
        device = images[0].device
        import torch._dynamo as torch_dynamo

        @torch_dynamo.disable
        def process_inputs(
            images_inner, img_pad_masks_inner, raw_texts_inner, frame_age_seconds_inner
        ):
            bsize = images_inner[0].shape[0]
            cpu_img_pad_masks = [pad.detach().to(device="cpu") for pad in img_pad_masks_inner]
            if raw_texts_inner is None:
                raise ValueError("RynnBrain inference requires canonical raw task text")
            raw_texts_local = list(raw_texts_inner)
            if len(raw_texts_local) == 1 and bsize > 1:
                raw_texts_local = raw_texts_local * bsize
            if len(raw_texts_local) != bsize:
                raise ValueError("Raw task text must match the observation batch")
            conversations = []
            batch_videos = []
            for b_idx in range(bsize):
                content = []
                sample_videos = []
                for camera, camera_pad in zip(images_inner, cpu_img_pad_masks, strict=True):
                    prepared = [
                        None if bool(camera_pad[b_idx, index]) else camera[b_idx, index].detach().contiguous()
                        for index in range(num_frames)
                    ]
                    last_valid = next(frame for frame in prepared if frame is not None)
                    frames = []
                    for frame in prepared:
                        if frame is None:
                            frames.append(last_valid)
                        else:
                            last_valid = frame
                            frames.append(frame)
                    video = torch.stack(frames, dim=0)
                    content.append({"type": "video", "video": video})
                    sample_videos.append(video)
                content.append({"type": "text", "text": raw_texts_local[b_idx]})
                conversations.append([{"role": "user", "content": content}])
                batch_videos.append(sample_videos)
            processor = self.paligemma_with_expert.rynnbrain_processor
            prompts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            video_inputs = processor.video_processor(
                videos=batch_videos,
                do_sample_frames=False,
                do_resize=False,
                do_convert_rgb=False,
                do_rescale=False,
                do_normalize=False,
                return_tensors="pt",
            )
            merge_length = processor.video_processor.merge_size**2
            video_grid_thw = video_inputs["video_grid_thw"]
            prompts_list = prompts
            grid_keys = []
            grid_offset = 0
            for sample_videos in batch_videos:
                next_grid_offset = grid_offset + len(sample_videos)
                sample_grid = video_grid_thw[grid_offset:next_grid_offset]
                grid_keys.append(
                    tuple((tuple((int(value) for value in row)) for row in sample_grid.tolist()))
                )
                grid_offset = next_grid_offset
            if grid_offset != len(video_grid_thw):
                raise ValueError(
                    f"RynnBrain video grid count mismatch: consumed={grid_offset}, available={len(video_grid_thw)}"
                )
            cache_enabled = processor.tokenizer.padding_side == "right"
            token_cache = None
            cache_keys = []
            cached_rows = []
            if cache_enabled:
                token_cache = getattr(self, "_rynnbrain_video_token_cache", None)
                if token_cache is None:
                    token_cache = {}
                    self._rynnbrain_video_token_cache = token_cache
                cache_keys = [
                    (prompt, grid_key) for prompt, grid_key in zip(prompts_list, grid_keys, strict=True)
                ]
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
                        video_placeholder = (
                            processor.vision_start_token
                            + "<|placeholder|>" * frame_seqlen
                            + processor.vision_end_token
                        )
                        wrapped_video_token = f"{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}"
                        replace_token = (
                            wrapped_video_token
                            if wrapped_video_token in processed_prompts[prompt_index]
                            else processor.video_token
                        )
                        processed_prompts[prompt_index] = processed_prompts[prompt_index].replace(
                            replace_token, video_placeholder, 1
                        )
                        grid_index += 1
                    processed_prompts[prompt_index] = processed_prompts[prompt_index].replace(
                        "<|placeholder|>", processor.video_token
                    )
                if grid_index != len(video_grid_thw):
                    raise ValueError(
                        f"RynnBrain prompt/grid mismatch: consumed={grid_index}, available={len(video_grid_thw)}"
                    )
                tokenizer_kwargs = {
                    "padding": True,
                    "return_tensors": "pt",
                    "return_token_type_ids": False,
                }
                bos_token = processor.tokenizer.bos_token
                single_prompt = processed_prompts[0] if processed_prompts else None
                if (
                    bos_token is not None
                    and isinstance(single_prompt, str)
                    and single_prompt.startswith(bos_token)
                ):
                    tokenizer_kwargs["add_special_tokens"] = False
                text_inputs = processor.tokenizer(processed_prompts, **tokenizer_kwargs)
                processor._check_special_mm_tokens(processed_prompts, text_inputs, modalities=["video"])
                rynnbrain_inputs = {**text_inputs, **video_inputs}
                if cache_enabled:
                    staged_rows = {}
                    for row_index, cache_key in enumerate(cache_keys):
                        valid_mask = text_inputs["attention_mask"][row_index].bool()
                        staged_rows[cache_key] = (
                            text_inputs["input_ids"][row_index, valid_mask].contiguous()
                        )
                    staged_tokens = sum((row.numel() for row in staged_rows.values()))
                    if (
                        len(staged_rows) <= _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES
                        and staged_tokens <= _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS
                    ):
                        cached_tokens = int(getattr(self, "_rynnbrain_video_token_cache_tokens", 0))
                        replaced_tokens = sum(
                            (token_cache[key].numel() for key in staged_rows if key in token_cache)
                        )
                        projected_tokens = cached_tokens - replaced_tokens + staged_tokens
                        projected_entries = len(set(token_cache).union(staged_rows))
                        if (
                            projected_entries > _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_ENTRIES
                            or projected_tokens > _RYNNBRAIN_VIDEO_TOKEN_CACHE_MAX_TOKENS
                        ):
                            token_cache.clear()
                            projected_tokens = staged_tokens
                        token_cache.update(staged_rows)
                        self._rynnbrain_video_token_cache_tokens = projected_tokens
            else:
                token_rows = cached_rows
                padded_length = max(row.shape[0] for row in token_rows)
                pad_token_id = processor.tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("RynnBrain tokenizer requires a pad token for batched video inputs")
                input_ids = torch.full((bsize, padded_length), pad_token_id, dtype=token_rows[0].dtype)
                attention_mask = torch.zeros_like(input_ids)
                for row_index, token_row in enumerate(token_rows):
                    valid_length = token_row.shape[0]
                    input_ids[row_index, :valid_length] = token_row
                    attention_mask[row_index, :valid_length] = 1
                rynnbrain_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    **video_inputs,
                }
            frame_age_enabled = bool(self.config.short_term_memory_frame_age_embedding_enable)
            if frame_age_enabled:
                if frame_age_seconds_inner is None:
                    raise ValueError(
                        "STM frame-age embedding is enabled but frame-age metadata was not passed to embed_prefix"
                    )
                if frame_age_seconds_inner.ndim != 2 or tuple(frame_age_seconds_inner.shape) != (
                    bsize,
                    num_frames,
                ):
                    raise ValueError(
                        f"STM frame-age metadata must have shape [{bsize}, {num_frames}], got {tuple(frame_age_seconds_inner.shape)}"
                    )
                packed_age_rows = [
                    frame_age_seconds_inner[b_idx]
                    for (b_idx, sample_videos) in enumerate(batch_videos)
                    for _ in sample_videos
                ]
                if not packed_age_rows:
                    raise ValueError(
                        "STM frame-age embedding requires at least one valid packed camera video"
                    )
                packed_ages = torch.stack(packed_age_rows, dim=0).to(dtype=torch.float32)
                video_grid_thw = rynnbrain_inputs.get("video_grid_thw")
                if video_grid_thw is None or int(video_grid_thw.shape[0]) != int(packed_ages.shape[0]):
                    raise ValueError(
                        f"Packed STM frame ages do not align with video_grid_thw: ages={tuple(packed_ages.shape)} grid={(None if video_grid_thw is None else tuple(video_grid_thw.shape))}"
                    )
                rynnbrain_inputs["video_frame_age_seconds"] = packed_ages
            cpu_metadata_keys = {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"}
            return {
                k: (
                    (v.contiguous() if k in cpu_metadata_keys else v.contiguous().to(device))
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for (k, v) in rynnbrain_inputs.items()
            }

        rynnbrain_inputs = process_inputs(
            images, img_pad_masks, raw_texts, short_term_memory_frame_age_seconds
        )
        pad = rynnbrain_inputs["attention_mask"].bool()
        return (rynnbrain_inputs, pad)


    def _run_feature_inject(self, inputs, base_pad, *, la_query=None):
        rb = self.paligemma_with_expert.rynnbrain
        result = run_rynnbrain_inject_full_forward(
            rb, inputs, base_pad, la_query=la_query
        )
        hidden = action_prefix_hidden_from_inject_result(result)
        pad = base_pad
        if result.la_len:
            pad = torch.cat(
                [pad, torch.ones(pad.shape[0], result.la_len, dtype=torch.bool, device=pad.device)], dim=1
            )
        if pad.shape[1] != hidden.shape[1]:
            raise RuntimeError("Action prefix hidden and mask lengths differ")
        return (hidden, pad, torch.zeros_like(pad, dtype=torch.bool), result)
