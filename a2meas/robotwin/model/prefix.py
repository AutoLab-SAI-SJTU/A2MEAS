# Derived from LeRobot / OpenPI, Apache-2.0. See README.md.
"""RynnBrain visual features and LaWAM query injection.

The causal sequence is [vision/text (including image history) | LaWAM queries].
"""

from __future__ import annotations
import os
import torch
from torch import Tensor, nn
from a2meas.models.action_expert import (
    RynnBrainInjectForwardResult, extend_visual_pos_masks, build_inject_sequence,
)


def _qwen_metadata_fastpath_enabled() -> bool:
    """Reuse the processor's exact positional metadata by default."""
    return os.environ.get(
        "LEROBOT_QWEN_METADATA_FASTPATH", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


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
        image_embeds, deepstack_image_embeds = model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    pixel_values_videos = rynnbrain_inputs.get("pixel_values_videos")
    if pixel_values_videos is not None:
        video_grid_thw = rynnbrain_inputs.get("video_grid_thw")
        video_embeds, deepstack_video_embeds = (
            model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
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
        for img_embed, vid_embed in zip(
            deepstack_image_embeds, deepstack_video_embeds, strict=True
        ):
            embed_joint = img_embed.new_zeros(
                visual_pos_masks.sum(), img_embed.shape[-1]
            ).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = deepstack_video_embeds
    return (
        inputs_embeds,
        visual_pos_masks,
        deepstack_visual_embeds,
    )


def compute_base_mrope_position_ids(
    rynnbrain_model: nn.Module, rynnbrain_inputs: dict[str, Tensor]
) -> Tensor:
    """Compute mRoPE position ids for the base (processor) sequence."""
    cached_positions = rynnbrain_inputs.get("position_ids")
    input_ids = rynnbrain_inputs["input_ids"]
    if (
        _qwen_metadata_fastpath_enabled()
        and isinstance(cached_positions, Tensor)
        and (cached_positions.shape == (3, *input_ids.shape))
    ):
        return cached_positions.to(device=input_ids.device)
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
        raise ValueError(
            f"Expected base_position_ids [3, B, L], got {tuple(base_position_ids.shape)}"
        )
    batch_size = base_position_ids.shape[1]
    device = base_position_ids.device
    dtype = base_position_ids.dtype
    valid = base_pad_masks.bool()
    if _qwen_metadata_fastpath_enabled() and dtype in (torch.int32, torch.int64):
        if base_position_ids.shape[-1] == 0:
            starts = torch.zeros(batch_size, dtype=dtype, device=device)
        else:
            masked = base_position_ids.masked_fill(
                ~valid.unsqueeze(0), torch.iinfo(dtype).min
            )
            starts = torch.where(valid.any(dim=-1), masked.amax(dim=(0, 2)) + 1, 0)
        offsets = torch.arange(num_aux_tokens, device=device, dtype=dtype)
        aux_position_ids = (starts[None, :, None] + offsets[None, None, :]).expand(
            3, -1, -1
        )
        return torch.cat([base_position_ids, aux_position_ids], dim=-1)
    aux_chunks: list[Tensor] = []
    for batch_idx in range(batch_size):
        if valid[batch_idx].any():
            pos_valid = base_position_ids[:, batch_idx, valid[batch_idx]]
            start = int(pos_valid.max().item()) + 1
        else:
            start = 0
        aux_pos = (
            torch.arange(num_aux_tokens, device=device, dtype=dtype)
            .view(1, -1)
            .expand(3, -1)
            + start
        )
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
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    base_len = base_position_ids.shape[-1]
    num_aux = seq_len - base_len
    position_ids = extend_mrope_position_ids_for_aux_tokens(
        base_position_ids, pad_masks[:, :base_len], num_aux
    )
    visual_pos_masks = extend_visual_pos_masks(
        visual_pos_masks, batch_size, num_aux, device=device
    )
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
    return outputs.hidden_states[-1]


def run_rynnbrain_inject_full_forward(
    rynnbrain_model: nn.Module,
    rynnbrain_inputs: dict[str, Tensor],
    base_pad_masks: Tensor,
    *,
    la_query: Tensor | None = None,
) -> RynnBrainInjectForwardResult:
    """Run the visual and LaWAM queries through the language model once."""
    (
        base_inputs_embeds,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = build_base_inputs_embeds_from_processor_inputs(
        rynnbrain_model, rynnbrain_inputs
    )
    stage_base_len = int(base_inputs_embeds.shape[1])
    packed = build_inject_sequence(
        base_inputs_embeds, base_pad_masks, la_query=la_query
    )
    inputs_embeds, pad_masks, action_base_len, la_len = packed
    base_position_ids = compute_base_mrope_position_ids(
        rynnbrain_model, rynnbrain_inputs
    )
    full_hidden_pre_norm = forward_rynnbrain_inject_sequence(
        rynnbrain_model,
        inputs_embeds=inputs_embeds,
        pad_masks=pad_masks,
        base_position_ids=base_position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    return RynnBrainInjectForwardResult(
        full_hidden_pre_norm=full_hidden_pre_norm,
        base_len=action_base_len,
        stage_base_len=stage_base_len,
        la_len=la_len,
    )
