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


from dataclasses import dataclass, field


from a2meas.behavior.schema import PreTrainedConfig


DEFAULT_IMAGE_SIZE = 224


def _resolve_action_head_indices(
    configured_gripper_indices: list[int] | None, action_dim: int
) -> tuple[list[int], list[int]]:
    gripper_indices = [int(idx) for idx in (configured_gripper_indices or [])]
    seen: set[int] = set()
    for idx in gripper_indices:
        if idx < 0 or idx >= int(action_dim):
            raise ValueError(
                f"gripper_action_indices contains out-of-range index {idx} for action_dim={action_dim}"
            )
        if idx in seen:
            raise ValueError(f"gripper_action_indices contains duplicate index {idx}")
        seen.add(idx)
    joint_indices = [idx for idx in range(int(action_dim)) if idx not in seen]
    return (joint_indices, gripper_indices)


BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("base", (0, 1, 2)),
    ("torso", (3, 4, 5, 6)),
    ("arm", (7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21)),
    ("gripper", (14, 22)),
)


def _resolve_behavior_b1k_semantic_action_head_groups(
    action_dim: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if int(action_dim) != 23:
        raise ValueError(
            f"enable_behavior_b1k_semantic_action_heads=True requires Behavior 23D action layout, got action_dim={action_dim}"
        )
    return BEHAVIOR_B1K_SEMANTIC_ACTION_HEAD_GROUPS


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class A2WLAConfig(PreTrainedConfig):
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"
    rynnbrain_path: str = "assets/rynnbrain"
    enable_lawam_stage2: bool = False
    lawam_config_path: str = "shared/world_model.json"
    lawam_primary_camera_key: str = "observation.rgb.zed_link_camera_0"
    lawam_multi_view_camera_keys: list[str] = field(default_factory=list)
    lawam_image_resolution: int = 256
    lawam_num_action_queries: int = 8
    lawam_num_prefix_tokens: int = 16
    lawam_prefix_view_mode: str = "primary_only"
    lawam_prefix_token_budget_mode: str = "per_view"
    lawam_visual_prefix_mode: str = "compressed"
    lawam_inject_current_features: bool = True
    lawam_decoder_bf16_autocast: bool = False
    lawam_prefix_bf16_autocast: bool = False
    lawam_mha_need_weights: bool = True
    lawam_inject_mode: str = "prefix_append"
    lawam_alternate_first: str = "vlm"
    lawam_alternate_period: int = 1
    enable_trajectory_2d_aux: bool = False
    trajectory_2d_num_points: int = 10
    enable_rynnbrain_prefix_kv_cache: bool = False
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 0.004
    max_period: float = 4.0
    suffix_attention_mask: str = "causal"
    enable_temporal_memory: bool = False
    temporal_memory_pool_size: int = 4
    enable_short_term_memory: bool = False
    short_term_memory_inference_num_frames: int = 6
    short_term_memory_frame_age_embedding_enable: bool = False
    short_term_memory_frame_age_max_seconds: float = 300.0
    short_term_memory_frame_age_embedding_dim: int = 64
    short_term_memory_frame_age_fps: float = 30.0
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    enable_state_in_prefix_tokens: bool = False
    enable_global_task_conditioning: bool = False
    global_task_conditioning_mode: str = "language_task_id"
    global_task_conditioning_injection: str = "adarms"
    global_task_condition_hidden_dim: int = 256
    global_task_num_tasks: int = 100
    global_task_text_to_id: dict[str, int] | None = None
    enable_speed_meta: bool = False
    speed_meta_map: dict[str, int] | None = None
    enable_gripper_first_close_meta: bool = False
    enable_split_action_heads: bool = False
    enable_behavior_b1k_semantic_action_heads: bool = False
    gripper_action_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Unsupported model dtype: {self.dtype}")
        if self.action_expert_variant != "gemma_300m":
            raise ValueError(f"Unsupported action expert: {self.action_expert_variant}")
        if (
            self.chunk_size < 1
            or self.num_inference_steps < 1
            or min(self.max_state_dim, self.max_action_dim) < 1
        ):
            raise ValueError("Chunk size, diffusion steps and state/action dimensions must be positive")
        if len(self.image_resolution) != 2 or min(self.image_resolution) < 1:
            raise ValueError("image_resolution must contain two positive dimensions")
        if self.suffix_attention_mask not in {"causal", "block"}:
            raise ValueError("Unsupported suffix_attention_mask")
        if self.global_task_conditioning_mode not in {"language", "task_id", "language_task_id"}:
            raise ValueError("Unsupported global task conditioning mode")
        if self.global_task_conditioning_injection != "adarms":
            raise ValueError("BEHAVIOR inference uses AdaRMS task conditioning")
        if min(self.global_task_condition_hidden_dim, self.global_task_num_tasks) < 1:
            raise ValueError("Global task dimensions must be positive")
        if self.enable_global_task_conditioning and self.global_task_text_to_id is not None:
            from .conditioning import validate_task_mapping

            self.global_task_text_to_id = validate_task_mapping(
                self.global_task_text_to_id, self.global_task_num_tasks
            )
        if self.enable_temporal_memory:
            if self.temporal_memory_pool_size < 1 or self.short_term_memory_inference_num_frames != 6:
                raise ValueError("Temporal memory requires a positive pool size and six inference frames")
            self.enable_short_term_memory = True
        if self.short_term_memory_inference_num_frames < 1:
            raise ValueError("Inference frame count must be positive")
        if (
            self.short_term_memory_frame_age_embedding_dim < 1
            or min(self.short_term_memory_frame_age_max_seconds, self.short_term_memory_frame_age_fps) <= 0
        ):
            raise ValueError("Frame-age dimensions, time scale and FPS must be positive")
        if self.short_term_memory_frame_age_embedding_enable and (not self.enable_short_term_memory):
            raise ValueError("Frame-age embedding requires short-term memory")
        if self.enable_trajectory_2d_aux and (not self.enable_lawam_stage2):
            raise ValueError("Trajectory query parameters require LaWAM")
        if self.enable_lawam_stage2:
            if not self.lawam_config_path:
                raise ValueError("LaWAM requires its architecture config")
            if self.lawam_num_action_queries < 1 or self.lawam_num_prefix_tokens < 1:
                raise ValueError("LaWAM query and prefix counts must be positive")
            if self.lawam_visual_prefix_mode not in {"compressed", "none", "raw"}:
                raise ValueError("Unsupported LaWAM visual prefix mode")
            if self.lawam_prefix_view_mode not in {"primary_only", "all_views"}:
                raise ValueError("Unsupported LaWAM view mode")
            if self.lawam_prefix_token_budget_mode not in {"per_view", "shared_total"}:
                raise ValueError("Unsupported LaWAM token budget mode")
            if self.lawam_prefix_token_budget_mode == "shared_total" and (
                self.lawam_prefix_view_mode != "all_views" or self.lawam_visual_prefix_mode != "compressed"
            ):
                raise ValueError("Shared LaWAM token budget requires compressed multi-view prefixes")
            if (
                self.lawam_inject_mode not in {"prefix_append", "alternate_kv"}
                or self.lawam_alternate_first not in {"vlm", "lawam"}
                or self.lawam_alternate_period < 1
            ):
                raise ValueError("Invalid LaWAM KV routing")
            if self.lawam_inject_mode == "alternate_kv" and self.lawam_visual_prefix_mode == "none":
                raise ValueError("Alternate KV requires a LaWAM visual prefix")
            if self.lawam_primary_camera_key in self.lawam_multi_view_camera_keys:
                raise ValueError("LaWAM auxiliary cameras must exclude its primary camera")

    def validate_features(self):
        from a2meas.behavior.observations import POLICY_CAMERA_KEYS

        if set(self.image_features) != set(POLICY_CAMERA_KEYS):
            raise ValueError("BEHAVIOR checkpoints require the three canonical RGB camera features")
        if "observation.state" not in self.input_features or "action" not in self.output_features:
            raise ValueError("Checkpoint features must include observation.state and action")
        action_dim = int(self.output_features["action"].shape[0])
        if self.enable_split_action_heads:
            joints, grippers = _resolve_action_head_indices(self.gripper_action_indices, action_dim)
            if not joints or not grippers:
                raise ValueError("Split action heads require joint and gripper dimensions")
        if self.enable_behavior_b1k_semantic_action_heads:
            _resolve_behavior_b1k_semantic_action_head_groups(action_dim)
        if self.enable_trajectory_2d_aux and (
            action_dim != 23 or "observation.rgb.zed_link_camera_0" not in self.image_features
        ):
            raise ValueError("Trajectory query parameters require the BEHAVIOR 23D layout and head camera")
