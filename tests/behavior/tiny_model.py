"""Small CPU inference model with real Gemma attention, AdaRMS and prefix caches."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from a2meas.models.gemma import GemmaConfig, GemmaExpert, GemmaRMSNorm

from a2meas.schema import FeatureType, PolicyFeature
from a2meas.behavior.model import policy as modeling
from a2meas.behavior.model.policy import A2WLAPolicy
from a2meas.behavior.model.config import A2WLAConfig


class TinyBackbone(nn.Module):
    """Small stand-in backbone with real Gemma layers, AdaRMS and DynamicCache."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = GemmaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            use_adarms=True,
            adarms_cond_dim=16,
            attn_implementation="eager",
        )
        self.gemma_expert = GemmaExpert(cfg).float()
        self.gemma_expert.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rynnbrain = SimpleNamespace(
            dtype=torch.float32, config=SimpleNamespace(vision_config=SimpleNamespace())
        )
        self.paligemma = SimpleNamespace(language_model=self.gemma_expert.model)
        self.text_calls = 0
        # Use nonzero modulation weights to exercise AdaRMS.
        for module in self.gemma_expert.modules():
            if isinstance(module, GemmaRMSNorm) and module.dense is not None:
                nn.init.normal_(module.dense.weight, std=0.05)
        self.conditions = []

    def embed_language_tokens(self, tokens):
        return self.gemma_expert.model.embed_tokens(tokens)

    def embed_prefix(self, images, *, raw_texts, **kwargs):
        task_tokens = {"put cup": [1, 2], "open drawer": [2, 3]}
        tokens = torch.tensor(
            [task_tokens[text] for text in raw_texts], device=self.gemma_expert.model.embed_tokens.weight.device
        )
        prefix = self.embed_language_tokens(tokens)
        mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
        return prefix, mask, torch.zeros_like(mask)

    def get_text_prefix_embs(self, raw_texts, device):
        self.text_calls += 1
        ids = torch.tensor([[sum(map(ord, text)) % 30 + 1, len(text) % 30 + 1] for text in raw_texts])
        return self.embed_language_tokens(ids), torch.ones_like(ids, dtype=torch.bool)

    def forward(
        self,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        adarms_cond=None,
        use_cache=False,
        **kwargs,
    ):
        prefix, suffix = inputs_embeds
        if prefix is not None and suffix is not None:
            # Match RynnBrain's existing uncached path: the entire concatenated
            # expert sequence uses the suffix condition, including prefix tokens.
            self.conditions.append(adarms_cond[1].detach().clone())
            output = self.gemma_expert.model(
                inputs_embeds=torch.cat([prefix, suffix], dim=1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                adarms_cond=adarms_cond[1],
            )
            hidden = output.last_hidden_state
            return [hidden[:, : prefix.shape[1]], hidden[:, prefix.shape[1] :]], None
        cache = past_key_values
        prefix_out = None
        if prefix is not None:
            n = prefix.shape[1]
            output = self.gemma_expert.model(
                inputs_embeds=prefix,
                attention_mask=attention_mask[:, :, :n, :n],
                position_ids=position_ids[:, :n],
                use_cache=True,
                adarms_cond=None,
            )
            cache, prefix_out = output.past_key_values, output.last_hidden_state
        if suffix is None:
            return [prefix_out, None], cache
        self.conditions.append(adarms_cond[1].detach().clone())
        output = self.gemma_expert.model(
            inputs_embeds=suffix,
            attention_mask=attention_mask[:, :, -suffix.shape[1] :, :],
            position_ids=position_ids[:, -suffix.shape[1] :],
            past_key_values=cache,
            use_cache=True,
            adarms_cond=adarms_cond[1],
        )
        return [prefix_out, output.last_hidden_state], output.past_key_values


@contextmanager
def tiny_construction():
    tiny_cfg = modeling.ActionExpertConfig(16, 2, 32, 2, 1, 8)
    with (
        patch.object(modeling, "ActionExpertConfig", return_value=tiny_cfg),
        patch.object(modeling, "A2WLABackbone", TinyBackbone),
    ):
        yield


def config(enabled=True, **kwargs):
    cfg = A2WLAConfig(
        device="cpu",
        dtype="float32",
        enable_global_task_conditioning=enabled,
        global_task_num_tasks=4,
        global_task_condition_hidden_dim=8,
        chunk_size=3,
        max_action_dim=4,
        max_state_dim=4,
        global_task_text_to_id={"put cup": 2, "open drawer": 0},
        **kwargs,
    )
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    return cfg


def make_model(enabled=True, **kwargs):
    with tiny_construction():
        model = modeling.A2WLAModel(config(enabled, **kwargs))

    model.embed_prefix = model.paligemma_with_expert.embed_prefix
    return model


def inputs():
    return dict(
        images=[torch.zeros(2, 6, 3, 1, 1)],
        img_pad_masks=[],
        raw_texts=["put cup", "open drawer"],
        global_task_inputs={"texts": ["put cup", "open drawer"], "ids": torch.tensor([2, 0])},
    )


def make_qwen_model(attention="eager"):
    from a2meas.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from a2meas.models.temporal_memory import Qwen3VLTemporalMemory
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    torch.manual_seed(57)
    cfg = Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            pad_token_id=0,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        ).to_dict(),
        vision_config=Qwen3VLVisionConfig(
            depth=6,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=8,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            deepstack_visual_indexes=[1, 3],
            mem_enable_short_term_memory=True,
            mem_frame_age_embedding_enable=True,
            mem_frame_age_embedding_dim=8,
        ).to_dict(),
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
        mem_compress_video_tokens=True,
    )
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    model.set_attn_implementation(attention)
    vision = model.model.visual
    vision.temporal_memory = Qwen3VLTemporalMemory(16).eval()
    # Exercise the learned memory/age branches with nonzero checkpoint-like parameters.
    torch.nn.init.normal_(vision.temporal_memory.out_proj.weight, std=0.03)
    torch.nn.init.normal_(vision.mem_frame_age_encoder.out_proj.weight, std=0.03)
    return model


def make_behavior_policy(enabled):
    torch.manual_seed(63)
    cfg = config(
        enabled,
        enable_state_in_prefix_tokens=True,
        enable_rynnbrain_prefix_kv_cache=True,
        enable_split_action_heads=True,
        enable_behavior_b1k_semantic_action_heads=True,
        gripper_action_indices=[14, 22],
    )
    cfg.max_action_dim = 32
    cfg.max_state_dim = 32
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(23,))}
    cfg.input_features = {
        f"observation.rgb.{key}_link_camera_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for key in ["zed", "left_realsense", "right_realsense"]
    }
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(23,))
    with tiny_construction():
        policy = A2WLAPolicy(cfg).eval()

    policy.model.embed_prefix = policy.model.paligemma_with_expert.embed_prefix
    return policy


def behavior_batch(policy):
    batch = {
        k: torch.linspace(0, 1, 6 * 3 * 224 * 224).reshape(1, 6, 3, 224, 224).expand(2, -1, -1, -1, -1)
        for k, f in policy.config.input_features.items()
        if f.type == FeatureType.VISUAL
    }
    batch.update(
        {
            "observation.state": torch.linspace(-1, 1, 2 * 6 * 23).reshape(2, 6, 23),
            "task": ["put cup", "open drawer"],
            "global_task_text": ["put cup", "open drawer"],
            "global_task_id": torch.tensor([2, 0]),
        }
    )
    return batch
