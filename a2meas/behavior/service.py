"""Checkpoint loading and batched model inference, shared by BEHAVIOR routes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from . import CONFIG_ROOT


@dataclass
class ModelArgs:
    resources: Path
    model_dir: Path
    checkpoint_path: Path
    num_inference_steps: int
    device: str = "cuda"


class ModelService:

    def __init__(self, args: ModelArgs):
        import torch
        from a2meas.behavior.schema import PreTrainedConfig
        from a2meas.behavior.model.policy import A2WLAPolicy
        import a2meas.behavior.model.prompts  # noqa: F401 -- Register saved prompt processors.

        policy_cfg = PreTrainedConfig.from_pretrained(args.model_dir, device=args.device)
        policy_cfg.rynnbrain_path = str(args.resources / policy_cfg.rynnbrain_path)
        policy_cfg.lawam_config_path = str(CONFIG_ROOT.parent / policy_cfg.lawam_config_path)
        policy_cfg.num_inference_steps = int(args.num_inference_steps)
        self.policy = A2WLAPolicy.from_pretrained(
            args.model_dir, checkpoint_path=args.checkpoint_path, config=policy_cfg
        )
        pre_overrides: dict[str, Any] = {}
        pre_overrides["pi05_prepare_state_tokenizer_processor_step"] = {
            "enable_speed_meta": policy_cfg.enable_speed_meta,
            "enable_gripper_first_close_meta": policy_cfg.enable_gripper_first_close_meta,
        }
        pre_overrides["device_processor"] = {"device": args.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            pretrained_path=str(args.model_dir), preprocessor_overrides=pre_overrides
        )
        self.device = torch.device(args.device)
        self.amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[policy_cfg.dtype]

    def predict(self, sample) -> np.ndarray:
        return self.predict_flow(self.prepare_flow(sample))

    def prepare_flow(self, sample):
        import torch

        with torch.no_grad():
            model_in = self.preprocessor(sample)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                return self.policy.prepare_flow(model_in)

    def predict_flow(self, flow, *, hidden_residual=None) -> np.ndarray:
        import torch

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                pred_chunk = self.policy.model.integrate_flow(flow, hidden_residual=hidden_residual)
                pred_chunk = pred_chunk[..., :self.policy.config.output_features["action"].shape[0]]
            post_chunk = self.postprocessor(pred_chunk)
            if post_chunk.shape != pred_chunk.shape or post_chunk.ndim != 3 or post_chunk.shape[0] != 1:
                raise ValueError("Postprocessor must preserve (1, T, A) action chunk shape")
        return post_chunk.squeeze(0).detach().float().cpu().numpy()


def make_pre_post_processors(pretrained_path, *, preprocessor_overrides):
    from a2meas.processing import PolicyProcessorPipeline
    from a2meas.processing.pipeline import (
        batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action,
    )

    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename="policy_preprocessor.json",
            overrides=preprocessor_overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
