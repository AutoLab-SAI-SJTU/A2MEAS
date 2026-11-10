"""One BEHAVIOR VLA route, its checkpoint and action queue."""

from collections import deque
import numpy as np
from . import load_routes, resolve_task
from .actions import (
    ActionChunker, EffectiveGripperCloseCounter, FirstEffectiveGripperCloseTracker,
    build_gripper_recovery, gaussian_filter_action_chunk,
)
from .observations import ObservationAdapter
from .resources import checkpoint_path
from .service import ModelService, ModelArgs


class VLAPolicy:

    def __init__(self, route, resources, *, adapter=False, num_inference_steps=None):
        import torch

        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Expose one allocated CUDA GPU with native BF16 support")
        self.resources = resources
        self.num_inference_steps = num_inference_steps
        self.adapter_config = None
        self.correction = None
        if adapter:
            from .correction import load_config
            self.adapter_config = load_config()
        effective, _ = self.resolve_route(route)
        checkpoint = checkpoint_path(resources, effective["model"])
        self.service = ModelService(
            ModelArgs(resources=resources, model_dir=checkpoint.parent, checkpoint_path=checkpoint,
                      num_inference_steps=effective["num_inference_steps"])
        )
        self.configure_route(route)

    def resolve_route(self, route):
        task = None
        if self.adapter_config is not None:
            task = next((item for item in self.adapter_config["tasks"].values() if item["task"] == route["task"]), None)
        if task is not None:
            route = dict(route, model=self.adapter_config["model"],
                         num_inference_steps=self.adapter_config["num_inference_steps"],
                         execution=self.adapter_config["execution"])
        if self.num_inference_steps is not None:
            route = dict(route, num_inference_steps=self.num_inference_steps)
        return route, task

    def configure_route(self, route):
        route, task = self.resolve_route(route)
        self.task_id = resolve_task(route["task"], load_routes())[0]
        if task is not None:
            if self.correction is None:
                from .correction.controller import CorrectionController
                if (not self.service.policy.model._lawam_alternate_kv_enabled()
                        or len(self.service.policy.model.paligemma_with_expert.gemma_expert.model.layers) != 18):
                    raise ValueError("Adapter requires the trained eighteen-layer alternate-KV expert")
                self.correction = CorrectionController(self.service, self.resources, self.adapter_config, task)
            else:
                self.correction.configure_task(task)
        elif self.correction is not None:
            self.correction.close()
            self.correction = None
        self.observations = ObservationAdapter(self.service, route)
        self.execution = route["execution"]
        self.chunking = ActionChunker(arm_step_limit=self.execution["arm_step_limit"])
        self.first_close = (
            FirstEffectiveGripperCloseTracker() if self.service.policy.config.enable_gripper_first_close_meta else None
        )
        self.reset()

    def warmup(self):
        import torch
        from .observations import CAMERA_KEYS

        size = self.service.policy.config.image_resolution[0]
        image = np.full((size, size, 3), 128, dtype=np.uint8)
        observation = {**dict.fromkeys(CAMERA_KEYS, image),
                       "robot_r1::proprio": np.zeros(61, dtype=np.float32)}
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.observations.observe(observation)
                sample = self.observations.sample()
                if self.first_close is not None:
                    sample["gripper_first_close"] = 0
                self.service.predict(sample)
        finally:
            self.reset()

    def reset(self):
        self.observations.reset()
        self.actions = deque()
        self.last_action = None
        self.recovery = None
        self.close_counter = EffectiveGripperCloseCounter()
        if self.first_close is not None:
            self.first_close.reset()
        if self.correction is not None:
            self.correction.reset()

    def close(self):
        if self.correction is not None:
            self.correction.close()

    def step(self, observation):
        state = observation["robot_r1::proprio"]
        self.close_counter.observe_state(state)
        if self.first_close is not None:
            was_closed = self.first_close.has_closed_once
            self.first_close.observe_state(state)
            if not was_closed and self.first_close.has_closed_once and self.recovery is None:
                self.actions.clear()
        if self.recovery is not None and self.recovery.complete(state):
            self.actions.clear()
            self.recovery = None
        self.observations.observe(observation)
        if self.correction is not None:
            self.correction.observe(observation)
        if not self.actions:
            self.recovery = build_gripper_recovery(
                self.task_id, state, self.close_counter.total_count, self.last_action,
            )
            if self.recovery is None:
                sample = self.observations.sample()
                if self.first_close is not None:
                    sample["gripper_first_close"] = int(self.first_close.has_closed_once)
                if self.correction is None:
                    predicted, mask = self.service.predict(sample), None
                else:
                    predicted, mask = self.correction.predict(sample)
                if self.last_action is None:
                    self.recovery = build_gripper_recovery(
                        self.task_id, state, self.close_counter.total_count, predicted[0],
                    )
            if self.recovery is not None:
                chunk = self.recovery.actions
                if self.correction is None:
                    chunk = gaussian_filter_action_chunk(
                        chunk, window=self.execution["filter_window"], sigma=self.execution["filter_sigma"],
                        skip_dims=self.execution["filter_skip_dims"],
                    )
            else:
                chunk = gaussian_filter_action_chunk(
                    predicted,
                    window=self.execution["filter_window"],
                    sigma=self.execution["filter_sigma"],
                    skip_dims=self.execution["filter_skip_dims"],
                    initial=self.last_action,
                )
                if mask is None:
                    chunk = self.chunking.process(chunk, gripper_full_source_on_transition=self.correction is None)
                else:
                    chunk[:, mask != 0] = predicted[:, mask != 0]
                    chunk = chunk[:self.correction.horizon]
            if not np.isfinite(chunk).all():
                raise ValueError("Model returned non-finite actions")
            self.actions.extend(chunk)
        action = self.actions.popleft()
        self.last_action = action.copy()
        if self.correction is not None:
            self.correction.action_sent(action)
        if self.recovery is None:
            self.close_counter.observe_action(action)
        if self.first_close is not None:
            self.first_close.observe_action(action, is_recovery=self.recovery is not None)
        if not self.actions:
            self.recovery = None
        return action
