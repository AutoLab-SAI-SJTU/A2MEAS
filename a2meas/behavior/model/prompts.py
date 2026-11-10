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

"""Canonical task metadata and checkpoint prompt processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from a2meas.behavior.model.conditioning import (
    GLOBAL_TASK_ID,
    GLOBAL_TASK_TEXT,
    clean_task_text,
    validate_task_mapping,
)
from a2meas.processing import ProcessorStep, ProcessorStepRegistry
from a2meas.processing.pipeline import EnvTransition, TransitionKey


@ProcessorStepRegistry.register(name="pi05_global_task_processor")
@dataclass
class Pi05GlobalTaskProcessorStep(ProcessorStep):
    """Capture the canonical goal before prompt augmentation."""

    task_text_to_id: dict[str, int]
    num_tasks: int = 100

    def __post_init__(self):
        self.task_text_to_id = validate_task_mapping(self.task_text_to_id, self.num_tasks)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        data = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        texts = [clean_task_text(task) for task in data["task"]]
        data[GLOBAL_TASK_TEXT] = texts
        unknown = sorted(set(texts) - self.task_text_to_id.keys())
        if unknown:
            raise ValueError(f"Unknown global task text: {unknown}")
        data[GLOBAL_TASK_ID] = torch.tensor(
            [self.task_text_to_id[text] for text in texts], dtype=torch.long
        )
        result[TransitionKey.COMPLEMENTARY_DATA] = data
        return result


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class A2WLAPromptProcessor(ProcessorStep):
    enable_speed_meta: bool = False
    enable_gripper_first_close_meta: bool = False

    @staticmethod
    def _batch_values(value, count):
        if value is None:
            return [None] * count
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().reshape(-1).tolist()
        else:
            values = np.asarray(value).reshape(-1).tolist()
        if len(values) != count:
            raise ValueError("Prompt metadata must match the observation batch")
        return values

    def __call__(self, transition):
        transition = transition.copy()
        data = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        tasks = data["task"]
        speeds = self._batch_values(data["speed_bin"] if self.enable_speed_meta else None, len(tasks))
        closes = self._batch_values(data.get("gripper_first_close"), len(tasks))
        prompts = []
        for index, task in enumerate(tasks):
            close = closes[index] if self.enable_gripper_first_close_meta else None
            if self.enable_gripper_first_close_meta and close not in (0, 1):
                raise ValueError("First-close metadata must be 0 or 1")
            text = task.strip().replace("_", " ").replace("\n", " ")
            prompt = f"Task: {text}"
            if self.enable_speed_meta and speeds[index] is not None:
                prompt += f", Speed: {int(speeds[index])}"
            if close is not None:
                prompt += f", GripperClose: {close}"
            prompts.append(prompt)
        data["task"] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = data
        return transition
