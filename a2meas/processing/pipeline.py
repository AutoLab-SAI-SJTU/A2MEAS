#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Load and run saved observation and action processing pipelines."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias, TypedDict, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from a2meas.schema import OBS_PREFIX, SHORT_TERM_MEMORY_FRAME_AGE_SECONDS


class TransitionKey(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    COMPLEMENTARY_DATA = "complementary_data"


PolicyAction: TypeAlias = torch.Tensor


class EnvTransition(TypedDict):
    observation: dict[str, Any] | None
    action: PolicyAction | None
    complementary_data: dict[str, Any]


def create_transition(*, observation=None, action=None, complementary_data=None) -> EnvTransition:
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:
    return transition[TransitionKey.ACTION]


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    metadata_keys = {"task", "speed_bin", "gripper_first_close", SHORT_TERM_MEMORY_FRAME_AGE_SECONDS}
    return create_transition(
        observation={key: value for key, value in batch.items() if key.startswith(OBS_PREFIX)},
        complementary_data={
            key: value for key, value in batch.items() if key in metadata_keys or key.endswith("_is_pad")
        },
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    return {**transition[TransitionKey.OBSERVATION], **transition[TransitionKey.COMPLEMENTARY_DATA]}


class ProcessorStepRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def register_step(step_class):
            if name in cls._registry:
                raise ValueError(f"Processor step already registered: {name}")
            cls._registry[name] = step_class
            return step_class

        return register_step

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]


class ProcessorStep(ABC):
    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition: ...


@dataclass
class PolicyProcessorPipeline:
    steps: list[ProcessorStep] = field(default_factory=list)
    name: str = "PolicyProcessorPipeline"
    to_transition: Callable = field(default_factory=lambda: batch_to_transition, repr=False)
    to_output: Callable = field(default_factory=lambda: transition_to_batch, repr=False)

    def __call__(self, data):
        transition = self.to_transition(data)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config_filename,
        *,
        overrides=None,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    ):
        directory = Path(pretrained_model_name_or_path)
        config = json.loads((directory / config_filename).read_text())
        entries = config["steps"]
        if not isinstance(entries, list):
            raise ValueError("Processor configuration must contain a list of steps")
        overrides = overrides or {}
        unknown = overrides.keys() - {entry["registry_name"] for entry in entries}
        if unknown:
            raise KeyError(f"Overrides do not match saved processor steps: {sorted(unknown)}")
        steps = []
        for entry in entries:
            name = entry["registry_name"]
            step = ProcessorStepRegistry.get(name)(**{**entry.get("config", {}), **overrides.get(name, {})})
            if "state_file" in entry:
                step.load_state_dict(load_file(directory / entry["state_file"]))
            steps.append(step)
        return cls(steps=steps, name=config["name"], to_transition=to_transition, to_output=to_output)
