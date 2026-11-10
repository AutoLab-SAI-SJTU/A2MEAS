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

"""Batching, device transfer and quantile normalization."""

from dataclasses import dataclass, field

from torch import Tensor
import torch

from a2meas.schema import (
    OBS_STATE,
    SHORT_TERM_MEMORY_FRAME_AGE_SECONDS,
    ACTION,
)
from .pipeline import ProcessorStep, ProcessorStepRegistry, TransitionKey
from a2meas.schema import FeatureType, NormalizationMode, PolicyFeature


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class AddBatchDimensionProcessorStep(ProcessorStep):
    def __call__(self, transition):
        result = transition.copy()
        observation = dict(transition[TransitionKey.OBSERVATION])
        if observation[OBS_STATE].ndim == 1:
            observation[OBS_STATE] = observation[OBS_STATE].unsqueeze(0)
        result[TransitionKey.OBSERVATION] = observation
        data = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        if isinstance(data["task"], str):
            data["task"] = [data["task"]]
        for key in ("speed_bin", "gripper_first_close", SHORT_TERM_MEMORY_FRAME_AGE_SECONDS):
            value = data.get(key)
            if isinstance(value, Tensor) and (
                value.ndim == 0 or (key == SHORT_TERM_MEMORY_FRAME_AGE_SECONDS and value.ndim == 1)
            ):
                data[key] = value.unsqueeze(0)
        result[TransitionKey.COMPLEMENTARY_DATA] = data
        return result


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    device: str = "cpu"

    def __post_init__(self):
        self.tensor_device = torch.device(self.device)

    def _process_tensor(self, tensor):
        return tensor.to(
            device=self.tensor_device, non_blocking=self.tensor_device.type == "cuda"
        )

    def __call__(self, transition):
        result = transition.copy()
        action = transition[TransitionKey.ACTION]
        if action is not None:
            result[TransitionKey.ACTION] = self._process_tensor(action)
        for key in (TransitionKey.OBSERVATION, TransitionKey.COMPLEMENTARY_DATA):
            data = transition[key]
            if data is not None:
                result[key] = {
                    name: self._process_tensor(value) if isinstance(value, torch.Tensor) else value
                    for name, value in data.items()
                }
        return result


@dataclass
class _QuantileNormalization:
    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    _quantiles: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.features = {
            key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            if isinstance(value, dict) else value
            for key, value in self.features.items()
        }
        self.norm_map = {FeatureType(key): NormalizationMode(value) for key, value in self.norm_map.items()}

    def load_state_dict(self, state):
        quantiles = {}
        for key, feature in self.features.items():
            if self.norm_map[feature.type] == NormalizationMode.IDENTITY:
                continue
            lower, upper = state[f"{key}.q01"].float(), state[f"{key}.q99"].float()
            if lower.shape != feature.shape or upper.shape != feature.shape:
                raise ValueError(f"Normalization statistics do not match the feature shape: {key}")
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (upper < lower).any():
                raise ValueError(f"Invalid normalization quantiles: {key}")
            quantiles[key] = (lower, upper)
        self._quantiles = quantiles

    def _transform(self, tensor, key, *, inverse=False):
        if self.norm_map[self.features[key].type] == NormalizationMode.IDENTITY:
            return tensor
        lower, upper = self._quantiles[key]
        lower, upper = lower.to(tensor.device, tensor.dtype), upper.to(tensor.device, tensor.dtype)
        self._quantiles[key] = (lower, upper)
        span = upper - lower
        if inverse:
            return (tensor + 1.0) * span / 2.0 + lower
        zero_span = span == 0
        safe_span = torch.where(zero_span, torch.ones_like(span), span)
        normalized = 2.0 * (tensor - lower) / safe_span - 1.0
        return torch.where(zero_span, torch.zeros_like(normalized), normalized)


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_QuantileNormalization, ProcessorStep):
    def __call__(self, transition):
        result = transition.copy()
        observation = dict(transition[TransitionKey.OBSERVATION])
        for key in self.features:
            observation[key] = self._transform(torch.as_tensor(observation[key]), key)
        result[TransitionKey.OBSERVATION] = observation
        return result


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_QuantileNormalization, ProcessorStep):
    def __call__(self, transition):
        result = transition.copy()
        result[TransitionKey.ACTION] = self._transform(transition[TransitionKey.ACTION], ACTION, inverse=True)
        return result
