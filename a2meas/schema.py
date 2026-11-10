# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Feature schema and local A2WLA checkpoint configuration."""

from dataclasses import dataclass, field
from enum import Enum

import torch


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


@dataclass
class PolicyConfig:
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    device: str = "cpu"

    def __post_init__(self):
        torch.device(self.device)

    @property
    def type(self):
        return self.get_choice_name(type(self))

    @property
    def image_features(self):
        return {
            key: feature for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL
        }



OBS_PREFIX = "observation."
OBS_STATE = "observation.state"
SHORT_TERM_MEMORY_FRAME_AGE_SECONDS = "short_term_memory_frame_age_seconds"
ACTION = "action"
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
