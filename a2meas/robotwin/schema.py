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

from dataclasses import dataclass
import json
from pathlib import Path

import draccus


ROBOTWIN_CAMERAS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
ROBOTWIN_GRIPPERS = [6, 13]
ROBOTWIN_DELTA_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


from a2meas.schema import PolicyConfig


@dataclass
class PreTrainedConfig(PolicyConfig, draccus.ChoiceRegistry):
    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, device=None):
        config = json.loads((Path(pretrained_name_or_path) / "config.json").read_text())
        if device is not None:
            config["device"] = device
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        if config.get("type") != "pi05":
            raise ValueError("RoboTwin requires checkpoint policy type pi05")
        config_class = cls.get_choice_class(config.pop("type"))
        return draccus.decode(config_class, config)
