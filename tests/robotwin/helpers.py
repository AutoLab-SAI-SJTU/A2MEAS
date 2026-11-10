"""Small CPU-only server and observation fixtures shared by RoboTwin tests."""
from types import SimpleNamespace

import numpy as np
import torch
from vla_eval.model_servers.predict import PredictModelServer
from a2meas.robotwin.server import CAMERAS, RoboTwinModelServer
from a2meas.robotwin.rsi.server import RoboTwinRSIServer
from a2meas.robotwin.schema import ROBOTWIN_CAMERAS


def native_server():
    result = RoboTwinModelServer.__new__(RoboTwinModelServer)
    result._histories, result._state_histories = {}, {}
    result._history_size, result._flow_frames = 31, 1
    result.offsets = [-30, 0]
    result.camera_keys = ("head", "left", "right")
    result._torch, result._padding = torch, False
    return result


def rsi_server():
    obj = object.__new__(RoboTwinRSIServer)
    PredictModelServer.__init__(obj, chunk_size=40)
    obj._torch, obj._histories, obj._state_histories = torch, {}, {}
    obj._batch_sizes, obj._graph_denoiser = {}, None
    obj.offsets, obj._history_size, obj._flow_frames = [-30, 0], 31, 1
    obj.camera_keys, obj._padding, obj._fast_sample = ROBOTWIN_CAMERAS, False, True
    obj._policy = SimpleNamespace(config=SimpleNamespace(short_term_memory_inference_num_frames=2))
    obj._preprocessor = SimpleNamespace(steps=[SimpleNamespace(enable_speed_meta=True,
        default_speed_bin=500, speed_meta_dropout_prob=.15)])
    return obj


def observation(value):
    return {
        "images": {name: np.full((2, 3, 3), value, np.uint8) for name in CAMERAS},
        "joint_state": np.full(14, value, np.float32),
        "task_description": "test task",
    }
