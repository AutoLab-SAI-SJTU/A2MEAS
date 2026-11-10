"""BEHAVIOR camera preprocessing, state projection and causal observation history."""

from collections import deque

import numpy as np

from a2meas.schema import SHORT_TERM_MEMORY_FRAME_AGE_SECONDS


POLICY_CAMERA_KEYS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)


HISTORY_OFFSETS = (-256, -205, -154, -60, -30, 0)


def _camera_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Own the native-resolution RGB frame; simulator buffers may be reused."""
    if (image.ndim != 3 or image.shape[:2] != (size, size)
            or image.shape[-1] not in (3, 4) or image.dtype != np.uint8):
        raise ValueError(f"Camera observations must be uint8 RGB or RGBA arrays with resolution {size} × {size}")
    return image[..., :3].copy()


def _project_behavior_b1k_state(state: np.ndarray) -> np.ndarray:
    """Project validated 61D challenge proprioception to the trained 23D state."""
    base_qvel = state[0:3]
    trunk_qpos = state[53:57]
    left_arm_qpos = state[3:10]
    left_gripper = np.asarray([state[24:26].sum()], dtype=np.float32)
    right_arm_qpos = state[28:35]
    right_gripper = np.asarray([state[49:51].sum()], dtype=np.float32)
    return np.concatenate(
        [base_qvel, trunk_qpos, left_arm_qpos, left_gripper, right_arm_qpos, right_gripper],
        axis=0,
    ).astype(np.float32, copy=False)


CAMERA_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


class ObservationAdapter:
    """Build six-frame model inputs from the official BEHAVIOR observation schema."""

    def __init__(self, service, route):
        config = service.policy.config
        self._task = route["task"]
        self._image_size = config.image_resolution[0]
        self._frame_age_embedding_enabled = config.short_term_memory_frame_age_embedding_enable
        self._frame_age_fps = config.short_term_memory_frame_age_fps
        self._speed_bin = config.speed_meta_map[self._task] if config.enable_speed_meta else None
        capacity = 1 - min(HISTORY_OFFSETS)
        self._image_history = {key: deque(maxlen=capacity) for key in POLICY_CAMERA_KEYS}
        self._state_history = deque(maxlen=capacity)
        self.reset()

    def reset(self):
        for history in self._image_history.values():
            history.clear()
        self._state_history.clear()

    def observe(self, observation):
        """Record every simulation step, including steps that execute queued actions."""
        frames = [_camera_rgb(observation[key], self._image_size) for key in CAMERA_KEYS]
        state = _project_behavior_b1k_state(np.asarray(observation["robot_r1::proprio"], dtype=np.float32))
        for key, frame in zip(POLICY_CAMERA_KEYS, frames, strict=True):
            self._image_history[key].append(frame)
        self._state_history.append(state)

    def sample(self):
        """Materialize model inputs when a new action chunk is needed."""
        import torch

        newest = len(self._state_history) - 1
        indices = [max(0, newest + offset) for offset in HISTORY_OFFSETS]
        sample = {
            "observation.state": torch.from_numpy(np.stack([self._state_history[i] for i in indices])).unsqueeze(0),
            "task": self._task,
        }
        if self._speed_bin is not None:
            sample["speed_bin"] = self._speed_bin
        # Pad unavailable history with the first frame; valid black frames remain unmasked.
        history_is_pad = torch.tensor(
            [newest + offset < 0 for offset in HISTORY_OFFSETS],
            dtype=torch.bool,
        ).unsqueeze(0)
        for key, history in self._image_history.items():
            frames = np.stack([history[i] for i in indices]).transpose(0, 3, 1, 2)
            frames = np.ascontiguousarray(frames, dtype=np.float32)
            frames /= 255.0
            sample[key] = torch.from_numpy(frames).unsqueeze(0)
            sample[f"{key}_is_pad"] = history_is_pad
        if self._frame_age_embedding_enabled:
            ages = [float(newest - index) / self._frame_age_fps for index in indices]
            sample[SHORT_TERM_MEMORY_FRAME_AGE_SECONDS] = torch.tensor(ages, dtype=torch.float32).unsqueeze(0)
        return sample
