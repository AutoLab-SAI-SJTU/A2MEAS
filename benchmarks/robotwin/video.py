from __future__ import annotations

from typing import Any

import numpy as np


def compose_camera_frame(observation: dict[str, Any], width: int, height: int) -> np.ndarray:
    import cv2

    canvas = np.zeros((height * 3, width * 2, 3), dtype=np.uint8)
    panels = (
        ("head_camera", "HEAD", 0, 0, width * 2, height * 2),
        ("left_camera", "LEFT WRIST", 0, height * 2, width, height),
        ("right_camera", "RIGHT WRIST", width, height * 2, width, height),
    )
    for camera, label, left, top, panel_width, panel_height in panels:
        pixels = np.asarray(observation["observation"][camera]["rgb"])
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError(f"{camera}: expected uint8 HWC RGB, got {pixels.shape}/{pixels.dtype}")
        source_height, source_width = pixels.shape[:2]
        if source_height < 1 or source_width < 1:
            raise ValueError(f"{camera}: empty RGB frame")
        scale = min(panel_width / source_width, panel_height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = cv2.resize(pixels, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        offset_left = left + (panel_width - resized_width) // 2
        offset_top = top + (panel_height - resized_height) // 2
        canvas[offset_top:offset_top + resized_height, offset_left:offset_left + resized_width] = resized
        cv2.rectangle(canvas, (left, top), (left + min(panel_width - 1, 145), top + 23), (0, 0, 0), -1)
        cv2.putText(canvas, label, (left + 6, top + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class MultiCameraVideoWriter:
    def __init__(self, process: Any, environment: Any, width: int, height: int) -> None:
        self.stdin = self
        self._process = process
        self._environment = environment
        self._width = width
        self._height = height

    def write(self, head_frame: bytes) -> int:
        frame = compose_camera_frame(self._environment.now_obs, self._width, self._height)
        self._process.stdin.write(frame.tobytes())
        return len(head_frame)

    def close(self) -> None:
        self._process.stdin.close()

    def wait(self) -> int:
        return self._process.wait()
