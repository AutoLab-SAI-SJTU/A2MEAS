"""BEHAVIOR action smoothing, chunk execution and first-close prompt metadata."""

from dataclasses import dataclass, field

import numpy as np

from .observations import _project_behavior_b1k_state


BEHAVIOR_ACTION_DIM = 23


LEFT_GRIPPER_ACTION_INDEX = 14


RIGHT_GRIPPER_ACTION_INDEX = 22


GRIPPER_VARIATION_THRESHOLD = 0.2


class ActionChunker:
    """Resample chunks, retaining source cadence for gripper changes and arm limits."""

    source_steps = 32
    execute_steps = 16

    def __init__(self, *, arm_step_limit: float = 0.0) -> None:
        self.arm_step_limit = arm_step_limit
        if not np.isfinite(arm_step_limit) or arm_step_limit < 0:
            raise ValueError("Arm step limit must be finite and nonnegative")

    @staticmethod
    def _resample(values: np.ndarray, target_steps: int) -> np.ndarray:
        from scipy.interpolate import interp1d

        source_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, target_steps, dtype=np.float64)
        interpolator = interp1d(source_x, values, axis=0, kind="cubic")
        return np.asarray(interpolator(target_x), dtype=np.float32)

    def _within_compression_limits(self, actions: np.ndarray) -> bool:
        if self.arm_step_limit == 0:
            return True
        arms = list(range(7, 14)) + list(range(15, 22))
        arm_step_peak = float(np.max(np.abs(np.diff(actions[:, arms], axis=0))))
        return arm_step_peak <= self.arm_step_limit

    def process(self, chunk: np.ndarray, *, gripper_full_source_on_transition=True) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] != BEHAVIOR_ACTION_DIM:
            raise ValueError("Action chunking requires a (T, 23) BEHAVIOR action chunk")
        if chunk.shape[0] < self.source_steps:
            raise ValueError(f"Action chunk needs at least {self.source_steps} steps")
        source = chunk[:self.source_steps]
        if any(
            float(np.ptp(source[:, index])) > GRIPPER_VARIATION_THRESHOLD
            for index in (LEFT_GRIPPER_ACTION_INDEX, RIGHT_GRIPPER_ACTION_INDEX)
        ):
            steps = self.source_steps if gripper_full_source_on_transition else self.execute_steps
            return source[:steps].copy()
        steps = self.execute_steps
        while steps < self.source_steps:
            executed = self._resample(source, steps)
            executed[:, :3] *= self.source_steps / float(steps)
            if self._within_compression_limits(executed):
                return executed
            steps += 1
        return source.copy()


def gaussian_filter_action_chunk(
    actions: np.ndarray,
    *,
    window: int,
    sigma: float,
    skip_dims: list[int],
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth an action chunk in time, preserving the configured discrete dimensions."""
    if window == 0:
        return actions
    if window < 1 or window % 2 == 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Smoothing requires an odd positive window and a finite positive sigma")

    radius = window // 2
    positions = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(positions * positions) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= float(kernel.sum())
    out = actions.copy()
    for dim in range(actions.shape[1]):
        if dim in skip_dims:
            continue
        left = actions[0, dim] if initial is None else float(initial[dim])
        sequence = np.concatenate([
            np.full((radius,), left, dtype=np.float32),
            actions[:, dim],
            np.full((radius,), actions[-1, dim], dtype=np.float32),
        ])
        out[:, dim] = np.convolve(sequence, kernel, mode="valid").astype(np.float32)
    return out


GRIPPER_ACTION_LOW_THRESHOLD = -0.5


GRIPPER_ACTION_HIGH_THRESHOLD = 0.5


GRIPPER_ACTION_CONFIRM_STEPS = 5


GRIPPER_EMPTY_WIDTH_THRESHOLD_M = 0.001


@dataclass
class _GripperState:
    observed: int = 0
    committed: int = 0
    pending_state: int = 0
    pending_count: int = 0
    close_active: bool = False
    close_invalid: bool = False
    close_counted: bool = False
    count: int = 0
    nonempty_streak: int = 0

    def observe_action(self, value: float) -> None:
        if value <= GRIPPER_ACTION_LOW_THRESHOLD:
            self.observed = -1
        elif value >= GRIPPER_ACTION_HIGH_THRESHOLD:
            self.observed = 1
        if self.observed == 0:
            return
        if self.committed == 0:
            self.committed = self.observed
            self.pending_state = self.pending_count = 0
            return
        if self.observed == self.committed:
            self.pending_state = self.pending_count = 0
            return
        if self.observed != self.pending_state:
            self.pending_state = self.observed
            self.pending_count = 1
        else:
            self.pending_count += 1
        if self.pending_count < GRIPPER_ACTION_CONFIRM_STEPS:
            return
        self.committed = self.observed
        self.pending_state = self.pending_count = 0
        self.close_active = self.committed == -1
        self.close_invalid = False
        self.close_counted = False

    def observe_width(self, width: float) -> bool:
        if self.close_active and width < GRIPPER_EMPTY_WIDTH_THRESHOLD_M:
            self.close_invalid = True
            if self.close_counted:
                self.count -= 1
                self.close_counted = False
        if self.close_active and not self.close_invalid:
            if not self.close_counted:
                self.count += 1
                self.close_counted = True
            self.nonempty_streak += 1
        else:
            self.nonempty_streak = 0
        return self.nonempty_streak >= 3


@dataclass
class EffectiveGripperCloseCounter:
    _arms: list[_GripperState] = field(default_factory=lambda: [_GripperState(), _GripperState()])

    @property
    def total_count(self):
        return sum(arm.count for arm in self._arms)

    def observe_state(self, state):
        for arm, indices in zip(self._arms, ((24, 26), (49, 51)), strict=True):
            arm.observe_width(float(np.sum(state[indices[0]:indices[1]], dtype=np.float64)))

    def observe_action(self, action):
        for arm, index in zip(self._arms, (14, 22), strict=True):
            arm.observe_action(float(action[index]))


@dataclass
class FirstEffectiveGripperCloseTracker:
    """Latch after a five-action close debounce and three nonempty observed states.

    Read the label after observe_state(state_t) and before observe_action(action_t).
    State and action inputs use the validated 61D and 23D BEHAVIOR layouts.
    """

    has_closed_once: bool = False
    _arms: list[_GripperState] = field(
        default_factory=lambda: [_GripperState(observed=1, committed=1), _GripperState(observed=1, committed=1)]
    )

    def reset(self) -> None:
        self.has_closed_once = False
        self._arms = [_GripperState(observed=1, committed=1), _GripperState(observed=1, committed=1)]

    def observe_state(self, state: np.ndarray) -> bool:
        widths = (
            float(np.sum(state[24:26], dtype=np.float64)),
            float(np.sum(state[49:51], dtype=np.float64)),
        )
        for arm, width in zip(self._arms, widths, strict=True):
            if arm.observe_width(width):
                self.has_closed_once = True
        return self.has_closed_once

    def observe_action(self, action: np.ndarray, *, is_recovery=False) -> None:
        for index, action_index in enumerate((LEFT_GRIPPER_ACTION_INDEX, RIGHT_GRIPPER_ACTION_INDEX)):
            if not is_recovery:
                self._arms[index].observe_action(float(action[action_index]))
            elif action[action_index] >= GRIPPER_ACTION_HIGH_THRESHOLD:
                self._arms[index] = _GripperState(observed=1, committed=1)


_RECOVERY_LEFT_TASKS = frozenset({
    *range(1, 41), 42, 43, 44, 45, 47, 48,
    *range(50, 66), 67, 68, 70, 71, 72, 73, 74, 75, 76,
    *range(78, 89), 90, 91, 92, 93, 94, 96, 97, 98, 99,
})
_RECOVERY_RIGHT_TASKS = frozenset({
    *range(1, 30), 34, 35, 36, 37, 42, 43, 44, 47, 48, 49,
    50, *range(52, 66), 67, 68, 71, 73, 74, 75, 76,
    *range(78, 92), 93, 94, 97, 98, 99,
})
_RECOVERY_EXCLUSION_CLOSE_COUNTS = {
    0: 1, 30: 3, 31: 3, 32: 3, 33: 7, 38: 1, 39: 1, 40: 2, 41: 8,
    45: 5, 46: 4, 49: 10, 51: 3, 66: 1, 69: 1, 70: 4, 72: 4, 77: 1,
    89: 1, 92: 1, 95: 8, 96: 2,
}


@dataclass
class GripperRecovery:
    actions: np.ndarray
    targets: tuple[bool, bool]

    def complete(self, state):
        widths = (np.sum(state[24:26], dtype=np.float64), np.sum(state[49:51], dtype=np.float64))
        return all(not target or width > 0.099 for target, width in zip(self.targets, widths, strict=True))


def build_gripper_recovery(task_id, state, close_count, reference_action):
    """Reopen empty grippers at prediction boundaries under the released task rules."""
    action = _project_behavior_b1k_state(state)
    widths = np.asarray([2.0 * float(action[index]) / 0.1 - 1.0 for index in (14, 22)], dtype=np.float32)
    defer_exclusion = close_count < _RECOVERY_EXCLUSION_CLOSE_COUNTS.get(task_id, 0)
    targets = tuple((task_id in tasks or defer_exclusion) and width < np.float32(-0.98)
                    for tasks, width in zip((_RECOVERY_LEFT_TASKS, _RECOVERY_RIGHT_TASKS), widths, strict=True))
    if not any(targets):
        return None
    for index, target, width in zip((14, 22), targets, widths, strict=True):
        if target:
            action[index] = 1.0
        elif reference_action is not None:
            if not np.isfinite(reference_action[index]):
                return None
            action[index] = 1.0 if reference_action[index] >= 0 else -1.0
        elif width > np.float32(0.90):
            action[index] = 1.0
        elif width < np.float32(-0.98):
            action[index] = -1.0
        else:
            return None
    return GripperRecovery(np.tile(action, (ActionChunker.execute_steps, 1)), targets)
