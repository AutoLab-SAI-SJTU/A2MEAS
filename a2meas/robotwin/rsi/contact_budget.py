"""Fit an otherwise valid regrasp into the remaining native action budget.

Existing accepted plans remain unchanged. Only an over-budget fallback removes
four free-space endpoint dwells, keeps the contact/jaw dwells and all motion
endpoints, and then removes over-dense interpolation targets under old limits.
"""
import numpy as np

def plan_with_contact_budget(native_plan, kin, state, roles, found, priors, grasp_R, remaining, stereo=None, settings=None):
    try:
        return native_plan(kin, state, roles, found, priors, grasp_R, remaining, stereo=stereo, settings=settings)
    except ValueError as original:
        if str(original) != 'No reachable regrasp within original action budget':
            raise
        try:
            full, native_details = native_plan(kin, state, roles, found, priors, grasp_R, 1000000, stereo=stereo, settings=settings)
            faster, details = native_plan(kin, state, roles, found, priors, grasp_R, 1000000, stereo=stereo, settings=settings, transit_hold=0)
            if native_details['arm'] != details['arm']:
                raise ValueError('Changed original shortest native arm branch')
            cursor = 0
            for target in faster:
                while cursor < len(full) and (not np.array_equal(full[cursor], target)):
                    cursor += 1
                if cursor == len(full):
                    raise ValueError('Changed original geometric target')
                cursor += 1
            lengths = details['segment_lengths']
            ends = np.cumsum(lengths) - 1
            required = set(ends.tolist()) | set(np.r_[0, np.cumsum(lengths)[:-1]].tolist())
            compacted = compact_path(faster, kin[details['arm']], details['arm'], required_indices=required)
            if len(compacted) > remaining - 10 or len(compacted) >= len(full):
                raise ValueError('New contact-dwell path still over original budget')
        except ValueError:
            raise original
        details = {key: value for key, value in details.items() if key != 'segment_lengths'}
        return compacted, {**details, 'counted_steps': len(compacted), 'original_steps': len(full)}

from a2meas.robotwin.rsi.geometry import ee_world

def compact_path(rows, kin, side, required_indices=()):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 14 or (not np.isfinite(rows).all()) or (len(rows) < 2):
        raise ValueError('Expected finite original path with14D targets')
    n = len(rows)
    sl = slice(0, 6) if side == 'left' else slice(7, 13)
    other = [i for i in range(14) if i not in range(sl.start, sl.stop)]
    q = rows[:, sl]
    mandatory = np.zeros(n, dtype=bool)
    mandatory[[0, n - 1]] = True
    mandatory[list(required_indices)] = True
    identical = np.all(rows[1:] == rows[:-1], axis=1)
    jaws = np.any(rows[1:, [6, 13]] != rows[:-1, [6, 13]], axis=1)
    for change in [identical, jaws]:
        indices = np.flatnonzero(change)
        mandatory[indices] = True
        mandatory[indices + 1] = True
    poses = [ee_world(kin, row) for row in q]
    cache = {}

    def angle(A, B):
        return float(np.arccos(np.clip((np.trace(A[:3, :3] @ B[:3, :3].T) - 1) / 2, -1, 1)))

    def permitted(i, j):
        if j == i + 1:
            return True
        if mandatory[i + 1:j].any() or not np.all(rows[i:j + 1, other] == rows[i, other]):
            return False
        dq = q[j] - q[i]
        if np.max(np.abs(dq)) > 0.030001 or np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3]) > 0.012001 or angle(poses[j], poses[i]) > 0.150001:
            return False
        norm = float(dq @ dq)
        if norm < 1e-14:
            return False
        alpha = (q[i + 1:j] - q[i]) @ dq / norm
        if np.any(alpha < 0) or np.any(alpha > 1) or np.any(np.diff(alpha) < -1e-06):
            return False
        interpolated = q[i] + alpha[:, None] * dq
        if np.max(np.abs(interpolated - q[i + 1:j])) > 0.003:
            return False
        for k, expected in zip(range(i + 1, j), interpolated):
            key = expected.tobytes()
            if key not in cache:
                cache[key] = ee_world(kin, expected)
            pose = cache[key]
            if np.linalg.norm(pose[:3, 3] - poses[k][:3, 3]) > 0.001 or angle(pose, poses[k]) > 0.01:
                return False
        return True
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[0] = 1
    for i in range(n - 1):
        for j in range(i + 1, min(n, i + 9)):
            if dist[i] + 1 < dist[j] and permitted(i, j):
                dist[j] = dist[i] + 1
                prev[j] = i
    indices = []
    k = n - 1
    while k >= 0:
        indices.append(k)
        k = int(prev[k])
    indices = np.array(indices[::-1], dtype=int)
    out = rows[indices].copy()
    if not set(np.flatnonzero(mandatory)) <= set(indices):
        raise ValueError('Lost required target or contact dwell')
    if np.sum(np.all(out[1:] == out[:-1], axis=1)) != int(identical.sum()):
        raise ValueError('Changed repeated-target dwell count')
    return out
