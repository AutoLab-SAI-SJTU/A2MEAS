"""Slow the first can-release window by repeating exact native command rows."""
import copy
import os
from .geometry import RightArmKinematics, ee_world
import numpy as np
from .actor import Actor


def release_rows(native, state, arm, schedule, settings, observation):
    native = np.asarray(native, dtype=np.float32)
    if native.shape != (40, 14) or not np.isfinite(native).all():
        raise ValueError('Expected finite unchanged native40x14 prefix')
    schedule = copy.deepcopy(schedule)
    if arm is None or schedule['done']:
        return native, [], schedule
    factor = settings['repeat_factor']
    if len(native) + settings['maximum_native_rows'] * (factor - 1) + 80 > settings['max_steps'] + 1 - observation:
        return native, [], schedule
    grip, other = (6, 13) if arm == 'left' else (13, 6)
    rows, selected = [], []
    for index, target in enumerate(native):
        if not schedule['started'] and not schedule['done'] and target[grip] > .2 and target[other] > .8:
            schedule['started'] = True
        active = schedule['started'] and not schedule['done']
        rows.append(target)
        if not active:
            continue
        selected.append(index)
        rows.extend(target for _ in range(factor - 1))
        schedule['processed'] += 1
        if schedule['tail_remaining'] is None and target[grip] > .95:
            schedule['tail_remaining'] = settings['fully_open_tail_rows']
        elif schedule['tail_remaining'] is not None:
            schedule['tail_remaining'] -= 1
        if schedule['processed'] >= settings['maximum_native_rows'] or schedule['tail_remaining'] == 0:
            schedule['done'] = True
    return np.asarray(rows, dtype=np.float32), selected, schedule


def release_end_rows(native, state, arm, schedule, settings, observation):
    before = copy.deepcopy(schedule)
    rows, selected, after = release_rows(native, state, arm, before, settings, observation)
    decision = None
    if not before['done'] and after['done'] and selected and arm is not None:
        grip = 6 if arm == 'left' else 13
        last = selected[-1]
        cut = last + 1 + len(selected) * (settings['repeat_factor'] - 1)
        if rows[cut - 1, grip] > .95 and len(rows) - cut >= 4:
            decision = dict(last_native_release_index=last, prefix_rows=cut,
                            original_expanded_rows=len(rows), omitted_tail_rows=len(rows) - cut)
            rows = rows[:cut].copy()
    return np.asarray(rows, dtype=np.float32), selected, after, decision


def release_feedback_rows(native, state, arm, schedule, settings, observation, anchor, kin):
    before = copy.deepcopy(schedule)
    rows, selected, after = release_rows(native, state, arm, before, settings, observation)
    anchor = copy.deepcopy(anchor)
    decision = None
    if selected and arm is not None:
        grip, sl = (6, slice(0, 6)) if arm == 'left' else (13, slice(7, 13))
        if anchor is None:
            openings = [i for i in selected if native[i, grip] > .95]
            if openings:
                anchor = ee_world(kin[arm], native[openings[0], sl])[:3, 3].tolist()
        if not before['done'] and after['done'] and anchor is not None:
            release_end = selected[-1] + 1 + len(selected) * (settings['repeat_factor'] - 1)
            for index in range(release_end - 1, len(rows) - 4):
                if rows[index, grip] <= .95:
                    continue
                point = ee_world(kin[arm], rows[index, sl])[:3, 3]
                if point[2] - anchor[2] < .06:
                    continue
                cut = index + 1
                decision = dict(last_native_release_index=selected[-1], prefix_rows=cut,
                                original_expanded_rows=len(rows), omitted_tail_rows=len(rows) - cut,
                                first_fully_open_ownwrist_position=anchor,
                                new_boundary_ownwrist_position=point.tolist(),
                                upward_clearance_m=float(point[2] - anchor[2]))
                rows = rows[:cut].copy()
                break
    return np.asarray(rows, np.float32), selected, after, anchor, decision


class CanReleaseCadence(Actor):
    def _load_model(self):
        super()._load_model()
        if self.settings.get('observe_after_release_clearance', False):
            urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
            self._release_kin = {side: RightArmKinematics(urdf, side=side) for side in ['left', 'right']}

    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_can_release_sessions'):
            self._can_release_sessions = {}
        self._can_release_sessions[ctx.session_id] = {
            'n': 0, 'closed_runs': {6: 0, 13: 0}, 'arm': None, 'open_wrist_anchor': None,
            'schedule': {'started': False, 'done': False, 'processed': 0, 'tail_remaining': None}}

    async def on_episode_end(self, result, ctx):
        self._can_release_sessions.pop(ctx.session_id, None)

    async def on_observation(self, obs, ctx):
        m = self._can_release_sessions[ctx.session_id]
        m['n'] += 1
        m['state'] = self._state_vector_from_obs(obs).copy()
        for grip, other, side in [(6, 13, 'left'), (13, 6, 'right')]:
            m['closed_runs'][grip] = m['closed_runs'][grip] + 1 if m['state'][grip] < .2 else 0
            if m['arm'] is None and m['closed_runs'][grip] >= self.settings['minimum_closed_observations'] and m['state'][other] > .8:
                m['arm'] = side
        return await super().on_observation(obs, ctx)

    def _normalize_result(self, result, ctx):
        sid = ctx.session_id
        m = self._can_release_sessions[sid]
        self._session_chunk_sizes[sid] = self.chunk_size
        native = super()._normalize_result(result, ctx)
        before = copy.deepcopy(m['schedule'])
        if self.settings.get('observe_at_release_end', False):
            rows, selected, m['schedule'], decision = release_end_rows(
                native['actions'], m['state'], m['arm'], before, self.settings, m['n'])
        elif self.settings.get('observe_after_release_clearance', False):
            rows, selected, m['schedule'], m['open_wrist_anchor'], decision = release_feedback_rows(
                native['actions'], m['state'], m['arm'], before, self.settings, m['n'],
                m['open_wrist_anchor'], self._release_kin)
        else:
            rows, selected, m['schedule'] = release_rows(native['actions'], m['state'], m['arm'], before, self.settings, m['n'])
        if len(rows) != self.chunk_size:
            self._chunk_buffers.pop(sid, None)
            self._session_chunk_sizes[sid] = len(rows)
            self._try_serve_from_buffer(ctx)
        return {**native, 'actions': rows, 'joints': rows}
