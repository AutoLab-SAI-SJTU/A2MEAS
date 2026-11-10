"""Counted upward withdrawal of an RGB-associated, already-open bread hand.

This is own-arm stall recovery, not a declaration of bread/pan contact or pose.
It never opens a closed gripper and never moves the other arm during recovery.
"""
from collections import deque
import os

import numpy as np

from .actor import Actor
from .geometry import HEAD_C, HEAD_RCW, RightArmKinematics, camera_k, ee_world, project
from .recovery import move_pose
from .vision import MugRGBPerception


def box_gap(pixel, box):
    box = np.asarray(box)
    return float(np.linalg.norm(np.maximum(np.maximum(box[:2] - pixel, pixel - box[2:]), 0.)))


def _pan_only_role(contacts, sources):
    if sources['bread'] or len(contacts) != 2:
        return None
    pairs = []
    for pan_arm, bread_arm in [('left', 'right'), ('right', 'left')]:
        for pan in sources['frying pan']:
            box = np.asarray(pan['box'])
            if pan['confidence'] < .55 or not 45 < float(np.prod(box[2:] - box[:2])) < 40000:
                continue
            pan_gap = box_gap(contacts[pan_arm], box)
            other_gap = box_gap(contacts[bread_arm], box)
            if pan_gap < 20 and other_gap > 20:
                pairs.append((pan_gap + 2 * (1 - pan['confidence']), dict(
                    bread_arm=bread_arm, pan_arm=pan_arm, bread_box=None, pan_box=box.tolist(),
                    pan_gap_px=pan_gap, other_contact_pan_gap_px=other_gap,
                    projected_native_contacts={k: v.tolist() for k, v in contacts.items()},
                    role_source='initial_RGB_pan_unique_contact_and_two_hand_task',
                    original_bread_detection_empty=True)))
    if not pairs or len({row['bread_arm'] for _, row in pairs}) != 1:
        return None
    return min(pairs, key=lambda pair: pair[0])[1]


def associate_bread_arm(raw, state, sources, kin, projection, *, pan_only_role_fallback=False):
    contacts = {}
    for side, sl, grip in [('left', slice(0, 6), 6), ('right', slice(7, 13), 13)]:
        closes = np.flatnonzero(raw[:, grip] < .2)
        if state[grip] < .8 or not len(closes) or int(closes[0]) >= 40:
            continue
        start = int(closes[0])
        pose = min((ee_world(kin[side], raw[i, sl]) for i in range(start, min(start + 8, len(raw)))),
                   key=lambda t: t[2, 3])
        contacts[side] = project(projection, (pose @ np.array([.14, 0, 0, 1.]))[:3])
    if len(contacts) != 2:
        return None
    pairs = []
    for side, other in [('left', 'right'), ('right', 'left')]:
        for bread in sources['bread']:
            for pan in sources['frying pan']:
                a, b = np.asarray(bread['box']), np.asarray(pan['box'])
                intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.))
                union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
                if intersection / max(union, 1.) > .05 or np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) < 45:
                    continue
                bread_gap = box_gap(contacts[side], a)
                pan_gap = box_gap(contacts[other], b)
                if (bread_gap < 20 and pan_gap < 20 and box_gap(contacts[side], b) > 20
                        and box_gap(contacts[other], a) > 20):
                    score = bread_gap + pan_gap + 2 * (2 - bread['confidence'] - pan['confidence'])
                    pairs.append((score, dict(bread_arm=side, pan_arm=other,
                        bread_box=a.tolist(), pan_box=b.tolist(), bread_gap_px=bread_gap, pan_gap_px=pan_gap,
                        projected_native_contacts={k: v.tolist() for k, v in contacts.items()})))
    if not pairs or len({r['bread_arm'] for _, r in pairs}) != 1:
        return _pan_only_role(contacts, sources) if pan_only_role_fallback else None
    return min(pairs, key=lambda x: x[0])[1]


def retreat_eligibility(memory, raw, kin, settings):
    if (memory['done'] or memory['source'] is None or not 160 <= memory['n'] <= 360
            or len(memory['states']) < 33):
        return None
    side = memory['source']['bread_arm']
    sl, grip, other = (slice(0, 6), 6, 13) if side == 'left' else (slice(7, 13), 13, 6)
    state = memory['state']
    if not memory['armed'][grip] or memory['open_runs'][grip] < 16 or state[other] >= .2:
        return None
    if float(np.min(raw[:40, grip])) <= .7:
        return None
    history = np.asarray(memory['states'])
    if np.max(np.linalg.norm(history[:, sl] - state[sl], axis=1)) >= .1:
        return None
    current = ee_world(kin[side], state[sl])
    old = np.array([ee_world(kin[side], q)[:3, 3] for q in history[:, sl]])
    planned = np.array([ee_world(kin[side], q)[:3, 3] for q in raw[:40, sl]])
    past_spread = float(np.max(np.linalg.norm(old - current[:3, 3], axis=1)))
    planned_spread = float(np.max(np.linalg.norm(planned - current[:3, 3], axis=1)))
    planned_up = float(np.max(planned[:, 2] - current[2, 3]))
    if past_spread >= .008 or planned_spread >= .025 or planned_up >= .015:
        return None
    return dict(arm=side, past33_command_wrist_spread_m=past_spread,
                native40_planned_wrist_spread_m=planned_spread,
                native40_max_planned_up_m=planned_up,
                consecutive_open_issued_targets=memory['open_runs'][grip])


def open_hand_retreat(kin, state, side, remaining):
    sl, grip = (slice(0, 6), 6) if side == 'left' else (slice(7, 13), 13)
    if state[grip] <= .8:
        raise ValueError('Retreat only supports an already-open native hand')
    pose = ee_world(kin[side], state[sl])
    rows = move_pose(kin[side], state, side, pose[:3, 3] + [0, 0, .06],
                     R=pose[:3, :3], grip=1., hold=8)
    if len(rows) + 80 > remaining:
        raise ValueError('Insufficient original native budget plus80 native-resume reserve')
    return rows


class BreadReleaseRetreat(Actor):
    def _load_model(self):
        super()._load_model()
        self.vision_model = self.vision(MugRGBPerception)
        urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
        self.kin = {s: RightArmKinematics(urdf, side=s) for s in ['left', 'right']}
        self.projection = camera_k(320, 240) @ np.c_[HEAD_RCW, -HEAD_RCW @ HEAD_C]
        self._sessions = {}


    async def on_episode_start(self, config, ctx):
        self._sessions[ctx.session_id] = dict(n=0, source=None, done=False, phase=None,
            states=deque(maxlen=33), closed_runs={6: 0, 13: 0}, open_runs={6: 0, 13: 0},
            armed={6: False, 13: False})

    async def on_observation(self, obs, ctx):
        memory = self._sessions[ctx.session_id]
        memory['n'] += 1
        state = self._state_vector_from_obs(obs).copy()
        memory['state'] = state
        memory['states'].append(state)
        for grip in [6, 13]:
            memory['closed_runs'][grip] = memory['closed_runs'][grip] + 1 if state[grip] < .2 else 0
            memory['open_runs'][grip] = memory['open_runs'][grip] + 1 if state[grip] > .8 else 0
            if memory['closed_runs'][grip] >= 16:
                memory['armed'][grip] = True
        if memory['n'] == 1:
            image = self._image_from_obs_dict(obs['images'], self.image_name, 0)
            sources = {}
            for prompt in ['bread', 'frying pan']:
                detected = self.vision_model.detect(image, prompt)
                sources[prompt] = [dict(box=box.tolist(), confidence=float(score))
                    for score, box in zip(detected['scores'], detected['boxes'])
                    if float(score) >= .55 and 45 < float(np.prod(box[2:] - box[:2])) < 40000]
            memory['initial_sources'] = sources
        buffer = self._chunk_buffers.get(ctx.session_id)
        if memory['phase'] == 'script' and (buffer is None or buffer.empty):
            memory['phase'] = None
            self._image_histories.pop(ctx.session_id, None)
            self._state_histories.pop(ctx.session_id, None)
        return await super().on_observation(obs, ctx)

    def _normalize_result(self, result, ctx):
        memory = self._sessions[ctx.session_id]
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        raw = np.asarray(result['actions'])
        if memory['source'] is None and memory['n'] <= 120:
            source = associate_bread_arm(raw, memory['state'], memory['initial_sources'], self.kin, self.projection,
                pan_only_role_fallback=self.settings.get('pan_only_role_fallback', False))
            if source is not None:
                memory['source'] = source
        native = super()._normalize_result(result, ctx)
        eligible = retreat_eligibility(memory, raw, self.kin, self.settings)
        if eligible is None:
            return native
        memory['done'] = True
        try:
            actions = open_hand_retreat(self.kin, memory['state'], eligible['arm'],
                                        self.settings['max_steps'] - memory['n'])
        except ValueError as exc:
            return native
        memory['phase'] = 'script'
        self._chunk_buffers.pop(ctx.session_id, None)
        self._session_chunk_sizes[ctx.session_id] = len(actions)
        self._try_serve_from_buffer(ctx)
        return {**native, 'actions': actions, 'joints': actions}

    async def on_episode_end(self, result, ctx):
        self._sessions.pop(ctx.session_id, None)
