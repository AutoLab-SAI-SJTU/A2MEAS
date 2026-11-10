"""Stabilize one RGB-associated native grasp, then return to the native policy."""
import os

import numpy as np
import torch
from PIL import Image

from .actor import Actor
from .geometry import HEAD_C, HEAD_RCW, RightArmKinematics, camera_k, ee_world, project
from .recovery import ALIASES, move_pose, object_prompt
from .vision import MugRGBPerception


def grasp_sources(vision, image, prompts):
    """Keep the frozen full-frame/two-tile detector order and duplicate selection."""
    found = []
    old = vision.processor.confidence_threshold
    vision.processor.confidence_threshold = .2
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for x0, y0, x1, y1 in [(0, 0, image.shape[1], image.shape[0]),
                                   (0, 64, 176, 240), (144, 64, 320, 240)]:
                encoded = vision.processor.set_image(Image.fromarray(image[y0:y1, x0:x1]))
                for prompt in prompts:
                    raw = vision.processor.set_text_prompt(prompt, state=encoded)
                    d = {k: raw[k].detach().float().cpu().numpy() for k in ['scores', 'boxes', 'masks']}
                    for score, box, mask in zip(d['scores'], d['boxes'], d['masks']):
                        box = box + np.array([x0, y0, x0, y0])
                        center = (box[:2] + box[2:]) / 2
                        size = box[2:] - box[:2]
                        area = int((np.squeeze(mask) > .5).sum())
                        if score < .3 or not (12 < center[0] < 310 and 35 < center[1] < 235):
                            continue
                        if not 45 < np.prod(size) < 40000 or area < 20:
                            continue
                        item = {'pixel': center, 'box': box, 'confidence': float(score), 'area': area}
                        duplicate = next((i for i, x in enumerate(found)
                                          if np.linalg.norm(center - x['pixel']) < 12), None)
                        if duplicate is None:
                            found.append(item)
                        elif score > found[duplicate]['confidence']:
                            found[duplicate] = item
    finally:
        vision.processor.confidence_threshold = old
    return [x for x in found if x['area'] < 6000]


class ProactiveGrasp(Actor):
    def _load_model(self):
        super()._load_model()
        self.vision_model = self.vision(MugRGBPerception)
        urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
        self.kin = {s: RightArmKinematics(urdf, side=s) for s in ['left', 'right']}
        self.projection = camera_k(320, 240) @ np.c_[HEAD_RCW, -HEAD_RCW @ HEAD_C]
        self._sessions = {}

    async def on_episode_start(self, config, ctx):
        self._sessions[ctx.session_id] = {'n': 0, 'source': None, 'done': False, 'phase': None}

    async def on_episode_end(self, result, ctx):
        self._sessions.pop(ctx.session_id, None)

    def _normalize_result(self, result, ctx):
        m = self._sessions[ctx.session_id]
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        forecast = np.asarray(result['actions'])
        if m['source'] is None and m['n'] <= 240:
            choices = []
            for side, sl, grip in [('left', slice(0, 6), 6), ('right', slice(7, 13), 13)]:
                closes = np.flatnonzero(forecast[:, grip] < .2)
                if m['state'][grip] < .2 or not len(closes) or int(closes[0]) >= self.chunk_size:
                    continue
                close = int(closes[0])
                window = [(i, ee_world(self.kin[side], forecast[i, sl]))
                          for i in range(close, min(close + 8, len(forecast)))]
                _, pose = min(window, key=lambda x: x[1][2, 3])
                pixel = project(self.projection, pose[:3, 3] + .12 * pose[:3, 0])
                for source in m['initial_sources']:
                    distance = float(np.linalg.norm(np.maximum(
                        np.maximum(source['box'][:2] - pixel, pixel - source['box'][2:]), 0)))
                    if distance < 20 and source['confidence'] >= .55:
                        choices.append((distance + .01 * np.linalg.norm(pixel - source['pixel']),
                                        side, source, pose.copy()))
            if choices:
                _, m['arm'], m['source'], m['grasp_pose'] = min(choices, key=lambda x: x[0])
        native = super()._normalize_result(result, ctx)
        if m['source'] is None or m['done']:
            return native
        m['done'] = True
        if m['state'][6 if m['arm'] == 'left' else 13] < .8:
            return native
        try:
            actions = self.plan(m['state'], m)
        except ValueError as exc:
            return native
        m['phase'] = 'script'
        self._chunk_buffers.pop(ctx.session_id, None)
        self._session_chunk_sizes[ctx.session_id] = len(actions)
        self._try_serve_from_buffer(ctx)
        return {**native, 'actions': actions, 'joints': actions}

    def plan(self, state, m):
        side = m['arm']
        sl, grip = (slice(0, 6), 6) if side == 'left' else (slice(7, 13), 13)
        current = state.copy()
        current[grip] = 1.
        rows = [np.repeat(current[None], 6, axis=0)]
        pose = ee_world(self.kin[side], current[sl])
        pick = m['grasp_pose'][:3, 3].copy()
        for point, rotation, hold in [(pose[:3, 3] + [0, 0, .04], None, 4),
                (pick + [0, 0, .05], m['grasp_pose'][:3, :3], 6),
                (pick, m['grasp_pose'][:3, :3], 8)]:
            segment = move_pose(self.kin[side], current, side, point, R=rotation, grip=1., hold=hold)
            rows.append(segment)
            current = segment[-1].copy()
        current[grip] = 0.
        rows.append(np.repeat(current[None], 12, axis=0))
        rows.append(move_pose(self.kin[side], current, side, pick + [0, 0, .06],
                              R=m['grasp_pose'][:3, :3], grip=0., hold=8))
        actions = np.concatenate(rows)
        if len(actions) + 80 > self.settings['max_steps'] - m['n']:
            raise ValueError('Insufficient original action budget for observed grasp')
        return actions

    async def on_observation(self, obs, ctx):
        sid = ctx.session_id
        m = self._sessions[sid]
        m['n'] += 1
        m['state'] = self._state_vector_from_obs(obs).copy()
        if m['n'] == 1:
            name = object_prompt(obs.get('task_description', ''))
            prompts = ALIASES.get(name, [name] if name else [])
            image = self._image_from_obs_dict(obs['images'], self.image_name, 0)
            with self.server._actor_load_context():
                m['initial_sources'] = grasp_sources(self.vision_model, image, prompts)
        buffer = self._chunk_buffers.get(sid)
        if m['phase'] == 'script' and (buffer is None or buffer.empty):
            m['phase'] = None
            self._image_histories.pop(sid, None)
            self._state_histories.pop(sid, None)
        return await super().on_observation(obs, ctx)
