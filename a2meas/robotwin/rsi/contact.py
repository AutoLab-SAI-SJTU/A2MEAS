"""Stall-triggered local contact search; task geometry lives in JSON."""
from collections import deque
import os
import numpy as np
from .actor import Actor
from .geometry import BASE_R, BASE_P, RightArmKinematics, ee_world


class ContactSearch(Actor):
    def _load_model(self):
        super()._load_model()
        urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
        self.kinematics = {side: RightArmKinematics(urdf, side=side) for side in ('left', 'right')}
        self._sessions = {}

    async def on_episode_start(self, config, ctx):
        self._sessions[ctx.session_id] = {'n': 0, 'states': deque(maxlen=self.settings['stall_window']), 'done': False, 'phase': None}

    def _normalize_result(self, result, ctx):
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        return super()._normalize_result(result, ctx)

    def trajectory(self, state, side, target_world):
        cfg = self.settings
        sl, grip = (slice(0, 6), 6) if side == 'left' else (slice(7, 13), 13)
        start = state[sl]
        pose, _ = self.kinematics[side].forward(start)
        pose[:3, 3] = BASE_R.T @ (np.asarray(target_world) - BASE_P)
        q = self.kinematics[side].solve_pose(start, pose)
        count = max(cfg['minimum_move_steps'], int(np.ceil(np.max(np.abs(q - start)) / cfg['joint_increment'])))
        if count > cfg['maximum_move_steps']:
            raise ValueError('Local contact search exceeds joint motion bound')
        rows = []
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            row = state.copy()
            row[sl] = start + alpha * (q - start)
            row[grip] = 0.
            rows.append(row)
        rows.extend(rows[-1].copy() for _ in range(cfg['hold_steps']))
        return np.stack(rows)

    def search_offset(self, memory, state, phase):
        original = np.asarray(self.settings['offsets'][phase], dtype=float)
        if not self.settings.get('command_frame_enabled', False):
            return original
        if 'command_search_rotation' not in memory:
            sl = slice(0, 6) if memory['side'] == 'left' else slice(7, 13)
            forward = ee_world(self.kinematics[memory['side']], state[sl])[:3, 0]
            norm = float(np.linalg.norm(forward[:2]))
            rotation = np.eye(3)
            if norm >= 0.5:
                direction = forward[:2] / norm
                if direction[1] < 0:
                    direction = -direction
                lateral = np.array([direction[1], -direction[0], 0.])
                rotation = np.stack([lateral, np.r_[direction, 0.], [0., 0., 1.]], axis=1)
            memory['command_search_rotation'] = rotation
        return memory['command_search_rotation'] @ original

    async def on_observation(self, obs, ctx):
        sid = ctx.session_id
        memory, cfg = self._sessions[sid], self.settings
        memory['n'] += 1
        state = self._state_vector_from_obs(obs).copy()
        memory['states'].append(state)
        memory.setdefault('initial', state.copy())
        begin, end, interval = cfg['trigger']
        if memory['phase'] is None and not memory['done'] and begin <= memory['n'] <= end and memory['n'] % interval == 0 and len(memory['states']) == cfg['stall_window']:
            instruction = obs.get('task_description', '').lower()
            movement = [np.linalg.norm(state[:6] - memory['initial'][:6]), np.linalg.norm(state[7:13] - memory['initial'][7:13])]
            side = 'left' if 'left' in instruction else 'right' if 'right' in instruction else ('left', 'right')[int(np.argmax(movement))]
            sl, grip = (slice(0, 6), 6) if side == 'left' else (slice(7, 13), 13)
            spread = np.max(np.linalg.norm(np.stack(memory['states'])[:, sl] - state[sl], axis=1))
            if movement[0 if side == 'left' else 1] > cfg['activation_threshold'] and state[grip] < cfg['gripper_threshold'] and spread < cfg['stall_threshold']:
                memory.update(side=side, anchor=ee_world(self.kinematics[side], state[sl])[:3, 3], phase=0)
                self._chunk_buffers.pop(sid, None)
        if memory['phase'] is not None and (sid not in self._chunk_buffers or self._chunk_buffers[sid].empty):
            phase = memory['phase']
            try:
                rows = self.trajectory(state, memory['side'], memory['anchor'] + self.search_offset(memory, state, phase))
            except ValueError:
                memory.update(phase=None, done=True)
            else:
                memory['phase'] = phase + 1 if phase + 1 < len(cfg['offsets']) else None
                memory['done'] = memory['phase'] is None
                self._try_serve_from_buffer(ctx)
                self._append_observation(obs, ctx)
                self._session_chunk_sizes[sid] = len(rows)
                await self._process_and_send({'actions': rows, 'joints': rows}, ctx)
                return
        await super().on_observation(obs, ctx)

    async def on_episode_end(self, result, ctx):
        self._sessions.pop(ctx.session_id, None)
