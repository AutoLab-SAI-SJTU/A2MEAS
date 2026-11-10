"""Counted interpolation and grasp/release settling, with per-session state."""
import numpy as np
from .actor import Actor

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class MotionTiming(Actor):
    def _load_model(self):
        super()._load_model()
        self._sessions = {}

    async def on_episode_start(self, config, ctx):
        self._sessions[ctx.session_id] = {'n': 0, 'settled': {6: False, 13: False}}

    async def on_observation(self, obs, ctx):
        memory = self._sessions[ctx.session_id]
        memory['n'] += 1
        memory['state'] = self._state_vector_from_obs(obs).copy()
        await super().on_observation(obs, ctx)

    def _normalize_result(self, result, ctx):
        sid, cfg = ctx.session_id, self.settings
        memory = self._sessions[sid]
        self._session_chunk_sizes[sid] = self.chunk_size
        native = super()._normalize_result(result, ctx)
        previous = memory['state'].copy()
        closed = {g: previous[g] < .2 for g in (6, 13)}
        rows = []
        for target in np.asarray(native['actions']):
            if cfg['interpolate']:
                count = min(96, max(1, int(np.ceil(np.max(np.abs(target[JOINTS] - previous[JOINTS])) / cfg['max_joint_increment']))))
                rows.extend(previous + alpha * (target - previous) for alpha in np.linspace(0, 1, count + 1)[1:])
                previous = target.copy()
                for grip in (6, 13):
                    if not memory['settled'][grip] and target[grip] < .15:
                        previous[grip] = 0.
                        rows.extend(previous.copy() for _ in range(cfg['grasp_hold_steps']))
                        memory['settled'][grip] = True
            else:
                closing = [g for g in (6, 13) if target[g] < .2 and not closed[g]]
                opening = [g for g in (6, 13) if target[g] > .8 and closed[g]]
                if closing and cfg['pre_grasp_hold_steps']:
                    approach = target.copy()
                    approach[closing] = 1.
                    rows.extend(approach.copy() for _ in range(cfg['pre_grasp_hold_steps']))
                rows.append(target.copy())
                if closing:
                    rows.extend(target.copy() for _ in range(cfg['grasp_hold_steps']))
                if opening:
                    rows.extend(target.copy() for _ in range(cfg['release_hold_steps']))
                for g in closing:
                    closed[g] = True
                for g in opening:
                    closed[g] = False
        rows = np.asarray(rows, dtype=np.float32)[:max(1, cfg['max_steps'] + 1 - memory['n'])]
        self._chunk_buffers.pop(sid, None)
        self._session_chunk_sizes[sid] = len(rows)
        self._try_serve_from_buffer(ctx)
        return {**native, 'actions': rows, 'joints': rows}

    async def on_episode_end(self, result, ctx):
        self._sessions.pop(ctx.session_id, None)
