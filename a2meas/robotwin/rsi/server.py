"""RGB and kinematics recovery over the checkpoint-native policy server."""
from contextlib import contextmanager
from functools import cached_property
from threading import RLock
import numpy as np
from vla_eval.model_servers.predict import PredictModelServer
from a2meas.robotwin.server import RoboTwinModelServer
from a2meas.robotwin.rsi.routing import RSIRoutingMixin
from a2meas.robotwin.rsi.sampling import FixedGaussianSource, SessionGaussianSource

class RoboTwinRSIServer(RSIRoutingMixin, RoboTwinModelServer):

    def __init__(self, *args, **kwargs):
        self._actors, self._routes, self._vision_backend = ({}, {}, None)
        super().__init__(*args, **kwargs)

    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_session_gaussian_sources'):
            self._session_gaussian_sources = {}
        sources = self._session_gaussian_sources
        sources.pop(ctx.session_id, None)
        await super().on_episode_start(config, ctx)
        actor = self._routes.get(ctx.session_id)
        if actor is not None and actor.settings.get('session_gaussian_noise', False):
            sources[ctx.session_id] = SessionGaussianSource(self._torch)

    async def on_episode_end(self, result, ctx):
        try:
            await super().on_episode_end(result, ctx)
        finally:
            getattr(self, '_session_gaussian_sources', {}).pop(ctx.session_id, None)

    @cached_property
    def _rsi_cuda_lock(self):
        return RLock()

    def _actor_load_context(self):
        return self._rsi_cuda_lock

    @cached_property
    def _fixed_gaussian_source(self):
        return FixedGaussianSource(self._torch, self._policy.config.output_features['action'].shape[0])

    def _load_model(self):
        if getattr(self, '_rsi_model_ready', False):
            return
        self.image_history_num_frames = len(self.offsets)
        if self.image_history_num_frames != 2 or self.continuous_inference:
            raise ValueError('RSI requires synchronous two-frame inference')
        self._rsi_model_ready = True

    @contextmanager
    def _using_solver_steps(self, steps):
        if steps is None:
            yield
            return
        config = self._policy.config
        if self._policy.model.config is not config:
            raise ValueError('Solver override requires shared native model configuration')
        previous = config.num_inference_steps
        try:
            config.num_inference_steps = steps
            yield
        finally:
            config.num_inference_steps = previous

    @contextmanager
    def _using_recent_history(self, enabled):
        if not enabled:
            yield
            return
        if (self.offsets, self._history_size) != ([-30, 0], 31):
            raise ValueError('recent_visual_history requires the native STM2 history contract')
        previous = self.offsets
        try:
            self.offsets = [-15, 0]
            yield
        finally:
            self.offsets = previous

    def predict_batch(self, obs_batch, ctx_batch):
        with self._rsi_cuda_lock:
            routes = getattr(self, '_routes', {})
            settings = [routes[ctx.session_id].settings if ctx.session_id in routes else {} for ctx in ctx_batch]
            single = [bool(row.get('single_sample_inference', False)) for row in settings]
            steps = [row.get('num_inference_steps') for row in settings]
            if any((value is not None for value in steps)):
                steps = [self._policy.config.num_inference_steps if value is None else value for value in steps]
            if len(ctx_batch) > 1 and (any(single) or len(set(steps)) > 1):
                ordinary = {}
                for i, enabled in enumerate(single):
                    if not enabled:
                        ordinary.setdefault(steps[i], []).append(i)
                groups = list(ordinary.values()) + [[i] for i, enabled in enumerate(single) if enabled]
                results = [None] * len(ctx_batch)
                for indices in groups:
                    predicted = self.predict_batch([obs_batch[i] for i in indices], [ctx_batch[i] for i in indices])
                    for i, result in zip(indices, predicted, strict=True):
                        results[i] = result
                return results
            recent = any((row.get('recent_visual_history', False) for row in settings))
            if recent and len(ctx_batch) != 1:
                raise ValueError('recent_visual_history requires an isolated native request')
            with self._using_solver_steps(steps[0] if steps else None), self._using_recent_history(recent):
                enabled = [bool(row.get('fixed_gaussian_noise', False)) for row in settings]
                stream = any((row.get('session_gaussian_noise', False) for row in settings))
                if not any(enabled) and (not stream):
                    return super().predict_batch(obs_batch, ctx_batch)
                model = self._policy.model
                native = model.sample_noise
                if stream:
                    if len(ctx_batch) != 1 or any(enabled):
                        raise ValueError('Session Gaussian sampling requires an isolated native request')
                    source = self._session_gaussian_sources[ctx_batch[0].session_id]
                    model.sample_noise = lambda shape, device: source.sample(native, shape, device)
                else:
                    source = self._fixed_gaussian_source
                    model.sample_noise = lambda shape, device: source.sample(native, shape, device, enabled)
                try:
                    return super().predict_batch(obs_batch, ctx_batch)
                finally:
                    model.sample_noise = native

    def _normalize_result(self, result, ctx):
        with self._rsi_cuda_lock:
            return super()._normalize_result(result, ctx)

    def _normalize_native(self, result, ctx):
        native = PredictModelServer._normalize_result(self, result, ctx)
        actions = native['actions'].copy()
        return {**native, 'actions': actions, 'joints': actions}

    def _prediction_result(self, action):
        return {'actions': action.copy(), 'joints': action.copy()}

    @property
    def _image_histories(self):
        return self._histories
    image_name, wrist_image_name, right_wrist_image_name = ('head_camera', 'left_camera', 'right_camera')

    @staticmethod
    def _image_from_obs_dict(images, name, fallback_idx):
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'Expected native uint8 RGB for {name}')
        return image
