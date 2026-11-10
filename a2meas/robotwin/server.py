"""RoboTwin adapter for the external vla-evaluation-harness protocol.

Load normalization and architecture from the delivered checkpoint. Training
datasets and train-only depth teachers are never opened by this server.
"""
from __future__ import annotations
import json
import os
from collections import deque
import numpy as np
from vla_eval.model_servers.predict import PredictModelServer
from a2meas import PROJECT_ROOT
from a2meas.robotwin.schema import ROBOTWIN_CAMERAS
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
CAMERAS = ('head_camera', 'left_camera', 'right_camera')

def checkpoint_stats(checkpoint, filename, registry):
    from safetensors.torch import load_file
    pipeline = json.loads((checkpoint / filename).read_text())
    steps = [s for s in pipeline['steps'] if s.get('registry_name') == registry]
    if len(steps) != 1 or 'state_file' not in steps[0]:
        raise ValueError(f'Expected one saved {registry} with statistics in {filename}')
    stats = {}
    state_file = (checkpoint / steps[0]['state_file']).resolve(strict=True)
    state_file.relative_to(checkpoint.resolve())
    for name, tensor in load_file(str(state_file)).items():
        feature, stat = name.rsplit('.', 1)
        stats.setdefault(feature, {})[stat] = tensor
    return stats

class RoboTwinModelServer(PredictModelServer):

    def __init__(self, checkpoint: str, rynnbrain_path: str, lawam_config_path: str='configs/shared/world_model.json', lawam_vision_path: str='configs/shared/dinov3', device: str='cuda', chunk_size: int=40, image_history_num_frames: int=0, image_history_stride: int=30, num_inference_steps: int=0, inference_seed: int=-1, **kwargs):
        super().__init__(chunk_size=chunk_size, **kwargs)
        if self.continuous_inference or self.action_ensemble != 'newest':
            raise ValueError('RoboTwin evaluation requires synchronous newest-chunk execution')
        self.checkpoint = (PROJECT_ROOT / checkpoint).resolve(strict=True)
        rynnbrain_path = str((PROJECT_ROOT / rynnbrain_path).resolve(strict=True))
        lawam_config_path = str((PROJECT_ROOT / lawam_config_path).resolve(strict=True))
        lawam_vision_path = str((PROJECT_ROOT / lawam_vision_path).resolve(strict=True))
        self.device = device
        self._histories = {}
        self._state_histories = {}
        self._graph_denoiser = None
        self._initialize_policy(rynnbrain_path, lawam_config_path, image_history_num_frames, image_history_stride, num_inference_steps, inference_seed, lawam_vision_path)

    def _initialize_policy(self, rynnbrain, lawam_config, frames, stride, steps, seed, lawam_vision):
        import torch
        from a2meas.robotwin.schema import PreTrainedConfig
        from a2meas.robotwin.model.policy import A2WLAPolicy
        from a2meas.robotwin.processor import make_processors
        self._torch = torch
        config = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        self.camera_keys = ROBOTWIN_CAMERAS
        self.dtype = config.dtype
        config.device = self.device
        config.rynnbrain_path = rynnbrain
        config.lawam_config_path = lawam_config
        config.lawam_vision_path = lawam_vision
        if steps and steps != config.num_inference_steps:
            raise ValueError('NUM_INFERENCE_STEPS differs from the checkpoint')
        if frames and frames != 2:
            raise ValueError('RoboTwin requires two observation frames')
        if stride != 30:
            raise ValueError('RoboTwin requires an observation history stride of 30')
        self.offsets = list(config.short_term_memory_inference_delta_indices)
        if not 1 <= self.chunk_size <= config.chunk_size:
            raise ValueError('Execution chunk must fit the checkpoint prediction horizon')
        self._history_size = 1 - min(self.offsets)
        pre_stats = checkpoint_stats(self.checkpoint, 'policy_preprocessor.json', 'normalizer_processor')
        post_stats = checkpoint_stats(self.checkpoint, 'policy_postprocessor.json', 'unnormalizer_processor')
        self._policy = A2WLAPolicy.from_pretrained(str(self.checkpoint), config=config)
        self._preprocessor, self._postprocessor = make_processors(config, pre_stats, post_stats)
        if seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if os.environ.get('CUDA_GRAPH_DENOISE', 'false') == 'true':
            from a2meas.robotwin.graph_denoise import configure_graph_denoiser
            self._graph_denoiser = configure_graph_denoiser(self._policy, torch)
            self._policy.model.denoise_step = self._graph_denoiser

    async def on_episode_start(self, config, ctx):
        self._histories.pop(ctx.session_id, None)
        self._state_histories.pop(ctx.session_id, None)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result, ctx):
        try:
            await super().on_episode_end(result, ctx)
        finally:
            self._histories.pop(ctx.session_id, None)
            self._state_histories.pop(ctx.session_id, None)

    @staticmethod
    def _state_vector_from_obs(obs):
        return np.asarray(obs['joint_state'], dtype=np.float32)

    def _validate_observation(self, obs):
        instruction = obs.get('task_description')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Missing RoboTwin task instruction')
        images = obs.get('images', {})
        if any((name not in images for name in CAMERAS)):
            raise ValueError('Every control step requires all three named RGB cameras')
        for name in CAMERAS:
            image = np.asarray(images[name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or (0 in image.shape):
                raise ValueError(f'Expected nonempty HWC uint8 RGB for {name}')
        state = self._state_vector_from_obs(obs)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 14D state')

    def _append_observation(self, obs, ctx):
        state = self._state_vector_from_obs(obs)
        history = self._histories.setdefault(ctx.session_id, {name: deque(maxlen=self._history_size) for name in CAMERAS})
        for name in CAMERAS:
            history[name].append(np.asarray(obs['images'][name]).copy())
        self._state_histories.setdefault(ctx.session_id, deque(maxlen=1)).append(state.copy())

    def _prepare_sample(self, obs, ctx):
        self._append_observation(obs, ctx)
        sample = {'task': obs['task_description']}
        for name, key in zip(CAMERAS, self.camera_keys):
            history = list(self._histories[ctx.session_id][name])
            newest = len(history) - 1
            selected = [history[max(0, newest + offset)] for offset in self.offsets]
            frames = np.stack(selected).astype(np.float32)
            np.divide(frames, 255.0, out=frames)
            packed = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
            sample[key] = self._torch.from_numpy(packed)
        states = list(self._state_histories[ctx.session_id])
        sample['observation.state'] = self._torch.from_numpy(states[-1])
        return sample

    async def on_observation(self, obs, ctx):
        self._validate_observation(obs)
        await self._on_validated_observation(obs, ctx)

    async def _on_validated_observation(self, obs, ctx):
        buffer = self._chunk_buffers.get(ctx.session_id)
        if buffer is not None and (not buffer.empty):
            self._append_observation(obs, ctx)
            await ctx.send_action({'actions': self._try_serve_from_buffer(ctx)})
        else:
            await super().on_observation(obs, ctx)

    def predict_batch(self, obs_batch, ctx_batch):
        torch = self._torch
        samples = [self._prepare_sample(obs, ctx) for obs, ctx in zip(obs_batch, ctx_batch, strict=True)]
        with torch.inference_mode():
            inputs = self._preprocessor(torch.utils.data.default_collate(samples))
            with torch.autocast(device_type=torch.device(self.device).type, dtype=getattr(torch, self.dtype), enabled=self.dtype != 'float32'):
                normalized = self._policy.predict_action_chunk(inputs)
            actions = self._postprocessor(normalized).detach().float().cpu().numpy()
        if actions.shape != (len(samples), self._policy.config.chunk_size, 14) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid physical action chunk: {actions.shape}')
        return [self._prediction_result(action) for action in actions]

    def _prediction_result(self, action):
        return {'actions': action[:self.chunk_size].copy()}

    def get_action_spec(self):
        return {'actions': DimSpec('actions', 14, 'joint_positions')}

    def get_observation_spec(self):
        return {**{name: IMAGE_RGB for name in CAMERAS}, 'language': LANGUAGE, 'joint_state': STATE_JOINT}
