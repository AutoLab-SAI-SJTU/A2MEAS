"""Small recovery actors sharing a single native model and batch queue."""


class Actor:
    def __init__(self, server, settings):
        self.server = server
        self.settings = settings
        self.chunk_size = settings['execution_chunk_size']
        self._sessions = {}

    def __getattr__(self, name):
        return getattr(self.server, name)

    def _load_model(self):
        self.server._load_model()

    async def on_episode_start(self, config, ctx):
        pass

    async def on_episode_end(self, result, ctx):
        self._sessions.pop(ctx.session_id, None)

    async def on_observation(self, obs, ctx):
        await self.server._native_observation(obs, ctx)

    def _normalize_result(self, result, ctx):
        return self.server._normalize_native(result, ctx)

    def vision(self, cls):
        # Preserve policy RNG while lazily loading one shared segmentation model.
        from functools import wraps
        import torch
        if self.server._vision_backend is None:
            from .vision import MugRGBPerception
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.server._vision_backend = MugRGBPerception()
        detector = cls.__new__(cls)
        detector.__dict__.update(self.server._vision_backend.__dict__)
        detect = detector.detect

        @wraps(detect)
        def detect_on_shared_device(*args, **kwargs):
            # SAM also runs during observations, outside result normalization.
            # Its GPU work must finish before the inference thread captures.
            with self.server._actor_load_context():
                return detect(*args, **kwargs)

        detector.detect = detect_on_shared_device
        return detector

    async def _send_script(self, obs, ctx, actions):
        sid = ctx.session_id
        self._chunk_buffers.pop(sid, None)
        self._session_chunk_sizes[sid] = len(actions)
        self._try_serve_from_buffer(ctx)
        self._append_observation(obs, ctx)
        await self._process_and_send({'actions': actions}, ctx)
