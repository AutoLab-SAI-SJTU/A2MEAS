"""Task routing for the shared native policy and recovery actors."""
from contextlib import nullcontext
import importlib
import os

from .config import load_config, ACTORS




class RSIRoutingMixin:
    def _actor_load_context(self):
        return nullcontext()

    async def on_episode_start(self, config, ctx):
        await super().on_episode_start(config, ctx)
        try:
            if not hasattr(self, '_actors'):
                self._actors, self._routes, self._vision_backend = {}, {}, None
            self._session_chunk_sizes.pop(ctx.session_id, None)
            registry = load_config()
            task = config.get('task', {})
            name = task.get('name', '') if isinstance(task, dict) else str(task)
            settings = registry['tasks'].get(name)
            if not registry.get('enabled', True) or os.environ.get('ROBOTWIN_RSI', 'true').lower() == 'false':
                settings = None
            if settings is None or not settings.get('enabled', True):
                self._routes.pop(ctx.session_id, None)
                return
            if name not in self._actors:
                with self._actor_load_context():
                    module, actor_name = ACTORS[settings['kind']]
                    cls = getattr(importlib.import_module('a2meas.robotwin.rsi.' + module), actor_name)
                    settings = {**registry['defaults'], **settings}
                    actor = cls(self, settings)
                    actor._load_model()
                    self._actors[name] = actor
            actor = self._actors[name]
            self._routes[ctx.session_id] = actor
            self._session_chunk_sizes[ctx.session_id] = actor.chunk_size
            await actor.on_episode_start({'task': {'name': name}}, ctx)
        except Exception:
            await self.on_episode_end({}, ctx)
            raise

    async def on_episode_end(self, result, ctx):
        actor = self._routes.pop(ctx.session_id, None)
        try:
            if actor is not None:
                await actor.on_episode_end(result, ctx)
        finally:
            await super().on_episode_end(result, ctx)

    async def on_observation(self, obs, ctx):
        self._validate_observation(obs)
        actor = self._routes.get(ctx.session_id)
        if actor is not None:
            await actor.on_observation(obs, ctx)
        else:
            await self._native_observation(obs, ctx)

    async def _native_observation(self, obs, ctx):
        await super()._on_validated_observation(obs, ctx)

    def _normalize_result(self, result, ctx):
        actor = getattr(self, '_routes', {}).get(ctx.session_id)
        return actor._normalize_result(result, ctx) if actor is not None else self._normalize_native(result, ctx)
