"""Key recovery behavior and external boundaries."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from a2meas.robotwin.rsi.config import load_config
import importlib
from a2meas.robotwin.rsi.actor import Actor
from a2meas.robotwin.rsi.ranking_slots import retained_slot_plan
from a2meas.robotwin.rsi.config import ACTORS


@pytest.fixture(autouse=True)
def reference_configuration(monkeypatch):
    monkeypatch.setenv("ROBOTWIN_RSI_CONFIG", str(Path(__file__).with_name("fixtures") / "rsi_base.json"))
    load_config.cache_clear()
    yield
    load_config.cache_clear()


class FakeServer:
    def __init__(self):
        self._session_chunk_sizes = {}
        self._chunk_buffers = {}

    def _load_model(self):
        pass

    def _normalize_native(self, result, ctx):
        size = self._session_chunk_sizes[ctx.session_id]
        return {**result, 'actions': np.asarray(result['actions'], dtype=np.float32)[:size]}

    def _try_serve_from_buffer(self, ctx):
        self._chunk_buffers.setdefault(ctx.session_id, SimpleNamespace(empty=True))


@pytest.mark.parametrize('enabled', [False, True])
def test_native_bottle_closure_preserves_arm_targets_and_other_sessions(enabled):
    from a2meas.robotwin.rsi.recovery import BottleRecovery
    server = FakeServer()
    policy = BottleRecovery(server, dict(execution_chunk_size=32, fully_close_native_grasp=enabled))
    ctx = SimpleNamespace(session_id='one')
    policy._bottle_sessions = {'one': {'seen': {'left': True, 'right': True}}}
    other = object()
    server._chunk_buffers['two'] = other
    server._session_chunk_sizes['two'] = 40
    raw = np.arange(700, dtype=np.float32).reshape(50, 14) / 700
    raw[:7, 6] = [-.1, 0., .1, .1999, .2, .8, 1.]
    raw[:7, 13] = [1., .8, .2, .1999, .1, 0., -.1]
    original = raw.copy()
    out = policy._normalize_result({'actions': raw}, ctx)
    expected = raw[:32].copy()
    if enabled:
        for jaw in [6, 13]:
            expected[(expected[:, jaw] > 0) & (expected[:, jaw] < .2), jaw] = 0.
    np.testing.assert_array_equal(out['actions'], expected)
    np.testing.assert_array_equal(out['joints'], expected)
    np.testing.assert_array_equal(raw, original)
    assert server._chunk_buffers['two'] is other
    assert server._session_chunk_sizes == {'one': 32, 'two': 40}
    assert ('one' in server._chunk_buffers) == enabled


@pytest.fixture
def grasp_policy():
    pytest.importorskip('sam3')
    from a2meas.robotwin.rsi.grasp import ProactiveGrasp
    cfg = load_config()
    settings = {**cfg['defaults'], **cfg['tasks']['place_phone_stand']}
    server = FakeServer()
    server._image_histories, server._state_histories = {}, {}
    policy = ProactiveGrasp(server, settings)
    policy._sessions = {}
    ctx = SimpleNamespace(session_id='one')
    asyncio.run(policy.on_episode_start({}, ctx))
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = 1.
    policy._sessions['one'].update(state=state, n=41, initial_sources=[])
    return policy, server, ctx


def test_grasp_budget_or_ik_failure_leaves_native_buffer(grasp_policy, monkeypatch):
    policy, server, ctx = grasp_policy
    m = policy._sessions['one']
    m.update(source={}, arm='right')
    old_buffer = server._chunk_buffers['one'] = object()
    def decline(*_):
        raise ValueError('No feasible path within original budget')
    monkeypatch.setattr(policy, 'plan', decline)
    actions = np.repeat(m['state'][None], 50, axis=0)
    out = policy._normalize_result({'actions': actions}, ctx)
    np.testing.assert_array_equal(out['actions'], actions[:40])
    assert server._chunk_buffers['one'] is old_buffer
    assert m['done'] and m['phase'] is None


def test_grasp_replaces_buffer_once_then_clears_only_its_history(grasp_policy, monkeypatch):
    policy, server, ctx = grasp_policy
    m = policy._sessions['one']
    m.update(source={}, arm='right')
    script = np.repeat(m['state'][None], 86, axis=0)
    monkeypatch.setattr(policy, 'plan', lambda *_: script)
    actions = np.repeat(m['state'][None], 50, axis=0)
    out = policy._normalize_result({'actions': actions}, ctx)
    assert out['actions'] is script and out['joints'] is script
    assert m['phase'] == 'script' and server._session_chunk_sizes['one'] == 86
    server._image_histories.update(one=['old'], two=['other'])
    server._state_histories.update(one=['old'], two=['other'])
    server._state_vector_from_obs = lambda obs: m['state']
    called = []
    async def native(obs, context):
        called.append(context.session_id)
    server._native_observation = native
    asyncio.run(policy.on_observation({}, ctx))
    assert called == ['one'] and m['phase'] is None
    assert server._image_histories == {'two': ['other']}
    assert server._state_histories == {'two': ['other']}
    assert len(policy._normalize_result({'actions': actions}, ctx)['actions']) == 40
    asyncio.run(policy.on_episode_start({}, SimpleNamespace(session_id='two')))
    asyncio.run(policy.on_episode_end({}, ctx))
    assert set(policy._sessions) == {'two'}


def test_grasp_rgb_detector_restores_threshold_on_error(monkeypatch):
    pytest.importorskip('sam3')
    from a2meas.robotwin.rsi.grasp import grasp_sources
    processor = SimpleNamespace(confidence_threshold=.4)
    def fail(*_):
        raise RuntimeError('Segmentation unavailable')
    processor.set_image = fail
    with pytest.raises(RuntimeError, match='Segmentation unavailable'):
        grasp_sources(SimpleNamespace(processor=processor), np.zeros((240, 320, 3), dtype=np.uint8), ['phone'])
    assert processor.confidence_threshold == .4


def test_mug_preclose_preserves_native_forecast_and_resets_each_episode():
    pytest.importorskip('sam3')
    from a2meas.robotwin.rsi.mug import MugRecovery
    server = FakeServer()
    policy = MugRecovery(server, dict(execution_chunk_size=40, max_steps=900,
        observe_before_close=True))
    a, b = [SimpleNamespace(session_id=sid, step=0) for sid in ['one', 'two']]
    state = np.zeros(14, np.float32)
    state[[6, 13]] = 1.
    for ctx in [a, b]:
        asyncio.run(policy.on_episode_start({}, ctx))
        policy._sessions[ctx.session_id]['state'] = state.copy()
    raw = np.tile(state, (50, 1))
    raw[:, 7] = np.arange(50) / 100
    raw[20:, 6], raw[30:, 13] = .5, .4
    for step, length in [(0, 16), (16, 26), (42, 40)]:
        a.step = step
        result = policy._normalize_result({'actions': raw}, a)
        np.testing.assert_array_equal(result['actions'], raw[:length])
        np.testing.assert_array_equal(policy._sessions['one']['forecast'], raw[39, 7:13])
        np.testing.assert_array_equal(policy._sessions['one']['native_guard_plan'][1], raw)
    assert 'preclose_used' not in policy._sessions['two']
    assert len(policy._normalize_result({'actions': raw}, b)['actions']) == 16
    asyncio.run(policy.on_episode_end({}, a))
    asyncio.run(policy.on_episode_start({}, a))
    policy._sessions['one']['state'] = state.copy()
    assert len(policy._normalize_result({'actions': raw}, a)['actions']) == 16
    policy.settings['observe_before_close'] = False
    assert len(policy._normalize_result({'actions': raw}, b)['actions']) == 40


@pytest.mark.parametrize("kind", ACTORS)
def test_actor_episode_cleanup_is_isolated_and_idempotent(kind):
    module, name = ACTORS[kind]
    actor_type = getattr(importlib.import_module("a2meas.robotwin.rsi." + module), name)
    assert issubclass(actor_type, Actor)
    actor = actor_type(SimpleNamespace(), {"execution_chunk_size": 40, "stall_window": 33})
    if kind == "can_release_cadence":
        actor._can_release_sessions = actor._sessions
    if kind == "bottle":
        actor._bottle_sessions = {}
    memory = (actor._can_release_sessions if kind == "can_release_cadence" else
              actor._bottle_sessions if kind == "bottle" else actor._sessions)
    memory["other"] = {"retained": True}
    ctx = SimpleNamespace(session_id="current")
    for _ in range(2):
        asyncio.run(actor.on_episode_start({}, ctx))
        if kind != "native_prefix":
            assert "current" in memory
        assert asyncio.run(actor.on_episode_end({}, ctx)) is None
        assert memory == {"other": {"retained": True}}
        assert asyncio.run(actor.on_episode_end({}, ctx)) is None


def test_cyclic_ranking_uses_existing_slots_and_preserves_inputs():
    points = np.array([[.12, -.15, .74], [-.12, -.14, .735], [0., -.16, .73]])
    before = points.copy()
    state = np.arange(14, dtype=np.float32)
    own_before = state.copy()
    actions, moves, proof = retained_slot_plan(None, state, points, {}, 100, _transfer_stub)
    layout = points.copy()
    for move in moves:
        rank = move['rank']
        np.testing.assert_array_equal(move['source'], layout[rank])
        destination = np.asarray(move['destination'])
        assert all(i == rank or np.linalg.norm(layout[i, :2] - destination[:2]) >= .055 for i in range(3))
        layout[rank] = destination
    np.testing.assert_allclose(layout[:, 0], np.sort(before[:, 0]))
    np.testing.assert_allclose(layout[:, 1], np.median(before[:, 1]))
    np.testing.assert_array_equal(layout[:, 2], before[:, 2])
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(state, own_before)
    assert actions.shape == (sum(m['counted_steps'] for m in moves), 14)
    assert len(actions) == proof['total_steps'] < 90 and proof['expanded_transfers'] <= 64


def test_incomplete_plan_is_never_returned_when_budget_is_insufficient():
    points = np.array([[.12, -.14, .74], [-.12, -.14, .735], [0., -.14, .73]])
    with pytest.raises(ValueError, match='No complete visible-slot reorder'):
        retained_slot_plan(None, np.zeros(14, np.float32), points, {}, 13, _transfer_stub)


def _transfer_stub(kin, state, source, destination, rotations, remaining):
    assert remaining > 10
    return np.repeat(state[None], 2, axis=0), 'left'
