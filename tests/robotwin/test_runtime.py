"""Key runtime behavior and external boundaries."""

import asyncio
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from vla_eval.model_servers.predict import PredictModelServer
from a2meas.robotwin.server import CAMERAS, RoboTwinModelServer
from a2meas.robotwin.rsi.server import RoboTwinRSIServer
from tests.robotwin.helpers import native_server, rsi_server, observation
from a2meas.robotwin.rsi.sampling import SessionGaussianSource
from benchmarks.robotwin.benchmark import RoboTwinBenchmark


def test_dense_stm_history_and_session_isolation():
    policy = native_server()
    ctx = SimpleNamespace(session_id="first")
    for value in range(32):
        sample = policy._prepare_sample(observation(value), ctx)
    torch.testing.assert_close(
        sample["head"][:, 0, 0, 0], torch.tensor([1.0 / 255, 31.0 / 255])
    )
    torch.testing.assert_close(sample["observation.state"], torch.full((14,), 31.0))
    fresh = policy._prepare_sample(
        observation(10), SimpleNamespace(session_id="second")
    )
    torch.testing.assert_close(fresh["head"][0], fresh["head"][1])
    assert len(policy._histories["first"]["head_camera"]) == 31


@pytest.mark.parametrize("server_type", [RoboTwinModelServer, RoboTwinRSIServer])
@pytest.mark.parametrize("bad", ["missing_camera", "camera_dtype", "empty_image", "state_shape", "state_nan", "task_type", "task_empty"])
def test_invalid_observation_cannot_advance_actor_history_or_actions(server_type, bad):
    from vla_eval.model_servers.chunking import ActionChunkBuffer, get_ensemble_fn

    policy = server_type.__new__(server_type)
    policy.__dict__.update(native_server().__dict__)
    calls = []
    class Actor:
        async def on_observation(self, *args):
            calls.append(True)
    policy._routes = {"test": Actor()}
    ctx = SimpleNamespace(session_id="test")
    policy._append_observation(observation(10), ctx)
    policy.chunk_size, policy._session_chunk_sizes = 40, {}
    buffer = ActionChunkBuffer(40, get_ensemble_fn("newest", 0.1))
    actions = np.stack([np.full(14, 1, np.float32), np.full(14, 2, np.float32)])
    buffer.push_chunk(actions)
    policy._chunk_buffers = {"test": buffer}
    obs = observation(20)
    if bad == "missing_camera":
        del obs["images"]["head_camera"]
    elif bad == "camera_dtype":
        obs["images"]["head_camera"] = obs["images"]["head_camera"].astype(np.float32)
    elif bad == "empty_image":
        obs["images"]["head_camera"] = np.zeros((0, 3, 3), np.uint8)
    elif bad == "state_shape":
        obs["joint_state"] = np.zeros(13)
    elif bad == "state_nan":
        obs["joint_state"][0] = np.nan
    else:
        obs["task_description"] = None if bad == "task_type" else " "
    with pytest.raises(ValueError):
        asyncio.run(policy.on_observation(obs, ctx))
    np.testing.assert_array_equal(buffer.pop(), actions[0])
    assert not calls and len(policy._state_histories["test"]) == 1
    assert all(len(history) == 1 for history in policy._histories["test"].values())


@pytest.mark.parametrize('kind', ['contact', 'mug'])
def test_scripted_takeover_records_native_history_once_before_sending(kind, monkeypatch):
    from collections import deque
    from a2meas.robotwin.rsi.contact import ContactSearch
    from a2meas.robotwin.rsi.mug import MugRecovery

    obj = rsi_server()
    ctx, other = SimpleNamespace(session_id='takeover'), SimpleNamespace(session_id='other')
    obj._append_observation(observation(3), other)
    obs, actions, sent = observation(7), np.zeros((4, 14), np.float32), []

    async def send(result, context):
        assert context is ctx
        assert len(obj._histories[ctx.session_id]['head_camera']) == 1
        assert len(obj._state_histories[ctx.session_id]) == 1
        assert obj._session_chunk_sizes[ctx.session_id] == len(actions)
        assert result['actions'] is actions and result['joints'] is actions
        sent.append(result)

    monkeypatch.setattr(obj, '_process_and_send', send)
    if kind == 'contact':
        actor = ContactSearch(obj, dict(execution_chunk_size=40, trigger=[0, 0, 1],
                                       offsets=[[0, 0, 0]], stall_window=2))
        actor._sessions[ctx.session_id] = dict(n=0, states=deque(maxlen=2), done=False,
                                              phase=0, side='right', anchor=np.zeros(3))
        monkeypatch.setattr(actor, 'trajectory', lambda *args: actions)
    else:
        actor = MugRecovery(obj, dict(execution_chunk_size=40))
        actor._mode = 'align'
        actor._sessions[ctx.session_id] = dict(n=1, phase='settle', done=False,
                                              tip3d=None, open_seen=False, held=0)
        monkeypatch.setattr(actor, '_next_actions', lambda *args: actions)
    asyncio.run(actor.on_observation(obs, ctx))
    assert len(sent) == 1
    for image in obs['images'].values():
        image.fill(254)
    obs['joint_state'].fill(254)
    for name in CAMERAS:
        assert int(obj._histories[ctx.session_id][name][-1][0, 0, 0]) == 7
        assert int(obj._histories[other.session_id][name][-1][0, 0, 0]) == 3
    np.testing.assert_array_equal(obj._state_histories[ctx.session_id][-1], np.full(14, 7))


def test_native_forecast50_survives_until_actual40_prefix_trim():
    obj = rsi_server(); obj._load_model()
    assert obj.image_history_num_frames == 2
    raw = np.arange(700, dtype=np.float32).reshape(50, 14)
    native = native_server()
    native.chunk_size = 40
    assert native._prediction_result(raw)["actions"].shape == (40, 14)
    result = obj._prediction_result(raw)
    raw.fill(-9)
    assert result['actions'].shape == (50, 14) and result['actions'][0, 1] == 1
    normalized = obj._normalize_result(result, SimpleNamespace(session_id='native'))
    np.testing.assert_array_equal(normalized['actions'], result['actions'][:40])
    np.testing.assert_array_equal(normalized['joints'], normalized['actions'])


def test_observation_perception_waits_for_inference_device_lock(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from a2meas.robotwin.rsi.actor import Actor
    from a2meas.robotwin.server import RoboTwinModelServer

    obj = rsi_server()
    entered, release, vision_entered, vision_called = Event(), Event(), Event(), Event()

    def inference(self, observations, contexts):
        entered.set()
        assert release.wait(5)
        assert not vision_entered.is_set()
        return 'native prediction'

    class Detector:
        def detect(self, image, prompt):
            vision_entered.set()
            return image, prompt

    monkeypatch.setattr(RoboTwinModelServer, 'predict_batch', inference)
    obj._vision_backend = Detector()
    detector = Actor(obj, {'execution_chunk_size': 40}).vision(Detector)

    def perception():
        vision_called.set()
        return detector.detect('original RGB', 'cube')

    with ThreadPoolExecutor(max_workers=2) as pool:
        prediction = pool.submit(obj.predict_batch, [], [])
        assert entered.wait(5)
        detection = pool.submit(perception)
        try:
            assert vision_called.wait(5)
            assert not vision_entered.wait(.1)
        finally:
            release.set()
        assert prediction.result(timeout=5) == 'native prediction'
        assert detection.result(timeout=5) == ('original RGB', 'cube')
    assert vision_entered.is_set()


def test_session_history_is_released_when_base_cleanup_fails(monkeypatch):
    policy, ctx = native_server(), SimpleNamespace(session_id="test")
    policy._append_observation(observation(10), ctx)

    async def fail(*args):
        raise RuntimeError("base cleanup")

    monkeypatch.setattr(PredictModelServer, "on_episode_end", fail)
    with pytest.raises(RuntimeError, match="base cleanup"):
        asyncio.run(policy.on_episode_end({}, ctx))
    assert not policy._histories and not policy._state_histories


@pytest.mark.parametrize('padding', [False, True])
def test_session_stream_advances_and_ignores_other_episode_order(padding):
    def native(shape, device):
        noise = torch.normal(0., 1., size=shape, device=device)
        if padding:
            noise[..., 14:] = 0.
        return noise

    a, b = SessionGaussianSource(torch), SessionGaussianSource(torch)
    shape = (1, 50, 32)
    first = a.sample(native, shape, 'cpu')
    torch.testing.assert_close(b.sample(native, shape, 'cpu'), first, rtol=0, atol=0)
    global_before = torch.random.get_rng_state()
    native(shape, 'cpu')
    expected_global = torch.random.get_rng_state()
    torch.random.set_rng_state(global_before)
    second = a.sample(native, shape, 'cpu')
    assert torch.equal(expected_global, torch.random.get_rng_state())
    assert not torch.equal(first, second)
    native(shape, 'cpu')
    torch.testing.assert_close(b.sample(native, shape, 'cpu'), second, rtol=0, atol=0)
    torch.testing.assert_close(SessionGaussianSource(torch).sample(native, shape, 'cpu'), first, rtol=0, atol=0)
    assert bool(torch.count_nonzero(first[..., 14:])) is not padding
    with pytest.raises(ValueError, match='one request'):
        a.sample(native, (2, 50, 32), 'cpu')


def test_session_stream_restores_state_after_native_sampler_failure():
    source = SessionGaussianSource(torch)
    saved = source.cpu_state.clone()
    calls = []

    def native(shape, device):
        noise = torch.randn(shape, device=device)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('private draw failed')
        return noise

    before = torch.random.get_rng_state()
    torch.randn((1, 50, 32))
    expected = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    with pytest.raises(RuntimeError, match='private draw failed'):
        source.sample(native, (1, 50, 32), 'cpu')
    assert torch.equal(source.cpu_state, saved)
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_solver_groups_restore_shared_configuration_and_sampler_on_errors(monkeypatch):
    from a2meas.robotwin.rsi.server import RoboTwinRSIServer
    from a2meas.robotwin.server import RoboTwinModelServer
    server = object.__new__(RoboTwinRSIServer)
    server._torch = torch
    config = SimpleNamespace(num_inference_steps=10, flow_source_mode='blend',
        output_features={'action': SimpleNamespace(shape=[14])})
    native = lambda shape, device: torch.ones(shape, device=device)
    model = SimpleNamespace(config=config, sample_noise=native)
    server._policy = SimpleNamespace(config=config, model=model)
    server._routes = {str(i): SimpleNamespace(settings=settings) for i, settings in {
        0: dict(fixed_gaussian_noise=True, single_sample_inference=True),
        1: dict(num_inference_steps=20),
        3: dict(num_inference_steps=20, fixed_gaussian_noise=True, single_sample_inference=True),
        4: dict(num_inference_steps=20)}.items()}
    contexts = [SimpleNamespace(session_id=str(i)) for i in range(5)]
    calls = []

    def parent(self, observations, contexts):
        steps = self._policy.config.num_inference_steps
        calls.append((observations.copy(), steps))
        noise = model.sample_noise((len(contexts), 50, 32), 'cpu')
        return [(obs, steps, noise[i]) for i, obs in enumerate(observations)]

    monkeypatch.setattr(RoboTwinModelServer, 'predict_batch', parent)
    result = server.predict_batch(list(range(5)), contexts)
    assert calls == [([1, 4], 20), ([2], 10), ([0], 10), ([3], 20)]
    assert [(obs, steps) for obs, steps, _ in result] == [(0, 10), (1, 20), (2, 10), (3, 20), (4, 20)]
    assert config.num_inference_steps == 10 and model.sample_noise is native
    torch.testing.assert_close(result[0][2], result[3][2], rtol=0, atol=0)

    def failure(*args):
        assert config.num_inference_steps == 20
        raise RuntimeError('solver test failure')

    monkeypatch.setattr(RoboTwinModelServer, 'predict_batch', failure)
    for index in [1, 3]:
        with pytest.raises(RuntimeError, match='solver test failure'):
            server.predict_batch([index], [contexts[index]])
        assert config.num_inference_steps == 10 and model.sample_noise is native


@pytest.mark.parametrize(
    "action", [np.zeros(13), np.zeros(16), np.zeros((1, 14)), np.full(14, np.nan)]
)
def test_bad_actions_are_rejected_before_simulation(action):
    bench = RoboTwinBenchmark.__new__(RoboTwinBenchmark)
    with pytest.raises(ValueError, match="finite 14D"):
        bench.step({"actions": action})


def test_simulator_cleanup_runs_even_if_video_close_fails():
    bench = RoboTwinBenchmark.__new__(RoboTwinBenchmark)
    closed = []
    bench._env = SimpleNamespace(close_env=lambda **kwargs: closed.append(kwargs))
    bench._last_rendered_obs = object()

    def fail():
        raise RuntimeError("video close")

    bench._stop_eval_video = fail
    with pytest.raises(RuntimeError, match="video close"):
        bench.cleanup()
    assert closed == [{"clear_cache": True}]
    assert bench._env is None and bench._last_rendered_obs is None


def test_dense_simulation_steps_keep_targets_and_skip_only_terminal_rgb():
    bench = RoboTwinBenchmark("stamp_seal", expert_task_cache_path="unused", instruction_type="unseen")
    target, sent, renders = np.arange(14) * .01, [], []
    env = SimpleNamespace(take_action_cnt=0, step_lim=2, eval_success=False, eval_video_path=None)
    env.robot = SimpleNamespace(get_left_arm_jointState=lambda: target[:7].tolist(),
                                get_right_arm_jointState=lambda: target[7:].tolist())
    raw = {"observation": {key: {"rgb": np.zeros((2, 2, 3), np.uint8)} for key in ("head_camera", "left_camera", "right_camera")},
           "joint_action": {"vector": np.zeros(14)}, "language": "Stamp the paper."}

    def take_action(action, action_type):
        assert action_type == "qpos"
        sent.append(action.copy())
        env.take_action_cnt += 1

    env.take_action = take_action
    env.get_obs = lambda: (renders.append(env.take_action_cnt) or raw)
    bench._env, bench._last_rendered_obs = env, raw
    bench._task = {"seed": 100001, "episode_idx": 0}
    first, last = bench.step({"actions": target}), bench.step({"actions": target})
    assert not first.done and last.done and not last.info["success"]
    assert renders == [1]
    np.testing.assert_array_equal(sent, np.stack([target, target]))
    assert len(bench.make_obs(first.obs, bench._task)["images"]) == 3
    terminal = bench.make_obs(last.obs, bench._task)
    assert terminal["images"] == {}
    np.testing.assert_array_equal(terminal["joint_state"], target)
