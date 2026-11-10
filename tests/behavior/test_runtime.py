import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a2meas.behavior import resources
from a2meas.behavior.observations import ObservationAdapter, CAMERA_KEYS
from a2meas.behavior.protocol import Packer, unpackb
from a2meas.behavior import CONFIG_ROOT, load_routes, resolve_task
from a2meas.behavior.server import RouteManager, TaskRoutingError, single_rollout_observation
from a2meas.behavior.schema import PreTrainedConfig


def test_cpu_layout_separates_physical_cores_within_gpu_numa_and_cpuset(monkeypatch):
    import a2meas as runtime

    allowed = set(range(48))
    node = 0
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: allowed.copy())
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "00000000:61:00.0\n")

    def read(path):
        if path.name == "numa_node":
            assert str(path) == "/sys/bus/pci/devices/0000:61:00.0/numa_node"
            return str(node)
        cpu = int(path.parts[-3][3:])
        return f"{cpu % 12},{cpu % 12 + 12}"

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "exists", lambda path: int(path.parent.name[3:]) < 24)
    simulator, policies = runtime.cpu_layout(0, 2)
    assert policies == [{0, 1, 2, 12, 13, 14}, {3, 4, 5, 15, 16, 17}]
    assert simulator == {6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23}
    allowed = {0, 2, 4, 6}
    assert runtime.cpu_layout(0, 2) == ({4, 6}, [{0}, {2}])
    with pytest.raises(ValueError, match="physical CPU cores"):
        runtime.cpu_layout(0, 4)
    node = -1
    allowed = {24, 25, 26, 27}
    assert runtime.cpu_layout(0, 1) == ({26, 27}, [{24, 25}])



def test_child_inherits_cpu_affinity_and_parent_restores_it():
    import subprocess
    import sys
    from a2meas import cpu_affinity

    original = os.sched_getaffinity(0)
    selected = {min(original)}
    with pytest.raises(RuntimeError):
        with cpu_affinity(selected):
            child = subprocess.check_output([
                sys.executable, "-c", "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
            ], text=True)
            assert set(json.loads(child)) == selected
            raise RuntimeError("launch failed")
    assert os.sched_getaffinity(0) == original


@pytest.fixture
def evaluator_launches(monkeypatch):
    from benchmarks.behavior import evaluate
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            self.args, self.returncode = command, 0
            launched.append((command, kwargs))
            output = Path(command[command.index("--output-dir") + 1]) / "json"
            output.mkdir(parents=True, exist_ok=True)
            first = 301 if command[command.index("--mode") + 1] == "public_test" else 321
            for group in json.loads(kwargs["env"]["A2MEAS_TASK_GROUPS"]):
                for index in group["instances"]:
                    (output / f"{group['task']}_{first + index}_0.json").write_text(json.dumps({
                        'task': group['task'], 'instance_id': first + index, 'rollout_id': 0,
                        'success': False, 'q_score': {'final': 0.0},
                    }))

        def poll(self):
            return self.returncode

    monkeypatch.setattr(evaluate.subprocess, "Popen", Process)
    monkeypatch.setattr(evaluate, "stop_process", lambda process: None)
    monkeypatch.setattr(evaluate, "cpu_layout", lambda gpu, workers: (os.sched_getaffinity(0), [set()] * workers))
    return launched


def test_all_100_routes_resolve_and_reference_released_models(capsys):
    from a2meas.behavior import cli as serve
    routes = load_routes()
    manifest = json.loads((CONFIG_ROOT / "resources.json").read_text())["files"]
    assert set(map(int, routes["tasks"])) == set(range(100))
    for task_id, route in routes["tasks"].items():
        assert resolve_task(route["task"], routes)[0] == int(task_id)
        assert route["num_inference_steps"] > 0
        assert f"behavior/policies/{route['model']}/model.safetensors" in manifest
    serve.main(["list"])
    assert capsys.readouterr().out.splitlines() == [
        f"{task_id}: {route['task']}" for task_id, route in routes["tasks"].items()
    ]


@pytest.mark.parametrize("batched", [False, True])
def test_protocol_selects_single_rollout_inputs(batched):
    state = np.arange(61, dtype=np.float32)
    image = np.ones((6, 8, 3), dtype=np.uint8)
    original = {
        "robot_r1::proprio": state[None] if batched else state,
        **dict.fromkeys(CAMERA_KEYS, image[None] if batched else image),
        "unused_depth": np.zeros((2, 6, 8), dtype=np.float32),
        "task_id": 6,
    }
    decoded = unpackb(Packer().pack(original))
    result = single_rollout_observation(decoded)
    np.testing.assert_array_equal(result["robot_r1::proprio"], np.arange(61, dtype=np.float32))
    assert set(result) == {*CAMERA_KEYS, "robot_r1::proprio"}
    for key in CAMERA_KEYS:
        np.testing.assert_array_equal(result[key], image)
    with pytest.raises(ValueError):
        single_rollout_observation({"robot_r1::proprio": np.zeros((2, 61))})
    with pytest.raises(ValueError):
        single_rollout_observation(dict(original, **{"robot_r1::proprio": np.full(61, np.nan)}))
    if batched:
        with pytest.raises(ValueError, match="batch"):
            single_rollout_observation(dict(original, **{CAMERA_KEYS[0]: image}))


def test_protocol_rejects_unsupported_array_and_scalar_dtypes():
    import msgpack

    for dtype in ("O", "V8", "c16"):
        for marker in (b"__ndarray__", b"__npgeneric__"):
            payload = {marker: True, b"dtype": dtype, b"data": b"", b"shape": (0,)}
            with pytest.raises(ValueError, match="Unsupported array dtype"):
                unpackb(msgpack.packb(payload))
    for value in (np.int32(6), np.float32(0.5), np.bool_(True)):
        decoded = unpackb(Packer().pack(value))
        assert type(decoded) is type(value) and decoded == value


def test_route_binding_rejects_task_switch_until_reset(tmp_path):
    manager = RouteManager(tmp_path, os.sched_getaffinity(0))
    manager.active_task = manager.bound_task = 6
    manager.bound_instance = 0
    calls = []
    manager.request = lambda operation, payload=None, timeout=580: calls.append(operation)
    observation = {**dict.fromkeys(CAMERA_KEYS, np.zeros((6, 8, 3), dtype=np.uint8)),
                   "robot_r1::proprio": np.zeros(61, dtype=np.float32)}
    manager.forward(dict(observation, task_id=6))
    assert calls == ["step"]
    with pytest.raises(TaskRoutingError, match="reset required"):
        manager.forward({"task_id": 40})
    manager.reset()
    manager.active_task = 40
    manager.forward(dict(observation, task_id=40))
    assert manager.bound_task == 40
    with pytest.raises(TaskRoutingError, match="Task text"):
        manager.forward({"task_id": 40, "task": "wrong"})
    manager.reset()
    seeds = []
    manager.request = lambda operation, payload=None, timeout=580: seeds.append((operation, payload))
    manager.forward(dict(observation, task_id=40, instance_id=301))
    first_seed = next(payload for operation, payload in seeds if operation == "seed")
    with pytest.raises(TaskRoutingError, match="instance_id changed"):
        manager.forward(dict(observation, task_id=40, instance_id=302))
    manager.reset()
    seeds.clear()
    manager.forward(dict(observation, task_id=40, instance_id=301))
    assert next(payload for operation, payload in seeds if operation == "seed") == first_seed


@pytest.mark.parametrize("adapter,target", [(False, 37), (True, 1)])
def test_same_model_route_reuses_worker_and_resets_task(tmp_path, monkeypatch, adapter, target):
    manager = RouteManager(tmp_path, {0}, adapter=adapter)
    manager.process = object()
    manager.active_task = 0
    calls = []
    monkeypatch.setattr(manager, "request", lambda operation, task_id, timeout: calls.append((operation, task_id)))
    monkeypatch.setattr(manager, "close_worker", lambda: pytest.fail("matching model was reloaded"))
    manager.load(target, timeout=10)
    assert calls == [("route", target)] and manager.active_task == target


@pytest.mark.parametrize("terminate", [False, True], ids=["pipe-closed", "sigterm"])
@pytest.mark.parametrize("adapter", [False, True])
def test_worker_exits_normally_when_gateway_closes_its_pipe(tmp_path, monkeypatch, terminate, adapter):
    import multiprocessing
    import a2meas.behavior.policy as policy
    from a2meas.behavior.server import route_worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "compile"))
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    def load(*_, **kwargs):
        expected = "model_20" if adapter else load_routes()['tasks']['0']['model']
        assert Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).name == expected
        return SimpleNamespace(close=lambda: None, warmup=lambda: None)

    monkeypatch.setattr(policy, "VLAPolicy", load)

    def worker():
        parent.close()
        route_worker(child, 0, str(tmp_path), adapter=adapter)

    process = context.Process(target=worker)
    process.start()
    child.close()
    manager = RouteManager(tmp_path, os.sched_getaffinity(0))
    manager.process, manager.connection = process, parent
    try:
        assert parent.poll(5) and parent.recv() == ("ready", None)
        if terminate:
            process.terminate()
            process.join(timeout=5)
    finally:
        manager.close()
    assert process.exitcode == 0
    assert manager.process is manager.connection is None


def test_temporal_observations_fill_initial_frame_and_own_history():
    from a2meas.schema import SHORT_TERM_MEMORY_FRAME_AGE_SECONDS as age_key

    config = SimpleNamespace(
        input_features={
            "observation.rgb.zed_link_camera_0": None,
            "observation.rgb.left_realsense_link_camera_0": None,
            "observation.rgb.right_realsense_link_camera_0": None,
        },
        enable_short_term_memory=True,
        enable_temporal_memory=True,
        short_term_memory_inference_num_frames=6,
        enable_speed_meta=True,
        speed_meta_map={"picking_up_trash": 500},
        short_term_memory_frame_age_embedding_enable=True,
        short_term_memory_frame_age_fps=30,
        image_resolution=(224, 224),
    )
    route = {"task": "picking_up_trash"}
    adapter = ObservationAdapter(SimpleNamespace(policy=SimpleNamespace(config=config)), route)
    image = np.full((224, 224, 3), 64, dtype=np.uint8)
    observation = {**dict.fromkeys(CAMERA_KEYS, image), "robot_r1::proprio": np.arange(61, dtype=np.float32)}
    adapter.observe(observation)
    first = adapter.sample()
    key = "observation.rgb.zed_link_camera_0"
    assert first[key].shape == (1, 6, 3, 224, 224)
    torch.testing.assert_close(first[key][:, 0], first[key][:, -1], rtol=0, atol=0)
    assert first[key + "_is_pad"].tolist() == [[True, True, True, True, True, False]]
    image.fill(192)
    adapter.observe(observation)
    second = adapter.sample()
    torch.testing.assert_close(second[key][:, 0], first[key][:, 0], rtol=0, atol=0)
    assert second["observation.state"].shape == (1, 6, 23)
    torch.testing.assert_close(second[age_key], torch.tensor([[1 / 30] * 5 + [0]]), rtol=0, atol=0)
    invalid = dict(observation, **{CAMERA_KEYS[-1]: image.astype(np.float32)})
    with pytest.raises(ValueError, match="Camera observations"):
        adapter.observe(invalid)
    assert len(adapter._state_history) == 2
    assert all(len(history) == 2 for history in adapter._image_history.values())
    for name, value in adapter.sample().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, second[name], rtol=0, atol=0)
        else:
            assert value == second[name]
    for _ in range(256):
        adapter.observe(observation)
    wrapped = adapter.sample()
    assert not wrapped[key + "_is_pad"].any()
    torch.testing.assert_close(
        wrapped[age_key], torch.tensor([[256 / 30, 205 / 30, 154 / 30, 2, 1, 0]]), rtol=0, atol=0
    )
    adapter.reset()
    assert not adapter._state_history and all(not h for h in adapter._image_history.values())





def test_prepare_downloads_pinned_resources_and_rejects_corruption(tmp_path, monkeypatch):
    import huggingface_hub
    import a2meas.behavior.weights as weights

    files = {"assets/rynnbrain/config.json": b"backbone", "behavior/policies/model_00/config.json": b"config",
             "behavior/policies/model_00/model.safetensors": b"inference"}
    metadata = tmp_path / "config"
    (metadata / "policy").mkdir(parents=True)
    (metadata / "policy/config.json").write_bytes(files["behavior/policies/model_00/config.json"])
    manifest = {"files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    (metadata / "resources.json").write_text(json.dumps(manifest))
    (metadata / "release.json").write_text(json.dumps({"repo_id": "test/weights", "revision": "pinned"}))
    monkeypatch.setattr(resources, "CONFIG_ROOT", metadata)
    monkeypatch.setattr(resources, "RUNTIME_ROOT", tmp_path / "runtime")
    downloads, restored, scratch = [], [], []

    def download(**kwargs):
        downloads.append((kwargs["repo_id"], kwargs["revision"], kwargs["token"]))
        target = kwargs["local_dir"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[kwargs["filename"]])

    def restore(destination, temporary, release, manifest, models):
        assert models == ["model_00"]
        scratch.append(temporary)
        shard = temporary / "temporary-shard"
        if len(scratch) == 1:
            shard.write_bytes(b"shard")
            raise RuntimeError("interrupted download")
        assert shard.read_bytes() == b"shard"
        (destination / "behavior/policies/model_00/model.safetensors").write_bytes(
            files["behavior/policies/model_00/model.safetensors"])
        restored.extend(models)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(weights, "restore_models", restore)
    destination = tmp_path / "resources"
    with pytest.raises(RuntimeError, match="interrupted download"):
        resources.prepare(destination)
    assert (scratch[0] / "temporary-shard").exists()
    assert resources.prepare(destination) == 3
    assert downloads == [("test/weights", "pinned", None)] and restored == ["model_00"]
    assert scratch[0] == scratch[1] and not scratch[0].exists()
    assert resources.prepare(destination) == 3
    assert len(downloads) == len(restored) == 1
    resources.verify_shared(destination)
    shared = destination / "assets/rynnbrain/config.json"
    shared.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        resources.verify_shared(destination)
    shared.write_bytes(files["assets/rynnbrain/config.json"])
    stored = destination / "behavior/policies/model_00/model.safetensors"
    stored.write_bytes(b"x" * len(files["behavior/policies/model_00/model.safetensors"]))
    with pytest.raises(ValueError, match="checksum"):
        resources.prepare(destination)
    stored.unlink()
    stored.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Missing or invalid"):
        resources.prepare(destination)
    assert not (tmp_path / "outside").exists()


def test_weight_store_restores_shuffled_and_referenced_tensors():
    from io import BytesIO
    from a2meas.behavior.weights import decode_tensor

    reference = bytes([5, 6, 7, 8])
    delta = bytes.fromhex('28b52ffd200421000002010004')
    shuffled = bytes.fromhex('28b52ffd200421000001030204')
    index = {'tensors': [
        dict(file='shard', offset=0, stored_bytes=4, raw_bytes=4, itemsize=1,
             codec='raw', reference=None),
        dict(file='shard', offset=4, stored_bytes=len(delta), raw_bytes=4, itemsize=1,
             codec='zstd_zigzag_shuffle', reference=0),
        dict(file='shard', offset=4 + len(delta), stored_bytes=len(shuffled), raw_bytes=4,
             itemsize=2, codec='zstd_shuffle', reference=None),
    ]}
    sources = {'shard': BytesIO(reference + delta + shuffled)}
    assert decode_tensor(index, sources, 1) == bytes([6, 5, 7, 10])
    assert decode_tensor(index, sources, 2) == bytes([1, 2, 3, 4])
    with pytest.raises(EOFError, match='Truncated'):
        decode_tensor(index, {'shard': BytesIO(b'')}, 0)


def test_action_queue_reuses_predictions_and_resets_history():
    from a2meas.behavior.policy import VLAPolicy
    from a2meas.behavior.actions import ActionChunker

    class Service:
        calls = 0

        def predict(self, sample):
            self.calls += 1
            chunk = np.full((50, 23), 0.1, dtype=np.float32)
            chunk[:, [14, 22]] = 0.8
            return chunk

    class Observations:
        def reset(self):
            self.steps = self.samples = 0

        def observe(self, observation):
            self.steps += 1

        def sample(self):
            self.samples += 1
            return {}

    policy = VLAPolicy.__new__(VLAPolicy)
    policy.task_id = 1
    policy.service = Service()
    policy.observations = Observations()
    policy.chunking = ActionChunker(arm_step_limit=0.1)
    policy.execution = load_routes()["tasks"]["1"]["execution"]
    policy.first_close = None
    policy.correction = None
    policy.reset()
    state = np.zeros(61, dtype=np.float32)
    state[24:26] = state[49:51] = 0.05
    observation = {"robot_r1::proprio": state}
    for _ in range(16):
        action = policy.step(observation)
        np.testing.assert_allclose(action[:3], 0.2, rtol=1e-6)
        np.testing.assert_array_equal(action[[14, 22]], np.full(2, 0.8, dtype=np.float32))
    assert policy.service.calls == 1
    policy.step(observation)
    assert policy.service.calls == policy.observations.samples == 2
    assert policy.observations.steps == 17
    policy.reset()
    assert not policy.actions and policy.last_action is None and policy.observations.steps == 0


def test_websocket_reset_action_and_reconnection_replay(monkeypatch):
    import asyncio
    from websockets.asyncio.client import connect
    import a2meas.behavior.server as server

    class Policy:
        closed = False
        calls = 0
        closes = 0

        def alive(self):
            return not self.closed

        def reset(self):
            self.calls = 0

        def forward(self, observation):
            self.calls += 1
            return np.arange(23, dtype=np.float32)

        def close(self):
            self.closed = True
            self.closes += 1

    async def exercise():
        policy = Policy()
        ready = asyncio.Event()
        addresses = []
        original = server.serve

        async def listening(*args, **kwargs):
            listener = await original(*args, **kwargs)
            addresses.append(listener.sockets[0].getsockname())
            ready.set()
            return listener

        monkeypatch.setattr(server, "serve", listening)
        task = asyncio.create_task(server.run(policy, "127.0.0.1", 0))
        try:
            await asyncio.wait_for(ready.wait(), 5)
            uri = f"ws://127.0.0.1:{addresses[0][1]}"
            payload = Packer().pack({"task_id": 6})
            async with connect(uri) as websocket:
                assert unpackb(await websocket.recv()) == {"action_dim": 23, "expected_state_dim": 61, "image_size": 224}
                await websocket.send(Packer().pack({"reset": True}))
                await websocket.send(payload)
                response = await websocket.recv()
                np.testing.assert_array_equal(unpackb(response)["action"], np.arange(23, dtype=np.float32))
            async with connect(uri) as websocket:
                await websocket.recv()
                await websocket.send(payload)
                assert await websocket.recv() == response
                assert policy.calls == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert policy.closed and policy.closes == 1

    asyncio.run(exercise())




def test_saved_processor_requires_local_state_and_known_overrides(tmp_path):
    from safetensors.torch import save_file
    from a2meas.processing import PolicyProcessorPipeline
    from a2meas.processing.pipeline import policy_action_to_transition, transition_to_policy_action

    metadata = {
        "name": "post",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "features": {"action": {"type": "ACTION", "shape": [2]}},
                    "norm_map": {"ACTION": "QUANTILES"},
                },
                "state_file": "stats.safetensors",
            }
        ],
    }
    (tmp_path / "post.json").write_text(json.dumps(metadata))
    save_file(
        {"action.q01": torch.tensor([-2.0, -1.0]), "action.q99": torch.tensor([2.0, 3.0])},
        tmp_path / "stats.safetensors",
    )
    pipeline = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        "post.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    torch.testing.assert_close(
        pipeline(torch.tensor([[-1.0, 1.0]])), torch.tensor([[-2.0, 3.0]]), rtol=0, atol=0
    )
    with pytest.raises(KeyError, match="Overrides"):
        PolicyProcessorPipeline.from_pretrained(tmp_path, "post.json", overrides={"missing_step": {}})
    (tmp_path / "stats.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        PolicyProcessorPipeline.from_pretrained(tmp_path, "post.json")




@pytest.mark.parametrize("limit,gripper_change,expected_steps", [(0, False, 16), (0.1, False, 22), (0.1, True, 32)])
def test_chunking_preserves_gripper_timing_and_arm_limits(limit, gripper_change, expected_steps):
    from a2meas.behavior.actions import ActionChunker

    actions = np.zeros((50, 23), dtype=np.float32)
    actions[:, :3] = 1
    actions[:32, 7] = np.linspace(0, 2, 32)
    if gripper_change:
        actions[16:, 14] = 1
    chunk = ActionChunker(arm_step_limit=limit).process(actions)
    assert chunk.shape == (expected_steps, 23)
    np.testing.assert_allclose(chunk[:, :3], 32 / expected_steps)
    np.testing.assert_allclose(chunk[[0, -1], 7], [0, 2], atol=1e-6)
    if gripper_change:
        np.testing.assert_array_equal(chunk, actions[:32])
    elif limit:
        assert np.abs(np.diff(chunk[:, 7])).max() <= limit


def test_chunking_preserves_gripper_threshold_precision():
    from a2meas.behavior.actions import ActionChunker

    for dtype in (np.float32, np.float64):
        threshold = dtype(0.2)
        for value, transition in (
            (np.nextafter(threshold, dtype(-np.inf)), False),
            (threshold, dtype is np.float32),
            (np.nextafter(threshold, dtype(np.inf)), True),
        ):
            for index in (14, 22):
                actions = np.zeros((50, 23), dtype=dtype)
                actions[16:, index] = value
                for full_source in (False, True):
                    chunk = ActionChunker().process(actions, gripper_full_source_on_transition=full_source)
                    steps = 32 if transition and full_source else 16
                    assert chunk.shape == (steps, 23)
                    if transition:
                        np.testing.assert_array_equal(chunk, actions[:steps])


def test_first_close_prompt_requires_debounce_and_nonempty_states():
    from a2meas.behavior.actions import FirstEffectiveGripperCloseTracker

    tracker = FirstEffectiveGripperCloseTracker()
    state = np.zeros(61, dtype=np.float32)
    state[[24, 49]] = 0.02
    action = np.ones(23, dtype=np.float32)
    action[14] = -1
    for _ in range(5):
        assert not tracker.observe_state(state)
        tracker.observe_action(action)
    for expected in (False, False, True):
        assert tracker.observe_state(state) == expected
        tracker.observe_action(action)
    tracker.reset()
    for _ in range(5):
        assert not tracker.observe_state(state)
        tracker.observe_action(action)
    state[24] = 0
    assert not tracker.observe_state(state)
    state[24] = 0.02
    for _ in range(10):
        assert not tracker.observe_state(state)
        tracker.observe_action(action)


def test_quantile_processor_preserves_constant_features_and_requires_statistics():
    from a2meas.processing.steps import NormalizerProcessorStep
    from a2meas.processing.pipeline import batch_to_transition, transition_to_batch

    processor = NormalizerProcessorStep(
        features={"observation.state": {"type": "STATE", "shape": [2]}},
        norm_map={"STATE": "QUANTILES"},
    )
    stats = {"observation.state.q01": torch.tensor([0.0, 2.0]),
             "observation.state.q99": torch.tensor([0.0, 4.0])}
    processor.load_state_dict(stats)
    sample = {"observation.state": torch.tensor([[123.0, 3.0]]), "task": "test"}
    result = transition_to_batch(processor(batch_to_transition(sample)))
    torch.testing.assert_close(result["observation.state"], torch.zeros(1, 2), rtol=0, atol=0)
    with pytest.raises(KeyError, match="observation.state"):
        processor(batch_to_transition({"task": "test"}))
    with pytest.raises(KeyError, match="q99"):
        processor.load_state_dict({"observation.state.q01": stats["observation.state.q01"]})


@pytest.mark.parametrize("close", [0, 1])
def test_prompt_processor_preserves_task_metadata(close):
    from a2meas.behavior.model.prompts import A2WLAPromptProcessor
    from a2meas.processing.pipeline import batch_to_transition, transition_to_batch

    processor = A2WLAPromptProcessor(
        enable_speed_meta=True, enable_gripper_first_close_meta=True,
    )
    sample = {
        "observation.state": torch.tensor([[-1.0, 0.0, 1.0]]),
        "task": ["pick_cup"], "speed_bin": 500, "gripper_first_close": close,
    }
    prompt = transition_to_batch(processor(batch_to_transition(sample)))["task"]
    assert prompt == [f"Task: pick cup, Speed: 500, GripperClose: {close}"]


@pytest.mark.parametrize("custom", [False, True], ids=["project-resources", "custom-resources"])
@pytest.mark.parametrize("workers", [1, 3])
def test_prepare_and_serve_use_the_same_resources_from_another_directory(tmp_path, monkeypatch, custom, workers):
    import signal
    from a2meas.behavior import cli as serve
    import a2meas.behavior.server as server
    from a2meas import RESOURCE_ROOT

    monkeypatch.chdir(tmp_path)
    for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    directories = []
    ports, closed = [], []
    monkeypatch.setattr(resources, "prepare", lambda path, **kwargs: directories.append(path) or 0)
    monkeypatch.setattr(resources, "verify_shared", lambda path: None)

    class Manager:
        def __init__(self, path, cpus, *, adapter=False, num_inference_steps=None, seed=0, compile_mode="max-autotune-no-cudagraphs"):
            assert adapter is False
            directories.append(path)
            assert cpus == {100 + len(directories)}

        def close(self):
            closed.append(self)

    async def run(manager, host, port):
        ports.append(port)
        manager.close()

    monkeypatch.setattr(server, "RouteManager", Manager)
    monkeypatch.setattr(server, "run", run)
    monkeypatch.setattr(serve, "cpu_layout", lambda gpu, count: (set(), [{102 + i} for i in range(count)]))
    options = ["--resources", "weights"] if custom else []
    serve.main(["prepare", *options])
    serve.main(["serve", *options, "--workers", str(workers)])
    expected = tmp_path / "weights" if custom else RESOURCE_ROOT
    assert directories == [expected] * (workers + 1)
    assert ports == list(range(8000, 8000 + workers))
    assert len(closed) == workers


def test_explicit_simulator_cpu_pool_is_not_partitioned_again(tmp_path, monkeypatch, evaluator_launches):
    from benchmarks.behavior import evaluate

    cpus = set(sorted(os.sched_getaffinity(0))[:4])
    observed = []
    affinity = evaluate.cpu_affinity

    def capture(selected):
        observed.append(set(selected))
        return affinity(selected)

    monkeypatch.setattr(evaluate, "cpu_affinity", capture)
    monkeypatch.setattr(evaluate, "cpu_layout", lambda *args: pytest.fail("Shared simulator pool was split again"))
    evaluate.main(["--task", "2", "--output-dir", str(tmp_path / "results"),
                   "--simulator-cpus", *map(str, sorted(cpus))])
    assert observed == [cpus]
    assert len(evaluator_launches) == 1


def test_evaluator_uses_relative_paths_and_preserves_existing_results(tmp_path, monkeypatch, evaluator_launches):
    from benchmarks.behavior import evaluate

    monkeypatch.chdir(tmp_path)
    sdk_module = tmp_path / "sdk/OmniGibson/omnigibson/eval/eval.py"
    sdk_module.parent.mkdir(parents=True)
    sdk_module.touch()
    logo = tmp_path / "sdk/docs/assets/OmniGibson_logo.png"
    logo.parent.mkdir(parents=True)
    logo.touch()
    metadata = tmp_path / "data/2026-challenge-task-instances/metadata/B100_task_misc.csv"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    launched = evaluator_launches
    args = ["--task", "0", "--gpu", "1", "--behavior-dir", "sdk", "--data-dir", "data",
            "--output-dir", "results", "--instance", "0"]
    evaluate.main(args)
    command, kwargs = launched[0]
    assert command[1] == str(evaluate.PROJECT_ROOT / "benchmarks/behavior/gpu_evaluator.py")
    for flag, value in {
        "--task-name": load_routes()["tasks"]["0"]["task"],
        "--output-dir": str(tmp_path / "results"),
        "--instance-indices": "0",
        "--num-rollouts": "1",
        "--replay-action-chunk-size": "0",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert kwargs["env"]["OMNIGIBSON_DATA_PATH"] == str(tmp_path / "data")
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert kwargs["env"]["OMNIGIBSON_RENDER_VIEWER_CAMERA"] == "False"
    output = tmp_path / "results"
    (output / "json" / f"{load_routes()['tasks']['0']['task']}_301_0.json").unlink()
    task = load_routes()["tasks"]["0"]["task"]
    other = output / "json" / f"{task}_302_0.json"
    other.write_text("{}")
    evaluate.main(args)
    assert len(launched) == 2 and other.read_text() == "{}"
    result = output / "json" / f"{task}_301_0.json"
    result.write_text("{}")
    with pytest.raises(SystemExit):
        evaluate.main(args)
    assert result.read_text() == "{}" and len(launched) == 2
    (output / "videos").mkdir()
    (output / "videos" / f"{task}_303_0.mp4").touch()
    with pytest.raises(SystemExit):
        evaluate.main([*args, "--instance", "2"])
    logo.unlink()
    with pytest.raises(SystemExit):
        evaluate.main(args)
    sdk_module.unlink()
    with pytest.raises(SystemExit):
        evaluate.main(args)
    assert len(launched) == 2


@pytest.mark.parametrize("mode,split,index", [("public_test", "public", 0), ("hidden_test", "hidden", 19)])
def test_evaluator_defaults_resolve_from_repository_and_task_name(tmp_path, monkeypatch, mode, split, index, evaluator_launches):
    from benchmarks.behavior import evaluate

    project = tmp_path / "policy"
    project.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", project)
    monkeypatch.setattr(evaluate, "SDK_ROOT", tmp_path / "BEHAVIOR-1K")
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path / "BEHAVIOR-1K/datasets")
    sdk_module = tmp_path / "BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval.py"
    sdk_module.parent.mkdir(parents=True)
    sdk_module.touch()
    logo = tmp_path / "BEHAVIOR-1K/docs/assets/OmniGibson_logo.png"
    logo.parent.mkdir(parents=True)
    logo.touch()
    metadata = tmp_path / "BEHAVIOR-1K/datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    launched = evaluator_launches
    evaluate.main(["--task", load_routes()["tasks"]["0"]["task"], "--mode", mode, "--instance", str(index)])
    command, kwargs = launched[0]
    expected = project / "results/evaluations" / f"standard.{split}.a2meas.a2wla.{evaluate.date.today():%Y%m%d}"
    assert command[command.index("--output-dir") + 1] == str(expected)
    assert command[command.index("--mode") + 1] == mode
    assert command[command.index("--host") + 1] == "127.0.0.1" and command[command.index("--port") + 1] == "8000"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["OMNIGIBSON_DATA_PATH"] == str(tmp_path / "BEHAVIOR-1K/datasets")


def test_evaluator_assigns_each_instance_once_to_resident_workers(tmp_path, monkeypatch, evaluator_launches):
    from benchmarks.behavior import evaluate
    sdk = tmp_path / "sdk/OmniGibson/omnigibson/eval/eval.py"
    sdk.parent.mkdir(parents=True)
    sdk.touch()
    logo = tmp_path / "sdk/docs/assets/OmniGibson_logo.png"
    logo.parent.mkdir(parents=True)
    logo.touch()
    metadata = tmp_path / "data/2026-challenge-task-instances/metadata/B100_task_misc.csv"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    output = tmp_path / "results"
    monkeypatch.setenv("A2MEAS_WORKER_CACHE", str(tmp_path / "lane-cache"))
    original = evaluate.subprocess.Popen

    def start(command, **kwargs):
        copied = Path(kwargs["env"]["OMNIGIBSON_APPDATA_PATH"]) / "local"
        copied.mkdir(parents=True, exist_ok=True)
        marker = copied / "reused"
        if len(evaluator_launches) < 4:
            marker.write_text("warm")
        else:
            assert marker.read_text() == "warm"
        return original(command, **kwargs)

    monkeypatch.setattr(evaluate.subprocess, "Popen", start)
    args = ["--task", "0", "--behavior-dir", str(tmp_path / "sdk"), "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(output), "--workers", "4", "--endpoint", "localhost:9000",
            "--instance", *map(str, range(10))]
    evaluate.main(args)
    assert len(evaluator_launches) == 4
    for index, (command, kwargs) in enumerate(evaluator_launches):
        env = kwargs["env"]
        assert json.loads(env["A2MEAS_TASK_GROUPS"])[0]["instances"] == list(range(index, 10, 4))
        assert command[command.index("--num-envs") + 1] == "1"
        assert command[command.index("--port") + 1] == str(9000 + index)
        assert not Path(env["TMPDIR"]).exists()
    assert len({kwargs["env"]["OMNIGIBSON_APPDATA_PATH"] for _, kwargs in evaluator_launches}) == 4
    args[args.index("--output-dir") + 1] = str(tmp_path / "second-results")
    evaluate.main(args)
    assert len(evaluator_launches) == 8
    with pytest.raises(SystemExit):
        evaluate.main(args)

    processes = []

    def start_with_one_failure(command, **kwargs):
        process = original(command, **kwargs)
        processes.append(process)
        if len(processes) == 1:
            process.returncode = 1
        else:
            process.returncode = None
            polls = iter((None, 0))

            def poll():
                process.returncode = next(polls, 0)
                return process.returncode

            process.poll = poll
        return process

    monkeypatch.setattr(evaluate.subprocess, "Popen", start_with_one_failure)
    args[args.index("--output-dir") + 1] = str(tmp_path / "partial-failure")
    with pytest.raises(evaluate.subprocess.CalledProcessError):
        evaluate.main(args)
    assert [process.returncode for process in processes] == [1, 0, 0, 0]


def test_camera_inputs_require_the_official_rgb_schema():
    from a2meas.behavior.observations import _camera_rgb
    from benchmarks.behavior.gpu_evaluator import compact_rgb

    rgb = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    rgba = np.concatenate([rgb, np.full((224, 224, 1), 255, dtype=np.uint8)], axis=-1)
    for image in (rgb, rgba):
        frame = _camera_rgb(image, 224)
        np.testing.assert_array_equal(frame, rgb)
        np.testing.assert_array_equal(compact_rgb(image, 224), rgb)
        assert not np.shares_memory(frame, image)
    for image in (rgb.astype(np.float32), rgb.transpose(2, 0, 1), rgb[..., :1], rgb[:0], rgb[:112]):
        with pytest.raises(ValueError, match="Camera observations"):
            _camera_rgb(image, 224)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("explicit", [False, True], ids=["native-cache", "configured-cache"])
def test_runtime_cache_reuse_and_owned_temporary_cleanup(tmp_path, monkeypatch, fail, explicit):
    import os
    import tempfile
    import signal
    from contextlib import nullcontext
    import a2meas

    monkeypatch.setattr(a2meas, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-auth"))
    system_temporary = tmp_path / "system-temporary"
    system_temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_temporary))
    if explicit:
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "external"))
    else:
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    original = dict(os.environ)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_temporary = tempfile.tempdir
    other = tmp_path / ".cache/runs/other-process"
    other.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="run failed") if fail else nullcontext():
        with a2meas.runtime_environment():
            cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
            assert Path(os.environ["HF_HOME"]) == tmp_path / "hf-auth"
            assert cache == (tmp_path / "external" if explicit else tmp_path / "runtime/cache/inductor")
            temporary = Path(os.environ["TMPDIR"])
            assert temporary.parent == system_temporary
            assert not cache.is_relative_to(temporary)
            assert Path(os.environ["TRITON_CACHE_DIR"]) == tmp_path / "runtime/cache/triton"
            assert all(os.environ[key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            assert Path(tempfile.gettempdir()) == temporary
            with tempfile.NamedTemporaryFile() as scratch:
                assert Path(scratch.name).parent == temporary
            (cache / "kernel").write_text("compiled")
            (temporary / "scene.usd").write_text("temporary scene")
            if fail:
                raise RuntimeError("run failed")
    assert not temporary.exists() and other.is_dir()
    assert dict(os.environ) == original
    assert signal.getsignal(signal.SIGTERM) == original_sigterm
    assert tempfile.tempdir == original_temporary
    with a2meas.runtime_environment():
        assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == cache
        assert (cache / "kernel").read_text() == "compiled"
        assert Path(os.environ["TMPDIR"]) != temporary
        assert Path(tempfile.gettempdir()) == Path(os.environ["TMPDIR"])
    assert list(other.parent.iterdir()) == [other]



@pytest.mark.parametrize("cpu_count", [2, 64])
def test_simulator_keeps_cache_settings_and_isolates_run_files(tmp_path, monkeypatch, cpu_count):
    from benchmarks.behavior.gpu_evaluator import simulation_config
    monkeypatch.setenv("OMNIGIBSON_GPU_ID", "3")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpu_count)))
    config = simulation_config({"headless": True, "extra_args": ["--existing"]})
    assert config["active_gpu"] == 3 and config["physics_gpu"] == 0
    assert config["limit_cpu_threads"] == min(16, cpu_count)
    assert config["extra_args"] == ["--existing", f"--/log/file={tmp_path / 'kit.log'}",
                                    f"--/app/userConfigPath={tmp_path / 'user.config.json'}",
                                    '--/log/level=warning', '--/log/outputStreamLevel=error',
                                    '--/log/fileLogLevel=warning']


def test_relative_resources_hotdog_data_and_cache_from_another_directory(
    tmp_path, monkeypatch
):
    from a2meas.behavior import cli as serve
    import a2meas.behavior.server as server

    cwd = tmp_path / "directory with spaces"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    checkpoint = cwd / "model-files/adapter/adapter.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selected adapter")
    calls = []

    def prepare(path, *, adapter):
        assert path == cwd / "model-files" and adapter
        assert (path / "adapter/adapter.safetensors").read_bytes() == b"selected adapter"
        calls.append("prepare")
        return 2

    class Manager:
        def __init__(self, path, cpus, *, adapter, num_inference_steps=None, seed=0, compile_mode="max-autotune-no-cudagraphs"):
            assert path == cwd / "model-files" and adapter
            assert os.environ["OMNIGIBSON_DATA_PATH"] == str(cwd / "dataset")
            assert (
                Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
                == cwd.parent / "compiler-cache"
            )
            calls.append("serve")

        def close(self):
            calls.append("close")

    async def run(manager, *args):
        manager.close()

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "../compiler-cache")
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "OMNIGIBSON_DATA_PATH",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(resources, "prepare", prepare)
    monkeypatch.setattr(resources, "verify_shared", lambda path: None)
    monkeypatch.setattr(serve, "cpu_layout", lambda gpu, workers: ({0}, [{0}]))
    monkeypatch.setattr(server, "RouteManager", Manager)
    monkeypatch.setattr(server, "run", run)
    serve.main(
        [
            "prepare",
            "--resources",
            "./model-files",
            "--adapter",
        ]
    )
    serve.main(
        [
            "serve",
            "--resources",
            "./model-files",
            "--adapter",
            "--data-dir",
            "./dataset",
        ]
    )
    assert calls == ["prepare", "serve", "close"]


def test_runtime_rejects_whitespace_in_torch_compiler_cache(tmp_path, monkeypatch):
    import tempfile
    import a2meas

    monkeypatch.setattr(a2meas, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "cache with spaces"))
    previous = tempfile.tempdir
    with pytest.raises(ValueError, match="without whitespace"):
        with a2meas.runtime_environment():
            pytest.fail("Unsupported compiler cache was accepted")
    assert tempfile.tempdir == previous
    assert not (tmp_path / ".cache").exists()


def test_websocket_bind_failure_closes_policy():
    import asyncio
    import socket
    from a2meas.behavior.server import run

    closed = []
    policy = SimpleNamespace(close=lambda: closed.append(True))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            asyncio.run(run(policy, "127.0.0.1", listener.getsockname()[1]))
    assert closed == [True]


def test_multiworker_start_failure_closes_each_manager_once(monkeypatch, tmp_path):
    import socket
    from a2meas.behavior import cli
    import a2meas.behavior.server as implementation

    closed = []
    managers = []

    class Manager:
        def __init__(self, *args, **kwargs):
            managers.append(self)

        def alive(self):
            return True

        def close(self):
            closed.append(self)

    monkeypatch.setattr(implementation, "RouteManager", Manager)
    monkeypatch.setattr(resources, "verify_shared", lambda path: None)
    monkeypatch.setattr(cli, "cpu_layout", lambda *_: (set(), [set(), set()]))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            cli.main(["serve", "--port", str(listener.getsockname()[1] - 1), "--workers", "2"])
    assert len(managers) == 2
    assert sorted(map(id, closed)) == sorted(map(id, managers))


@pytest.mark.parametrize("change", [{"model": "agbqsy"}, {"scale": [1., 1., 1.]}, {"fixed_base": True}])
def test_scene_restore_rebuilds_changed_construction_and_keeps_compatible_objects(change):
    import ast
    from contextlib import nullcontext
    from copy import deepcopy
    from a2meas.behavior import SDK_ROOT

    path = SDK_ROOT / "OmniGibson/omnigibson/scenes/scene_base.py"
    if not path.is_file():
        pytest.skip("Install the BEHAVIOR SDK for its restore regression")
    scene_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == "Scene")
    restore = next(node for node in scene_class.body if isinstance(node, ast.FunctionDef) and node.name == "restore")
    def info(name, **args):
        return {"class_module": "objects", "class_name": "DatasetObject", "args": {"name": name, **args}}
    events = []
    class Layer:
        def Apply(self, edit):
            events.append("erase")
            return True
    scene_layer, asset_layer = Layer(), Layer()
    def prim(layer, path):
        return SimpleNamespace(GetPrimStack=lambda: [SimpleNamespace(layer=layer, path=path)])
    def obj(value):
        return SimpleNamespace(name=value["args"]["name"], get_init_info=lambda: value,
                               relative_prim_path="/" + value["args"]["name"], prim=SimpleNamespace(GetPrimStack=lambda: [
                                   SimpleNamespace(layer=scene_layer, path="/World/book"),
                                   SimpleNamespace(layer=asset_layer, path="/Asset")]))
    original = info("book", model="poanfs", scale=[.9, .9, .9], fixed_base=False)
    target = deepcopy(original)
    target["args"].update(change)
    objects = {"keep": obj(info("keep", expected_file_hash="before")), "book": obj(original),
               "old_template": obj(info("old_template"))}
    keep, book = objects["keep"], objects["book"]
    expected, ancestral = [target], [True]
    class Registry:
        def get_dict(self, key):
            return objects
        def __call__(self, key, name):
            return objects[name]
    class Scene:
        object_registry = Registry()
        _scene_prim = SimpleNamespace(prim=prim(scene_layer, "/World"))
        active_systems = {"old": object()}
        _task_metadata = {"stale": True}
        transition_rule_api = SimpleNamespace(clear=lambda: events.append("clear"),
                                             refresh_all_rules=lambda: events.append("refresh"))
        _check_versions_compatible = staticmethod(lambda versions: None)
        def clear_system(self, name):
            del self.active_systems[name]
            del objects["old_template"]
        def get_system(self, name, force_init):
            self.active_systems[name] = object()
            objects["new_template"] = obj(info("new_template"))
        def load_state(self, state, serialized):
            assert sim.is_playing() and objects["book"].get_init_info() == expected[0]
    playing = [True]
    def stop():
        playing[0] = False
        events.append("stop")
    def play():
        playing[0] = True
        events.append("play")
    def remove(entries):
        assert not playing[0]
        events.append("remove")
        for item in entries:
            del objects[item.name]
    def add(entries, scenes):
        assert not playing[0]
        events.append("add")
        objects.update((item.name, item) for item in entries)
    sim = SimpleNamespace(is_stopped=lambda: False, is_playing=lambda: playing[0], stop=stop, play=play,
                          batch_remove_objects=remove, batch_add_objects=add, editing_usd=nullcontext,
                          stage=SimpleNamespace(GetPrimAtPath=lambda path: prim(scene_layer, path) if ancestral[0] else None))
    gm = SimpleNamespace(ENABLE_TRANSITION_RULES=True, unlocked=nullcontext)
    namespace = dict(json=json, deepcopy=deepcopy, TorchEncoder=json.JSONEncoder, gm=gm,
                     og=SimpleNamespace(sim=sim), create_object_from_init_info=obj,
                     recursively_convert_to_torch=lambda value: value,
                     scene_relative_prim_path_to_absolute=lambda scene, path: "/World" + path,
                     lazy=SimpleNamespace(pxr=SimpleNamespace(Sdf=SimpleNamespace(
                         BatchNamespaceEdit=lambda: SimpleNamespace(Add=lambda path, target: None),
                         Path=SimpleNamespace(emptyPath="")))))
    exec(compile(ast.Module(body=[restore], type_ignores=[]), str(path), "exec"), namespace)
    Scene.restore = namespace["restore"]
    scene = Scene()
    saved = {"init_info": {"class_name": "Scene"}, "metadata": {"task": {"task": "new"}},
             "objects_info": {"init_info": {"keep": info("keep", expected_file_hash="after"),
                                            "book": target, "new_template": info("new_template")}},
             "state": {"registry": {"system_registry": {"new": {}}}}}
    scene.restore(saved)
    assert objects["keep"] is keep and objects["book"] is not book
    assert set(objects) == {"keep", "book", "new_template"}
    assert scene._task_metadata == {"task": "new"} and gm.ENABLE_TRANSITION_RULES
    assert events == ["clear", "stop", "remove", "erase", "add", "play", "refresh"]
    scene.restore(saved)
    assert events == ["clear", "stop", "remove", "erase", "add", "play", "refresh"]
    # Objects loaded by the first restore are ordinary prims and are deleted completely.
    ancestral[0] = False
    expected[0] = original
    saved["objects_info"]["init_info"]["book"] = original
    scene.restore(saved)
    assert events[-6:] == ["clear", "stop", "remove", "add", "play", "refresh"]
    # A previously removed object may still have an inactive ancestral prim.
    del objects["book"]
    ancestral[0] = True
    expected[0] = target
    saved["objects_info"]["init_info"]["book"] = target
    scene.restore(saved)
    assert events[-6:] == ["clear", "stop", "erase", "add", "play", "refresh"]
