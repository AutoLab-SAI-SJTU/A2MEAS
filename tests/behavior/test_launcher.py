"""Full-suite selection, denoising overrides and shared-judge scheduling."""

from contextlib import contextmanager
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace
import threading

import pytest

from benchmarks.behavior import run
from benchmarks import runtime


@pytest.mark.parametrize("headers_available", [False, True])
def test_setup_checks_managed_headers_before_creating_environments(tmp_path, headers_available):
    python = tmp_path / "runtime/python310/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport sys, sysconfig\n"
        "if sys.argv[1:] != ['-']:\n    sys.exit('reached venv creation')\n"
        + ("" if headers_available else f"sysconfig.get_path = lambda key: {str(tmp_path)!r}\n")
        + "exec(sys.stdin.read())\n"
    )
    python.chmod(0o755)
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    script = checkout / 'setup.sh'
    script.write_bytes((run.PROJECT_ROOT / 'setup.sh').read_bytes())
    result = subprocess.run(["bash", str(script)],
                            env=dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
                                     A2MEAS_RUNTIME=str(tmp_path / 'runtime')),
                            capture_output=True, text=True)
    assert result.returncode != 0
    if headers_available:
        assert "reached venv creation" in result.stderr
    else:
        assert "Missing Python development headers" in result.stderr
        assert "reached venv creation" not in result.stderr


def test_shared_gpu_locks_use_hardware_uuid_instead_of_container_index(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUNTIME_ROOT", tmp_path)

    def query(command, **kwargs):
        assert "--query-gpu=uuid" in command
        return "GPU-A" if command[1] in ("--id=1", "--id=4") else "GPU-B"

    monkeypatch.setattr(run.subprocess, "check_output", query)
    with run.reserve_gpus([1]):
        with pytest.raises(RuntimeError, match="GPU 4 is reserved"):
            with run.reserve_gpus([4]):
                pytest.fail("A remapped index acquired the same physical GPU")
        with run.reserve_gpus([0]):
            pass
    assert {p.name for p in (tmp_path / "cache/gpu-locks").iterdir()} == {"gpu-a.lock", "gpu-b.lock"}


def test_gpu_reservation_rejects_overlap_and_releases_partial_acquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(run.subprocess, "check_output",
                        lambda command, **kwargs: command[1].split("=")[1])
    with run.reserve_gpus([1]):
        with pytest.raises(RuntimeError, match="GPU 1 is reserved"):
            with run.reserve_gpus([0, 1]):
                pytest.fail("Overlapping evaluations acquired the same GPU")
        with run.reserve_gpus([0]):
            pass
    with run.reserve_gpus([0, 1]):
        pass


def test_adapter_reserves_a_separate_judge_gpu(monkeypatch):
    @contextmanager
    def reserve(gpus):
        assert set(gpus) == {0, 1}
        raise RuntimeError("checked GPU reservations")
        yield

    monkeypatch.setattr(run, "reserve_gpus", reserve)
    with pytest.raises(RuntimeError, match="checked GPU reservations"):
        run.main(["--adapter", "--task", "0", "--gpus", "0", "--judge-gpu", "1"])


@pytest.mark.parametrize("steps", [None, 2, 10])
def test_full_plan_preserves_routes_and_covers_each_instance_once(capsys, steps):
    args = ["--dry-run", "--adapter", "--gpus", *map(str, range(8)), "--workers-per-gpu", "4"]
    if steps is not None:
        args += ["--num-inference-steps", str(steps)]
    run.main(args)
    plan = json.loads(capsys.readouterr().out)
    assert [phase["name"] for phase in plan] == ["base", "adapter"]
    assert [len(phase["tasks"]) for phase in plan] == [88, 12]
    tasks = [task for phase in plan for task in phase["tasks"]]
    assert sorted(task["id"] for task in tasks) == list(range(100))
    assert sum(len(task["instances"]) for task in tasks) == 1000
    routes, adapter = run.load_routes(), run.load_config()
    for task in tasks:
        assert task["num_inference_steps"] == (steps if steps is not None else 2)
        assert task["instances"] == list(range(10))
        assert task["model"] == (adapter["model"] if str(task["id"]) in adapter["tasks"]
                                  else routes["tasks"][str(task["id"])]["model"])
    assert [lane["workers"] for lane in plan[0]["lanes"]] == [4] * 8
    assert [lane["workers"] for lane in plan[1]["lanes"]] == [4] * 7
    assert 7 not in {lane["gpu"] for lane in plan[1]["lanes"]}


def test_generic_plan_keeps_configured_steps_without_adapter(capsys):
    run.main(["--dry-run", "--gpus", "2", "5", "--task", "0", "6", "45",
              "--instance", "3", "8", "--workers-per-gpu", "3"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) == 1 and plan[0]["name"] == "base"
    assert plan[0]["lanes"] == [{"gpu": 2, "workers": 3}, {"gpu": 5, "workers": 3}]
    for task in plan[0]["tasks"]:
        route = run.load_routes()["tasks"][str(task["id"])]
        assert task["model"] == route["model"]
        assert task["num_inference_steps"] == route["num_inference_steps"]
        assert task["instances"] == [3, 8]


def test_public_base_suite_defaults_to_880_episodes(capsys):
    routes, adapter = run.load_routes(), run.load_config()
    selected = [task for task in routes["tasks"] if task not in adapter["tasks"]]
    run.main(["--dry-run", "--workers-per-gpu", "6",
              "--gpus", *map(str, range(8)), "--task", *selected])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan) == 1 and plan[0]["name"] == "base"
    assert len(plan[0]["tasks"]) == 88
    assert [lane["workers"] for lane in plan[0]["lanes"]] == [6] * 8
    assert sum(len(task["instances"]) for task in plan[0]["tasks"]) == 880
    for task in plan[0]["tasks"]:
        assert str(task["id"]) not in adapter["tasks"]
        assert task["instances"] == list(range(10))
        assert task["model"] == routes["tasks"][str(task["id"])]["model"]
        assert task["num_inference_steps"] == routes["tasks"][str(task["id"])]["num_inference_steps"]


@pytest.mark.parametrize("failure", [None, "simulator", "policy", "policy-exit"])
@pytest.mark.parametrize("physical_cores", [4, 12])
def test_phase_dispatch_and_failure_cleanup(tmp_path, monkeypatch, failure, physical_cores):
    args = SimpleNamespace(
        base_port=18000, port_stride=2, workers_per_gpu=2, instances_per_claim=2, seed=0, mode="public_test", compile_mode="default", max_steps=None, video=False,
        judge_port=8001, judge_concurrency=4,
        model_python=tmp_path / "model/bin/python", behavior_python=tmp_path / "sim/bin/python",
        behavior_dir=tmp_path / "sdk", data_dir=tmp_path / "data",
        resources=tmp_path / "resources", output_dir=tmp_path,
        evaluation_dir=tmp_path / "evaluations/standard.public.a2meas.a2wla.20261102",
        num_inference_steps=2,
    )
    phase = dict(name="adapter", lanes=[dict(gpu=i, workers=1 if i == 7 else 2) for i in range(8)],
                 tasks=[dict(id=i, model=f"model_{i:02d}", num_inference_steps=2, instances=[0, 1]) for i in range(12)])
    calls, closed = [], []
    failed_policy = []
    lock = threading.Lock()

    @contextmanager
    def running(command, env):
        is_policy = "serve.py" in command
        with lock:
            calls.append((command, env))
        try:
            if is_policy:
                poll = lambda: 1 if failure == "policy-exit" else None
            else:
                poll = lambda: None if failure == "policy-exit" else 1 if failure == "simulator" else 0
            yield SimpleNamespace(poll=poll, returncode=1 if failure == "simulator" else 0)
        finally:
            with lock:
                closed.append(command)

    def ready(_process, url, *_):
        with lock:
            if failure == "policy" and not failed_policy:
                failed_policy.append(True)
                raise RuntimeError("policy exited")

    monkeypatch.setattr(run, "running", running)
    monkeypatch.setattr(run, "wait_ready", ready)
    monkeypatch.setattr(run, "scene_contracts", lambda data: {i: ("scene", ("room",)) for i in range(100)})
    routes = run.load_routes()
    monkeypatch.setattr(run, "load_routes", lambda: routes)
    monkeypatch.setattr(Path, "read_text", lambda path: str(int(path.parts[-3][3:]) // 2))
    pools = {i: list(range(i * 24, i * 24 + 2 * physical_cores)) for i in range(8)}
    stop = threading.Event()
    if failure in (None, "simulator"):
        run.evaluate_phase(args, phase, pools, stop)
        assert not stop.is_set()
    else:
        error = RuntimeError
        with pytest.raises(error):
            run.evaluate_phase(args, phase, pools, stop)
        assert stop.is_set()
    jobs = [cmd for cmd, _ in calls if "evaluate.py" in cmd]
    episodes = [(int(cmd[cmd.index("--task") + 1]), int(instance)) for cmd in jobs
                for instance in cmd[cmd.index("--instance") + 1:]]
    assert len(episodes) == len(set(episodes))
    if failure is None:
        assert sorted(episodes) == [(task, instance) for task in range(12) for instance in range(2)]
    if failure is None:
        assert all("--simulator-cpus" in cmd for cmd in jobs)
        policies = [cmd for cmd, _ in calls if "serve.py" in cmd]
        assert 1 <= len(policies) <= 15
        assert len({cmd[cmd.index("--port") + 1] for cmd in policies}) == len(policies)
        assert all(cmd[cmd.index("--workers") + 1] == "1" for cmd, _ in calls)
        for command, env in calls:
            assert env["A2MEAS_REVIEW_CONCURRENCY"] == "4"
            if "serve.py" in command:
                assert "--adapter" in command
                assert command[command.index("--num-inference-steps") + 1] == "2"
    assert len(calls) == len(closed)
    assert all(not Path(env["A2MEAS_WORKER_CACHE"]).exists() for _, env in calls)


def test_worker_claims_next_task_before_peer_finishes(tmp_path, monkeypatch):
    args = SimpleNamespace(
        base_port=18000, port_stride=2, instances_per_claim=2, seed=0, mode="public_test", compile_mode="default", max_steps=None, video=False, judge_port=8001, judge_concurrency=4, num_inference_steps=None,
        model_python=tmp_path / "model-python", behavior_python=tmp_path / "sim-python",
        resources=tmp_path, data_dir=tmp_path, behavior_dir=tmp_path,
        output_dir=tmp_path, evaluation_dir=tmp_path / "evaluations",
    )
    phase = dict(name="base", lanes=[dict(gpu=0, workers=2)],
                 tasks=[dict(id=i, model=f"model_{i:02d}", num_inference_steps=2, instances=[0, 1])
                        for i in range(4)])
    slow_started, fast_advanced = threading.Event(), threading.Event()
    jobs, closed, caches = [], [], {}
    fast_jobs = 0

    @contextmanager
    def running(command, env):
        nonlocal fast_jobs
        is_policy = "serve.py" in command
        if is_policy:
            poll = lambda: None
        else:
            assert command[command.index("--workers") + 1] == "1"
            instances = command[command.index("--instance") + 1:]
            assert instances == ["0", "1"]
            jobs.append(int(command[command.index("--task") + 1]))
            endpoint = command[command.index("--endpoint") + 1]
            cache = Path(env["A2MEAS_WORKER_CACHE"])
            assert cache.is_dir()
            if endpoint in caches:
                assert caches[endpoint] == cache
                with (cache / "warm").open() as marker:
                    assert marker.read() == endpoint
            else:
                caches[endpoint] = cache
                (cache / "warm").write_text(endpoint)
            if endpoint.endswith(":18000"):
                slow_started.set()
                def poll():
                    assert fast_advanced.wait(5), "Fast worker waited for its slow peer"
                    return 0
            else:
                assert slow_started.wait(5)
                fast_jobs += 1
                if fast_jobs == 2:
                    fast_advanced.set()
                poll = lambda: 0
        try:
            yield SimpleNamespace(poll=poll, returncode=0)
        finally:
            closed.append(command)

    monkeypatch.setattr(run, "running", running)
    monkeypatch.setattr(run, "wait_ready", lambda *args: None)
    monkeypatch.setattr(run, "scene_contracts", lambda data: {i: ("scene", ("room",)) for i in range(100)})
    routes = run.load_routes()
    monkeypatch.setattr(run, "load_routes", lambda: routes)
    monkeypatch.setattr(Path, "read_text", lambda path: str(int(path.parts[-3][3:]) // 2))
    run.evaluate_phase(args, phase, {0: list(range(12))}, threading.Event())
    assert fast_advanced.is_set()
    assert sorted(jobs) == list(range(4))
    assert len(closed) == 6
    assert all(not cache.exists() for cache in caches.values())


def test_judge_exit_is_not_reported_as_adapter_evaluation_success(tmp_path, monkeypatch):
    import subprocess

    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        runtime.stop_process(process)
        assert process.poll() is not None
        with pytest.raises(RuntimeError, match="Service exited"):
            run.wait_ready(process, 'http://127.0.0.1:1/healthz', threading.Event(), 1)
    finally:
        runtime.stop_process(process)


def test_judge_preparation_downloads_only_missing_files_and_rejects_corruption(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub
    from a2meas.behavior import resources

    root = tmp_path / "configs/behavior"
    root.mkdir(parents=True)
    (root / "judge.json").write_text(json.dumps(dict(repo_id="test/judge", revision="pinned")))
    (root / "judge.sha256").write_text(hashlib.sha256(b"weights").hexdigest() + "  model.safetensors\n")
    monkeypatch.setattr(resources, "CONFIG_ROOT", root.parent / "behavior")
    calls = []

    def download(**kwargs):
        assert kwargs["repo_id"] == "test/judge" and kwargs["revision"] == "pinned"
        calls.append(kwargs["filename"])
        (kwargs["local_dir"] / kwargs["filename"]).write_bytes(b"weights")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    destination = tmp_path / "judge"
    resources.prepare_judge(destination)
    resources.prepare_judge(destination)
    assert calls == ["model.safetensors"]
    (destination / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resources.prepare_judge(destination)
    assert calls == ["model.safetensors"]


def test_summary_uses_the_planned_instance_count_and_rejects_missing_results(tmp_path):
    from benchmarks.behavior.results import summarize
    task = run.load_routes()["tasks"]["0"]["task"]
    (tmp_path / "run.json").write_text(json.dumps({"identity": {"mode": "public_test"}, "phases": [{"tasks": [{"id": 0, "instances": [0, 1]}]}]}))
    directory = tmp_path / "evaluations/standard.public.a2meas.a2wla.20261103/json"
    directory.mkdir(parents=True)
    for index, success in enumerate((True, False)):
        result = {"task": task, "instance_id": 301 + index, "rollout_id": 0,
                  "success": success, "q_score": {"final": float(success)}}
        (directory / f"{task}_{301 + index}_0.json").write_text(json.dumps(result))
    summary = summarize(tmp_path)
    assert summary["episodes"] == 2 and summary["success_rate"] == 0.5
    assert json.loads((directory / f"{task}_302_0.json").read_text())["success"] is False
    (directory / f"{task}_302_0.json").unlink()
    with pytest.raises(ValueError, match="Missing 1 planned result"):
        summarize(tmp_path)
    failed = {"task": task, "instance_id": 301, "rollout_id": 0, "success": False,
              "failure_reason": "policy_connection_lost", "q_score": {"final": 0.0}}
    path = directory / f"{task}_301_0.json"
    path.write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="Invalid evaluation result"):
        summarize(tmp_path)
    assert json.loads(path.read_text()) == failed


def test_resident_worker_requests_compatible_tasks_and_prefers_same_model(tmp_path, monkeypatch):
    import socket
    args = SimpleNamespace(base_port=18000, port_stride=1, instances_per_claim=2, seed=0, mode="public_test", compile_mode="default", max_steps=None, video=False, judge_port=8001, judge_concurrency=4,
                           num_inference_steps=None, model_python=tmp_path / "model", behavior_python=tmp_path / "sim",
                           resources=tmp_path, data_dir=tmp_path, behavior_dir=tmp_path,
                           output_dir=tmp_path, evaluation_dir=tmp_path / "evaluations")
    phase = {"name": "base", "lanes": [{"gpu": 0, "workers": 1}], "tasks": [
        {"id": 0, "model": "same", "num_inference_steps": 2, "instances": [0, 1]},
        {"id": 1, "model": "other", "num_inference_steps": 2, "instances": [0]},
        {"id": 2, "model": "same", "num_inference_steps": 2, "instances": [0]},
    ]}
    routes = run.load_routes()
    claimed = []
    @contextmanager
    def running(command, env):
        if "evaluate.py" not in command:
            yield SimpleNamespace(poll=lambda: None, returncode=0)
            return
        task_id = int(command[command.index("--task") + 1])
        claimed.append(task_id)
        while True:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(env["A2MEAS_DISPATCH_SOCKET"])
                client.sendall((json.dumps({"port": 18000, "task_id": task_id}) + "\n").encode())
                with client.makefile("rb") as stream:
                    job = json.loads(stream.readline())
            if job is None:
                break
            task_id = next(int(key) for key, route in routes["tasks"].items() if route["task"] == job["task"])
            claimed.append(task_id)
        yield SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(run, "running", running)
    monkeypatch.setattr(run, "wait_ready", lambda *args: None)
    monkeypatch.setattr(run, "scene_contracts", lambda data: {i: ("scene", ("room",)) for i in range(100)})
    monkeypatch.setattr(run, "load_routes", lambda: routes)
    monkeypatch.setattr(Path, "read_text", lambda path: str(int(path.parts[-3][3:]) // 2))
    run.evaluate_phase(args, phase, {0: list(range(8))}, threading.Event())
    assert claimed == [0, 2, 1]


@pytest.mark.parametrize('timeout', ['group_time_budget_exceeded', 'policy_query_timeout'])
@pytest.mark.parametrize('fail_again', [False, True])
@pytest.mark.parametrize('partial', [False, True])
def test_only_unfinished_simulator_episode_gets_one_retry(tmp_path, monkeypatch, timeout, fail_again, partial):
    args = SimpleNamespace(evaluation_dir=tmp_path, mode='public_test')
    phase = dict(name='base', tasks=[dict(id=0, instances=[0, 1])])
    task = run.load_routes()['tasks']['0']['task']
    directory = tmp_path / 'json'
    directory.mkdir()
    calls = []

    def evaluate(args, remaining, pools, stop, judge, *, attempts):
        indices = remaining['tasks'][0]['instances']
        calls.append(list(indices))
        for index in indices:
            attempts[(0, index)] = attempts.get((0, index), 0) + 1
            if index == 1 and (len(calls) == 1 or fail_again):
                if partial:
                    (directory / f'{task}_{301 + index}_0.json').write_text('{')
                continue
            result = dict(task=task, instance_id=301 + index, rollout_id=0,
                          success=False, q_score={'final': 0.0})
            if index == 0:
                result['failure_reason'] = timeout
            (directory / f'{task}_{301 + index}_0.json').write_text(json.dumps(result))

    monkeypatch.setattr(run, 'evaluate_phase', evaluate)
    if fail_again:
        with pytest.raises(RuntimeError, match='Simulator failed twice'):
            run.run_phase(args, phase, {}, threading.Event())
    else:
        run.run_phase(args, phase, {}, threading.Event())
    assert calls == [[0, 1], [1]]
    if partial:
        assert (directory / f'{task}_302_0.json.failed').read_text() == '{'
    assert json.loads((directory / f'{task}_301_0.json').read_text())['failure_reason'] == timeout


@pytest.mark.parametrize('timeout', ['group_time_budget_exceeded', 'policy_query_timeout'])
def test_timeout_requires_failed_zero_score(timeout):
    from benchmarks.behavior.results import validate_result
    result = dict(task='task', instance_id=301, rollout_id=0, success=False,
                  q_score={'final': 0.0}, failure_reason=timeout)
    assert validate_result(result, 'task', 301) is result
    for changes in ({'success': True}, {'q_score': {'final': 0.5}}):
        with pytest.raises(ValueError, match='Timeout must be a failed zero-score'):
            validate_result({**result, **changes}, 'task', 301)
