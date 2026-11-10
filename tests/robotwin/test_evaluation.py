"""Key evaluation behavior and external boundaries."""

import hashlib
import json
import pytest
import os
from pathlib import Path
import yaml
from benchmarks.robotwin.benchmark import RoboTwinBenchmark
from benchmarks.robotwin.prepare import prepare
from benchmarks.robotwin.results import collect
from benchmarks.robotwin.summarize import main


@pytest.mark.parametrize('preinstalled', [False, True])
def test_prepare_downloads_standard_policy_and_reuses_verified_local_weights(tmp_path, monkeypatch, preinstalled):
    from a2meas.robotwin import resources
    import huggingface_hub

    config = tmp_path / 'configs'
    destination = tmp_path / 'resources'
    expected = b'policy tensors'
    files = {
        'robotwin/policy/config.json': b'{}',
        'robotwin/policy/model.safetensors': expected,
        'assets/rynnbrain/config.json': b'{}',
        'robotwin/expert_cache/seeds.json': (json.dumps({'tasks': [1, 2]}, indent=2) + '\n').encode(),
        'robotwin/simulator/envs/task.py': b'reference SDK',
    }
    entry = lambda data: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    policy = destination / 'robotwin/policy/model.safetensors'
    cache = destination / 'robotwin/expert_cache/seeds.json'
    cache.parent.mkdir(parents=True)
    if preinstalled:
        cache.write_bytes(files['robotwin/expert_cache/seeds.json'])
        policy.parent.mkdir(parents=True)
        policy.write_bytes(expected)
    metadata = {
        'robotwin/resources.json': dict(files={
            'policy/model.safetensors': entry(expected), 'policy/config.json': entry(b'{}'),
            'expert_cache/seeds.json': entry(files['robotwin/expert_cache/seeds.json']),
            'simulator/envs/task.py': entry(files['robotwin/simulator/envs/task.py']),
        }),
        'robotwin/downloads.json': dict(files={k: entry(v) for k, v in files.items() if k.startswith(('robotwin/policy/', 'assets/'))}),
        'robotwin/episodes.json': {'seeds.json': {'tasks': [1, 2]}},
        'robotwin/release.json': dict(repo_id='policy', revision='policy-revision'),
    }
    for name in ("expert_cache/seeds.json", "simulator/envs/task.py"):
        source = (tmp_path / "third_party/robotwin" / name.removeprefix("simulator/")
                  if name.startswith("simulator/") else config / "robotwin/reference" / name)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(files["robotwin/" + name])
    for name, value in metadata.items():
        path = config / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    calls = []

    def download(*, repo_id, revision, filename, token, local_dir):
        assert (repo_id, revision) == ('policy', 'policy-revision')
        assert token is None
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])
        calls.append(filename)
        return str(target)

    monkeypatch.setattr(resources, 'CONFIG_ROOT', config)
    monkeypatch.setattr(resources, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    cache.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='resource|checksum'):
        resources.prepare(destination)
    assert calls == []
    cache.unlink()
    assert resources.prepare(destination) == 5
    assert policy.read_bytes() == expected
    assert ('robotwin/policy/model.safetensors' in calls) is (not preinstalled)
    assert not any(name.startswith(("robotwin/expert_cache/", "robotwin/simulator/")) for name in calls)
    calls.clear()
    assert resources.prepare(destination) == 5 and calls == []
    # A locally complete release is usable before public download coordinates exist.
    (config / 'robotwin/release.json').write_text(json.dumps(dict(repo_id=None, revision=None)))
    assert resources.prepare(destination) == 5 and calls == []
    backbone = destination / 'assets/rynnbrain/config.json'
    backbone.unlink()
    with pytest.raises(RuntimeError, match='has not been pinned'):
        resources.prepare(destination)
    assert calls == []
    backbone.write_bytes(b'{}')
    policy.write_bytes(b'broken')
    with pytest.raises(ValueError, match='resource'):
        resources.prepare(destination)
    assert policy.read_bytes() == b'broken' and not policy.with_suffix('.partial').exists()


ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_reads_simulator_root_at_construction(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOTWIN_ROOT', str(tmp_path / 'first'))
    first = RoboTwinBenchmark('stamp_seal', 'episodes.json')
    monkeypatch.setenv('ROBOTWIN_ROOT', str(tmp_path / 'second'))
    second = RoboTwinBenchmark('stamp_seal', 'episodes.json')
    assert first.robotwin_root == str(tmp_path / 'first')
    assert second.robotwin_root == str(tmp_path / 'second')


def test_runtime_rebases_yaml_without_mutating_source(tmp_path):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    embodiment = source / "assets/embodiments/aloha"
    embodiment.mkdir(parents=True)
    original = "urdf_path: /old/robot/assets/embodiments/aloha/robot.urdf\n"
    (embodiment / "config.yml").write_text(original)
    (embodiment / "robot.urdf").write_text("<robot/>")
    (source / "envs").mkdir()
    prepare(source, runtime)
    prepare(source, runtime)
    assert (embodiment / "config.yml").read_text() == original
    assert (
        "assets/embodiments/aloha/robot.urdf"
        in (runtime / "assets/embodiments/aloha/config.yml").read_text()
    )
    assert (runtime / "envs").is_symlink()
    assert not os.path.isabs(os.readlink(runtime / "envs"))
    with pytest.raises(ValueError, match="disjoint"):
        prepare(source, source / "child")


def result_fixture(tmp_path):
    results, shards = tmp_path / "results", tmp_path / "shards"
    results.mkdir()
    shards.mkdir()
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"tasks": [{"seed": 100000}, {"seed": 100001}]}))
    config = {
        "subname": "stamp_seal__demo_clean",
        "episodes_per_task": 2,
        "max_steps": 400,
        "params": {"task_name": "stamp_seal", "task_config": "demo_clean", "expert_task_cache_path": str(cache)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"benchmarks": [config]}))
    data = {
        "config": config,
        "num_errors": 0,
        "tasks": [
            {
                "task": "stamp_seal",
                "episodes": [
                    {
                        "episode_id": 0,
                        "steps": 1,
                        "metrics": {"success": False, "episode_index": 0, "seed": 100000},
                    },
                ],
            }
        ],
    }
    return path, results, shards, config, data


def test_collection_keeps_unsuccessful_rollouts(tmp_path):
    path, results, shards, config, data = result_fixture(tmp_path)
    original = shards / "RoboTwinBenchmark_stamp_seal__demo_clean_shard0of2.json"
    original.write_text(json.dumps(data))
    (shards / "RoboTwinBenchmark_stamp_seal__demo_clean_shard1of2.json").write_text('{"tasks":')
    _, episodes, _ = collect(path, results, shards)
    assert set(episodes) == {0} and episodes[0]["metrics"]["success"] is False
    data["tasks"][0]["episodes"][0]["metrics"]["success"] = True
    (shards / "RoboTwinBenchmark_stamp_seal__demo_clean_shard0of3.json").write_text(
        json.dumps(data)
    )
    _, episodes, _ = collect(path, results, shards)
    assert episodes[0]["metrics"]["success"] is False
    failed = json.loads(original.read_text())
    failed['num_errors'] = 1
    failed['tasks'][0]['episodes'] = [{
        'episode_id': 1, 'failure_reason': 'exception',
        'metrics': {'success': False, 'episode_index': 1, 'seed': 100001},
    }]
    (shards / "RoboTwinBenchmark_stamp_seal__demo_clean_shard1of2.json").write_text(json.dumps(failed))
    _, episodes, _ = collect(path, results, shards)
    assert set(episodes) == {0} and episodes[0]['metrics']['success'] is False


@pytest.mark.parametrize(
    "bad", ["negative_seed", "wrong_index", "empty_instruction", "boolean_index"]
)
def test_expert_cache_checks_episode_identity(tmp_path, bad):
    path = tmp_path / "cache.json"
    payload = {
        "version": 1,
        "task_name": "stamp_seal",
        "task_config": "demo_clean",
        "seed": 0,
        "instruction_type": "unseen",
        "test_num": 2,
        "instruction_override": None,
        "skip_expert_check": False,
        "tasks": [
            {
                "name": "stamp_seal",
                "suite": "robotwin",
                "seed": i + 10,
                "episode_idx": i,
                "instruction": "Stamp the paper.",
            }
            for i in range(2)
        ],
    }
    if bad == "negative_seed":
        payload["tasks"][1]["seed"] = -1
    if bad == "wrong_index":
        payload["tasks"][1]["episode_idx"] = 0
    if bad == "empty_instruction":
        payload["tasks"][1]["instruction"] = " "
    if bad == "boolean_index":
        payload["tasks"][0]["episode_idx"] = False
    path.write_text(json.dumps(payload))
    bench = RoboTwinBenchmark(
        "stamp_seal",
        test_num=2,
        instruction_type="unseen",
        expert_task_cache_path=str(path),
    )
    with pytest.raises(RuntimeError, match="Invalid seed, index or instruction"):
        bench._load_expert_task_cache()
    assert json.loads(path.read_text()) == payload


def test_run_manifest_records_processor_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.robotwin import run

    monkeypatch.setattr(run, 'PROJECT_ROOT', tmp_path)
    for name in ('a2meas/__init__.py', 'a2meas/schema.py', 'a2meas/resources.py',
                 'a2meas/grounding.py', 'serve.py', 'benchmarks/runtime.py',
                 'checkpoint/config.json', 'checkpoint/model.safetensors',
                 'backbone/config.json', 'backbone/chat_template.jinja',
                 'world.json', 'vision/config.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    output = tmp_path / 'run'
    output.mkdir()
    args = SimpleNamespace(output_dir=output, episodes=10, settings=['demo_clean'])
    env = dict(run.FLAGS, CHECKPOINT=str(tmp_path / 'checkpoint'),
               ROBOTWIN_ROOT=str(tmp_path / 'simulator'), RYNNBRAIN_PATH=str(tmp_path / 'backbone'),
               EXPERT_CACHE_DIR=str(tmp_path / 'episodes'), LAWAM_CONFIG_PATH=str(tmp_path / 'world.json'),
               LAWAM_VISION_PATH=str(tmp_path / 'vision'), HARNESS_SOURCE=str(tmp_path / 'harness'),
               ROBOTWIN_RSI='false', WORKERS_PER_GPU='1', BATCH_MAX_SIZE='1', BATCH_MAX_WAIT_TIME='0.04',
               EPISODES_PER_JOB='4', INFERENCE_SEED='0', ASSET_CACHE_CLEAR_INTERVAL='8',
               VLA_EVAL_IMAGE_FORMAT='raw', SERVER_RESPONSE_TIMEOUT='300')
    protocol = run.write_run_manifest(args, env)
    assert not (output / 'results').exists() and not (output / 'shards').exists()
    with pytest.raises(FileExistsError):
        run.write_run_manifest(args, env)
    original = json.loads((output / 'run_manifest.json').read_text())
    (tmp_path / 'backbone/chat_template.jinja').write_text('changed template')
    args.output_dir = tmp_path / 'new_run'
    args.output_dir.mkdir()
    run.write_run_manifest(args, env)
    updated = json.loads((args.output_dir / 'run_manifest.json').read_text())
    assert updated['rynnbrain_processor_assets_sha256'] != original['rynnbrain_processor_assets_sha256']


@pytest.fixture
def root(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "task_names.txt").write_text("task_a\n")
    return tmp_path


def render_summary(root, expected_seeds=3, expected_exit=0):
    output = root / "summary.md"
    assert (
        main(
            [
                "--results-dir",
                str(root / "results"),
                "--output",
                str(output),
                "--expected-seeds",
                str(expected_seeds),
                "--checkpoint",
                "checkpoint",
            ]
        )
        == expected_exit
    )
    return output.read_text()


def _write_result(
    path: Path,
    *,
    task: str,
    successes: list[bool],
    error_indices: set[int] | None = None,
    mtime: int | None = None,
    declared_errors: int | None = None
) -> None:
    error_indices = error_indices or set()
    episodes = [
        {
            "metrics": {"success": success, "episode_index": index, "seed": index},
            "failure_reason": "exception" if index in error_indices else None,
        }
        for index, success in enumerate(successes)
    ]
    cache = path.parent.parent / "seeds.json"
    cache.write_text(json.dumps({"tasks": [{"seed": index} for index in range(len(episodes))]}))
    payload = {
        "config": {"episodes_per_task": len(episodes), "params": {"task_name": task,
                   "task_config": "demo_clean", "expert_task_cache_path": str(cache)}},
        "tasks": [{"task": task, "episodes": episodes}],
        "num_episodes": len(episodes),
    }
    for episode_id, episode in enumerate(episodes):
        episode["episode_id"] = episode_id
    if declared_errors is not None:
        payload["num_errors"] = declared_errors
    path.write_text(json.dumps(payload))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_prefers_an_error_free_complete_result_over_a_newer_error_result(root) -> None:
    _write_result(
        root / "results" / "older_good.json", task="task_a", successes=[True, True, True], mtime=100
    )
    _write_result(
        root / "results" / "newer_error.json",
        task="task_a",
        successes=[False, False, False],
        error_indices={0, 1, 2},
        mtime=200,
    )
    summary = render_summary(root)
    assert "Completed jobs: 1/1" in summary
    assert "Micro success: **100.00%** (3/3)" in summary
    assert "Episode errors: 0" in summary


def test_marks_error_results_incomplete_and_excludes_error_episodes(root) -> None:
    _write_result(
        root / "results" / "error.json",
        task="task_a",
        successes=[True, True, False],
        error_indices={2},
    )
    payload = json.loads((root / "results" / "error.json").read_text())
    payload["tasks"][0]["episodes"][2]["episode_id"] = 0
    (root / "results" / "error.json").write_text(json.dumps(payload))
    summary = render_summary(root, expected_exit=1)
    assert "Completed jobs: 0/1" in summary
    assert "Episodes: 2/3" in summary
    assert "Episode errors: 1" in summary
    assert "rerun required" in summary
    assert "Partial evaluation" in summary


def test_excludes_job_when_only_top_level_error_count_is_available(root) -> None:
    _write_result(
        root / "results" / "error.json",
        task="task_a",
        successes=[True, True, True],
        declared_errors=1,
    )
    summary = render_summary(root, expected_exit=1)
    assert "Completed jobs: 0/1" in summary
    assert "Episodes: 0/3" in summary
    assert "Episode errors: 1" in summary
    assert "Micro success: **N/A**" in summary


def test_ignores_malformed_json_shapes_without_crashing(root) -> None:
    (root / "results" / "list.json").write_text("[]")
    (root / "results" / "bad_tasks.json").write_text(
        json.dumps(
            {
                "config": {"params": {"task_name": "task_a", "task_config": "demo_clean"}},
                "tasks": None,
            }
        )
    )
    summary = render_summary(root, expected_exit=1)
    assert "Completed jobs: 0/1" in summary
    assert "`task_a/demo_clean`: missing" in summary


def test_released_seed_list_supports_a_smaller_run_without_rewriting(tmp_path):
    from benchmarks.robotwin.benchmark import RoboTwinBenchmark
    path = tmp_path / "episodes.json"
    payload = {"version": 1, "task_name": "stamp_seal", "task_config": "demo_clean", "seed": 0,
               "instruction_type": "unseen", "test_num": 2, "instruction_override": None,
               "skip_expert_check": False,
               "tasks": [{"name": "stamp_seal", "suite": "robotwin", "seed": 10 + i,
                          "episode_idx": i, "instruction": "Stamp the paper."} for i in range(2)]}
    path.write_text(json.dumps(payload))
    benchmark = RoboTwinBenchmark("stamp_seal", test_num=1, instruction_type="unseen",
                                  expert_task_cache_path=str(path))
    assert benchmark._load_expert_task_cache() == payload["tasks"][:1]
    assert json.loads(path.read_text()) == payload


def test_evaluation_uses_the_prepared_resource_root(tmp_path, monkeypatch):
    from benchmarks.robotwin import run
    monkeypatch.delenv('RESOURCE_ROOT', raising=False)
    for name in ('CHECKPOINT', 'EXPERT_CACHE_DIR', 'RYNNBRAIN_PATH', 'RSI_SAM3_WEIGHTS', 'ROBOTWIN_ROOT'):
        monkeypatch.delenv(name, raising=False)
    args = run.options(['--resources', str(tmp_path / 'resources'), '--simulator-root', str(tmp_path / 'runtime')])
    env = run.environment(args)
    assert Path(env['CHECKPOINT']) == args.resources / 'robotwin/policy'
    assert Path(env['EXPERT_CACHE_DIR']) == args.resources / 'robotwin/expert_cache'
    assert Path(env['RYNNBRAIN_PATH']) == args.resources / 'assets/rynnbrain'
    assert Path(env['RSI_SAM3_WEIGHTS']) == args.resources / 'grounding/sam.pt'
    assert Path(env['ROBOTWIN_ROOT']) == args.simulator_root


def test_relative_paths_resolve_from_project_root(tmp_path, monkeypatch):
    import subprocess
    import sys
    from a2meas import PROJECT_ROOT
    from benchmarks.robotwin import run
    monkeypatch.chdir(tmp_path)
    for name in ('RESOURCE_ROOT', 'ROBOTWIN_ROOT'):
        monkeypatch.delenv(name, raising=False)
    args = run.options(['--resources', 'resources', '--simulator-root', 'runtime/cache/robotwin',
                        '--output-dir', 'results/relative-check'])
    assert args.resources == PROJECT_ROOT / 'resources'
    assert args.simulator_root == (PROJECT_ROOT / 'runtime/cache/robotwin').resolve()
    assert args.output_dir == PROJECT_ROOT / 'results/relative-check'
    env = dict(os.environ, A2MEAS_RUNTIME='../a2meas-runtime', PYTHONPATH=str(PROJECT_ROOT))
    runtime = subprocess.check_output(
        [sys.executable, '-c', 'from a2meas import RUNTIME_ROOT; print(RUNTIME_ROOT)'],
        cwd=tmp_path, env=env, text=True,
    ).strip()
    assert Path(runtime) == PROJECT_ROOT.parent / 'a2meas-runtime'
