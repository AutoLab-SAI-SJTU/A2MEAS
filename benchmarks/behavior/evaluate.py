"""Run the official BEHAVIOR evaluator against a VLA endpoint."""

import argparse
from datetime import date
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from a2meas import PROJECT_ROOT, cpu_affinity, cpu_layout, runtime_environment
from a2meas.behavior import DATA_ROOT, SDK_ROOT, resolve_task, load_routes, validate_sdk
from benchmarks.runtime import stop_process
from benchmarks.behavior.results import read_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--behavior-dir", type=Path, default=SDK_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--endpoint", default="127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="Override the official task step limit")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--simulator-cpus", type=int, nargs="+",
                        help="Shared simulator CPU pool assigned by the batch launcher")
    parser.add_argument("--mode", choices=["public_test", "hidden_test"], default="public_test")
    parser.add_argument("--instance", type=int, nargs="+", choices=range(20), default=list(range(10)))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    _, route = resolve_task(args.task, load_routes())
    split = {"public_test": "public", "hidden_test": "hidden"}[args.mode]
    sdk, data = args.behavior_dir.resolve(), args.data_dir.resolve()
    output = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT / "results/evaluations" / f"standard.{split}.a2meas.a2wla.{date.today():%Y%m%d}"
    ).resolve()
    if args.gpu < 0 or args.seed < 0:
        parser.error("--gpu and --seed must be nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(set(args.instance)) != len(args.instance):
        parser.error("--instance must contain distinct indices")
    try:
        endpoint = urlsplit(f"//{args.endpoint}")
        port = endpoint.port
        if endpoint.hostname is None or endpoint.path or port is None or not 1 <= port <= port + args.workers - 1 <= 65535:
            raise ValueError
    except ValueError:
        parser.error("--endpoint must be HOST:PORT with worker ports in 1..65535")
    host = args.endpoint.rsplit(":", 1)[0]
    try:
        validate_sdk(sdk)
    except ValueError as error:
        parser.error(str(error))
    if not (data / "2026-challenge-task-instances/metadata/B100_task_misc.csv").is_file():
        parser.error("--data-dir must point to the installed BEHAVIOR dataset root")
    instance_start = 301 if args.mode == "public_test" else 321
    resident = os.environ.get("A2MEAS_TASK_GROUPS")
    if resident:
        try:
            task_groups = json.loads(resident)
            if (args.workers != 1 or not isinstance(task_groups, list) or not task_groups
                    or task_groups[0] != {"task": route["task"], "instances": args.instance}):
                raise ValueError
            seen = set()
            for group in task_groups:
                task = resolve_task(group["task"], load_routes())[1]["task"]
                for index in group["instances"]:
                    if type(index) is not int or index not in range(20) or (task, index) in seen:
                        raise ValueError
                    seen.add((task, index))
                if not group["instances"]:
                    raise ValueError
        except (ValueError, TypeError, KeyError):
            parser.error("Invalid resident task groups")
        groups = [task_groups]
    else:
        count = min(args.workers, len(args.instance))
        groups = [[{"task": route["task"], "instances": args.instance[index::count]}] for index in range(count)]
    filenames = [f"{group['task']}_{instance_start + index}_0"
                 for worker in groups for group in worker for index in group["instances"]]
    for filename in filenames:
        if (output / "json" / f"{filename}.json").exists() or (output / "videos" / f"{filename}.mp4").exists():
            parser.error("A requested rollout already has results; select another --output-dir")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "benchmarks/behavior/gpu_evaluator.py"),
        "--task-name",
        route["task"],
        "--robot-config",
        str(PROJECT_ROOT / "benchmarks/behavior/r1pro.yaml"),
        "--mode",
        args.mode,
        "--num-rollouts",
        "1",
        "--replay-action-chunk-size",
        "0",
        "--output-dir",
        str(output),
        "--write-video" if args.video else "--no-write-video",
        "--headless",
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    simulator_cpus = (set(args.simulator_cpus) if args.simulator_cpus is not None
                      else cpu_layout(args.gpu, args.workers)[0])
    with runtime_environment(), cpu_affinity(simulator_cpus):
        processes = []
        try:
            for index, task_groups in enumerate(groups):
                instances = task_groups[0]["instances"]
                temporary = Path(os.environ["TMPDIR"]) / f"worker-{index}"
                temporary.mkdir()
                cache_root = os.environ.get("A2MEAS_WORKER_CACHE")
                appdata = (Path(cache_root) / f"worker-{index}" / "appdata"
                           if cache_root else temporary / "appdata")
                env = dict(
                    os.environ,
                    TMPDIR=str(temporary),
                    OMNIGIBSON_APPDATA_PATH=str(appdata),
                    OMNIGIBSON_GLOBAL_CACHE_PATH=str(appdata / "global/cache"),
                    XDG_CACHE_HOME=str(appdata / "xdg/cache"),
                    XDG_CONFIG_HOME=str(appdata / "xdg/config"),
                    A2MEAS_TASK_GROUPS=json.dumps(task_groups),
                    A2MEAS_EVAL_SEED=str(args.seed),
                    CUDA_VISIBLE_DEVICES=str(args.gpu),
                    OMNIGIBSON_GPU_ID=str(args.gpu),
                    OMNIGIBSON_DATA_PATH=str(data),
                    OMNIGIBSON_DEBUG="False",
                    OMNIGIBSON_NO_OMNI_LOGS="False",
                    OMNIGIBSON_ENABLE_TILED_RENDERING="True",
                    OMNIGIBSON_RENDER_VIEWER_CAMERA=os.environ.get("OMNIGIBSON_RENDER_VIEWER_CAMERA", "False"),
                )
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(PROJECT_ROOT), *[str(sdk / name) for name in ["OmniGibson", "bddl3", "joylo"]]]
                )
                for key in ["NO_PROXY", "no_proxy"]:
                    env[key] = ",".join(filter(None, [env.get(key, ""), "localhost", "127.0.0.1", "::1"]))
                worker_command = [*command, "--num-envs", "1", "--instance-indices", str(instances[0]),
                                  "--host", host, "--port", str(port + index)]
                processes.append(subprocess.Popen(worker_command, env=env, start_new_session=True))
            while any(process.poll() is None for process in processes):
                time.sleep(0.2)
            for process in processes:
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
        finally:
            for process in processes:
                stop_process(process)
            for index, process in enumerate(processes):
                log = Path(os.environ['TMPDIR']) / f'worker-{index}' / 'kit.log'
                if log.is_file():
                    logs = output / 'logs'
                    logs.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(log, logs / f'simulator-{process.pid}.log')
        # Kit can exit with status zero after an evaluator exception.
        for filename in filenames:
            task, instance, _ = filename.rsplit('_', 2)
            read_result(output / "json" / f"{filename}.json", task, int(instance))



if __name__ == "__main__":
    main()
