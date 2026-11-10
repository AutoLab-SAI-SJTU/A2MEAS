"""Evaluate BEHAVIOR routes and optionally deploy a shared Adapter judge."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import tempfile
import subprocess
import sys
import threading
import yaml

from benchmarks.runtime import Progress, inference_environment, reserve_gpus, running, wait_ready
from benchmarks.behavior.results import read_result, summarize
from a2meas import PROJECT_ROOT, RESOURCE_ROOT, RUNTIME_ROOT
from a2meas.behavior import DATA_ROOT, SDK_ROOT, load_routes, resolve_task, validate_sdk
from a2meas.behavior.correction import load_config
from a2meas.behavior.resources import checkpoint_path, prepare_judge, verify_shared
from a2meas.resources import verify_file


def phases(args):
    routing, adapter = load_routes(), load_config()
    selected = ({str(resolve_task(task, routing)[0]) for task in args.task}
                if args.task else set(routing["tasks"]))
    result = [dict(
        name=name,
        lanes=[dict(gpu=gpu, workers=args.judge_workers if enabled and gpu == args.judge_gpu
                    else args.workers_per_gpu) for gpu in args.gpus
               if not (enabled and gpu == args.judge_gpu and args.judge_workers == 0)],
        tasks=[dict(id=int(task_id), model=adapter["model"] if enabled else route["model"],
                    num_inference_steps=(args.num_inference_steps or
                                         (adapter["num_inference_steps"] if enabled else route["num_inference_steps"])),
                    instances=args.instance)
               for task_id, route in routing["tasks"].items()
               if task_id in selected and (args.adapter and task_id in adapter["tasks"]) == enabled],
    ) for name, enabled in [("base", False), ("adapter", True)]]
    result = [phase for phase in result if phase["tasks"]]
    if any(not phase['lanes'] for phase in result):
        raise ValueError('A dedicated judge needs another GPU; use --judge-workers 1 to share one GPU')
    return result


def cpu_pools(gpus):
    nodes = {}
    for gpu in gpus:
        bus = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}", "--query-gpu=pci.bus_id", "--format=csv,noheader",
        ], text=True).strip()[-12:].lower()
        nodes[gpu] = int((Path("/sys/bus/pci/devices") / bus / "numa_node").read_text())
    pools = {}
    for node in set(nodes.values()):
        peers = [gpu for gpu in gpus if nodes[gpu] == node]
        cores = {}
        for cpu in sorted(os.sched_getaffinity(0)):
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            if node >= 0 and not (topology / f"node{node}").exists():
                continue
            siblings = (topology / "topology/thread_siblings_list").read_text()
            cores.setdefault(siblings, []).append(cpu)
        physical = list(cores.values())
        for index, gpu in enumerate(peers):
            pools[gpu] = [cpu for core in physical[index::len(peers)] for cpu in core]
    return pools


def source_digest():
    paths = [*PROJECT_ROOT.joinpath('a2meas').glob('*.py'),
             *PROJECT_ROOT.joinpath('a2meas/behavior').rglob('*.py'),
             *PROJECT_ROOT.joinpath('a2meas/models').rglob('*.py'),
             *PROJECT_ROOT.joinpath('a2meas/processing').rglob('*.py'),
             *PROJECT_ROOT.joinpath('benchmarks/behavior').glob('*.py'),
             PROJECT_ROOT / 'benchmarks/runtime.py', PROJECT_ROOT / 'evaluate.py', PROJECT_ROOT / 'serve.py',
             *PROJECT_ROOT.joinpath('configs/behavior').glob('*.json'),
             *PROJECT_ROOT.joinpath('configs/shared').rglob('*.json'),
             PROJECT_ROOT / 'benchmarks/behavior/r1pro.yaml',
             PROJECT_ROOT / 'third_party/behavior/evaluation.patch']
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def run_phase(args, phase, pools, stop, judge=None):
    attempts = {}
    remaining = phase
    routes = load_routes()['tasks']
    while remaining['tasks']:
        evaluate_phase(args, remaining, pools, stop, judge, attempts=attempts)
        tasks = []
        for job in phase['tasks']:
            task = routes[str(job['id'])]['task']
            missing = []
            for index in job['instances']:
                instance = (301 if args.mode == 'public_test' else 321) + index
                name = f'{task}_{instance}_0'
                path = args.evaluation_dir / 'json' / f'{name}.json'
                if path.exists():
                    try:
                        read_result(path, task, instance)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    else:
                        continue
                if attempts.get((job['id'], index), 0) >= 2:
                    raise RuntimeError(f'Simulator failed twice for {task}/{instance}')
                if path.exists():
                    path.rename(path.with_suffix('.json.failed'))
                video = args.evaluation_dir / 'videos' / f'{name}.mp4'
                if video.exists():
                    video.rename(video.with_suffix('.failed.mp4'))
                missing.append(index)
            if missing:
                tasks.append(dict(job, instances=missing))
        remaining = dict(phase, tasks=tasks)


def evaluation_jobs(tasks, workers, instances_per_claim):
    """Interleave small task groups so idle workers can claim remaining episodes."""
    groups = [[dict(task, instances=task['instances'][start:start + instances_per_claim])
               for start in range(0, len(task['instances']), instances_per_claim)] for task in tasks]
    jobs = [job for row in zip_longest(*groups) for job in row if job is not None]
    while len(jobs) < workers:
        largest = max(range(len(jobs)), key=lambda i: len(jobs[i]['instances']), default=None)
        if largest is None or len(jobs[largest]['instances']) < 2:
            break
        job = jobs[largest]
        middle = (len(job['instances']) + 1) // 2
        jobs[largest] = dict(job, instances=job['instances'][:middle])
        jobs.append(dict(job, instances=job['instances'][middle:]))
    return jobs


def scene_contracts(data_dir):
    metadata = Path(data_dir) / '2026-challenge-task-instances/metadata'
    available = yaml.safe_load((metadata / 'available_tasks.yaml').read_text())
    routes = load_routes()['tasks']
    contracts = {}
    for task_id, route in routes.items():
        name = route['task']
        scene = available[name][0]['scene_model']
        template = (metadata.parent / 'scenes' / scene / 'json' /
                    f'{scene}_task_{name}_0_0_template.json')
        contracts[int(task_id)] = scene if template.is_file() else None
    return contracts


def evaluate_phase(args, phase, pools, stop, judge=None, *, attempts=None):
    worker_count = sum(spec['workers'] for spec in phase['lanes'])
    jobs = evaluation_jobs(phase['tasks'], worker_count, args.instances_per_claim)
    contracts = scene_contracts(args.data_dir)
    routes = load_routes()['tasks']
    lock = threading.Lock()
    last_model = {}
    attempts = {} if attempts is None else attempts

    def record(task_id, instance):
        key = (task_id, instance)
        attempts[key] = attempts.get(key, 0) + 1

    lanes = []
    for index, spec in enumerate(phase['lanes']):
        port = args.base_port + index * args.port_stride
        cores = {}
        for cpu in pools[spec['gpu']]:
            siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()
            cores.setdefault(siblings, []).append(cpu)
        physical = list(cores.values())
        if len(physical) < spec['workers'] + 2:
            raise ValueError(f"GPU {spec['gpu']}: need one physical CPU core per worker plus two simulator cores")
        simulator_cpus = [cpu for core in physical[spec['workers']:] for cpu in core]
        for offset in range(spec['workers']):
            shared = spec['workers'] + offset % (len(physical) - spec['workers'])
            worker_cpus = [cpu for core in (physical[offset], physical[shared]) for cpu in core]
            lanes.append((port + offset, spec['gpu'], worker_cpus, simulator_cpus))

    def take(port, contract=None, current_task=None):
        with lock:
            eligible = [i for i, job in enumerate(jobs)
                        if contract is None or contracts[job['id']] == contract]
            if not eligible:
                return None
            same_task = next((i for i in eligible if jobs[i]['id'] == current_task), None)
            index = (same_task if same_task is not None else
                     next((i for i in eligible if last_model.get(port) ==
                           (jobs[i]['model'], jobs[i]['num_inference_steps'])), eligible[0]))
            job = jobs.pop(index)
            last_model[port] = (job['model'], job['num_inference_steps'])
            return job

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            port, task_id = int(request['port']), int(request['task_id'])
            if 'completed' in request:
                task = routes[str(task_id)]['task']
                instance = int(request['completed'])
                path = args.evaluation_dir / 'json' / f'{task}_{instance}_0.json'
                result = read_result(path, task, instance)
                args.progress.update([result['success']])
                self.wfile.write(b'null\n')
                return
            if 'instance' in request:
                with lock:
                    record(task_id, int(request['instance']))
                self.wfile.write(b'null\n')
                return
            contract = contracts[task_id]
            job = None if contract is None else take(port, contract, task_id)
            payload = None if job is None else {'task': routes[str(job['id'])]['task'], 'instances': job['instances']}
            if job is not None:
                with lock:
                    record(job['id'], job['instances'][0])
            self.wfile.write((json.dumps(payload) + '\n').encode())

    class DispatchServer(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    def lane(port, gpu, cpus, simulator_cpus, dispatch_path):
        job = take(port)
        if job is None:
            return
        prefix = ['taskset', '--cpu-list', ','.join(map(str, cpus))]
        env = dict(os.environ, QWEN_BASE_URL=f'http://127.0.0.1:{args.judge_port}/v1',
                   A2MEAS_REVIEW_CONCURRENCY=str(args.judge_concurrency),
                   A2MEAS_DISPATCH_SOCKET=str(dispatch_path), A2MEAS_DISPATCH_PORT=str(port))
        env.update(inference_environment(env))
        command = [*prefix, str(args.model_python), 'serve.py', 'behavior', 'serve',
                   '--resources', str(args.resources), '--data-dir', str(args.data_dir),
                   '--gpu', str(gpu), '--workers', '1', '--port', str(port), '--seed', str(args.seed),
                   '--compile-mode', args.compile_mode]
        if args.num_inference_steps is not None:
            command.extend(['--num-inference-steps', str(args.num_inference_steps)])
        if phase['name'] == 'adapter':
            command.append('--adapter')
        try:
            with ExitStack() as stack:
                env['A2MEAS_WORKER_CACHE'] = stack.enter_context(tempfile.TemporaryDirectory(prefix='a2meas-worker-'))
                server = stack.enter_context(running(command, env))
                wait_ready(server, f'http://127.0.0.1:{port}/healthz', stop, 120)
                while job is not None and not stop.is_set():
                    task_id = job['id']
                    simulator_prefix = ['taskset', '--cpu-list', ','.join(map(str, simulator_cpus))]
                    command = [*simulator_prefix, str(args.behavior_python), 'evaluate.py', 'behavior',
                               '--task', str(task_id), '--gpu', str(gpu), '--workers', '1', '--seed', str(args.seed), '--mode', args.mode,
                               '--endpoint', f'127.0.0.1:{port}',
                               '--behavior-dir', str(args.behavior_dir), '--data-dir', str(args.data_dir),
                               '--output-dir', str(args.evaluation_dir),
                               '--simulator-cpus', *map(str, simulator_cpus),
                               '--instance', *map(str, job['instances'])]
                    if args.max_steps is not None:
                        command.extend(['--max-steps', str(args.max_steps)])
                    if args.video:
                        command.append('--video')
                    with lock:
                        record(task_id, job['instances'][0])
                    with running(command, env) as simulator:
                        while simulator.poll() is None:
                            if stop.wait(1):
                                raise InterruptedError('Evaluation interrupted')
                            if server.poll() is not None:
                                raise RuntimeError(f'Policy on port {port} exited during evaluation')
                            if judge is not None and judge.poll() is not None:
                                raise RuntimeError('Judge exited during Adapter evaluation')
                        if simulator.returncode:
                            wait_ready(server, f'http://127.0.0.1:{port}/healthz', stop, 0)
                            if judge is not None and judge.poll() is not None:
                                raise RuntimeError('Judge exited during Adapter evaluation')
                    job = take(port)
        except BaseException:
            stop.set()
            raise

    with tempfile.TemporaryDirectory(prefix='a2meas-dispatch-') as temporary:
        socket_path = Path(temporary) / 'queue.sock'
        with DispatchServer(str(socket_path), Handler) as dispatch:
            thread = threading.Thread(target=dispatch.serve_forever, daemon=True)
            thread.start()
            try:
                with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
                    try:
                        futures = [executor.submit(lane, *spec, socket_path) for spec in lanes]
                        for future in as_completed(futures):
                            future.result()
                    finally:
                        if sys.exc_info()[0] is not None:
                            stop.set()
            finally:
                dispatch.shutdown()
                thread.join()


def judge_command(args):
    return [str(args.judge_python.parent / "sglang"), "serve", "--model-type", "llm",
            "--log-level", "error", "--log-level-http", "error",
            "--model-path", str(args.resources / "judge"), "--served-model-name", "qwen3.8-27b",
            "--host", "127.0.0.1", "--port", str(args.judge_port), "--tp-size", "1",
            "--dtype", "bfloat16", "--context-length", str(args.judge_context_length),
            "--max-total-tokens", str(args.judge_max_total_tokens),
            "--chunked-prefill-size", "8192", "--mem-fraction-static", str(args.judge_memory_fraction),
            "--max-running-requests", str(args.judge_concurrency + 1),
            "--max-mamba-cache-size", str(args.judge_concurrency + 1),
            "--disable-radix-cache", "--mamba-ssm-dtype", "bfloat16",
            "--mamba-backend", "triton", "--linear-attn-backend", "triton",
            "--attention-backend", "triton", "--mm-attention-backend", "sdpa",
            "--sampling-backend", "pytorch", "--reasoning-parser", "qwen3",
            "--cuda-graph-backend-prefill", "disabled", "--cuda-graph-backend-decode", "full",
            "--cuda-graph-max-bs-decode", str(args.judge_concurrency),
            "--cuda-graph-bs-decode", *map(str, range(1, args.judge_concurrency + 1)),
            "--constrained-json-disable-any-whitespace"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-inference-steps", type=int, help="Override denoising steps; omit to use configured routes")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--task", nargs="+", help="Task IDs or official names; omit for all tasks")
    parser.add_argument("--instance", type=int, nargs="+", choices=range(20), default=list(range(10)))
    parser.add_argument("--mode", choices=["public_test", "hidden_test"], default="public_test")
    parser.add_argument("--adapter", action="store_true", help="Enable configured Adapter tasks and deploy Qwen automatically")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--instances-per-claim", type=int, default=2,
                        help="Maximum instances assigned at a time to a resident worker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="Override the official task step limit")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"],
                        default="max-autotune-no-cudagraphs")
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--behavior-python", type=Path, default=RUNTIME_ROOT / "behavior/bin/python")
    parser.add_argument("--judge-python", type=Path, default=RUNTIME_ROOT / "judge/bin/python")
    parser.add_argument("--behavior-dir", type=Path, default=SDK_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--resources", type=Path, default=RESOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-port", type=int, default=18000)
    parser.add_argument("--judge-port", type=int, default=8001)
    parser.add_argument("--judge-gpu", type=int, help="Allocated judge GPU; defaults to the last evaluation GPU")
    parser.add_argument("--judge-workers", type=int, default=0,
                        help="Workers sharing the judge GPU; default 0 dedicates it to the judge")
    parser.add_argument("--judge-memory-fraction", type=float, default=0.8)
    parser.add_argument("--judge-concurrency", type=int, default=28)
    parser.add_argument("--judge-context-length", type=int, default=32768)
    parser.add_argument("--judge-max-total-tokens", type=int, default=131072)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--seed must be nonnegative and --max-steps must be positive")
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus must name distinct nonnegative GPU indices")
    if args.judge_gpu is None:
        args.judge_gpu = args.gpus[-1]
    if (min(args.workers_per_gpu, args.instances_per_claim, args.judge_concurrency) < 1
            or args.judge_workers < 0 or args.judge_gpu < 0):
        parser.error("Worker counts and judge concurrency must be positive; GPU indices must be nonnegative")
    if args.num_inference_steps is not None and args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if len(set(args.instance)) != len(args.instance):
        parser.error("--instance must contain distinct indices")
    if not 0 < args.judge_memory_fraction < 1 or not 0 < args.judge_context_length <= args.judge_max_total_tokens:
        parser.error("Use a judge memory fraction in (0, 1) and a token pool at least as large as its context")
    plan = phases(args)
    args.port_stride = max(lane['workers'] for phase in plan for lane in phase['lanes'])
    ports = list(range(args.base_port, args.base_port + len(args.gpus) * args.port_stride))
    if args.adapter:
        ports.append(args.judge_port)
    if min(ports) < 1 or max(ports) > 65535 or len(set(ports)) != len(ports):
        parser.error("Policy and judge ports must be distinct and within 1..65535")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output_dir = (args.output_dir or PROJECT_ROOT / "results" /
                       f"behavior_n{args.num_inference_steps}_{datetime.now():%Y%m%dT%H%M%S}").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    allocated_gpus = set(args.gpus)
    if any(phase['name'] == 'adapter' for phase in plan):
        allocated_gpus.add(args.judge_gpu)
    with reserve_gpus(allocated_gpus), ExitStack() as stack:
        # Preserve the venv entry point: resolving its Python symlink selects the base interpreter.
        for name in ("model_python", "behavior_python"):
            path = getattr(args, name).absolute()
            if not os.access(path, os.X_OK):
                parser.error(f"{name}: executable not found: {path}")
            setattr(args, name, path)
        subprocess.check_call([str(args.model_python), "-c",
            "import sys, sysconfig; from pathlib import Path; "
            "header = Path(sysconfig.get_path('include')) / 'Python.h'; "
            "sys.exit(0 if header.is_file() else f'Missing {header}; recreate the managed Python runtime "
            "with setup.sh before evaluation')"
        ])
        for name in ("behavior_dir", "data_dir", "resources"):
            setattr(args, name, getattr(args, name).resolve())
        verify_shared(args.resources)
        try:
            validate_sdk(args.behavior_dir)
        except ValueError as error:
            parser.error(str(error))
        if not (args.data_dir / "2026-challenge-task-instances/metadata/B100_task_misc.csv").is_file():
            parser.error("--data-dir must point to the installed BEHAVIOR dataset root")
        for port in ports:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))
        identity = {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
                    for key in ('seed', 'mode', 'compile_mode', 'max_steps', 'video', 'num_inference_steps', 'adapter', 'resources', 'data_dir', 'behavior_dir',
                                'judge_context_length', 'judge_max_total_tokens')}
        identity['source_sha256'] = source_digest()
        identity['inference_environment'] = inference_environment(os.environ)
        sdk_files = ['eval/eval.py', 'eval/evaluator.py', 'eval/policies.py', 'envs/env_base.py',
                     'tasks/behavior_task.py', 'scenes/scene_base.py', 'scenes/interactive_traversable_scene.py',
                     'objects/dataset_object.py', 'robots/robot.py', 'utils/usd_utils.py',
                     'macros.py', 'prims/xform_prim.py', 'simulator.py', 'systems/macro_particle_system.py']
        sdk_hash = hashlib.sha256()
        for name in sdk_files:
            path = args.behavior_dir / 'OmniGibson/omnigibson' / name
            sdk_hash.update(name.encode() + b'\0' + path.read_bytes())
        for name in ('available_tasks.yaml', 'B100_task_misc.csv', 'task.jsonl'):
            path = args.data_dir / '2026-challenge-task-instances/metadata' / name
            sdk_hash.update(name.encode() + b'\0' + path.read_bytes())
        available = yaml.safe_load((args.data_dir / '2026-challenge-task-instances/metadata/available_tasks.yaml').read_text())
        routes = load_routes()['tasks']
        for phase in plan:
            for job in phase['tasks']:
                task = routes[str(job['id'])]['task']
                scene = available[task][0]['scene_model']
                base = args.data_dir / '2026-challenge-task-instances'
                template = f'{scene}_task_{task}_0_0_template.json'
                paths = [base / 'scenes' / scene / 'json' / template]
                for index in job['instances']:
                    instance = (301 if args.mode == 'public_test' else 321) + index
                    name = f'{scene}_task_{task}_0_{instance}_template-tro_state.json'
                    split = 'public' if args.mode == 'public_test' else 'private'
                    paths.append(base / 'scene_test' / split / scene / 'json' / f'{scene}_task_{task}_instances' / name)
                for path in paths:
                    sdk_hash.update(str(path.relative_to(base)).encode() + b'\0' + path.read_bytes())
        identity['sdk_and_metadata_sha256'] = sdk_hash.hexdigest()
        run_file = args.output_dir / 'run.json'
        split = 'public' if args.mode == 'public_test' else 'hidden'
        args.evaluation_dir = args.output_dir / 'evaluations' / f"standard.{split}.a2meas.a2wla.{datetime.now():%Y%m%d}"
        run_file.write_text(json.dumps(dict(
            options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            identity=identity, phases=plan,
        ), indent=2) + "\n")
        args.progress = Progress('BEHAVIOR', sum(len(job['instances']) for phase in plan for job in phase['tasks']))
        if any(phase['name'] == 'adapter' for phase in plan):
            args.judge_python = args.judge_python.absolute()
            if not os.access(args.judge_python, os.X_OK):
                parser.error(f'judge_python: executable not found: {args.judge_python}')
            subprocess.check_call([str(args.judge_python), '-c',
                'import sys; from importlib.metadata import version; '
                'assert sys.version_info[:2] == (3, 11); '
                'assert version("sglang") == "0.5.19"; '
                'assert version("torch") == "2.13.0+cu130"'])
            adapter = load_config()
            checkpoint_path(args.resources, adapter['model'])
            verify_file(args.resources / 'adapter/adapter.safetensors', adapter['weights'])
            verify_file(args.resources / 'grounding/sam.pt', adapter['grounding'])
            prepare_judge(args.resources / 'judge')
        pools, stop = cpu_pools(args.gpus), threading.Event()
        def interrupt(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        def finish():
            summary = summarize(args.output_dir)
            (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + "\n")
            print(f"BEHAVIOR: {summary['successes']}/{summary['episodes']} successful "
                  f"({summary['success_rate']:.2%}), mean Q-score={summary['mean_task_q_score']:.4f}",
                  flush=True)

        for phase in plan:
            if phase['name'] == 'base':
                run_phase(args, phase, pools, stop)
        if not any(phase['name'] == 'adapter' for phase in plan):
            finish()
            return
        cache = RUNTIME_ROOT / "cache/judge"
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.judge_gpu), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", HF_HOME=str(RUNTIME_ROOT / "cache/huggingface"),
                   TQDM_DELAY="inf",
                   SGLANG_CACHE_DIR=str(cache), TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                   XDG_CACHE_HOME=str(RUNTIME_ROOT / "cache"), TRITON_CACHE_DIR=str(cache / "triton"),
                   TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
                   A2MEAS_REVIEW_CONCURRENCY=str(args.judge_concurrency))
        site = subprocess.check_output([
            str(args.judge_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))",
        ], text=True).strip()
        cuda = Path(site) / "nvidia/cu13"
        env.update(CUDA_HOME=str(cuda), PATH=f"{cuda / 'bin'}:{args.judge_python.parent}:{os.environ['PATH']}")
        with running(judge_command(args), env) as judge:
            wait_ready(judge, f'http://127.0.0.1:{args.judge_port}/v1/models', stop, 1200)
            run_phase(args, next(phase for phase in plan if phase["name"] == "adapter"), pools, stop, judge)
        finish()


if __name__ == "__main__":
    main()
