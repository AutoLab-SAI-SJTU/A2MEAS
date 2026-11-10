"""Run the released RoboTwin policy through the native evaluation harness."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import socket
import subprocess
import tempfile
import threading

import yaml

from a2meas import PROJECT_ROOT, RUNTIME_ROOT
from a2meas.robotwin import SIMULATOR_ROOT
from a2meas.resources import sha256
from benchmarks.runtime import Progress, inference_environment, reserve_gpus, running, wait_ready
from .check_checkpoint import validate_checkpoint
from .results import collect, publish
from .summarize import main as summarize


FLAGS = {
    'ROBOTWIN_RSI': 'true', 'ROBOTWIN_KEEP_PREVIOUS_SCENE': 'true',
    'CUDA_GRAPH_DENOISE': 'true', 'FAST_STEP_RENDER': 'true', 'FAST_OBS_NO_COPY': 'true',
    'MINIMAL_OBSERVATION': 'true', 'POLICY_CAMERAS_ONLY': 'true',
}


def options(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=list(map(int, os.environ.get('GPU_IDS', '0').split(','))))
    parser.add_argument('--workers-per-gpu', type=int, default=int(os.environ.get('WORKERS_PER_GPU', '1')))
    parser.add_argument('--tasks', nargs='+', default=list(filter(None, os.environ.get('TASK_FILTER', '').split(','))))
    settings = os.environ.get('SETTING_FILTER', 'both')
    parser.add_argument('--settings', nargs='+', choices=['demo_clean', 'demo_randomized'],
                        default=['demo_clean', 'demo_randomized'] if settings == 'both' else [settings])
    parser.add_argument('--episodes', type=int, default=int(os.environ.get('TEST_NUM', '10')))
    parser.add_argument('--resources', type=Path, default=Path(os.environ.get('RESOURCE_ROOT', PROJECT_ROOT / 'resources')))
    parser.add_argument('--simulator-root', type=Path, default=Path(os.environ.get('ROBOTWIN_ROOT', SIMULATOR_ROOT)))
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get(
        'OUTPUT_ROOT', PROJECT_ROOT / 'results' / datetime.now(timezone.utc).strftime('robotwin_%Y%m%dT%H%M%SZ'))))
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN', 'false') == 'true')
    args = parser.parse_args(argv)
    if (not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)
            or args.workers_per_gpu < 1 or args.episodes < 1):
        parser.error('Use distinct nonnegative GPUs and positive worker/episode counts')
    args.tasks = [name.strip() for name in args.tasks]
    if len(set(args.tasks)) != len(args.tasks) or any(not name for name in args.tasks):
        parser.error('Tasks must be distinct, nonempty names')
    if not args.settings or len(set(args.settings)) != len(args.settings) or set(args.settings) - {'demo_clean', 'demo_randomized'}:
        parser.error('Settings must be distinct')
    args.resources = (PROJECT_ROOT / args.resources).resolve()
    args.simulator_root = (PROJECT_ROOT / args.simulator_root).resolve()
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    return args


def environment(args):
    env = dict(os.environ)
    env.update(inference_environment(env, robotwin=True))
    for name, default in {**FLAGS, 'EVAL_VIDEO_LOG': 'false'}.items():
        value = env.setdefault(name, default)
        if value not in ('true', 'false'):
            raise ValueError(f'{name} must be true or false')
    defaults = {
        'WORKERS_PER_GPU': str(args.workers_per_gpu), 'BATCH_MAX_SIZE': str(args.workers_per_gpu),
        'BATCH_MAX_WAIT_TIME': '0.04', 'EPISODES_PER_JOB': '4', 'INFERENCE_SEED': '0',
        'ASSET_CACHE_CLEAR_INTERVAL': '8', 'SERVER_RESPONSE_TIMEOUT': '300',
        'VLA_EVAL_IMAGE_FORMAT': 'raw', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    }
    for name, value in defaults.items():
        env.setdefault(name, value)
    env['WORKERS_PER_GPU'] = str(args.workers_per_gpu)
    for name in ('BATCH_MAX_SIZE', 'EPISODES_PER_JOB', 'ASSET_CACHE_CLEAR_INTERVAL', 'SERVER_RESPONSE_TIMEOUT'):
        if int(env[name]) < 1:
            raise ValueError(f'{name} must be positive')
    if float(env['BATCH_MAX_WAIT_TIME']) < 0 or int(env['INFERENCE_SEED']) < -1:
        raise ValueError('Invalid batching delay or inference seed')
    if env['VLA_EVAL_IMAGE_FORMAT'] not in ('raw', 'png'):
        raise ValueError('Evaluation requires lossless raw or png images')
    for name, expected in [('IMAGE_HISTORY_NUM_FRAMES', 2), ('IMAGE_HISTORY_STRIDE', 30)]:
        if int(env.get(name, expected)) != expected:
            raise ValueError(f'The released policy requires {name}={expected}')
    resource = args.resources
    paths = {
        'ROBOTWIN_ROOT': args.simulator_root,
        'CHECKPOINT': resource / 'robotwin/policy', 'EXPERT_CACHE_DIR': resource / 'robotwin/expert_cache',
        'RYNNBRAIN_PATH': resource / 'assets/rynnbrain',
        'LAWAM_CONFIG_PATH': PROJECT_ROOT / 'configs/shared/world_model.json',
        'LAWAM_VISION_PATH': PROJECT_ROOT / 'configs/shared/dinov3',
        'RSI_SAM3_WEIGHTS': resource / 'grounding/sam.pt',
        'ROBOTWIN_RSI_CONFIG': PROJECT_ROOT / 'configs/robotwin/rsi.json',
        'MODEL_PYTHON': RUNTIME_ROOT / 'model/bin/python',
        'ROBOTWIN_PYTHON': RUNTIME_ROOT / 'simulator/bin/python',
        'LD_PRELOAD': RUNTIME_ROOT / 'simulator/oidn/lib/libOpenImageDenoise.so',
        'XDG_CACHE_HOME': RUNTIME_ROOT / 'cache/xdg',
    }
    # Python executable symlinks must retain their venv directory.
    for name, default in paths.items():
        value = (PROJECT_ROOT / Path(default if name == 'ROBOTWIN_ROOT' else env.get(name, default))).absolute()
        env[name] = str(value if name.endswith('PYTHON') else value.resolve())
    for name in ('MODEL_PYTHON', 'ROBOTWIN_PYTHON'):
        if not os.access(env[name], os.X_OK):
            raise FileNotFoundError(f'{name} is not executable: {env[name]}; run bash setup.sh')
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', NO_PROXY='127.0.0.1,localhost',
               no_proxy='127.0.0.1,localhost')
    pythonpath = [str(PROJECT_ROOT)]
    if env.get('HARNESS_ROOT'):
        harness_root = (PROJECT_ROOT / env['HARNESS_ROOT']).resolve()
        harness_source = harness_root / 'src/vla_eval'
        if not harness_source.is_dir():
            raise ValueError(f'HARNESS_ROOT does not contain src/vla_eval: {harness_root}')
        pythonpath.append(str(harness_source.parent))
    else:
        locations = [Path(subprocess.check_output([
            env[name], '-I', '-c',
            'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vla_eval").origin).parent)',
        ], text=True).strip()) for name in ('MODEL_PYTHON', 'ROBOTWIN_PYTHON')]
        def source_hash(path):
            files = [p for p in path.rglob('*.py') if p.name != '_version.py']
            if not files:
                raise ValueError(f'VLA evaluation harness source missing: {path}')
            return combined_digest(files)
        if source_hash(locations[0]) != source_hash(locations[1]):
            raise ValueError('Model and simulator environments have different VLA evaluation harness versions')
        harness_source = locations[0]
    env['HARNESS_SOURCE'] = str(harness_source)
    env['PYTHONPATH'] = os.pathsep.join([*pythonpath, env.get('PYTHONPATH', '')])
    env['PATH'] = str(Path(env['ROBOTWIN_PYTHON']).parent) + os.pathsep + env['PATH']
    Path(env['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
    if 'VK_ICD_FILENAMES' not in env:
        for name in ('/etc/vulkan/icd.d/nvidia_icd.json', '/usr/share/vulkan/icd.d/nvidia_icd.json'):
            if Path(name).is_file():
                env['VK_ICD_FILENAMES'] = name
                break
    return env


def combined_digest(paths):
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode() + b'\0' + sha256(path).encode() + b'\0')
    return digest.hexdigest()


def write_run_manifest(args, env):
    output, checkpoint = args.output_dir, Path(env['CHECKPOINT'])
    simulator, backbone = Path(env['ROBOTWIN_ROOT']), Path(env['RYNNBRAIN_PATH'])
    source = PROJECT_ROOT / 'a2meas'
    inference_files = [*(source / 'robotwin').rglob('*.py'), *(source / 'models').glob('*.py'),
                       *(source / 'processing').glob('*.py'),
                       *(source / name for name in ('__init__.py', 'schema.py', 'resources.py', 'grounding.py'))]
    manifest = {
        'schema_version': 10,
        'checkpoint': os.path.relpath(checkpoint, PROJECT_ROOT), 'checkpoint_config_sha256': sha256(checkpoint / 'config.json'),
        'checkpoint_weights_sha256': sha256(checkpoint / 'model.safetensors'),
        'checkpoint_weights_size': (checkpoint / 'model.safetensors').stat().st_size,
        'checkpoint_processors_sha256': combined_digest([*checkpoint.glob('policy_*.json'), *checkpoint.glob('policy_*.safetensors')]),
        'rynnbrain_path': os.path.relpath(backbone, PROJECT_ROOT), 'rynnbrain_config_sha256': sha256(backbone / 'config.json'),
        'rynnbrain_processor_assets_sha256': combined_digest(p for p in backbone.rglob('*')
            if p.is_file() and p.suffix in {'.json', '.txt', '.model', '.jinja'} and not p.name.endswith('.safetensors.index.json')),
        'expert_cache_sha256': combined_digest(Path(env['EXPERT_CACHE_DIR']).glob('*.json')),
        'lawam_config_sha256': sha256(Path(env['LAWAM_CONFIG_PATH'])),
        'lawam_vision_config_sha256': sha256(Path(env['LAWAM_VISION_PATH']) / 'config.json'),
        'simulator_source_sha256': combined_digest((simulator / 'envs').rglob('*.py')),
        'simulator_config_sha256': combined_digest((simulator / 'task_config').glob('*.yml')),
        'embodiment_config_sha256': combined_digest(p for p in (simulator / 'assets/embodiments').rglob('*')
            if p.is_file() and p.suffix in {'.yaml', '.yml', '.urdf', '.srdf'}),
        'inference_source_sha256': combined_digest(inference_files),
        'evaluation_source_sha256': combined_digest([*Path(__file__).parent.glob('*.py'), PROJECT_ROOT / 'benchmarks/runtime.py']),
        'server_sha256': sha256(PROJECT_ROOT / 'serve.py'),
        'harness_source_sha256': combined_digest(p for p in Path(env['HARNESS_SOURCE']).rglob('*.py') if p.name != '_version.py'),
        'launcher_sha256': sha256(Path(__file__)),
        'rsi': json.loads((output / 'rsi_manifest.json').read_text()) if env['ROBOTWIN_RSI'] == 'true' else {'enabled': False},
        'execution': {key: env[key] for key in [*FLAGS, 'WORKERS_PER_GPU', 'BATCH_MAX_SIZE', 'BATCH_MAX_WAIT_TIME',
            'EPISODES_PER_JOB', 'INFERENCE_SEED', 'ASSET_CACHE_CLEAR_INTERVAL', 'VLA_EVAL_IMAGE_FORMAT']},
        'inference_environment': inference_environment(env, robotwin=True),
        'protocol': dict(test_num=args.episodes, settings=args.settings, instruction_type=env.get('INSTRUCTION_TYPE', 'unseen'),
            policy_timeout_outcome='failure',
            image_history_num_frames=2, image_history_stride=30, chunk_size=int(env.get('CHUNK_SIZE', '40')),
            num_inference_steps=int(env.get('NUM_INFERENCE_STEPS', '10')),
            server_response_timeout=int(env['SERVER_RESPONSE_TIMEOUT']), eval_video_log=env.get('EVAL_VIDEO_LOG', 'false') == 'true'),
    }
    path = output / 'run_manifest.json'
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest['protocol']


def prepare_jobs(args, env, protocol):
    from vla_eval.config import EvalConfig
    from .benchmark import RoboTwinBenchmark
    limits = yaml.safe_load((Path(env['ROBOTWIN_ROOT']) / 'task_config/_eval_step_limit.yml').read_text())
    if len(limits) != 50 or any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError('The released simulator must provide 50 valid task step limits')
    unknown = set(args.tasks) - limits.keys()
    if unknown:
        raise ValueError(f'Unknown tasks: {sorted(unknown)}')
    tasks = args.tasks or sorted(limits, key=lambda name: (-limits[name], name))
    output = args.output_dir
    configs = []
    jobs = []
    for task in tasks:
        for setting in sorted(args.settings):
            cache = Path(env['EXPERT_CACHE_DIR']) / f"{task}__{setting}__seed0__{protocol['instruction_type']}__n{args.episodes}.json"
            if not cache.is_file():
                cache = Path(env['EXPERT_CACHE_DIR']) / f"{task}__{setting}__seed0__{protocol['instruction_type']}__n100.json"
            entry = dict(benchmark='benchmarks.robotwin.benchmark:RoboTwinBenchmark', subname=f'{task}__{setting}',
                episodes_per_task=args.episodes, max_steps=limits[task], action_dim=14,
                params=dict(task_name=task, task_config=setting, seed=0, instruction_type=protocol['instruction_type'],
                    test_num=args.episodes, expert_task_cache_path=os.path.relpath(cache, PROJECT_ROOT), eval_video_log=protocol['eval_video_log'],
                    eval_video_save_dir=os.path.relpath(output / 'videos' / task / setting, PROJECT_ROOT),
                    asset_cache_clear_interval=int(env['ASSET_CACHE_CLEAR_INTERVAL']),
                    **{name: env[key] == 'true' for name, key in (
                        ('keep_previous_scene', 'ROBOTWIN_KEEP_PREVIOUS_SCENE'), ('fast_step_render', 'FAST_STEP_RENDER'),
                        ('fast_obs_no_copy', 'FAST_OBS_NO_COPY'), ('minimal_observation', 'MINIMAL_OBSERVATION'),
                        ('policy_cameras_only', 'POLICY_CAMERAS_ONLY'))}))
            EvalConfig.from_dict(entry)
            RoboTwinBenchmark(**entry['params'])._load_expert_task_cache()
            path = output / 'configs' / f'{task}__{setting}.yaml'
            path.write_text(yaml.safe_dump(dict(server=dict(url='ws://127.0.0.1:0', timeout=protocol['server_response_timeout']),
                output_dir=os.path.relpath(output / 'shards', PROJECT_ROOT), benchmarks=[entry]), sort_keys=False))
            configs.append(path)
            total = (args.episodes + int(env['EPISODES_PER_JOB']) - 1) // int(env['EPISODES_PER_JOB'])
            jobs.extend((path, sid, total) for sid in range(total))
    (output / 'task_names.txt').write_text('\n'.join(sorted(tasks)) + '\n')
    return configs, jobs


def execute(args, env, protocol, configs, jobs, stop):
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate RoboTwin shard jobs in the evaluation plan')
    for path, sid, total in jobs:
        result = args.output_dir / 'shards' / f'RoboTwinBenchmark_{path.stem}_shard{sid}of{total}.json'
        if result.exists():
            raise FileExistsError(f'Shard result already exists: {result}; use a new output directory')
    progress = Progress('RoboTwin', len(configs) * args.episodes)
    queue = Queue()
    for job in jobs:
        queue.put(job)
    port_start = int(env.get('BASE_PORT', '8210'))
    ports = [port_start + index for index in range(len(args.gpus))]
    if not 1 <= min(ports) <= max(ports) <= 65535:
        raise ValueError('Invalid model-server port range')
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    with reserve_gpus(args.gpus), ExitStack() as stack, tempfile.TemporaryDirectory(prefix='a2meas-robotwin-') as temporary:
        servers = []
        for gpu, port in zip(args.gpus, ports):
            local = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=tempfile.gettempdir())
            command = [env['MODEL_PYTHON'], str(PROJECT_ROOT / 'serve.py'), 'robotwin', 'serve',
                '--checkpoint', env['CHECKPOINT'], '--rynnbrain_path', env['RYNNBRAIN_PATH'],
                '--lawam_config_path', env['LAWAM_CONFIG_PATH'], '--lawam_vision_path', env['LAWAM_VISION_PATH'],
                '--image_history_num_frames', '2', '--image_history_stride', '30',
                '--num_inference_steps', str(protocol['num_inference_steps']), '--chunk_size', str(protocol['chunk_size']),
                '--device', 'cuda', '--host', '127.0.0.1', '--port', str(port),
                '--max_batch_size', env['BATCH_MAX_SIZE'], '--max_wait_time', env['BATCH_MAX_WAIT_TIME'],
                '--inference_seed', env['INFERENCE_SEED']]
            server = stack.enter_context(running(command, local))
            servers.append(server)
            wait_ready(server, f'http://127.0.0.1:{port}/config', stop, int(env.get('SERVER_STARTUP_TIMEOUT', '600')))
        def worker(index):
            gpu, port, server = args.gpus[index], ports[index], servers[index]
            while not stop.is_set():
                try:
                    path, sid, total = queue.get_nowait()
                except Empty:
                    return True
                if server.poll() is not None:
                    stop.set()
                    raise RuntimeError(f'Model server on GPU {gpu} exited')
                expected = set(range(sid, args.episodes, total))
                with tempfile.TemporaryDirectory(dir=temporary, prefix='shard-') as scratch:
                    command = [env['ROBOTWIN_PYTHON'], '-m', 'benchmarks.robotwin.benchmark',
                        '--config', str(path), '--server-url', f'ws://127.0.0.1:{port}',
                        '--output-dir', str(args.output_dir / 'shards'),
                        '--shard-id', str(sid), '--num-shards', str(total)]
                    with running(command, dict(env, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=scratch,
                                               XDG_CACHE_HOME=str(Path(scratch) / 'cache'))) as process:
                        while process.poll() is None and not stop.wait(1):
                            if server.poll() is not None:
                                stop.set()
                                raise RuntimeError(f'Model server on GPU {gpu} exited')
                        if stop.is_set():
                            return False
                        if process.returncode:
                            stop.set()
                            raise subprocess.CalledProcessError(process.returncode, command)
                _, episodes, _ = collect(path, args.output_dir / 'results', args.output_dir / 'shards')
                if not expected.issubset(episodes):
                    stop.set()
                    raise RuntimeError(f'Incomplete RoboTwin shard: {path.stem} {sid}/{total}')
                progress.update(episodes[index]['metrics']['success'] for index in sorted(expected))
            return False

        with ThreadPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu) as pool:
            futures = [pool.submit(worker, index) for _ in range(args.workers_per_gpu) for index in range(len(args.gpus))]
            successful = True
            for future in as_completed(futures):
                try:
                    successful &= future.result()
                except BaseException:
                    stop.set()
                    raise
        if not successful:
            raise RuntimeError('Evaluation did not complete all scheduled episodes')


def main(argv=None):
    args = options(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}; use a new output directory')
    if args.simulator_root == SIMULATOR_ROOT:
        from .prepare import prepare
        prepare(args.resources / 'robotwin/simulator', args.simulator_root)
    env = environment(args)
    os.environ.update({key: env[key] for key in ['ROBOTWIN_ROOT', 'ROBOTWIN_RSI', 'VLA_EVAL_IMAGE_FORMAT', 'RSI_SAM3_WEIGHTS']})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    configs = []
    try:
        for name in ('results', 'shards', 'configs', 'videos'):
            (args.output_dir / name).mkdir(exist_ok=True)
        report = validate_checkpoint(Path(env['CHECKPOINT']), 2, int(env.get('CHUNK_SIZE', '40')), int(env.get('NUM_INFERENCE_STEPS', '10')))
        if not report['compatible']:
            raise ValueError(report['errors'])
        subprocess.run([env['ROBOTWIN_PYTHON'], '-m', 'vla_eval.cli.main', 'run', '--help'], env=env, stdout=subprocess.DEVNULL, check=True)
        if env['ROBOTWIN_RSI'] == 'true':
            from a2meas.robotwin.rsi.prepare import prepare
            prepare(args.output_dir, Path(env['ROBOTWIN_RSI_CONFIG']), ','.join(args.tasks))
            env['ROBOTWIN_RSI_CONFIG'] = str(args.output_dir / 'rsi_config.json')
        protocol = write_run_manifest(args, env)
        configs, jobs = prepare_jobs(args, env, protocol)
        if args.dry_run:
            return 0
        try:
            if jobs:
                execute(args, env, protocol, configs, jobs, stop)
        finally:
            for path in configs:
                config, episodes, template = collect(path, args.output_dir / 'results', args.output_dir / 'shards')
                publish(config, episodes, template, args.output_dir / 'results')
            summary_status = summarize(['--results-dir', str(args.output_dir / 'results'), '--output', str(args.output_dir / 'summary.md'),
                '--expected-seeds', str(args.episodes), '--expected-settings', ','.join(args.settings),
                '--checkpoint', os.path.relpath(env['CHECKPOINT'], PROJECT_ROOT)])
        return 143 if stop.is_set() else summary_status
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
