"""Run a resident BEHAVIOR worker on one renderer and one policy endpoint."""
import copy
import json
import logging
import os
import socket
from pathlib import Path
import subprocess
import sys
import warnings

from benchmarks.behavior.results import validate_result
from benchmarks.runtime import Progress


def configure_logging():
    for name in ('AutoNode', 'omnigibson', 'omnigibson.eval.eval',
                 'omnigibson.eval.evaluator', 'omnigibson.eval.utils.network_utils'):
        logging.getLogger(name).setLevel(logging.ERROR)
    for handler in logging.getLogger('omnigibson').handlers:
        handler.setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message=r'.*Casting input x to numpy array\.',
                            category=UserWarning, module=r'gymnasium\.spaces\.box')


def compact_rgb(rgb, size):
    import numpy as np
    native = rgb.shape[:2] == (size, size)
    if native and rgb.shape[-1] == 3:
        return np.ascontiguousarray(rgb[..., :3])
    from PIL import Image
    image = Image.fromarray(rgb).convert('RGB')
    if native:
        return np.asarray(image, dtype=np.uint8)
    scale = min(size / image.height, size / image.width)
    height, width = max(1, round(image.height * scale)), max(1, round(image.width * scale))
    resized = np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.uint8)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    result[top:top + height, left:left + width] = resized
    return result


def encode_video(path, height, width, rate):
    import av
    import numpy as np
    with av.open(path, mode='w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.height, stream.width, stream.pix_fmt = height, width, 'yuv420p'
        frame_bytes = height * width * 3
        while data := sys.stdin.buffer.read(frame_bytes):
            if len(data) != frame_bytes:
                raise ValueError('Incomplete RGB video frame')
            rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def simulation_config(config):
    # Isaac Sim Vulkan ignores CUDA_VISIBLE_DEVICES; PhysX uses its remapped index.
    config = dict(config, active_gpu=int(os.environ['OMNIGIBSON_GPU_ID']), physics_gpu=0,
                  anti_aliasing=4, limit_cpu_threads=min(16, len(os.sched_getaffinity(0))))
    temporary = Path(os.environ['TMPDIR'])
    config['extra_args'] = [*config.get('extra_args', []),
                            f'--/log/file={temporary / "kit.log"}',
                            f'--/app/userConfigPath={temporary / "user.config.json"}',
                            '--/log/level=warning', '--/log/outputStreamLevel=error',
                            '--/log/fileLogLevel=warning']
    return config


def run():
    configure_logging()
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    import isaacsim
    simulation_app = isaacsim.SimulationApp

    def start(config, *args, **kwargs):
        app = simulation_app(simulation_config(config), *args, **kwargs)
        configure_logging()
        return app

    isaacsim.SimulationApp = start
    import cv2
    import numpy as np
    import torch
    import warp
    warp.config.quiet = True
    import omnigibson
    from omnigibson.eval import eval as official_cli
    from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids, TASK_NAMES_TO_INDICES
    from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER, TASK_NAMES_TO_ROOMS, seed_everything
    from omnigibson.eval.utils.score_utils import load_human_stats
    from omnigibson.utils.asset_utils import get_task_instance_path
    from gello.utils.og_teleop_utils import load_available_tasks
    from a2meas.behavior import episode_seed
    from omnigibson.eval.policies import WebsocketPolicy
    configure_logging()

    if omnigibson.__version__ != '3.9.3':
        raise RuntimeError('BEHAVIOR SDK 3.9.3 is required')
    cv2.setNumThreads(1)

    class CompactPolicy(WebsocketPolicy):
        def __init__(self, *args, evaluator, **kwargs):
            super().__init__(*args, **kwargs)
            self.evaluator = evaluator

        def forward(self, obs):
            size = self.policy.get_server_metadata()['image_size']
            compact = {}
            for key, value in obs.items():
                value = value[0] if isinstance(value, torch.Tensor) else value
                if key.endswith(('::intrinsic', '::rgb_detection', '::depth_linear')):
                    continue
                if key.endswith('::rgb'):
                    value = compact_rgb(value.detach().cpu().numpy(), size)
                compact[key] = value
            compact["instance_id"] = self.evaluator.instance_eval_states[0].instance_id
            return super().forward(compact).unsqueeze(0)

    class ResidentEvaluator(BatchedEvaluator):
        def __init__(self, cfg):
            if int(cfg.num_envs) != 1:
                raise ValueError('Each resident worker requires one environment')
            self.pristine_scene = None
            super().__init__(cfg)

        def load_policy(self):
            if self.cfg.policy_name == 'local':
                return super().load_policy()
            return CompactPolicy(host=self.cfg.model.host, port=self.cfg.model.port,
                                 allow_reconnect=True, action_chunk_size=0, evaluator=self)

        def load_batch(self, env_idx_to_instance, **kwargs):
            scene = self.env.scenes[0]
            if self.pristine_scene is None:
                self.pristine_scene = copy.deepcopy(scene.save(as_dict=True))
            else:
                scene.restore(copy.deepcopy(self.pristine_scene), update_initial_file=True)
                self.env.task.assign_object_scope_with_cache(self.env, 0)
            super().load_batch(env_idx_to_instance, **kwargs)

        def switch_task(self, name):
            tasks = load_available_tasks()
            scene = self.env.scenes[0]
            if tasks[name][0]['scene_model'] != scene.scene_model:
                raise ValueError('Resident task switch requires the same scene')
            template = self.env.task.get_cached_activity_scene_filename(scene.scene_model, name, 0, 0)
            path = get_task_instance_path(scene.scene_model, template)
            if path is None:
                raise FileNotFoundError(f'Missing task template: {template}')
            target = json.loads(Path(path).read_text())
            scene.load_room_instances = list(TASK_NAMES_TO_ROOMS[name])
            task_metadata = target['metadata']['task']
            objects = target['objects_info']['init_info']
            objects = {key: value for key, value in objects.items()
                       if scene._should_load_object(value, task_metadata)}
            target['objects_info']['init_info'] = objects
            object_state = target['state']['registry']['object_registry']
            target['state']['registry']['object_registry'] = {key: object_state[key] for key in objects}
            current = scene.save(as_dict=True)
            for robot in scene.robots:
                target['objects_info']['init_info'][robot.name] = current['objects_info']['init_info'][robot.name]
                target['state']['registry']['object_registry'][robot.name] = current['state']['registry']['object_registry'][robot.name]
            scene.restore(target, update_initial_file=True)
            task = self.env.task
            task.update_activity(self.env, name, 0)
            accepted, feedback = task.initialize_activity(self.env)
            if not accepted:
                raise RuntimeError(f'Task initialization failed: {feedback}')
            self.cfg.task.name = name
            self.human_stats = load_human_stats(name)
            self.max_steps = (int(self.human_stats['length'] * EVAL_TIMEOUT_MULTIPLIER)
                              if self.cfg.max_steps is None else int(self.cfg.max_steps))
            task._termination_config['max_steps'] = self.max_steps
            task._termination_conditions['timeout']._max_steps = self.max_steps
            for state in self.instance_eval_states:
                state.metrics = self.load_metrics(state.env_accessor)
            self.pristine_scene = None

        def run(self, instances_to_run, write_video=False, video_path=None, metrics_dir=None,
                rollout_id=0, video_fps=30):
            groups = json.loads(os.environ['A2MEAS_TASK_GROUPS'])
            results = {}
            dispatch_path = os.environ.get('A2MEAS_DISPATCH_SOCKET')
            progress = Progress('BEHAVIOR', sum(len(group['instances']) for group in groups))

            def dispatch(request):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(10)
                    client.connect(dispatch_path)
                    client.sendall((json.dumps(request) + '\n').encode())
                    with client.makefile('rb') as stream:
                        return json.loads(stream.readline())

            while groups:
                group = groups.pop(0)
                task = group['task']
                instances = resolve_instance_ids(task, group['instances'], mode=self.cfg.mode)
                for index, instance in enumerate(instances):
                    if index and dispatch_path:
                        dispatch({'port': int(os.environ['A2MEAS_DISPATCH_PORT']),
                                  'task_id': TASK_NAMES_TO_INDICES[task], 'instance': group['instances'][index]})
                    filename = f'{task}_{instance}_{rollout_id}'
                    if ((metrics_dir and (Path(metrics_dir) / f'{filename}.json').exists())
                            or (write_video and (Path(video_path) / f'{filename}.mp4').exists())):
                        raise FileExistsError(f'Rollout already exists: {filename}')
                    seed_everything(episode_seed(int(os.environ['A2MEAS_EVAL_SEED']),
                                                 TASK_NAMES_TO_INDICES[task], instance))
                    if index == 0 and task != self.cfg.task.name:
                        self.switch_task(task)
                    result = super().run([instance], write_video, video_path, metrics_dir, rollout_id, video_fps)
                    validate_result(result[instance], task, instance)
                    if dispatch_path:
                        dispatch({'port': int(os.environ['A2MEAS_DISPATCH_PORT']),
                                  'task_id': TASK_NAMES_TO_INDICES[task], 'completed': instance})
                    else:
                        progress.update([result[instance]['success']])
                    results[(task, instance)] = result[instance]
                if dispatch_path and not groups:
                    next_group = dispatch({'port': int(os.environ['A2MEAS_DISPATCH_PORT']),
                                           'task_id': TASK_NAMES_TO_INDICES[task]})
                    if next_group is not None:
                        groups.append(next_group)
            return results

        def _set_video_writer(self, state, writer):
            if state.video_writer is not None:
                encoder = state.video_writer
                encoder.stdin.close()
                code = encoder.wait()
                if code:
                    raise subprocess.CalledProcessError(code, encoder.args)
            state.video_writer = None
            if writer is not None:
                container, stream = writer
                path, rate = container.name, stream.average_rate
                container.close()
                state.video_writer = subprocess.Popen(
                    [sys.executable, __file__, '--encode-video', path, '448', '672', str(rate)],
                    stdin=subprocess.PIPE)

        def _write_video(self, state):
            images = [cv2.resize(state.obs[self.robot_camera_names[role] + '::rgb'].detach().cpu().numpy(),
                                 (224, 224)) for role in ('left_wrist', 'right_wrist', 'head')]
            frame = np.hstack([np.vstack(images[:2]), cv2.resize(images[2], (448, 448))])
            state.video_writer.stdin.write(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).tobytes())

    official_cli.BatchedEvaluator = ResidentEvaluator
    try:
        official_cli.main()
    finally:
        isaacsim.SimulationApp = simulation_app


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--encode-video':
            _, _, path, height, width, rate = sys.argv
            encode_video(path, int(height), int(width), int(rate))
        else:
            run()
    except Exception:
        # Kit fast shutdown otherwise returns zero for Python exceptions.
        sys.excepthook(*sys.exc_info())
        sys.exit(1)
