"""RoboTwin 2.0 joint-position evaluation with dense policy-camera observations."""

from __future__ import annotations
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np
from vla_eval.benchmarks.base import StepBenchmark, StepResult
from a2meas import PROJECT_ROOT
from a2meas.robotwin import SIMULATOR_ROOT
from benchmarks.robotwin.video import MultiCameraVideoWriter
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task



class RoboTwinBenchmark(StepBenchmark):
    """Evaluate expert-valid 14D joint-position episodes with three RGB cameras.

    Simulator resources load lazily. Cached seeds and instructions remain read-only."""

    def __init__(
        self,
        task_name: str,
        expert_task_cache_path: str,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "unseen",
        test_num: int = 100,
        reset_unstable_retries: int = 2,
        fast_step_render: bool = False,
        fast_obs_no_copy: bool = False,
        asset_cache_clear_interval: int = 1,
        minimal_observation: bool = False,
        policy_cameras_only: bool = False,
        eval_video_log: bool = False,
        eval_video_save_dir: str | None = None,
        eval_video_layout: str = "head",
        keep_previous_scene: bool = False,
    ) -> None:
        import re

        super().__init__()
        image_format = os.environ.get("VLA_EVAL_IMAGE_FORMAT", "png")
        if image_format not in {"raw", "png"}:
            raise ValueError("RoboTwin evaluation requires lossless raw or png images")
        from vla_eval.protocol.numpy_codec import set_image_format

        set_image_format(image_format)
        if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", task_name):
            raise ValueError(f"Invalid task_name: {task_name!r}")
        if not re.fullmatch("[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_name = task_name
        self.robotwin_root = str((PROJECT_ROOT / os.environ.get("ROBOTWIN_ROOT", SIMULATOR_ROOT)).resolve())
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = int(test_num)
        if self.test_num < 1:
            raise ValueError(f"test_num must be positive, got {self.test_num}")
        self.reset_unstable_retries = int(reset_unstable_retries)
        if self.reset_unstable_retries < 0:
            raise ValueError(
                f"reset_unstable_retries must be non-negative, got {self.reset_unstable_retries}"
            )
        self.fast_step_render = bool(fast_step_render)
        self.fast_obs_no_copy = bool(fast_obs_no_copy)
        self.asset_cache_clear_interval = int(asset_cache_clear_interval)
        self.keep_previous_scene = bool(keep_previous_scene)
        self._deferred_scene_close: tuple[Any, bool] | None = None
        if self.asset_cache_clear_interval < 1:
            raise ValueError(
                f"asset_cache_clear_interval must be positive, got {self.asset_cache_clear_interval}"
            )
        self.minimal_observation = bool(minimal_observation)
        self.policy_cameras_only = bool(policy_cameras_only)
        self.expert_task_cache_path = str((PROJECT_ROOT / expert_task_cache_path).resolve())
        self._episode_tasks: list[Task] | None = None
        self.eval_video_log = bool(eval_video_log)
        self.eval_video_save_dir = str((PROJECT_ROOT / eval_video_save_dir).resolve()) if eval_video_save_dir else None
        if eval_video_layout not in ("head", "three_camera"):
            raise ValueError(f"Unsupported eval_video_layout: {eval_video_layout!r}")
        self.eval_video_layout = eval_video_layout
        self._video_tile_size: tuple[int, int] | None = None
        self._video_size: str | None = None
        self._current_eval_video_path: str | None = None
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None
        self._last_rendered_obs: Any = None
        self._episodes_since_cache_clear = 0

    @staticmethod
    def _joint_state_from_env(env: Any) -> np.ndarray:
        return np.asarray(
            env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
            dtype=np.float64,
        ).flatten()

    def _reuse_last_rgb_observation(self, joint_state: np.ndarray) -> dict[str, Any]:
        """Build a state-only step observation while advancing server STM indices."""
        if self._last_rendered_obs is None:
            raise RuntimeError(
                "Cannot reuse RGB before the initial RoboTwin observation is rendered"
            )
        return {
            "observation": self._last_rendered_obs["observation"],
            "language": self._last_rendered_obs.get("language", ""),
            "joint_action": {"vector": joint_state},
            "_vla_eval_reuse_previous_images": True,
        }

    def _keep_policy_cameras_only(self) -> None:
        """Drop unused static cameras from capture lists after scene creation."""
        if (
            not self.policy_cameras_only
            or self.eval_video_log
        ):
            return
        cameras = self._env.cameras
        kept = [
            (camera, name, config)
            for camera, name, config in zip(
                cameras.static_camera_list,
                cameras.static_camera_name,
                cameras.static_camera_config,
            )
            if name == "head_camera"
        ]
        if len(kept) != 1:
            raise RuntimeError(
                f"policy_cameras_only expected exactly one head_camera, found {[name for _, name, _ in kept]}"
            )
        cameras.static_camera_list = [item[0] for item in kept]
        cameras.static_camera_name = [item[1] for item in kept]
        cameras.static_camera_config = [item[2] for item in kept]
        cameras.head_camera_id = 0

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, resolve embodiment."""
        if self._args is not None:
            return
        if self.robotwin_root not in sys.path:
            sys.path.insert(0, self.robotwin_root)
        os.chdir(self.robotwin_root)
        import yaml

        config_path = os.path.join(
            self.robotwin_root, "task_config", f"{self.task_config}.yml"
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)
        args["task_name"] = self.task_name
        args["task_config"] = self.task_config
        from envs import CONFIGS_PATH
        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)
        hcam = args["camera"]["head_camera_type"]
        wcam = args["camera"]["wrist_camera_type"]
        if hcam not in _camera_config:
            raise ValueError(f"Unknown RoboTwin head_camera_type: {hcam!r}")
        if wcam not in _camera_config:
            raise ValueError(f"Unknown RoboTwin wrist_camera_type: {wcam!r}")
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        if self.eval_video_log:
            video_save_dir = self.eval_video_save_dir or os.path.join(
                self.robotwin_root, "results", "vla_eval_videos", self.task_name
            )
            os.makedirs(video_save_dir, exist_ok=True)
            args["eval_video_save_dir"] = video_save_dir
            self._video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"
            if self.eval_video_layout == "three_camera":
                self._video_tile_size = (args["head_camera_w"], args["head_camera_h"])
                self._video_size = (
                    f"{args['head_camera_w'] * 2}x{args['head_camera_h'] * 3}"
                )
        args["eval_mode"] = True
        envs_module = importlib.import_module(f"envs.{self.task_name}")
        self._env_class = getattr(envs_module, self.task_name)
        self._args = args

    def _create_env(self) -> Any:
        assert self._env_class is not None
        env = self._env_class()
        env.fast_eval_step_render = self.fast_step_render
        env.fast_eval_obs_no_copy = (
            self.fast_obs_no_copy
            and (not self.eval_video_log)
        )
        env.fast_eval_minimal_obs = (
            self.minimal_observation
            and (not self.eval_video_log)
        )
        return env


    def _start_eval_video(self, episode_idx: int) -> None:
        if self._env is None or self._env.eval_video_path is None:
            return
        if self._video_size is None:
            raise RuntimeError(
                "eval_video_log is enabled but video size was not initialised"
            )
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise FileNotFoundError("Video recording requires FFmpeg")
        video_path = os.path.join(
            self._env.eval_video_path, f"episode{episode_idx}.mp4"
        )
        ffmpeg = subprocess.Popen(
            [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                self._video_size,
                "-framerate",
                "10",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                video_path,
            ],
            stdin=subprocess.PIPE,
        )
        self._current_eval_video_path = video_path
        writer: Any = ffmpeg
        if self.eval_video_layout == "three_camera":
            if self._video_tile_size is None:
                raise RuntimeError("Three-camera video tile size was not initialised")
            writer = MultiCameraVideoWriter(ffmpeg, self._env, *self._video_tile_size)
        self._env._set_eval_video_ffmpeg(writer)

    def _stop_eval_video(self, success: bool | None = None) -> None:
        if self._env is None or self._env.eval_video_path is None:
            return
        if not hasattr(self._env, "eval_video_ffmpeg"):
            return
        try:
            self._env._del_eval_video_ffmpeg()
        except Exception as e:
            pass
        if (
            self._current_eval_video_path
            and success is not None
            and os.path.isfile(self._current_eval_video_path)
        ):
            tagged_path = self._current_eval_video_path.replace(
                ".mp4", f"_{str(success).lower()}.mp4"
            )
            try:
                if os.path.isfile(tagged_path):
                    os.remove(tagged_path)
                os.rename(self._current_eval_video_path, tagged_path)
            except OSError as e:
                pass
        self._current_eval_video_path = None

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._stop_eval_video()
            finally:
                try:
                    self._env.close_env(clear_cache=True)
                finally:
                    self._env = None
                    self._last_rendered_obs = None

    _EXPERT_TASK_CACHE_VERSION = 1

    def _load_expert_task_cache(self) -> list[Task]:
        """Read the released expert-valid seed and instruction list without modifying it."""
        cache_path = Path(self.expert_task_cache_path)
        try:
            payload = json.loads(cache_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to read RoboTwin expert-task cache {cache_path}: {exc}"
            ) from exc
        expected = {
            "version": self._EXPERT_TASK_CACHE_VERSION,
            "task_name": self.task_name,
            "task_config": self.task_config,
            "seed": self.seed,
            "instruction_type": self.instruction_type,
            "instruction_override": None,
            "skip_expert_check": False,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"RoboTwin expert-task cache metadata mismatch for {cache_path}: {key}={payload.get(key)!r}, expected {value!r}"
                )
        tasks = payload.get("tasks")
        if (not isinstance(tasks, list) or type(payload.get("test_num")) is not int
                or payload["test_num"] != len(tasks) or len(tasks) < self.test_num):
            raise RuntimeError(f"RoboTwin expert-task cache {cache_path} does not contain {self.test_num} valid tasks")
        tasks = tasks[:self.test_num]
        required_keys = {"name", "suite", "seed", "episode_idx", "instruction"}
        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or not required_keys.issubset(task):
                raise RuntimeError(
                    f"Invalid task #{index} in RoboTwin expert-task cache {cache_path}"
                )
            if task["name"] != self.task_name or task["suite"] != "robotwin":
                raise RuntimeError(
                    f"Unexpected task #{index} in RoboTwin expert-task cache {cache_path}"
                )
            if (
                type(task["episode_idx"]) is not int
                or task["episode_idx"] != index
                or type(task["seed"]) is not int
                or (task["seed"] < 0)
                or (not isinstance(task["instruction"], str))
                or (not task["instruction"].strip())
            ):
                raise RuntimeError(
                    f"Invalid seed, index or instruction for task #{index} in {cache_path}"
                )
        return tasks


    def get_tasks(self) -> list[Task]:
        if self._episode_tasks is None:
            self._episode_tasks = self._load_expert_task_cache()
        return [{"name": self.task_name, "suite": "robotwin"}]


    def reset(self, task: Task) -> Any:
        if self._episode_tasks is None:
            self.get_tasks()
        episode_index = int(task.get("episode_idx", 0))
        if self._episode_tasks is None or not 0 <= episode_index < len(
            self._episode_tasks
        ):
            raise IndexError(
                f"RoboTwin episode index outside prepared seed list: {episode_index}"
            )
        task = self._episode_tasks[episode_index].copy()
        self._task = task
        try:
            return self._reset_environment(task)
        finally:
            if self._deferred_scene_close is not None:
                previous, clear_asset_cache = self._deferred_scene_close
                self._deferred_scene_close = None
                try:
                    previous.close_env(clear_cache=clear_asset_cache)
                except Exception as exc:
                    pass

    def _reset_environment(self, task: Task) -> Any:
        self._init_robotwin()
        assert self._args is not None
        if self._env is not None:
            self._episodes_since_cache_clear += 1
            clear_asset_cache = (
                self._episodes_since_cache_clear >= self.asset_cache_clear_interval
            )
            if self.keep_previous_scene and (not clear_asset_cache):
                self._deferred_scene_close = (self._env, clear_asset_cache)
            else:
                try:
                    self._env.close_env(clear_cache=clear_asset_cache)
                except Exception as exc:
                    pass
            if clear_asset_cache:
                self._episodes_since_cache_clear = 0
            self._env = None
        for attempt in range(self.reset_unstable_retries + 1):
            self._env = self._create_env()
            try:
                self._env.setup_demo(
                    now_ep_num=task.get("episode_idx", 0),
                    seed=task["seed"],
                    is_test=True,
                    **self._args,
                )
                break
            except Exception as exc:
                if exc.__class__.__name__ != "UnStableError":
                    raise
                try:
                    self._env.close_env(clear_cache=True)
                except Exception:
                    pass
                self._env = None
                if attempt >= self.reset_unstable_retries:
                    raise
        assert self._env is not None
        self._keep_policy_cameras_only()
        self._start_eval_video(int(task.get("episode_idx", 0)))
        self._env.set_instruction(instruction=task["instruction"])
        raw_obs = self._env.get_obs()
        self._last_rendered_obs = raw_obs
        return raw_obs

    def step(self, action: Action) -> StepResult:
        raw = action["actions"]
        act = np.asarray(raw, dtype=np.float64)
        if act.shape != (14,) or not np.isfinite(act).all():
            raise ValueError(f"Expected finite 14D joint positions, got {act.shape}")
        self._env.take_action(act, action_type="qpos")
        joint_state_after = self._joint_state_from_env(self._env)
        success = bool(self._env.eval_success)
        done = success or self._env.take_action_cnt >= self._env.step_lim
        if done:
            raw_obs = self._reuse_last_rgb_observation(joint_state_after)
        else:
            raw_obs = self._env.get_obs()
            self._last_rendered_obs = raw_obs
        if done:
            self._stop_eval_video(success=success)
        return StepResult(
            obs=raw_obs,
            reward=1.0 if success else 0.0,
            done=done,
            info={"success": success},
        )

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        reuse_previous_images = bool(
            raw_obs.get("_vla_eval_reuse_previous_images", False)
        )
        return {
            "images": {}
            if reuse_previous_images
            else {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            "task_description": raw_obs.get("language")
            or (self._task or task).get("instruction", ""),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {
            "success": step_result.info.get("success", False),
            "seed": self._task["seed"],
            "episode_index": self._task["episode_idx"],
        }

    def get_metadata(self) -> dict[str, Any]:
        import yaml

        limits = yaml.safe_load((Path(self.robotwin_root) / "task_config/_eval_step_limit.yml").read_text())
        return {
            "max_steps": limits[self.task_name],
            "task_name": self.task_name,
            "action_dim": 14,
            "max_episodes_per_task": self.test_num,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": DimSpec("actions", 14, "joint_positions")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "joint_state": STATE_JOINT,
            "language": LANGUAGE,
        }


def main(argv=None):
    import argparse
    import logging
    import anyio
    import yaml
    from vla_eval.orchestrator import Orchestrator
    from vla_eval.results.collector import ResultCollector

    parser = argparse.ArgumentParser(description='Execute one RoboTwin evaluation shard')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard-id', type=int, required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_id < args.num_shards:
        parser.error('Require 0 <= shard-id < num-shards')

    logging.basicConfig(level=logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    config['server']['url'] = args.server_url
    config['output_dir'] = args.output_dir
    # The parent reports verified run-wide counts, not a separate table per shard.
    original = ResultCollector.print_summary
    ResultCollector.print_summary = lambda self: None
    try:
        anyio.run(Orchestrator(config, shard_id=args.shard_id, num_shards=args.num_shards).run)
    finally:
        ResultCollector.print_summary = original


if __name__ == '__main__':
    main()
