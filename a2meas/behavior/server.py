"""Single-rollout BEHAVIOR gateway with one isolated VLA worker."""

import asyncio
import hashlib
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import websockets
from websockets.asyncio.server import serve
from .protocol import Packer, unpackb
from .observations import CAMERA_KEYS
from a2meas import cpu_affinity
from a2meas.behavior import episode_seed, load_routes



def route_worker(connection, task_id, resources, adapter=False, num_inference_steps=None,
                 compile_mode="max-autotune-no-cudagraphs"):
    os.setsid()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    policy = None
    try:
        routing = load_routes()
        slot = f"{os.uname().nodename}-{os.environ['CUDA_VISIBLE_DEVICES']}-{sorted(os.sched_getaffinity(0))}"
        slot = hashlib.sha256(slot.encode()).hexdigest()[:16]
        model = routing['tasks'][str(task_id)]['model']
        if adapter:
            from .correction import load_config
            config = load_config()
            if str(task_id) in config['tasks']:
                model = config['model']
        cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR']) / slot / model
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
        os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
        os.environ['A2MEAS_COMPILE_MODE'] = compile_mode
        from .policy import VLAPolicy

        policy = VLAPolicy(routing["tasks"][str(task_id)], Path(resources),
                           adapter=adapter, num_inference_steps=num_inference_steps)
        policy.warmup()
        connection.send(("ready", None))
        while True:
            operation, payload = connection.recv()
            try:
                if operation == "route":
                    policy.configure_route(routing["tasks"][str(payload)])
                    result = None
                elif operation == "seed":
                    import torch
                    torch.manual_seed(payload)
                    torch.cuda.manual_seed_all(payload)
                    result = None
                elif operation == "reset":
                    policy.reset()
                    result = None
                elif operation == "step":
                    result = policy.step(payload)
                else:
                    raise ValueError("Unknown worker operation")
                connection.send(("ok", result))
            except Exception as error:
                logging.getLogger(__name__).exception('BEHAVIOR inference failed during %s', operation)
                connection.send(("error", str(error)))
    except EOFError:
        pass
    except Exception:
        raise
    finally:
        if policy is not None:
            policy.close()
        connection.close()


def single_rollout_observation(observation):
    """Validate one rollout and select the worker's RGB and proprioceptive inputs."""
    state = np.asarray(observation["robot_r1::proprio"], dtype=np.float32)
    if state.shape not in ((61,), (1, 61)) or not np.isfinite(state).all():
        raise ValueError("Expected finite 61D proprioception with optional batch of one")
    result = {key: observation[key] for key in CAMERA_KEYS}
    if state.ndim == 2:
        for key, value in result.items():
            if value.ndim != 4 or value.shape[0] != 1:
                raise ValueError("Inconsistent single-rollout batch for " + key)
            result[key] = value[0]
        state = state[0]
    result["robot_r1::proprio"] = state
    return result


class TaskRoutingError(ValueError):
    pass


def observation_task_id(observation):
    value = np.asarray(observation.get("task_id"))
    if (value.size != 1 or not np.issubdtype(value.dtype, np.integer)
            or not 0 <= value.item() < 100):
        raise TaskRoutingError("Expected one integer task_id in 0..99")
    return int(value.item())


class RouteManager:
    def __init__(self, resources, cpus, *, adapter=False, num_inference_steps=None, seed=0,
                 compile_mode="max-autotune-no-cudagraphs"):
        self.resources = str(resources)
        self.cpus = cpus
        self.adapter_config = None
        if adapter:
            from .correction import load_config
            self.adapter_config = load_config()
        self.seed = seed
        self.compile_mode = compile_mode
        self.bound_instance = None
        self.num_inference_steps = num_inference_steps
        self.routes = load_routes()["tasks"]
        self.task_names = {key: route["task"] for key, route in self.routes.items()}
        self.active_task = self.bound_task = None
        self.process = self.connection = None
        self.broken = self.closed = False
        self.lifecycle = threading.Lock()

    def alive(self):
        with self.lifecycle:
            process = self.process
            return not self.closed and not self.broken and (process is None or process.is_alive())

    def route_key(self, task_id):
        route = self.routes[str(task_id)]
        if self.adapter_config is not None and str(task_id) in self.adapter_config["tasks"]:
            route = self.adapter_config
        return route["model"], self.num_inference_steps or route["num_inference_steps"]

    def load(self, task_id, timeout):
        if (self.process is not None and self.active_task is not None
                and self.route_key(self.active_task) == self.route_key(task_id)):
            self.request("route", task_id, timeout=timeout)
            self.active_task = task_id
            return
        self.close_worker()
        try:
            with self.lifecycle:
                if self.closed:
                    raise RuntimeError("Gateway is closing")
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                self.connection = parent
                process = context.Process(
                    target=route_worker, args=(child, task_id, self.resources, self.adapter_config is not None,
                                              self.num_inference_steps, self.compile_mode)
                )
                try:
                    with cpu_affinity(self.cpus):
                        process.start()
                    self.process = process
                finally:
                    child.close()
            if not parent.poll(timeout):
                raise TimeoutError("Route loading/warmup exceeded the request budget")
            if parent.recv()[0] != "ready":
                raise RuntimeError("Route worker did not initialize")
            with self.lifecycle:
                if self.closed:
                    raise RuntimeError("Gateway is closing")
                self.active_task = task_id
        except BaseException:
            self.broken = True
            self.close_worker()
            raise

    def request(self, operation, payload=None, timeout=580):
        connection = self.connection
        if self.closed or connection is None:
            raise RuntimeError("Route is unavailable")
        connection.send((operation, payload))
        if not connection.poll(timeout):
            self.broken = True
            self.close_worker()
            raise TimeoutError("Route request exceeded its time budget")
        status, result = connection.recv()
        if status != "ok":
            raise RuntimeError(result)
        return result

    def reset(self):
        self.bound_task = self.bound_instance = None
        if self.process is not None:
            self.request("reset")

    def forward(self, observation):
        task_id = observation_task_id(observation)
        if self.bound_task is not None and task_id != self.bound_task:
            raise TaskRoutingError("task_id changed within a rollout; reset required")
        explicit_task = observation.get("task")
        if explicit_task is not None and explicit_task != self.task_names[str(task_id)]:
            raise TaskRoutingError("Task text does not match task_id")
        instance = np.asarray(observation.get("instance_id", 0))
        if instance.size != 1 or not np.issubdtype(instance.dtype, np.integer) or instance.item() < 0:
            raise TaskRoutingError("Expected a nonnegative integer instance_id")
        instance = int(instance.item())
        if self.bound_instance is not None and instance != self.bound_instance:
            raise TaskRoutingError("instance_id changed within a rollout; reset required")
        observation = single_rollout_observation(observation)
        started = time.monotonic()
        if self.active_task != task_id:
            self.load(task_id, timeout=540)
            self.request("reset", timeout=max(1, 580 - (time.monotonic() - started)))
        if self.bound_instance is None:
            self.request("seed", episode_seed(self.seed, task_id, instance),
                         timeout=max(1, 580 - (time.monotonic() - started)))
        self.bound_task, self.bound_instance = task_id, instance
        return self.request("step", observation, timeout=max(1, 580 - (time.monotonic() - started)))

    def close_worker(self):
        with self.lifecycle:
            process, connection = self.process, self.connection
            self.process = self.connection = None
            self.active_task = None
        if connection is not None:
            connection.close()
        if process is not None:
            process.join(timeout=15)
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    if process.is_alive():
                        process.terminate()
                process.join(timeout=15)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join()

    def close(self):
        with self.lifecycle:
            self.closed = True
        self.close_worker()


async def run(policy, host, port):
    lock = asyncio.Lock()
    active = None
    connections = 0
    initialized = failed = False
    last_digest = last_response = None
    pending = set()

    def health(connection, request):
        if request.path == "/healthz":
            if not policy.alive():
                return connection.respond(503, "Unavailable\n")
            return connection.respond(200, "OK\n")

    async def transact(data, observation, reset, replay_allowed):
        nonlocal initialized, failed, last_digest, last_response
        async with lock:
            if reset:
                initialized, failed = False, True
                last_digest = last_response = None
                await asyncio.to_thread(policy.reset)
                initialized, failed = True, False
                return None
            if not initialized or failed:
                raise RuntimeError("A rollout reset is required")
            digest = hashlib.sha256(data).digest()
            if replay_allowed and digest == last_digest and last_response is not None:
                return last_response
            try:
                action = await asyncio.to_thread(policy.forward, observation)
                if action.shape != (23,) or action.dtype != np.float32 or not np.isfinite(action).all():
                    raise ValueError("Policy must return a finite 23D float32 action")
                response = Packer().pack({"action": action})
            except Exception:
                failed = True
                raise
            last_digest, last_response = digest, response
            return response

    async def handler(websocket):
        nonlocal active, connections
        previous, active = active, websocket
        # Protocol retries resend identical request bytes.
        replay_allowed = connections > 0
        connections += 1
        if previous is not None:
            await previous.close(code=1012, reason="Rollout connection replaced")
        try:
            await websocket.send(Packer().pack({"action_dim": 23, "expected_state_dim": 61, "image_size": 224}))
            async for data in websocket:
                if active is not websocket:
                    break
                if not isinstance(data, bytes):
                    raise ValueError("Expected binary MessagePack")
                observation = unpackb(data)
                if not isinstance(observation, dict):
                    raise ValueError("Expected an observation map")
                reset = "reset" in observation
                if reset and (set(observation) != {"reset"} or observation["reset"] is not True):
                    raise ValueError("Expected standalone reset=true")
                transaction = asyncio.create_task(transact(data, observation, reset, replay_allowed))
                pending.add(transaction)
                transaction.add_done_callback(pending.discard)
                response = await asyncio.shield(transaction)
                replay_allowed = False
                if response is not None:
                    await websocket.send(response)
        except TaskRoutingError as error:
            await websocket.close(code=1008, reason=str(error))
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logging.getLogger(__name__).exception('BEHAVIOR policy request failed')
            await websocket.close(code=1011, reason="Policy request failed")
        finally:
            if active is websocket:
                active = None

    listener = None
    try:
        listener = await serve(
            handler,
            host,
            port,
            max_size=None,
            compression=None,
            ping_interval=None,
            close_timeout=2,
            process_request=health,
        )
        while policy.alive():
            await asyncio.sleep(0.2)
        raise RuntimeError("Route worker exited")
    finally:
        try:
            await asyncio.to_thread(policy.close)
        finally:
            if listener is not None:
                listener.close()
                await listener.wait_closed()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
