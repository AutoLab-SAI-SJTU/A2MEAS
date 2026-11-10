"""Owned process lifetimes and physical GPU reservations shared by both benchmarks."""
import fcntl
from contextlib import ExitStack, contextmanager, suppress
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from a2meas import PROJECT_ROOT, RUNTIME_ROOT


class Progress:
    def __init__(self, benchmark, total):
        self.benchmark, self.total = benchmark, total
        self.completed = self.successes = 0
        self.lock = threading.Lock()

    def update(self, outcomes):
        outcomes = list(outcomes)
        if not outcomes:
            return
        with self.lock:
            self.completed += len(outcomes)
            self.successes += sum(outcomes)
            print(f"[{self.benchmark}] Completed {self.completed}/{self.total} | "
                  f"Successes {self.successes} Failures {self.completed - self.successes} | "
                  f"Success rate {self.successes / self.completed:.2%}", flush=True)


def inference_environment(env, *, robotwin=False):
    """Make inherited model switches explicit in both workers and run identities."""
    defaults = {
        'LAM_DINO_BF16': 'true', 'LAM_DINO_FAST_PENULTIMATE': 'true',
        'LAM_DINO_ENCODE_MICROBATCH': '32',
        'TORCH_COMPILE_DISABLE': '1',
    }
    if robotwin:
        defaults.update(
            LEROBOT_ACTION_EXPERT_ATTN_IMPLEMENTATION='eager',
            LEROBOT_QWEN_METADATA_FASTPATH='true',
            QWEN_VISION_SDPA_ASSUME_EQUAL_SEGMENTS='0', QWEN_VISION_SDPA_BATCH_SEGMENTS='0',
        )
    values = {name: env.get(name, default) for name, default in defaults.items()}
    # PyTorch 2.7 treats any nonempty value, including "0", as disabling compilation.
    if values['TORCH_COMPILE_DISABLE'] == '0':
        values['TORCH_COMPILE_DISABLE'] = ''
    return values


@contextmanager
def reserve_gpus(gpus):
    # Shared storage coordinates containers; UUIDs distinguish physical GPUs across hosts.
    directory = RUNTIME_ROOT / "cache/gpu-locks"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            uuid = subprocess.check_output([
                "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader",
            ], text=True).strip().lower()
            stream = stack.enter_context((directory / f"{uuid}.lock").open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"GPU {gpu} is reserved by another A²-MEAS evaluation; run configurations sequentially") from None
        yield


def stop_process(process):
    # Every managed process starts its own session; children may outlive its leader.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def running(command, env):
    with subprocess.Popen(command, cwd=PROJECT_ROOT, env=env,
                          start_new_session=True) as process:
        try:
            yield process
        finally:
            stop_process(process)


def wait_ready(process, url, stop, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while not stop.is_set():
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with {process.returncode}")
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Service readiness timed out")
        stop.wait(1)
    raise InterruptedError("Evaluation interrupted")
