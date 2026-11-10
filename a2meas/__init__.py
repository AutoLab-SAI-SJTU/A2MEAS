"""Shared paths and runtime resources."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESOURCE_ROOT = PROJECT_ROOT / "resources"
RUNTIME_ROOT = (PROJECT_ROOT / Path(os.environ.get("A2MEAS_RUNTIME", "runtime")).expanduser()).resolve()


def cpu_layout(gpu, workers):
    """Assign separate GPU-local physical cores to simulation and each policy."""
    bus = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=pci.bus_id", "--format=csv,noheader",
    ], text=True).strip()[-12:].lower()
    node = int((Path("/sys/bus/pci/devices") / bus / "numa_node").read_text())
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        if node >= 0 and not (topology / f"node{node}").exists():
            continue
        siblings = (topology / "topology/thread_siblings_list").read_text()
        cores.setdefault(siblings, set()).add(cpu)
    cores = list(cores.values())
    width = min(3, len(cores) // (workers + 1))
    if not width:
        raise ValueError(f"Need at least {workers + 1} GPU-local physical CPU cores for {workers} workers")
    policies = [set().union(*cores[index * width:(index + 1) * width]) for index in range(workers)]
    simulator = set().union(*cores[workers * width:workers * width + 16])
    return simulator, policies


@contextmanager
def cpu_affinity(cpus):
    """Let child processes inherit their CPU pool before importing libraries."""
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


@contextmanager
def runtime_environment():
    """Scope CLI caches, process signals and owned temporary files."""
    cache = RUNTIME_ROOT / "cache"
    previous_temporary = tempfile.tempdir
    temporary_root = Path(tempfile.gettempdir())
    # Torch's compiler splits cache paths on whitespace.
    paths = {
        "TORCHINDUCTOR_CACHE_DIR": Path(os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR", cache / "inductor"
        )).resolve(),
        "CUDA_CACHE_PATH": cache / "cuda",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "TRITON_CACHE_DIR": cache / "triton",
        "PYTHONPYCACHEPREFIX": cache / "python",
        "HF_HOME": Path(os.environ.get(
            "HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
        )).expanduser().resolve(),
        "XDG_CACHE_HOME": cache / "xdg",
        "OMNIGIBSON_APPDATA_PATH": cache / "omnigibson",
    }
    if any(char.isspace() for char in str(paths["TORCHINDUCTOR_CACHE_DIR"])):
        raise ValueError("TORCHINDUCTOR_CACHE_DIR must point to a path without whitespace")
    for name, path in paths.items():
        if name != "HF_HOME":
            path.mkdir(parents=True, exist_ok=True)
    # NVCC limits the length of its temporary directory path.
    with tempfile.TemporaryDirectory(prefix="a2meas-", dir=temporary_root) as temporary:
        environment = dict(
            {key: str(path) for key, path in paths.items()},
            PATH=str(Path(sys.executable).absolute().parent) + os.pathsep + os.environ['PATH'],
            TMPDIR=temporary,
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            TORCHINDUCTOR_COMPILE_THREADS="4",
        )
        previous = {key: os.environ.get(key) for key in environment}
        previous_pycache = sys.pycache_prefix
        previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        try:
            os.environ.update(environment)
            tempfile.tempdir = temporary
            sys.pycache_prefix = str(paths["PYTHONPYCACHEPREFIX"])
            yield
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            tempfile.tempdir = previous_temporary
            sys.pycache_prefix = previous_pycache
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
