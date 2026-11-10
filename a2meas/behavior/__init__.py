"""BEHAVIOR task catalog."""

import hashlib
import json
from a2meas import PROJECT_ROOT, RUNTIME_ROOT

CONFIG_ROOT = PROJECT_ROOT / "configs/behavior"
SDK_ROOT = RUNTIME_ROOT / "behavior_sdk"
DATA_ROOT = PROJECT_ROOT / "resources/behavior"


def validate_sdk(root):
    for name in ("OmniGibson/omnigibson/eval/eval.py", "docs/assets/OmniGibson_logo.png"):
        if not (root / name).is_file():
            raise ValueError(f"BEHAVIOR SDK is missing {name}: {root}")


def load_routes():
    routing = json.loads((CONFIG_ROOT / "routing.json").read_text())
    for route in routing['tasks'].values():
        route['num_inference_steps'] = routing['num_inference_steps']
        route['execution'] = routing['execution'].copy()
    return routing


def resolve_task(value, routing):
    matches = [
        (int(key), route) for key, route in routing["tasks"].items() if str(value) in (key, route["task"])
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown task: {value}. Run 'python serve.py behavior list' to list tasks.")
    return matches[0]


def episode_seed(seed, task_id, instance_id):
    """Derive a stable per-rollout seed independent of worker assignment."""
    value = f"{seed}:{task_id}:{instance_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")
