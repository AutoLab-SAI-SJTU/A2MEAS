"""Collect native harness shards and publish complete RoboTwin results.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from a2meas import PROJECT_ROOT


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def episode_seeds(params, count):
    cache = json.loads((PROJECT_ROOT / params["expert_task_cache_path"]).read_text())
    tasks = cache["tasks"]
    if not isinstance(tasks, list) or len(tasks) < count:
        raise ValueError("Episode cache does not cover the requested run")
    seeds = [item["seed"] for item in tasks[:count]]
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Episode cache contains an invalid seed")
    return seeds


def valid_episode(item, count, seeds):
    if not isinstance(item, dict) or item.get("failure_reason"):
        return False
    idx, metrics = item.get("episode_id"), item.get("metrics", {})
    return (type(idx) is int and 0 <= idx < count
            and isinstance(metrics, dict) and type(metrics.get("success")) is bool
            and type(metrics.get("episode_index")) is int and metrics["episode_index"] == idx
            and type(metrics.get("seed")) is int and metrics["seed"] == seeds[idx])


def collect(config_path, results_dir, shards_dir):
    config = yaml.safe_load(config_path.read_text())["benchmarks"][0]
    job, params = config["subname"], config["params"]
    count = config["episodes_per_task"]
    seeds = episode_seeds(params, count)
    episodes, template = {}, None
    paths = sorted(shards_dir.glob(f"RoboTwinBenchmark_{job}_shard*of*.json"))
    for path in paths:
        data = read(path)
        if not isinstance(data, dict):
            continue
        saved = data.get("config", {})
        if (not isinstance(saved, dict) or saved.get("subname") != job or saved.get("params") != params
                or saved.get("episodes_per_task") != count
                or saved.get("max_steps") != config["max_steps"]):
            continue
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict) or tasks[0].get("task") != params["task_name"]:
            continue
        items = tasks[0].get("episodes", [])
        if not isinstance(items, list):
            continue
        failed = sum(bool(e.get("failure_reason")) for e in items if isinstance(e, dict))
        if data.get("num_errors", failed) != failed:
            continue
        for item in items:
            if isinstance(item, dict) and item.get('failure_reason') == 'timeout':
                idx, metrics = item.get('episode_id'), item.get('metrics')
                if (type(idx) is not int or not 0 <= idx < count or not isinstance(metrics, dict)
                        or set(metrics) != {'success'} or metrics['success'] is not False):
                    continue
                item = {**item, 'termination_reason': 'policy_query_timeout',
                        'metrics': {'success': False, 'episode_index': idx, 'seed': seeds[idx]}}
                item.pop('failure_reason')
            if not valid_episode(item, count, seeds):
                continue
            idx = item["episode_id"]
            if idx not in episodes:
                episodes[idx] = item
                template = data
    return config, episodes, template


def publish(config, episodes, template, results_dir):
    count = config["episodes_per_task"]
    if len(episodes) != count:
        return False
    from vla_eval.results.merge import merge_shards

    # Aggregate only verified global episode IDs.
    shards = []
    for idx in range(count):
        part = {**template, "shard": {"id": idx, "total": count}}
        part.pop("partial", None)
        part["tasks"] = [{"task": config["params"]["task_name"], "episodes": [episodes[idx]]}]
        shards.append(part)
    merged = merge_shards(shards)
    merged["num_episodes"], merged["num_errors"] = count, 0
    path = results_dir / f"RoboTwinBenchmark_{config['subname']}_sync_merged.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n")
    tmp.replace(path)
    return True
