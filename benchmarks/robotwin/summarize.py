#!/usr/bin/env python3
"""Summarize the latest complete RoboTwin result for every task/setting pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.robotwin.results import episode_seeds, valid_episode


SETTINGS = ("demo_clean", "demo_randomized")


def _read_result(path: Path, expected_seeds: int) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("partial"):
        return None
    config = data.get("config", {})
    if not isinstance(config, dict):
        return None
    params = config.get("params", {})
    if not isinstance(params, dict):
        return None
    task = params.get("task_name")
    setting = params.get("task_config")
    if not task or setting not in SETTINGS:
        return None
    if type(config.get("episodes_per_task")) is not int or config["episodes_per_task"] != expected_seeds:
        return None
    task_results = data.get("tasks", [])
    if not isinstance(task_results, list) or len(task_results) != 1:
        return None
    episodes: list[dict[str, Any]] = []
    for task_result in task_results:
        if (
            not isinstance(task_result, dict)
            or task_result.get("task") != task
            or not isinstance(task_result.get("episodes", []), list)
        ):
            return None
        task_episodes = task_result.get("episodes", [])
        if any(not isinstance(episode, dict) for episode in task_episodes):
            return None
        for episode in task_episodes:
            metrics = episode.get("metrics")
            episode_id = episode.get("episode_id")
            if (
                not isinstance(metrics, dict)
                or not isinstance(metrics.get("success"), bool)
                or isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
            ):
                return None
        episodes.extend(task_episodes)
    try:
        raw_reported_episodes = data.get("num_episodes", len(episodes))
        if isinstance(raw_reported_episodes, bool):
            raise ValueError
        reported_episodes = int(raw_reported_episodes)
    except (TypeError, ValueError):
        return None
    if reported_episodes != len(episodes):
        return None
    episode_errors = sum(bool(episode.get("failure_reason")) for episode in episodes)
    try:
        raw_declared_errors = data.get("num_errors", episode_errors)
        if isinstance(raw_declared_errors, bool):
            raise ValueError
        declared_errors = int(raw_declared_errors)
    except (TypeError, ValueError):
        declared_errors = max(1, episode_errors)
    if declared_errors < 0:
        declared_errors = max(1, episode_errors)
    errors = max(episode_errors, declared_errors)
    # If the aggregate says there were errors but individual episodes do not
    # identify them, exclude the entire incomplete job from score aggregation.
    valid_episodes = (
        [episode for episode in episodes if not episode.get("failure_reason")]
        if errors == episode_errors
        else []
    )
    try:
        seeds = episode_seeds(params, expected_seeds)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if any(not valid_episode(episode, expected_seeds, seeds) for episode in valid_episodes):
        return None
    valid_episode_ids = [episode["episode_id"] for episode in valid_episodes]
    if len(valid_episode_ids) != len(set(valid_episode_ids)):
        return None
    return {
        "complete": errors == 0 and set(valid_episode_ids) == set(range(expected_seeds)),
        "task": str(task),
        "setting": str(setting),
        "episodes": len(episodes),
        "valid_episodes": len(valid_episodes),
        "success": sum(
            episode["metrics"]["success"] is True for episode in valid_episodes
        ),
        "errors": errors,
        "path": path,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected-settings", default="demo_clean")
    args = parser.parse_args(argv)

    expected_settings = tuple(item.strip() for item in args.expected_settings.split(",") if item.strip())
    if args.expected_seeds <= 0:
        raise ValueError("--expected-seeds must be positive")
    if not expected_settings:
        raise ValueError("--expected-settings must contain at least one setting")
    if len(expected_settings) != len(set(expected_settings)):
        raise ValueError("--expected-settings must not contain duplicates")
    invalid_settings = sorted(set(expected_settings) - set(SETTINGS))
    if invalid_settings:
        raise ValueError(f"Unknown settings: {invalid_settings}")

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(args.results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime):
        result = _read_result(path, args.expected_seeds)
        if result is not None and result["setting"] in expected_settings:
            candidates.setdefault((result["task"], result["setting"]), []).append(result)

    expected_tasks_path = args.results_dir.parent / "task_names.txt"
    expected_tasks = (
        [line.strip() for line in expected_tasks_path.read_text().splitlines() if line.strip()]
    )
    if not expected_tasks:
        raise ValueError("No expected RoboTwin tasks were found")
    if len(expected_tasks) != len(set(expected_tasks)):
        raise ValueError(f"Duplicate tasks in {expected_tasks_path}")
    expected_task_set = set(expected_tasks)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for key, results in candidates.items():
        if key[0] not in expected_task_set:
            continue
        complete_results = [
            result
            for result in results
            if result["complete"]
        ]
        selected[key] = complete_results[-1] if complete_results else results[-1]

    expected_jobs = len(expected_tasks) * len(expected_settings)
    complete_jobs = sum(
        result["complete"]
        for result in selected.values()
    )
    total_episodes = sum(result["valid_episodes"] for result in selected.values())
    total_success = sum(result["success"] for result in selected.values())
    total_errors = sum(result["errors"] for result in selected.values())

    lines = [
        "# RoboTwin Evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Checkpoint: `{args.checkpoint or 'not specified'}`",
        f"- Protocol: {len(expected_tasks)} tasks × {','.join(expected_settings)} × {args.expected_seeds} episodes",
        f"- Completed jobs: {complete_jobs}/{expected_jobs}",
        f"- Episodes: {total_episodes}/{expected_jobs * args.expected_seeds}",
        (
            f"- Micro success: **{100 * total_success / total_episodes:.2f}%** "
            f"({total_success}/{total_episodes})"
            if total_episodes
            else "- Micro success: **N/A**"
        ),
        f"- Episode errors: {total_errors}",
        "",
        "| Task | "
        + " | ".join(setting.removeprefix("demo_").title() for setting in expected_settings)
        + " | Mean | Episodes | Errors |",
        "|---|" + "---:|" * (len(expected_settings) + 3),
    ]
    if complete_jobs != expected_jobs:
        lines[10:10] = [
            "> [!WARNING]",
            "> Partial evaluation: do not report this aggregate as a complete RoboTwin score.",
            "",
        ]

    for task in expected_tasks:
        rates: list[float | None] = []
        episodes = errors = 0
        for setting in expected_settings:
            result = selected.get((task, setting))
            if result is None or not result["episodes"]:
                rates.append(None)
                continue
            rates.append(
                result["success"] / result["valid_episodes"]
                if result["valid_episodes"]
                else None
            )
            episodes += result["valid_episodes"]
            errors += result["errors"]
        available = [rate for rate in rates if rate is not None]
        mean = sum(available) / len(available) if available else None
        formatted = ["—" if rate is None else f"{100 * rate:.1f}%" for rate in rates]
        formatted_mean = "—" if mean is None else f"{100 * mean:.1f}%"
        lines.append(f"| {task} | " + " | ".join(formatted) + f" | {formatted_mean} | {episodes} | {errors} |")

    lines.extend(["", "## Missing or incomplete jobs", ""])
    missing = []
    for task in expected_tasks:
        for setting in expected_settings:
            result = selected.get((task, setting))
            if result is None:
                missing.append(f"- `{task}/{setting}`: missing")
            elif result["errors"]:
                missing.append(
                    f"- `{task}/{setting}`: {result['errors']} infrastructure errors; rerun required"
                )
            elif result["episodes"] != args.expected_seeds:
                missing.append(f"- `{task}/{setting}`: {result['episodes']}/{args.expected_seeds} episodes")
    lines.extend(missing or ["None."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    return 0 if complete_jobs == expected_jobs else 1


if __name__ == "__main__":
    raise SystemExit(main())
