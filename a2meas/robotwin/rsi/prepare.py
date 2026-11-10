"""Validate and freeze RSI configuration and perception weights."""
from __future__ import annotations

import hashlib
import importlib
from a2meas import CONFIG_ROOT, PROJECT_ROOT
from a2meas.resources import verify_file
import json
import os
from pathlib import Path

from a2meas.robotwin.rsi.config import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, config, selected_tasks):
    registry = load_config(config)
    config_bytes = (json.dumps(registry, indent=2) + "\n").encode()
    selected = set(filter(None, (task.strip() for task in selected_tasks.split(","))))
    active = {task: row for task, row in registry["tasks"].items()
              if registry.get("enabled", True) and row.get("enabled", True)
              and (not selected or task in selected)}
    visual = any(row["kind"] in {"placement", "ranking", "bottle", "mug", "proactive_grasp", "bread_release_retreat", "basket_inward_lift"} for row in active.values())
    package = Path(__file__).parent
    sources = list(package.glob("*.py"))
    contract = {"enabled": True, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "source_sha256": {str(path.relative_to(package.parent)): digest(path)
                                  for path in sorted(sources)}, "vision": None}
    if visual:
        import sam3
        source = Path(sam3.__file__).resolve().parent
        weights = Path(os.environ["RSI_SAM3_WEIGHTS"]).resolve(strict=True)
        expected = json.loads((CONFIG_ROOT / "shared/sam.json").read_text())
        verify_file(weights, expected)
        for name in ("sam3.model_builder", "sam3.model.vl_combiner",
                     "sam3.model.sam3_multiplex_detector", "sam3.model.sam3_image_processor"):
            importlib.import_module(name)
        contract["vision"] = {
            "source": os.path.relpath(source, PROJECT_ROOT),
            "source_sha256": {str(path.relative_to(source)): digest(path)
                              for path in sorted(source.rglob("*.py"))},
            "weights": {"path": os.path.relpath(weights, PROJECT_ROOT), "size": weights.stat().st_size,
                        "sha256": expected["sha256"]},
        }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"rsi_config.json": config_bytes,
                 "rsi_manifest.json": (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()}
    for name, content in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"RSI inputs changed: {path}; use a new OUTPUT_ROOT")
        if not path.exists():
            tmp = path.with_suffix(".json.tmp")
            try:
                tmp.write_bytes(content)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
