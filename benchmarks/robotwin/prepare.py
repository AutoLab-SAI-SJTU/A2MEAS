"""Create a writable RoboTwin view without changing the simulator/assets source."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from filelock import FileLock
from a2meas import PROJECT_ROOT
from a2meas.robotwin import SIMULATOR_ROOT


def prepare(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Simulator source and runtime must be disjoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / ".robotwin_runtime"
        if not marker.exists() and any(destination.iterdir()):
            raise ValueError(f"Refusing a nonempty directory without runtime marker: {destination}")
        marker.touch()

        def link(src, dst):
            target = os.path.relpath(src, dst.parent)
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise FileExistsError(dst)
                if os.readlink(dst) == target:
                    return
                temporary = dst.with_name(dst.name + f".tmp.{os.getpid()}")
                try:
                    temporary.symlink_to(target, target_is_directory=src.is_dir())
                    temporary.replace(dst)
                finally:
                    temporary.unlink(missing_ok=True)
            elif dst.exists():
                raise FileExistsError(dst)
            else:
                dst.symlink_to(target, target_is_directory=src.is_dir())

        def rebase(value):
            if isinstance(value, str) and Path(value).is_absolute() and "/assets/" in value:
                return str(Path("assets") / value.split("/assets/", 1)[1])
            if isinstance(value, dict):
                return {key: rebase(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rebase(item) for item in value]
            return value

        for path in source.iterdir():
            if path.name not in {"assets", "results", "__pycache__", ".git"}:
                link(path, destination / path.name)
        (destination / "assets").mkdir(exist_ok=True)
        for path in (source / "assets").iterdir():
            if path.name != "embodiments":
                link(path, destination / "assets" / path.name)
        embodiments = source / "assets/embodiments"
        for path in embodiments.rglob("*"):
            target = destination / "assets/embodiments" / path.relative_to(embodiments)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.suffix in {".yaml", ".yml"}:
                content = yaml.safe_dump(rebase(yaml.safe_load(path.read_text())), sort_keys=False)
                if target.exists() and target.read_text() == content:
                    continue
                temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
                try:
                    temporary.write_text(content)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                link(path, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("ROBOTWIN_ROOT", SIMULATOR_ROOT)))
    args = parser.parse_args()
    prepare(PROJECT_ROOT / args.robotwin_root, PROJECT_ROOT / args.output_root)
