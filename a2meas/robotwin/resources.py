"""Prepare the RoboTwin policy, reference SDK, episode lists and shared assets."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from filelock import FileLock
from a2meas import CONFIG_ROOT, PROJECT_ROOT
from a2meas.resources import download_files, download_grounding, copy_verified


def prepare(destination, *, sam=False):
    destination = Path(destination)
    manifest = json.loads((CONFIG_ROOT / "robotwin/resources.json").read_text())["files"]
    files = json.loads((CONFIG_ROOT / "robotwin/downloads.json").read_text())["files"]
    release = json.loads((CONFIG_ROOT / "robotwin/release.json").read_text())
    episodes = json.loads((CONFIG_ROOT / "robotwin/episodes.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".prepare.lock")), TemporaryDirectory(prefix="robotwin-episodes-") as temporary:
        for name, entry in manifest.items():
            if name.startswith("policy/"):
                continue
            target = destination / "robotwin" / name
            target.resolve().relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            source = (PROJECT_ROOT / "third_party/robotwin" / name.removeprefix("simulator/")
                      if name.startswith("simulator/") else PROJECT_ROOT / "licenses/robotwin_episodes" / Path(name).name)
            if name.startswith("expert_cache/") and name.endswith(".json"):
                source = Path(temporary) / Path(name).name
                source.write_text(json.dumps(episodes[source.name], indent=2) + "\n")
            copy_verified(source, target, entry, refresh=name.startswith("simulator/"))
        download_files(destination, files, release, None)
        if sam:
            entry = json.loads((CONFIG_ROOT / "shared/sam.json").read_text())
            download_grounding(destination, entry)
    return len(manifest) + sum(name.startswith("assets/") for name in files) + int(sam)
