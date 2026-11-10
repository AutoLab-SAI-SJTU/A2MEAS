"""Optional correction for the configured BEHAVIOR tasks."""

import json
from .. import CONFIG_ROOT


def load_config():
    config = json.loads((CONFIG_ROOT / "adapter.json").read_text())
    config["grounding"] = json.loads((CONFIG_ROOT.parent / "shared/sam.json").read_text())
    return config
