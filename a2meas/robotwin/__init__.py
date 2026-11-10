"""A2WLA inference for RoboTwin."""

import hashlib

from a2meas import PROJECT_ROOT, RUNTIME_ROOT

_checkout = PROJECT_ROOT.stat()
_identity = f"{PROJECT_ROOT}:{_checkout.st_dev}:{_checkout.st_ino}"
SIMULATOR_ROOT = (RUNTIME_ROOT / "cache/robotwin" / hashlib.sha256(_identity.encode()).hexdigest()[:12]).resolve()
