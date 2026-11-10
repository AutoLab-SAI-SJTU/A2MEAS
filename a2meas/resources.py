"""Verified local resources and pinned Hugging Face downloads."""

import hashlib
from contextlib import contextmanager
import shutil
from filelock import FileLock

from a2meas import RUNTIME_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Missing or invalid resource: {path}")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Resource checksum mismatch: {path}")


def copy_verified(source, target, entry, *, refresh=False):
    if not refresh and (target.exists() or target.is_symlink()):
        verify_file(target, entry)
        return
    temporary = target.with_suffix(".partial")
    try:
        shutil.copyfile(source, temporary)
        verify_file(temporary, entry)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def downloaded_file(release, name, **auth):
    from huggingface_hub import hf_hub_download

    cache = RUNTIME_ROOT / "cache/downloads" / release["repo_id"] / release["revision"]
    source = cache / name
    source.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(source) + '.lock'):
        hf_hub_download(repo_id=release['repo_id'], revision=release['revision'], filename=name,
                        local_dir=cache, **auth)
        yield source
        source.unlink()


def download_files(destination, files, release, token):
    for name, entry in files.items():
        target = destination / name
        target.parent.resolve().relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() and not target.is_symlink():
            # Keep downloads from writing through links into shared resource trees.
            if not release.get("repo_id") or not release.get("revision"):
                raise RuntimeError("Public download release has not been pinned yet; install verified local resources")
            with downloaded_file(release, name, token=token) as source:
                copy_verified(source, target, entry)
        else:
            verify_file(target, entry)


def download_grounding(destination, entry):
    target = destination / "grounding/sam.pt"
    target.parent.resolve().relative_to(destination.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        with downloaded_file(entry, entry['filename']) as source:
            copy_verified(source, target, entry)
    else:
        verify_file(target, entry)
