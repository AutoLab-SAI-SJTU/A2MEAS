"""Download pinned resources and restore verified checkpoints before inference."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from zipfile import ZipFile

from filelock import FileLock

from a2meas import RUNTIME_ROOT
from a2meas.behavior import CONFIG_ROOT, load_routes
from a2meas.resources import verify_file, download_files, download_grounding, copy_verified, sha256, downloaded_file



def verify_shared(resources):
    files = json.loads((CONFIG_ROOT / "resources.json").read_text())["files"]
    for name, entry in files.items():
        if name.startswith("assets/rynnbrain/"):
            target = Path(resources) / name
            target.parent.resolve().relative_to(Path(resources).resolve())
            verify_file(target, entry)


def prepare(destination, *, adapter=False):
    manifest = json.loads((CONFIG_ROOT / "resources.json").read_text())
    release = json.loads((CONFIG_ROOT / "release.json").read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".prepare.lock")):
        files = manifest["files"]
        assets = {name: entry for name, entry in files.items() if name.startswith("assets/")}
        download_files(destination, assets, release, None)
        sidecars = {sha256(path): path for path in (CONFIG_ROOT / "policy").iterdir() if path.is_file()}
        missing = []
        for name, entry in files.items():
            if not name.startswith("behavior/policies/"):
                continue
            target = destination / name
            target.parent.resolve().relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("/model.safetensors"):
                if target.exists() or target.is_symlink():
                    verify_file(target, entry)
                else:
                    missing.append(target.parent.name)
            else:
                copy_verified(sidecars[entry["sha256"]], target, entry)
        if missing:
            from .weights import restore_models

            cache = RUNTIME_ROOT / "cache/behavior-weights" / release['revision']
            cache.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(RUNTIME_ROOT / "cache/behavior-weights.lock")):
                cache.mkdir(exist_ok=True)
                restore_models(destination, cache, release, files, missing)
                shutil.rmtree(cache)
        if adapter:
            from .correction import load_config

            config = load_config()
            download_files(destination, {"adapter/adapter.safetensors": config["weights"]}, release, None)
            download_grounding(destination, config["grounding"])
    return len(manifest["files"]) + (2 if adapter else 0)


def checkpoint_path(resources, model_name):
    resources = Path(resources)
    prefix = f"behavior/policies/{model_name}/"
    manifest = json.loads((CONFIG_ROOT / "resources.json").read_text())["files"]
    directory = resources / prefix
    directory.resolve().relative_to(resources.resolve())
    entries = {name.removeprefix(prefix): entry for name, entry in manifest.items() if name.startswith(prefix)}
    if "model.safetensors" not in entries:
        raise KeyError(f"Unknown released model: {model_name}")
    for name, entry in entries.items():
        verify_file(directory / name, entry)
    return directory / "model.safetensors"


def prepare_judge(destination):
    root = CONFIG_ROOT
    release = json.loads((root / "judge.json").read_text())
    destination.resolve().relative_to(destination.parent.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".prepare.lock")):
        for line in (root / "judge.sha256").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            expected, name = line.split(maxsplit=1)
            target = destination / name
            if not target.exists() and not target.is_symlink():
                with downloaded_file(release, name) as source:
                    copy_verified(source, target, {'bytes': source.stat().st_size, 'sha256': expected})
            elif target.is_symlink() or sha256(target) != expected:
                raise ValueError(f"Judge checkpoint checksum mismatch: {name}")


def prepare_dataset(destination):
    """Install pinned official assets atomically, then retain the evaluation subset."""
    from huggingface_hub import hf_hub_download
    from omnigibson.utils.asset_utils import download_key, print_user_agreement

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / 'omnigibson.key').is_file():
        print_user_agreement()
        if input('Accept the BEHAVIOR dataset terms? [y/n] ').lower() != 'y':
            raise RuntimeError('BEHAVIOR dataset terms were not accepted')
        download_key()
    release = json.loads((CONFIG_ROOT / 'datasets.json').read_text())
    cache = RUNTIME_ROOT / 'cache/behavior-assets'
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / 'prepare.lock')):
        for name, entry in release['files'].items():
            target = destination / entry['directory']
            target.resolve().relative_to(destination.resolve())
            marker = target / 'installation.json'
            record = json.dumps(dict(revision=release['revision'], sha256=entry['sha256'])) + '\n'
            staging = destination / f"{entry['directory']}.partial"
            if marker.is_file() and marker.read_text() == record:
                if staging.exists():
                    shutil.rmtree(staging)
                continue
            archive = cache / 'downloads' / name
            ready = staging / 'installation.json'
            if not ready.is_file() or ready.read_text() != record:
                if staging.exists():
                    shutil.rmtree(staging)
                hf_hub_download(repo_id=release['repo_id'], repo_type='dataset',
                    revision=release['revision'], filename=name, local_dir=cache / 'downloads')
                verify_file(archive, entry)
                staging.mkdir()
                with ZipFile(archive) as bundle:
                    bundle.extractall(staging)
                ready.write_text(record)
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
            archive.unlink(missing_ok=True)
        prune_dataset(destination)


def prune_dataset(destination):
    """Keep task templates and reachable object models without changing model sampling."""
    import yaml
    from bddl.object_taxonomy import ObjectTaxonomy
    from bddl.transition_rules import load_cooking_recipes, load_machine_recipes, load_mixing_recipes

    destination = Path(destination)
    challenge = destination / '2026-challenge-task-instances'
    available = yaml.safe_load((challenge / 'metadata/available_tasks.yaml').read_text())
    tasks = {route['task']: available[route['task']][0]['scene_model'] for route in load_routes()['tasks'].values()}
    assets = destination / 'behavior-1k-assets'
    required = set()
    for task, scene in tasks.items():
        for split in (challenge / 'scenes', challenge / 'scene_test/public', challenge / 'scene_test/private'):
            path = split / scene / 'json' / f'{scene}_task_{task}_0_0_template.json'
            if split != challenge / 'scenes' and not path.exists():
                continue
            data = json.loads(path.read_text())
            for obj in data['objects_info']['init_info'].values():
                if obj['class_name'] == 'DatasetObject':
                    args = obj['args']
                    required.add((args['category'], args['model']))
    taxonomy = ObjectTaxonomy()
    recipes = load_cooking_recipes() + load_machine_recipes() + load_mixing_recipes()
    transitions = [(set().union(*(set(taxonomy.get_categories(s)) for s in recipe.input_synsets)),
                    set().union(*(set(taxonomy.get_categories(s)) for s in recipe.output_synsets)))
                   for recipe in recipes]
    checked = set()
    while True:
        previous = set(required)
        categories = {category for category, _ in required}
        for inputs, outputs in transitions:
            if inputs <= categories:
                for category in outputs:
                    required.update((category, model.name) for model in (assets / 'objects' / category).glob('*')
                                    if model.is_dir())
        for category, model in required - checked:
            directory = assets / 'objects' / category / model
            if not (directory / 'usd' / f'{model}.encrypted.usd').is_file():
                raise FileNotFoundError(f'Missing required object model: {category}/{model}')
            metadata = json.loads((directory / 'misc/metadata.json').read_text())
            required.update((part['category'], part['model']) for part in metadata['object_parts'])
            checked.add((category, model))
        if required == previous:
            break
    unused = []
    for category in (assets / 'objects').iterdir():
        if category.is_dir():
            for model in category.iterdir():
                if model.is_dir() and (category.name, model.name) not in required:
                    unused.append(model)
    scenes = set(tasks.values())
    for scene in (assets / 'scenes').iterdir():
        if scene.is_dir() and scene.name not in scenes:
            unused.append(scene)
    unused.extend((challenge / 'scenes').glob('*/json/*_instances'))
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(shutil.rmtree, unused))
    for category in (assets / 'objects').iterdir():
        if category.is_dir() and not any(category.iterdir()):
            category.rmdir()
    robots = destination / 'omnigibson-robot-assets'
    for name in ('source', 'test'):
        if (robots / name).exists():
            shutil.rmtree(robots / name)
    for model in (robots / 'models').iterdir():
        if model.name not in {'r1pro', 'background'}:
            shutil.rmtree(model)
    for path in robots.rglob('*'):
        if path.name in {'.DS_Store', '.project', '.cproject'}:
            path.unlink()
    for path in robots.rglob('.thumbs'):
        shutil.rmtree(path)
