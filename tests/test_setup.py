"""Exercise the public setup entry point without installing packages or models."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_checkout(tmp_path):
    checkout = tmp_path / 'checkout with spaces'
    checkout.mkdir()
    shutil.copyfile(ROOT / 'setup.sh', checkout / 'setup.sh')
    runtime = tmp_path / 'runtime files'
    log = tmp_path / 'commands.jsonl'
    # Replace only environment executables. Use real shell parsing, checksums
    # and zip extraction to test the download/extraction trust boundary.
    executable = f'''#!{sys.executable}
import json, os, sys, zipfile
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['SETUP_TEST_LOG']).open('a') as stream:
    stream.write(json.dumps(dict(executable=sys.argv[0], args=args, hf_home=os.environ.get('HF_HOME'))) + '\\n')
if args[:3] == ['-m', 'zipfile', '-e']:
    with zipfile.ZipFile(args[3]) as archive:
        archive.extractall(args[4])
if args[:3] == ['serve.py', 'robotwin', 'prepare']:
    Path('resources/robotwin/simulator').mkdir(parents=True, exist_ok=True)
if args == ['-']:
    sys.stdin.read()
'''
    for name in ('model/bin/python', 'model/bin/hf', 'simulator/bin/python', 'behavior/bin/python', 'judge/bin/python'):
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(executable)
        path.chmod(0o755)
        (path.parent.parent / 'pyvenv.cfg').touch()
    archive_root = runtime / 'cache/robotwin-assets'
    archive_root.mkdir(parents=True)
    checksums = []
    for name in ('background_texture', 'embodiments', 'objects'):
        path = archive_root / f'{name}.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'{name}/asset.txt', name)
        checksums.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  runtime/cache/robotwin-assets/{path.name}')
    config = checkout / 'configs/robotwin'
    config.mkdir(parents=True)
    (config / 'assets.sha256').write_text('\n'.join(checksums) + '\n')
    env = dict(os.environ, A2MEAS_RUNTIME=str(runtime), SETUP_TEST_LOG=str(log), HF_HOME=str(tmp_path / 'hf'))
    return checkout, runtime, log, env


@pytest.mark.parametrize('corrupt', [False, True])
def test_robotwin_verifies_archives_before_extracting(setup_checkout, corrupt):
    checkout, runtime, log, env = setup_checkout
    if corrupt:
        (runtime / 'cache/robotwin-assets/objects.zip').write_bytes(b'corrupt')
    result = subprocess.run(['bash', str(checkout / 'setup.sh'), '--download-only', '--policy-only'],
                            env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]['args'] == ['serve.py', 'robotwin', 'prepare']
    assert all(call['hf_home'] == env['HF_HOME'] for call in calls)
    assets = checkout / 'resources/robotwin/simulator/assets'
    assert (checkout / 'resources').is_dir() and not (checkout / 'resources').is_symlink()
    assert assets.parent.is_dir()
    if corrupt:
        assert result.returncode != 0
        assert not assets.exists()
        assert not any('benchmarks.robotwin.prepare' in call['args'] for call in calls)
    else:
        assert result.returncode == 0, result.stderr
        assert (assets / 'objects/asset.txt').read_text() == 'objects'
        assert calls[-1]['args'] == ['-m', 'benchmarks.robotwin.prepare', '--robotwin-root',
                                    'resources/robotwin/simulator']


def test_all_downloads_include_sam_adapter_judge_and_behavior_assets(setup_checkout):
    checkout, runtime, log, env = setup_checkout
    result = subprocess.run(['bash', str(checkout / 'setup.sh'), '--benchmark', 'all', '--adapter', '--download-only'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line)['args'] for line in log.read_text().splitlines()]
    assert not (checkout / 'resources').is_symlink()
    assert ['serve.py', 'robotwin', 'prepare', '--sam'] in calls
    assert ['serve.py', 'behavior', 'prepare', '--adapter'] in calls
    assert ['-'] in calls  # Judge preparation uses the model environment.
    assert calls[-1][0] == '-c'  # Keep stdin available for the dataset license prompt.
    assert not any('pip' in call for call in calls)
    log.unlink()
    (runtime / 'judge/bin/python').unlink()
    result = subprocess.run(['bash', str(checkout / 'setup.sh'), '--benchmark', 'behavior', '--adapter', '--download-only'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Judge environment missing' in result.stderr
    assert not log.exists()


@pytest.mark.parametrize('args,code', [(['--help'], 0), (['--benchmark'], 2),
    (['--benchmark', 'unknown'], 2), (['--adapter'], 2),
    (['--skip-downloads', '--download-only'], 2)])
def test_setup_options_have_no_side_effects(tmp_path, args, code):
    script = tmp_path / 'setup.sh'
    shutil.copyfile(ROOT / 'setup.sh', script)
    result = subprocess.run(['bash', str(script), *args], capture_output=True, text=True)
    assert result.returncode == code
    assert sorted(path.name for path in tmp_path.iterdir()) == ['setup.sh']


@pytest.mark.parametrize('benchmark,adapter,policy_only,expected', [
    ('robotwin', False, False, {'model', 'simulator'}),
    ('robotwin', False, True, {'model', 'simulator'}),
    ('behavior', False, False, {'model', 'behavior'}),
    ('all', True, False, {'model', 'simulator', 'behavior', 'judge'}),
])
def test_install_selects_isolated_environments_without_resource_downloads(setup_checkout, benchmark, adapter, policy_only, expected):
    checkout, runtime, log, env = setup_checkout
    binaries = checkout / 'bin'
    binaries.mkdir()
    shim = (runtime / 'model/bin/python').read_text()
    shim += """
if args[:2] == ['-m', 'venv']:
    target = Path(args[2]) / 'bin'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'python').write_text(Path(__file__).read_text())
    (target / 'python').chmod(0o755)
"""
    for name in ('python3.10', 'python3.11'):
        path = binaries / name
        path.write_text(shim)
        path.chmod(0o755)
    oidn = checkout / 'oidn.tar.gz'
    with tarfile.open(oidn, 'w:gz') as archive:
        data = b'library fixture'
        entry = tarfile.TarInfo('oidn/lib/libOpenImageDenoise.so')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    (checkout / 'configs/robotwin/oidn.sha256').write_text(
        hashlib.sha256(oidn.read_bytes()).hexdigest() + '  oidn.tar.gz\n')
    for name, body in {
        'conda': ("target = Path(args[args.index('--prefix') + 1]) / 'bin'\n"
                  "target.mkdir(parents=True, exist_ok=True)\n"
                  f"(target / 'python').write_text({shim!r})\n"
                  "(target / 'python').chmod(0o755)"),
        'curl': f"import shutil; shutil.copyfile({str(oidn)!r}, args[args.index('--output') + 1])",
        'git': "if args[0] == 'init':\n    sdk = Path(args[1]); sdk.mkdir(exist_ok=True); (sdk / 'setup.sh').write_text('exit 0\\n')",
    }.items():
        path = binaries / name
        path.write_text(f'#!{sys.executable}\nimport sys\nfrom pathlib import Path\nargs=sys.argv[1:]\n' + body + '\n')
        path.chmod(0o755)
    args = ['bash', str(checkout / 'setup.sh'), '--benchmark', benchmark, '--skip-downloads']
    if adapter:
        args.append('--adapter')
    if policy_only:
        args.append('--policy-only')
    result = subprocess.run(args, env=dict(env, PATH=f"{binaries}:{env['PATH']}"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert (checkout / 'runtime').resolve() == runtime
    assert not Path(os.readlink(checkout / 'runtime')).is_absolute()
    created = {Path(call['args'][2]).name for call in calls if call['args'][:2] == ['-m', 'venv']}
    assert created == expected
    requirements = [call['args'][call['args'].index('-r') + 1] for call in calls
                    if Path(call['executable']) == runtime / 'model/bin/python' and '-r' in call['args']]
    assert requirements == ['requirements/grounding.txt' if adapter or (benchmark != 'behavior' and not policy_only)
                            else 'requirements/model.txt']
    assert not any('serve.py' in call['args'] or 'download' in call['args'] for call in calls)
    if 'simulator' in expected:
        assert (runtime / 'simulator/oidn/lib/libOpenImageDenoise.so').read_bytes() == b'library fixture'
