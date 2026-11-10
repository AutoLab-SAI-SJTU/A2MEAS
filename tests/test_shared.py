import json

import pytest


@pytest.mark.parametrize('explicit_home', [False, True])
def test_runtime_preserves_huggingface_login_credentials(tmp_path, explicit_home):
    import os
    import subprocess
    import sys

    env = {key: value for key, value in os.environ.items()
           if key not in ('HF_HOME', 'HF_TOKEN_PATH', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')}
    env['XDG_CACHE_HOME'] = str(tmp_path / 'cache')
    home = tmp_path / 'huggingface' if explicit_home else tmp_path / 'cache/huggingface'
    if explicit_home:
        env['HF_HOME'] = str(home)
    home.mkdir(parents=True)
    (home / 'token').write_text('local-test-credential')
    subprocess.run([sys.executable, '-c', '''
import os, sys
from pathlib import Path
from huggingface_hub import get_token
from huggingface_hub.constants import HF_HOME
from a2meas import runtime_environment
assert get_token() == 'local-test-credential'
with runtime_environment():
    assert os.environ['HF_HOME'] == HF_HOME
    assert get_token() == 'local-test-credential'
    assert os.environ['PATH'].split(os.pathsep)[0] == str(Path(sys.executable).absolute().parent)
'''], env=env, check=True)


def test_benchmarks_keep_independent_checkpoint_registries():
    from a2meas.behavior.model.config import A2WLAConfig as BehaviorConfig
    from a2meas.robotwin.model.config import A2WLAConfig as RoboTwinConfig
    from a2meas.behavior.schema import PreTrainedConfig as BehaviorRegistry
    from a2meas.robotwin.schema import PreTrainedConfig as RoboTwinRegistry

    assert BehaviorRegistry.get_choice_class("pi05") is BehaviorConfig
    assert RoboTwinRegistry.get_choice_class("pi05") is RoboTwinConfig


def test_download_refuses_missing_files_under_external_shared_directory(tmp_path, monkeypatch):
    import huggingface_hub
    from a2meas.resources import download_files

    destination, shared = tmp_path / 'resources', tmp_path / 'shared'
    destination.mkdir()
    shared.mkdir()
    (destination / 'simulator').symlink_to(shared, target_is_directory=True)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kwargs: pytest.fail('Shared path reached downloader'))
    with pytest.raises(ValueError, match='subpath'):
        download_files(destination, {'simulator/NOTICE': {}}, {}, None)
    assert not list(shared.iterdir())


@pytest.mark.parametrize("benchmark", ["behavior", "robotwin"])
def test_serving_entry_dispatches_benchmark_arguments(monkeypatch, benchmark):
    import importlib
    import serve

    implementation = importlib.import_module(f"a2meas.{benchmark}.cli")
    calls = []
    monkeypatch.setattr(implementation, "main", lambda argv: calls.append(argv))
    serve.main([benchmark, "serve", "--help"])
    assert calls == [["serve", "--help"]]


def test_evaluator_sigterm_stops_simulator_process_group(tmp_path):
    from contextlib import suppress
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    ready = tmp_path / "simulator-ready.json"
    child = """import json,os,signal,subprocess,sys
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
path = Path(sys.argv[1])
path.with_suffix('.tmp').write_text(json.dumps(dict(pid=os.getpid(), child=child.pid, temporary=os.environ['TMPDIR'])))
path.with_suffix('.tmp').replace(path)
signal.pause()
"""
    parent = """import os,sys
from pathlib import Path
import a2meas
from benchmarks.runtime import running
a2meas.PROJECT_ROOT = Path(sys.argv[1])
with a2meas.runtime_environment():
    with running([sys.executable, "-c", sys.argv[2], sys.argv[3]], dict(os.environ)) as process:
        process.wait()
"""
    with subprocess.Popen(
        [sys.executable, "-c", parent, str(tmp_path), child, str(ready)],
        env=dict(os.environ, TMPDIR=str(tmp_path)),
    ) as process:
        data = None
        try:
            deadline = time.monotonic() + 5
            while not ready.exists():
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            data = json.loads(ready.read_text())
            process.terminate()
            assert process.wait(timeout=5) == 143
            assert not Path(data["temporary"]).exists()
            for pid in (data["pid"], data["child"]):
                state = Path(f"/proc/{pid}/stat")
                assert not state.exists() or state.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
        finally:
            process.kill()
            if data is not None:
                with suppress(ProcessLookupError):
                    os.killpg(data["pid"], signal.SIGKILL)


@pytest.mark.parametrize(
    "mode,expected_suffix",
    [
        ("causal", [[True, False], [True, True]]),
        ("block", [[True, True], [True, True]]),
    ],
)
def test_shared_prefix_and_suffix_masks_preserve_padding(mode, expected_suffix):
    import torch
    from a2meas.models.action_expert import build_suffix_cross_attn_mask_2d

    mask = build_suffix_cross_attn_mask_2d(
        suffix_pad_masks=torch.tensor([[True, True]]),
        suffix_att_masks=torch.tensor([[True, True]]),
        prefix_pad_masks=torch.tensor([[True, False]]),
        suffix_attention_mode=mode,
    )
    expected = torch.tensor(
        [[[True, False, *expected_suffix[0]], [True, False, *expected_suffix[1]]]]
    )
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)


@pytest.mark.parametrize("budget", ["per_view", "shared_total"])
def test_lawam_missing_views_remain_finite_and_masked(budget):
    import torch
    from a2meas.models.world_conditioning import (
        LaWAMPrefixAdapter,
        encode_lawam_multi_view_visual_cond,
    )
    from a2meas.models.action_expert import append_prefix_tokens

    adapter = LaWAMPrefixAdapter(vision_dim=16, output_dim=16, num_tokens=2, num_heads=2)
    features = torch.zeros(2, 3, 4, 16)
    valid = torch.tensor([[False, False, False], [True, False, True]])
    tokens, mask, _ = encode_lawam_multi_view_visual_cond(
        adapter,
        h_t=features,
        uhat=features,
        prefix_mode="compressed",
        inject_current=True,
        camera_embedding=torch.zeros(3, 16),
        current_valid=valid,
        future_valid=valid,
        device=features.device,
        dtype=features.dtype,
        token_budget_mode=budget,
        vision_camera_embedding=torch.zeros(3, 16),
    )
    assert torch.isfinite(tokens).all() and not mask[0].any()
    expected = (
        torch.ones(4, dtype=torch.bool)
        if budget == "shared_total"
        else valid[1].repeat_interleave(2).repeat(2)
    )
    torch.testing.assert_close(mask[1], expected)
    _, padded, attention = append_prefix_tokens(
        torch.zeros(2, 1, 16),
        torch.ones(2, 1, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.bool),
        tokens,
        token_pad_mask=mask,
    )
    torch.testing.assert_close(padded[:, 1:], mask)
    assert padded[:, 0].all() and not attention.any()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("query_count", [None, 0, 3])
def test_shared_query_injection_preserves_prefix_padding_and_rng(dtype_name, query_count):
    import torch
    from a2meas.models.action_expert import build_inject_sequence
    from a2meas.behavior.model.inputs import build_inject_sequence as behavior
    from a2meas.robotwin.model.prefix import build_inject_sequence as robotwin

    assert behavior is robotwin is build_inject_sequence
    prefix = torch.arange(64).reshape(2, 4, 8).to(getattr(torch, dtype_name))
    mask = torch.tensor([[True, False, True, True], [False, True, True, True]])
    query = None if query_count is None else torch.arange(query_count * 8).reshape(query_count, 8).float()
    rng = torch.get_rng_state().clone()
    embeddings, padding, length, added = build_inject_sequence(prefix, mask, la_query=query)
    torch.testing.assert_close(embeddings[:, :4], prefix, rtol=0, atol=0)
    torch.testing.assert_close(padding[:, :4], mask, rtol=0, atol=0)
    assert length == 4 + (query_count or 0) and added == (query_count or 0)
    assert torch.equal(torch.get_rng_state(), rng)
    if query is None:
        assert embeddings is prefix and padding is mask
    else:
        torch.testing.assert_close(
            embeddings[:, 4:], query.to(prefix.dtype).expand(2, -1, -1), rtol=0, atol=0
        )
        assert padding[:, 4:].all()


@pytest.mark.parametrize('returncode', [0, 143])
def test_evaluation_entry_preserves_robotwin_exit_status(monkeypatch, returncode):
    import runpy
    import sys
    from benchmarks.robotwin import run

    monkeypatch.setattr(run, 'main', lambda argv=None: returncode)
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', 'robotwin'])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(run.PROJECT_ROOT / 'evaluate.py'), run_name='__main__')
    assert error.value.code == returncode
