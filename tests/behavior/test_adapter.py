from a2meas.behavior import CONFIG_ROOT
"""Correction boundaries, flow reuse and native action execution."""

from concurrent.futures import Future
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from a2meas.behavior import load_routes
from a2meas.behavior.correction import load_config
from a2meas.behavior.correction.actions import action_summary, guarded_actions
from a2meas.behavior.correction.controller import CorrectionController
from a2meas.behavior.correction.gate import ProgressGate
from a2meas.behavior.correction.review import NeedSelector, validate_review
from a2meas.behavior.correction.review_worker import ReviewWorker, screening_negative
from a2meas.behavior.model.adapter import HiddenAdapter, condition_arrays
from a2meas.models.world_conditioning import LaWAMKVBank
from a2meas.behavior.server import RouteManager
from tests.behavior.tiny_model import make_model, inputs


def decision(**changes):
    return dict(
        skill_id="press",
        local_manipulation=True,
        need_correction=True,
        execution_status="stalled",
        progress="no_progress",
        confidence=0.95,
        need_confidence=0.9,
        same_target_as_previous=True,
        holding_left="stable",
        holding_right="empty",
        evidence="The finger misses the radio button.",
        target="radio button",
        rearm_reason="no_block",
        rearm_confidence=0.9,
        failure_type="control_contact_failure",
        recovery_stage="align",
        failure_confidence=0.9,
        **changes,
    )


def test_adapter_scope_and_base_routes(tmp_path):
    config = load_config()
    assert set(map(int, config["tasks"])) == {0, 1, 6, 17, 18, 22, 30, 32, 34, 35, 40, 45}
    assert config["num_inference_steps"] == 2 and config["execution_horizon"] == 16
    assert config["model"] == "model_20"
    assert f"behavior/policies/{config['model']}/model.safetensors" in json.loads(
        (CONFIG_ROOT / "resources.json").read_text()
    )["files"]
    manager = RouteManager(tmp_path, {0}, adapter=True)
    assert set(map(int, manager.task_names)) == set(range(100))
    assert manager.process is None


@pytest.mark.parametrize("steps", [None, 2, 10])
def test_adapter_switch_changes_only_the_twelve_configured_tasks(tmp_path, monkeypatch, steps):
    import a2meas.behavior.policy as implementation
    import a2meas.behavior.correction.controller as correction

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    loaded = []
    corrections = []

    def service(args):
        loaded.append(args)
        config = SimpleNamespace(
            image_resolution=(224, 224), short_term_memory_frame_age_embedding_enable=False,
            short_term_memory_frame_age_fps=30, enable_speed_meta=False,
            enable_gripper_first_close_meta=False,
        )
        model = SimpleNamespace(
            _lawam_alternate_kv_enabled=lambda: True,
            paligemma_with_expert=SimpleNamespace(
                gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=[None] * 18))
            ),
        )
        return SimpleNamespace(policy=SimpleNamespace(config=config, model=model))

    def controller(service, resources, config, task):
        corrections.append(task["id"])
        return SimpleNamespace(reset=lambda: None, close=lambda: None,
                               configure_task=lambda item: corrections.append(item["id"]))

    monkeypatch.setattr(implementation, "ModelService", service)
    monkeypatch.setattr(implementation, "checkpoint_path", lambda resources, name: resources / name / "model.safetensors")
    monkeypatch.setattr(correction, "CorrectionController", controller)
    routes = load_routes()
    adapter = load_config()
    for enabled in (False, True):
        for task_id, route in routes["tasks"].items():
            policy = implementation.VLAPolicy(route, tmp_path, adapter=enabled, num_inference_steps=steps)
            active = enabled and task_id in adapter["tasks"]
            assert (policy.correction is not None) is active
            assert loaded[-1].model_dir.name == (adapter["model"] if active else route["model"])
            assert loaded[-1].num_inference_steps == (steps if steps is not None else adapter["num_inference_steps"] if active else route["num_inference_steps"])
            assert policy.execution == (adapter["execution"] if active else route["execution"])
    assert corrections == [int(task_id) for task_id in routes["tasks"] if task_id in adapter["tasks"]]
    count = len(loaded)
    policy = implementation.VLAPolicy(routes["tasks"]["0"], tmp_path, adapter=True, num_inference_steps=steps)
    correction = policy.correction
    policy.configure_route(routes["tasks"]["1"])
    assert len(loaded) == count + 1 and policy.correction is correction and corrections[-1] == 1
    assert routes == load_routes()


def test_hidden_adapter_roundtrip_and_condition_capacity(tmp_path):
    torch.manual_seed(9)
    adapter = HiddenAdapter().eval()
    assert sum(p.numel() for p in adapter.parameters()) == 1962496
    context = dict(instruction="Align with the button.", direction=None, grounding=None)
    arrays = condition_arrays(
        "turning_on_radio",
        context,
        np.zeros((16, 46)),
        np.zeros(16),
        np.zeros((50, 23)),
    )
    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in arrays.items()}
    hidden = torch.randn(1, 50, 1024)
    with torch.no_grad():
        code = adapter(batch)
        torch.testing.assert_close(adapter.inject(hidden, code), hidden, rtol=0, atol=0)
    save_file(adapter.state_dict(), tmp_path / "adapter.safetensors")
    restored = HiddenAdapter().eval()
    restored.load_state_dict(load_file(tmp_path / "adapter.safetensors"), strict=True)
    torch.testing.assert_close(restored(batch), code, rtol=0, atol=0)
    with pytest.raises(ValueError, match="capacity"):
        condition_arrays(
            "x" * 2049,
            context,
            arrays["history"],
            arrays["history_mask"],
            arrays["base_actions"],
        )


def test_correction_reuses_noise_and_conditions_and_injects_layer_nine():
    torch.manual_seed(71)
    model = make_model(enabled=False, enable_rynnbrain_prefix_kv_cache=True).eval()
    expert = model.paligemma_with_expert
    expert.gemma_expert.model.layers = torch.nn.ModuleList(
        [deepcopy(expert.gemma_expert.model.layers[0]) for _ in range(18)]
    )
    for index, layer in enumerate(expert.gemma_expert.model.layers):
        layer.self_attn.layer_idx = index
    expert.num_expert_layers = 18
    model.config.enable_lawam_stage2 = True
    model.config.lawam_inject_mode = "alternate_kv"
    model.lawam_lam = torch.nn.Identity()
    model.lawam_kv_bank = LaWAMKVBank(
        num_layers=18,
        hidden_dim=16,
        num_kv_heads=1,
        head_dim=8,
        alternate_first="vlm",
        alternate_period=1,
    ).eval()
    model._lawam_cond_embs = torch.linspace(-1, 1, 128).reshape(2, 4, 16)
    model._lawam_cond_pad_mask = torch.ones(2, 4, dtype=torch.bool)
    noise = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    calls = []
    with torch.no_grad():
        flow = model.prepare_flow(**inputs(), noise=noise)
        base = model.integrate_flow(flow, num_steps=10)
        corrected = model.integrate_flow(
            flow,
            num_steps=10,
            hidden_residual=lambda hidden: calls.append(hidden.shape) or hidden + 0.1,
        )
        repeated = model.integrate_flow(flow, num_steps=10)
    assert len(calls) == 10 and all(shape[-1] == 16 for shape in calls)
    assert not torch.equal(base, corrected)
    torch.testing.assert_close(base, repeated, rtol=0, atol=0)
    torch.testing.assert_close(flow.noise, noise, rtol=0, atol=0)


def test_progress_gate_releases_after_sixteen_steps_and_requires_two_rearm_votes():
    gate = ProgressGate()
    review = decision()
    assert gate.update(review, 0)[0]
    assert gate.update(review, 16) == (False, "exit_no_progress")
    assert gate.update(review, 32) == (False, "skip_same_problem_latched")
    fresh = dict(review, target="another radio", rearm_reason="new_target")
    assert gate.update(fresh, 48) == (False, "skip_same_problem_latched")
    assert gate.update(fresh, 64) == (False, "skip_rearmed_base")
    assert gate.blocked_problem is None
    assert gate.update(fresh, 80)[0]


def test_blocked_background_screen_preserves_state_and_requires_fresh_rearm():
    gate = ProgressGate()
    review = decision()
    gate.update(review, 0)
    gate.update(review, 16)
    state = deepcopy(vars(gate))
    for changes, expected in (
        ({"rearm_reason": "same_problem"}, True),
        ({"rearm_reason": "uncertain"}, True),
        ({"rearm_reason": "new_target", "target": "another radio"}, False),
        ({"rearm_reason": "visible_progress"}, False),
        ({"progress": "progressing"}, False),
    ):
        assert screening_negative(gate, dict(review, **changes)) is expected
        assert vars(gate) == state
    with pytest.raises(ValueError, match="strictly increase"):
        gate.update(review, 16)
    assert vars(gate) == state


def test_review_contract_excludes_evaluator_truth_and_rejects_contradictions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:8001/v1")
    selector = NeedSelector(
        definitions=[
            dict(id="press", correction_text="Align the finger with the button.")
        ]
    )
    request = selector.build_request(
        task="turning_on_radio",
        policy_prompt="turning_on_radio",
        images=np.zeros((1, 3, 224, 224, 3), np.uint8),
        frame_steps=[0],
        state=np.zeros(23),
        history=np.zeros((16, 46)),
        history_mask=np.zeros(16, bool),
        candidate=np.zeros((50, 23)),
        gate_context={},
    )
    fields = json.loads(request["messages"][1]["content"][0]["text"])
    prompt = request["messages"][0]["content"]
    for definition in ("stable=", "slipping=", "empty=", "uncertain=holding"):
        assert definition in prompt
    assert fields["trained_recovery_modes"] == []
    assert (
        len([x for x in request["messages"][1]["content"] if x["type"] == "image_url"])
        == 3
    )
    assert not {"reward", "object_poses", "expert_actions", "qscore"} & fields.keys()
    validate_review(decision(), ["press"])
    with pytest.raises(ValueError, match="Contradictory"):
        validate_review(dict(decision(), execution_status="normal"), ["press"])
    with pytest.raises(ValueError, match="confidence"):
        validate_review(dict(decision(), confidence=float("nan")), ["press"])


def test_background_review_cannot_apply_delayed_positive_and_reset_discards_inflight():
    controller = CorrectionController.__new__(CorrectionController)
    controller.progress_gate = ProgressGate()
    controller._review_worker = SimpleNamespace(
        future=None,
        invalidate=lambda: None,
        poll=lambda **kwargs: dict(
            step=0,
            episode_id="episode",
            prompt="task",
            decision=decision(),
            kwargs={},
        ),
    )
    controller.previous_review = dict(
        decision(),
        skill_id="none",
        local_manipulation=False,
        need_correction=False,
        execution_status="normal",
    )
    controller._review_prompt = "task"
    controller._nonlocal_review_streak = 1
    controller.previous_review_step = 0
    controller.episode_id = "episode"
    controller.step = 16
    assert controller._background_review({}, "task") is False
    assert controller.progress_gate.active_skill is None
    worker = ReviewWorker(None)
    try:
        controller._review_worker = worker
        tasks = load_config()["tasks"]
        controller.configure_task(tasks["0"])
        controller.configure_task(tasks["1"])
        worker.selector.validate_response(dict(decision(), skill_id=tasks["1"]["skill"]["id"]))
        worker.future = Future()
        worker.future.set_running_or_notify_cancel()
        future = worker.future
        worker.invalidate()
        future.set_result({"generation": 0})
        assert worker.poll() is None
    finally:
        worker.close()


def test_action_guard_checks_the_entire_execution_prefix_and_keeps_support_hand():
    base = np.zeros((50, 23), np.float32)
    proposed = base.copy()
    proposed[:, 7:14] = 0.001
    proposed[:, 15:22] = 0.005
    fit = dict(
        verdict="compatible",
        confidence=0.9,
        arms="left",
        left_gripper="keep_base",
        right_gripper="keep_base",
        evidence="Only the left hand aligns with the button.",
    )
    geometry = dict(
        ee_delta_max_mm=1, rotation_delta_max_rad=0.01, residual_step_jump_rad=0.001
    )
    summary = dict(left=geometry, right=geometry)
    mask = np.ones(23, np.float32)
    fixed, mask, guard = guarded_actions(base, proposed, mask, fit, summary)
    assert guard["accepted"]
    np.testing.assert_array_equal(fixed[:, 14:], base[:, 14:])
    np.testing.assert_array_equal(fixed[:, :7], base[:, :7])

    class Geometry:
        def fk(self, action, side):
            transform = np.eye(4)
            transform[0, 3] = action[7 if side == "left" else 15]
            return transform

    proposed[15, 7] = 0.04
    summary = action_summary(base, proposed, Geometry())
    assert summary["prefix_steps"] == 16 and summary["left"]["ee_delta_max_mm"] > 30
    assert not guarded_actions(base, proposed, mask, fit, summary)[2]["accepted"]
    proposed[15, 7] = 0.001
    proposed[15, 15] = 0.04
    fit["arms"] = "both"
    summary = action_summary(base, proposed, Geometry())
    fixed, mask, guard = guarded_actions(base, proposed, np.ones(23, np.float32), fit, summary)
    assert not guard["accepted"] and not mask.any()
    np.testing.assert_array_equal(fixed, base)


@pytest.mark.parametrize("corrupt", [None, "adapter", "grounding"])
def test_prepare_adapter_resources_checks_identity_and_separate_hub_credentials(
    tmp_path, monkeypatch, corrupt
):
    import hashlib
    import huggingface_hub
    import a2meas.behavior.resources as resources
    import a2meas.behavior as routing
    import a2meas.behavior.correction as correction

    def entry(data):
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    config = tmp_path / "configs"
    config.mkdir()
    (config / "policy").mkdir()
    index = json.dumps({"models": {}}).encode()
    (config / "resources.json").write_text(
        json.dumps({"files": {"assets/index.json": entry(index)}})
    )
    (config / "release.json").write_text(
        json.dumps(dict(repo_id="test/base", revision="base-sha"))
    )
    (config / "routing.json").write_text(json.dumps({"models": {}}))
    monkeypatch.setattr(resources, "CONFIG_ROOT", config)
    monkeypatch.setattr(routing, "CONFIG_ROOT", config)
    monkeypatch.setattr(
        correction,
        "load_config",
        lambda: dict(
            weights=entry(b"adapter"),
            grounding=dict(
                repo_id="test/sam",
                revision="sam-sha",
                filename="grounding.pt",
                **entry(b"grounding"),
            ),
        ),
    )
    cache = tmp_path / "cached-grounding"
    cache.write_bytes(b"corrupt!" if corrupt == "grounding" else b"grounding")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if kwargs["repo_id"] == "test/sam":
            assert (
                kwargs["revision"] == "sam-sha" and kwargs["filename"] == "grounding.pt"
                and "token" not in kwargs
            )
            path = kwargs['local_dir'] / kwargs['filename']
            path.write_bytes(cache.read_bytes())
            return str(path)
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        assert kwargs["token"] is None
        if kwargs["filename"] == "adapter/adapter.safetensors":
            assert kwargs["repo_id"] == "test/base" and kwargs["revision"] == "base-sha"
            path.write_bytes(b"corrupt" if corrupt == "adapter" else b"adapter")
        else:
            path.write_bytes(index)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    destination = tmp_path / "resources"
    if corrupt:
        with pytest.raises(ValueError, match="invalid|checksum"):
            resources.prepare(destination, adapter=True)
        assert not (destination / "grounding/sam.pt").exists()
        assert not list(destination.rglob("*.partial"))
    else:
        assert resources.prepare(destination, adapter=True) == 3
        assert (destination / "adapter/adapter.safetensors").read_bytes() == b"adapter"
        assert (destination / "grounding/sam.pt").read_bytes() == b"grounding"
        assert resources.prepare(destination, adapter=True) == 3
        assert len(calls) == 3


def test_policy_executes_sixteen_raw_corrections_and_filters_only_base_channels():
    from a2meas.behavior.actions import ActionChunker, gaussian_filter_action_chunk
    from a2meas.behavior.policy import VLAPolicy

    base = np.tile(np.arange(50, dtype=np.float32)[:, None] / 50, (1, 23))
    mask = np.zeros(23, np.float32)
    mask[7:14] = 1
    predicted = base.copy()
    predicted[:, 7:14] += 0.01

    class Controller:
        horizon = 16
        state = np.zeros(23)
        calls = 0

        def reset(self):
            pass

        def observe(self, observation):
            return False

        def action_sent(self, action):
            pass

        def predict(self, sample):
            self.calls += 1
            return predicted.copy(), mask

    policy = VLAPolicy.__new__(VLAPolicy)
    policy.task_id = 0
    policy.observations = SimpleNamespace(
        reset=lambda: None, observe=lambda obs: None, sample=lambda: {}
    )
    policy.first_close = None
    policy.correction = Controller()
    policy.execution = load_config()["execution"]
    policy.chunking = ActionChunker(arm_step_limit=0.1)
    policy.reset()
    state = np.zeros(61, dtype=np.float32)
    state[24:26] = state[49:51] = 0.05
    observation = {"robot_r1::proprio": state}
    actual = np.stack([policy.step(observation) for _ in range(16)])
    assert policy.correction.calls == 1 and not policy.actions
    np.testing.assert_array_equal(actual[:, 7:14], predicted[:16, 7:14])
    expected = gaussian_filter_action_chunk(
        predicted, window=5, sigma=1, skip_dims=[0, 1, 2, 14, 22]
    )
    np.testing.assert_array_equal(actual[:, 15:22], expected[:16, 15:22])
    np.testing.assert_array_equal(actual[:, :3], base[:16, :3])
    policy.step(observation)
    assert policy.correction.calls == 2


def test_controller_generates_from_the_fresh_flow_and_handles_only_review_failures(
    monkeypatch,
):
    import a2meas.behavior.correction.controller as implementation
    from a2meas.behavior.correction.client import ReviewError

    flow = object()
    calls = []

    def predict_flow(value, *, hidden_residual=None):
        assert value is flow
        calls.append(hidden_residual)
        if hidden_residual is None:
            return np.zeros((50, 23), np.float32)
        hidden = hidden_residual(torch.zeros(1, 50, 16))
        return np.full((50, 23), hidden.mean().item(), np.float32)

    controller = CorrectionController.__new__(CorrectionController)
    controller.horizon = 16
    controller._review_worker = None
    controller.task = "turning_on_radio"
    controller.definition = dict(
        id="press", name="press", correction_text="Align the finger."
    )
    controller.service = SimpleNamespace(
        prepare_flow=lambda sample: flow,
        predict_flow=predict_flow,
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )

    class Adapter:
        def __call__(self, batch):
            assert batch["history"].shape == (1, 16, 46)
            assert batch["base_actions"].shape == (1, 50, 23)
            return torch.zeros(1, 16)

        def inject(self, hidden, code):
            return hidden + 0.001

    controller.adapter = Adapter()

    def ground(images, target, step):
        assert images.shape == (3, 224, 224, 3) and isinstance(target, str)
        return None

    controller.grounder = ground
    controller.selector = SimpleNamespace(infer=lambda **kwargs: decision())
    controller.fit_selector = SimpleNamespace(
        infer=lambda **kwargs: dict(
            verdict="compatible",
            confidence=0.9,
            arms="left",
            left_gripper="keep_base",
            right_gripper="keep_base",
            evidence="Local left hand alignment.",
        )
    )
    controller.kinematics = SimpleNamespace(fk=lambda action, side: np.eye(4))
    monkeypatch.setattr(
        implementation,
        "request_context",
        lambda *args, **kwargs: dict(
            instruction="Align the finger.",
            direction=None,
            grounding=None,
            confidence=0.9,
        ),
    )
    controller.reset()
    controller.step = 0
    controller.state = np.zeros(23, np.float32)
    controller.images = np.zeros((3, 224, 224, 3), np.uint8)
    fixed, mask = controller.predict({})
    assert len(calls) == 2 and calls[0] is None and callable(calls[1])
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(7, 14))
    assert (fixed[:, 7:14] > 0).all()
    np.testing.assert_array_equal(fixed[:, mask == 0], np.zeros((50, 16)))
    controller.step = 16
    inputs = controller._review_inputs(
        fixed, np.zeros((16, 46)), np.zeros(16, bool), controller.task
    )
    previous = inputs["gate_context"]["previous_review"]
    assert not {"holding_left", "holding_right", "evidence"} & previous.keys()
    assert previous["skill_id"] == "press" and previous["progress"] == "no_progress"
    assert controller.previous_review["holding_left"] == "stable"

    def unavailable(**kwargs):
        raise ReviewError("Malformed response")

    controller.selector.infer = unavailable
    base, mask = controller.predict({})
    assert (
        mask is None
        and not base.any()
        and controller.progress_gate.active_skill is None
    )
    assert controller.progress_gate.blocked_problem is None

    def broken(value, **kwargs):
        raise RuntimeError("Broken model invariant")

    controller.service.predict_flow = broken
    with pytest.raises(RuntimeError, match="invariant"):
        controller.predict({})


def test_optional_sam_installation_imports_its_model_builder():
    from importlib.util import find_spec

    if find_spec("sam3") is None:
        pytest.skip("SAM is an optional dependency")
    from sam3 import model_builder

    assert callable(model_builder._create_multiplex_tri_backbone)
