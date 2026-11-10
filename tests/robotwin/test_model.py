"""Key model behavior and external boundaries."""

import json
import pytest
from benchmarks.robotwin.check_checkpoint import validate_checkpoint
from a2meas.robotwin.schema import ROBOTWIN_CAMERAS
from types import SimpleNamespace
import torch
from safetensors.torch import save_file
from a2meas.robotwin.weights import load_inference_weights
from transformers.cache_utils import DynamicCache
from a2meas.robotwin.graph_denoise import GraphDenoiser
from a2meas.robotwin.model.config import A2WLAConfig
from a2meas.robotwin.model.policy import A2WLAPolicy, A2WLAModel
from a2meas.robotwin.processor import make_processors
from a2meas.processing.steps import NormalizerProcessorStep
from a2meas.schema import FeatureType, PolicyFeature


@pytest.fixture
def validate(tmp_path):
    (tmp_path / "model.safetensors").touch()

    def check(config, **kwargs):
        (tmp_path / "config.json").write_text(json.dumps(config))
        return validate_checkpoint(tmp_path, **kwargs)

    return check


def _valid_config() -> dict:
    return {
        "type": "pi05",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [14]},
            **{camera: {"type": "VISUAL", "shape": [480, 640, 3]} for camera in ROBOTWIN_CAMERAS},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "max_action_dim": 32,
        "max_state_dim": 32,
        "chunk_size": 50,
        "gripper_action_indices": [6, 13],
    }


def test_inference_config_roundtrip_preserves_query_parameters():
    import draccus
    config = A2WLAConfig.from_dict(_valid_config())
    assert draccus.encode(config)['trajectory_query_count'] == 10
    assert A2WLAConfig.from_dict(draccus.encode(config)) == config


def test_reports_malformed_indices_without_crashing(validate) -> None:
    for value in ("[6,13]", "6,13", "[left,right]", "[6,]", [6.5, 13], [True, 13], []):
        config = _valid_config()
        config["gripper_action_indices"] = value
        report = validate(config)
        assert not report["compatible"]
        assert any(("gripper_action_indices" in error for error in report["errors"]))


def test_rejects_missing_or_invalid_input_features(validate) -> None:
    for input_features in (None, {}, []):
        config = _valid_config()
        config["input_features"] = input_features
        report = validate(config)
        assert not report["compatible"]
        assert any(("input_features" in error for error in report["errors"]))


def test_rejects_missing_or_invalid_output_features(validate) -> None:
    for output_features in (None, {}, [], {"action": {"type": "ACTION", "shape": []}}):
        config = _valid_config()
        config["output_features"] = output_features
        report = validate(config)
        assert not report["compatible"]


def test_rejects_features_the_runtime_does_not_implement(validate):
    for field, value in (
        ("enable_fast_lm_head", True),
        ("enable_rynnbrain_lora", True),
        ("compile_model", True),
        ("enable_gripper_first_close_meta", True),
        ("short_term_memory_frame_age_embedding_enable", True),
        ("enable_clip_running_status_condition", True),
    ):
        config = {**_valid_config(), field: value}
        report = validate(config)
        assert not report["compatible"]
        assert "type" in config


def test_rejects_denoise_step_and_frame_mismatches_before_loading(validate):
    config = _valid_config()
    for arguments in ({"num_inference_steps": 4}, {"image_history_num_frames": 6}):
        assert not validate(config, **arguments)["compatible"]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(2, 3))
        self.model = torch.nn.Module()
        self.model.trajectory_queries = torch.nn.Parameter(torch.empty(2, 3))


@pytest.mark.parametrize('fault', [None, 'missing', 'unexpected', 'shape'])
def test_inference_weight_contract_and_saved_aliases(tmp_path, fault):
    path = tmp_path / 'model.safetensors'
    tensors = {
        'saved_embedding': torch.ones(2, 3),
        'model.trajectory_2d_tokens': torch.full((2, 3), 2.),
    }
    if fault == 'missing':
        del tensors['model.trajectory_2d_tokens']
    elif fault == 'unexpected':
        tensors['unknown.weight'] = torch.zeros(1)
    elif fault == 'shape':
        tensors['model.trajectory_2d_tokens'] = torch.zeros(3, 3)
    save_file(tensors, str(path), metadata={'embedding': 'saved_embedding'})
    policy = Policy()
    if fault:
        with pytest.raises(RuntimeError, match='size mismatch' if fault == 'shape' else fault):
            load_inference_weights(policy, path)
    else:
        load_inference_weights(policy, path)
        torch.testing.assert_close(policy.embedding, torch.ones(2, 3))
        torch.testing.assert_close(policy.model.trajectory_queries, torch.full((2, 3), 2.))


def test_local_weights_load_strictly(tmp_path):
    from safetensors import SafetensorError
    from safetensors.torch import save_model
    class TinyPolicy(A2WLAPolicy):
        def __init__(self, config):
            torch.nn.Module.__init__(self)
            self.projection = torch.nn.Linear(3, 2)

    model = TinyPolicy(None)
    save_model(model, str(tmp_path / "model.safetensors"))
    config = A2WLAConfig()
    rng = torch.get_rng_state().clone()
    loaded = TinyPolicy.from_pretrained(tmp_path, config=config)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(loaded.projection.weight, model.projection.weight)
    (tmp_path / "model.safetensors").write_bytes(b"invalid")
    with pytest.raises(SafetensorError, match="header"):
        TinyPolicy.from_pretrained(tmp_path, config=config)


def test_checkpoint_statistics_reject_files_outside_the_checkpoint(tmp_path):
    import json
    from safetensors.torch import save_file
    from a2meas.robotwin.server import checkpoint_stats

    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    outside = tmp_path / 'outside.safetensors'
    save_file({'observation.state.mean': torch.zeros(14)}, str(outside))
    pipeline = {'steps': [{'registry_name': 'normalizer_processor', 'state_file': '../outside.safetensors'}]}
    (checkpoint / 'policy_preprocessor.json').write_text(json.dumps(pipeline))
    with pytest.raises(ValueError):
        checkpoint_stats(checkpoint, 'policy_preprocessor.json', 'normalizer_processor')
    (checkpoint / 'linked.safetensors').symlink_to(outside)
    pipeline['steps'][0]['state_file'] = 'linked.safetensors'
    (checkpoint / 'policy_preprocessor.json').write_text(json.dumps(pipeline))
    with pytest.raises(ValueError):
        checkpoint_stats(checkpoint, 'policy_preprocessor.json', 'normalizer_processor')


def test_prompts_and_distinct_input_output_quantiles():
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    action = {"action": PolicyFeature(FeatureType.ACTION, (14,))}
    config = SimpleNamespace(
        input_features=features,
        output_features=action,
        normalization_mapping={"STATE": "QUANTILES", "ACTION": "QUANTILES"},
        action_target_mode="absolute",
        include_state_in_language_prompt=False,
        max_state_dim=32,
        enable_speed_meta=True,
        default_speed_bin=500,
        device="cpu",
    )
    pre = {
        key: {"q01": torch.zeros(14), "q99": torch.full((14,), 2.0)}
        for key in (*features, *action)
    }
    post = {"action": {"q01": torch.full((14,), 10.0), "q99": torch.full((14,), 30.0)}}
    preprocess, postprocess = make_processors(
        config, pre, post
    )
    result = preprocess(
        {"observation.state": torch.ones(2, 14), "task": ["pick_can", "Place\nthe mug"]}
    )
    torch.testing.assert_close(result["observation.state"], torch.zeros(2, 14))
    torch.testing.assert_close(
        postprocess(torch.zeros(2, 50, 14)), torch.full((2, 50, 14), 20.0)
    )
    assert set(result) == {"observation.state", "task"}
    assert result["task"] == [
        "Task: pick can, Speed: 500",
        "Task: Place the mug, Speed: 500",
    ]


@pytest.mark.parametrize(
    "lower,upper", [(float("nan"), 1.0), (0.0, float("inf")), (2.0, 1.0)]
)
def test_invalid_checkpoint_quantiles_are_rejected(lower, upper):
    step = NormalizerProcessorStep(
        features={"observation.state": PolicyFeature(FeatureType.STATE, (14,))},
        norm_map={"STATE": "QUANTILES"},
    )
    with pytest.raises(ValueError, match="Invalid normalization quantiles"):
        step.load_state_dict(
            {
                "observation.state.q01": torch.full((14,), lower),
                "observation.state.q99": torch.full((14,), upper),
            }
        )


def test_graph_errors_are_visible_without_eager_retry():
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("capture error")

    fake_torch = SimpleNamespace(compile=lambda *args, **kwargs: fail)
    graph = GraphDenoiser(fake_torch, lambda: calls.append("eager"))
    with pytest.raises(RuntimeError, match="capture error"):
        graph()
    assert not calls


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_alternate_lawam_forward_runs_both_prefix_sources(monkeypatch, backend):
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from a2meas.models.gemma import GemmaExpert
    from a2meas.models.world_conditioning import LaWAMKVBank

    monkeypatch.setenv("LEROBOT_ACTION_EXPERT_ATTN_IMPLEMENTATION", backend)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        expert = GemmaExpert(GemmaConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            use_adarms=True, adarms_cond_dim=16,
        )).eval()
        model = A2WLAModel.__new__(A2WLAModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(suffix_attention_mask="block", lawam_alternate_first="vlm", lawam_alternate_period=1)
        model.paligemma_with_expert = SimpleNamespace(
            gemma_expert=expert, num_expert_layers=2, rynnbrain=SimpleNamespace(dtype=torch.float32),
        )
        model.lawam_kv_bank = LaWAMKVBank(num_layers=2, hidden_dim=16, num_kv_heads=1, head_dim=8)
        cache = DynamicCache()
        for layer in range(2):
            cache.update(torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), layer)
        output = model._forward_suffix_alternate_lawam_kv(
            suffix_embs=torch.randn(1, 3, 16), suffix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
            suffix_att_masks=torch.zeros(1, 3, dtype=torch.bool), prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool),
            past_key_values=cache, adarms_cond=torch.randn(1, 16),
            lawam_kv_by_layer=model.lawam_kv_bank.build_kv_for_layers(
                torch.randn(1, 2, 16), rotary_model=expert.model, prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool),
            ), lawam_cond_pad_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    assert output.shape == (1, 3, 16) and torch.isfinite(output).all()
    assert cache.get_seq_length() == 4
