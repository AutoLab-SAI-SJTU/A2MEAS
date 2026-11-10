"""Model prediction and checkpoint loading regression tests."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tests.behavior.tiny_model import make_model, inputs

@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("steps", [2, 4, 10])
def test_model_predictions_survive_checkpoint_reload(cache, steps, tmp_path):
    torch.manual_seed(41)
    model = make_model(enable_rynnbrain_prefix_kv_cache=cache).eval()
    noise = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    with torch.inference_mode():
        expected = model.sample_actions(**inputs(), noise=noise, num_steps=steps)
    checkpoint = tmp_path / "model.safetensors"
    from safetensors.torch import save_model, load_model
    save_model(model, checkpoint)
    restored = make_model(enable_rynnbrain_prefix_kv_cache=cache).eval()
    load_model(restored, checkpoint, strict=True)
    with torch.inference_mode():
        actual = restored.sample_actions(**inputs(), noise=noise, num_steps=steps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dino_restores_weights_without_a_second_pretrained_download(tmp_path):
    from transformers import AutoModel, DINOv3ViTConfig
    from a2meas.models.vision import DINOv3Encoder

    config = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=16,
        patch_size=8,
    )
    original = AutoModel.from_config(config).eval()
    original.save_pretrained(tmp_path / "reference")
    architecture = tmp_path / "architecture"
    config.save_pretrained(architecture)
    torch.manual_seed(73)
    reference = AutoModel.from_pretrained(tmp_path / "reference", dtype=torch.float32).eval()
    torch.manual_seed(73)
    encoder = DINOv3Encoder(model_id=str(architecture))
    encoder.model.load_state_dict(original.state_dict(), strict=True)
    assert not (architecture / "model.safetensors").exists()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            encoder.model(image).last_hidden_state,
            reference(image).last_hidden_state,
            rtol=0,
            atol=0,
        )
        expected = reference(image, output_hidden_states=True)
        for layer in (-1, -2):
            tokens = expected.last_hidden_state if layer == -1 else expected.hidden_states[layer]
            torch.testing.assert_close(
                encoder.encode(image, n=layer), tokens[:, 5:].unsqueeze(1), rtol=0, atol=0
            )


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_qwen_memory_and_native_layers_preserve_predictions(attention, tmp_path):
    from tests.behavior.tiny_model import make_qwen_model

    expected = json.loads((Path(__file__).parent / "fixtures/tiny_qwen.json").read_text())
    model = make_qwen_model(attention)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    assert digest.hexdigest() == expected["state_sha256"]
    ids = torch.tensor([[1, 30] + [29] * 4 + [31, 30] + [29] * 4 + [31, 2]])
    pixels = torch.linspace(-1, 1, 2 * 3 * 4 * 4 * 3 * 2 * 8 * 8).reshape(-1, 3 * 2 * 8 * 8)
    for cache in (False, True):
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                pixel_values_videos=pixels,
                video_grid_thw=torch.tensor([[3, 4, 4], [3, 4, 4]]),
                video_frame_age_seconds=torch.tensor([[9.0, 8.0, 4.0, 3.0, 1.0, 0.0]] * 2),
                output_last_hidden_state_only=True,
                logits_to_keep=None,
                use_cache=cache,
            )
        torch.testing.assert_close(
            output.pre_norm_last_hidden_state,
            torch.tensor(expected["attention"][attention]["hidden"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.rope_deltas, torch.tensor(expected["attention"][attention]["rope"]), rtol=0, atol=0
        )

    # Load the backbone before attaching memory parameters.
    memory = model.model.visual.temporal_memory
    model.model.visual.temporal_memory = None
    model.save_pretrained(tmp_path)
    restored = type(model).from_pretrained(tmp_path, attn_implementation=attention).eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    restored.model.visual.temporal_memory = memory
    with torch.inference_mode():
        reloaded = restored(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values_videos=pixels,
            video_grid_thw=torch.tensor([[3, 4, 4], [3, 4, 4]]),
            video_frame_age_seconds=torch.tensor([[9.0, 8.0, 4.0, 3.0, 1.0, 0.0]] * 2),
            output_last_hidden_state_only=True,
            logits_to_keep=None,
            use_cache=True,
        )
    torch.testing.assert_close(
        reloaded.pre_norm_last_hidden_state, output.pre_norm_last_hidden_state, rtol=0, atol=0
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["base", "global-task"])
def test_behavior_action_interface_and_strict_checkpoint_loading(enabled, tmp_path):
    import copy
    import draccus
    from safetensors.torch import load_file, save_file, save_model
    from a2meas.behavior.model.policy import A2WLAPolicy
    from tests.behavior.tiny_model import make_behavior_policy, behavior_batch, tiny_construction

    expected = json.loads((Path(__file__).parent / "fixtures/behavior_policy.json").read_text())[str(enabled)]
    policy = make_behavior_policy(enabled)
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    assert digest.hexdigest() == expected["state_sha256"]
    batch = behavior_batch(policy)
    noise = torch.linspace(-1, 1, 2 * 3 * 32).reshape(2, 3, 32)
    expected_action = torch.tensor(expected["actions"])
    torch.testing.assert_close(
        policy.predict_action_chunk(batch, noise=noise, num_steps=2), expected_action, rtol=0, atol=0
    )
    missing_camera = dict(batch)
    missing_camera.pop("observation.rgb.zed_link_camera_0")
    with pytest.raises(KeyError):
        policy.predict_action_chunk(missing_camera, noise=noise, num_steps=2)
    (tmp_path / "config.json").write_text(json.dumps(draccus.encode(policy.config)))
    save_model(policy, tmp_path / "model.safetensors")
    with tiny_construction():
        restored = A2WLAPolicy.from_pretrained(tmp_path, checkpoint_path=tmp_path / "model.safetensors", config=policy.config)
    assert not restored.training
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    restored.model.embed_prefix = policy.model.embed_prefix
    torch.testing.assert_close(
        restored.predict_action_chunk(batch, noise=noise, num_steps=2), expected_action, rtol=0, atol=0
    )
    if enabled:
        changed = copy.deepcopy(policy.config)
        changed.global_task_text_to_id = {"put cup": 0, "open drawer": 2}
        with pytest.raises(ValueError, match="global task IDs"):
            A2WLAPolicy.from_pretrained(tmp_path, checkpoint_path=tmp_path / "model.safetensors", config=changed)
    weights = load_file(tmp_path / "model.safetensors")
    weights["model.state_in_proj.weight"] = weights["model.state_in_proj.weight"][:-1]
    save_file(weights, tmp_path / "model.safetensors")
    with tiny_construction(), pytest.raises(RuntimeError, match="size mismatch"):
        A2WLAPolicy.from_pretrained(tmp_path, checkpoint_path=tmp_path / "model.safetensors", config=policy.config)
    (tmp_path / "model.safetensors").unlink()
    with tiny_construction(), pytest.raises(FileNotFoundError):
        A2WLAPolicy.from_pretrained(tmp_path, checkpoint_path=tmp_path / "model.safetensors", config=policy.config)


@pytest.mark.parametrize("enabled", [False, True], ids=["base", "global-task"])
def test_alternate_lawam_kv_preserves_predictions(enabled):
    from a2meas.models.world_conditioning import LaWAMKVBank

    torch.manual_seed(71)
    model = make_model(enabled=enabled, enable_rynnbrain_prefix_kv_cache=True).eval()
    model.config.enable_lawam_stage2 = True
    model.config.lawam_inject_mode = "alternate_kv"
    model.lawam_lam = torch.nn.Identity()
    model.lawam_kv_bank = LaWAMKVBank(
        num_layers=2, hidden_dim=16, num_kv_heads=1, head_dim=8, alternate_first="vlm", alternate_period=1
    ).eval()
    model.paligemma_with_expert.num_expert_layers = 2
    model._lawam_cond_embs = torch.linspace(-1, 1, 2 * 4 * 16).reshape(2, 4, 16)
    model._lawam_cond_pad_mask = torch.ones(2, 4, dtype=torch.bool)
    expected = json.loads((Path(__file__).parent / "fixtures/alternate_kv.json").read_text())[str(enabled)]
    with torch.inference_mode():
        actions = model.sample_actions(
            **inputs(), noise=torch.linspace(-1, 1, 24).reshape(2, 3, 4), num_steps=4
        )
    torch.testing.assert_close(actions, torch.tensor(expected["actions"]), rtol=0, atol=0)


def test_lawam_current_frames_preserve_camera_order_and_explicit_padding():
    from types import SimpleNamespace
    from a2meas.behavior.model.world_conditioning import WorldModelMixin

    model = WorldModelMixin()
    model.config = SimpleNamespace(
        image_features={"left": None, "right": None, "head": None},
        lawam_primary_camera_key="head",
        lawam_multi_view_camera_keys=["left", "right"],
    )
    images = [torch.full((2, 6, 3, 4, 4), value) for value in (0.0, 0.5, -1.0)]
    pads = [torch.zeros((2, 6), dtype=torch.bool) for _ in images]
    pads[0][1, -1] = True
    current, auxiliary, valid = model._lawam_current_frames(images, pads)
    torch.testing.assert_close(current, images[2][:, -1])
    torch.testing.assert_close(auxiliary, torch.stack([images[0][:, -1], images[1][:, -1]], dim=1))
    torch.testing.assert_close(valid, torch.tensor([[True, True, True], [True, False, True]]))


def test_backbone_architecture_loads_without_redundant_base_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from a2meas.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from a2meas.behavior.model import policy as implementation

    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, tie_word_embeddings=True, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, deepstack_visual_indexes=[0]),
        tie_word_embeddings=True,
    )
    reference = Qwen3VLForConditionalGeneration._from_config(config, dtype=torch.bfloat16).eval()
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(implementation.Qwen3VLProcessor, "from_pretrained", lambda *_, **kw: SimpleNamespace())

    def reject_base_weights(*args, **kwargs):
        raise AssertionError("The full policy checkpoint supplies all backbone weights")

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", reject_base_weights)
    compiled = []

    def compile_layer(forward, **options):
        compiled.append((forward.__self__, options))
        return forward

    monkeypatch.setattr(torch, "compile", compile_layer)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: None)
    expert = SimpleNamespace(width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    model = implementation.A2WLABackbone(expert, str(tmp_path), precision="bfloat16")
    backbone = model.rynnbrain
    assert [layer for layer, _ in compiled] == [
        *(block.mlp for block in backbone.model.visual.blocks),
        *backbone.model.language_model.layers, *(layer.mlp for layer in model.gemma_expert.model.layers),
    ]
    assert all(options == {"mode": "max-autotune-no-cudagraphs", "dynamic": True}
               for _, options in compiled)
    assert backbone.get_input_embeddings().weight is backbone.lm_head.weight
    backbone.load_state_dict(reference.state_dict(), strict=True)
    backbone.eval()
    assert backbone.model.visual.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert backbone.model.language_model.rotary_emb.inv_freq.dtype == torch.float32
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = reference(input_ids=ids, output_last_hidden_state_only=True, use_cache=False)
        actual = backbone(input_ids=ids, output_last_hidden_state_only=True, use_cache=False)
    torch.testing.assert_close(actual.pre_norm_last_hidden_state, expected.pre_norm_last_hidden_state, rtol=0, atol=0)
