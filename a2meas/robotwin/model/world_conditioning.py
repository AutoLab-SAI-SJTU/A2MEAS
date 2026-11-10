"""LaWAM observed-image features and predicted future-prefix integration."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any
from a2meas.models.action_expert import extract_la_hidden_from_inject_result
import torch
from torch import Tensor, nn
from a2meas.models.world_conditioning import VLMToLAMQFormer, LaWAMPrefixAdapter, LaWAMKVBank, lawam_layer_uses_vlm_kv, prepare_lawam_rgb_batch, decode_future_tokens, encode_lawam_multi_view_visual_cond

class WorldModelMixin:

    def _lawam_layer_uses_vlm_kv(self, layer_idx: int) -> bool:
        return lawam_layer_uses_vlm_kv(layer_idx, alternate_first='vlm', alternate_period=1)

    def _init_lawam_stage2(self, *, expert_width: int, action_expert_config) -> None:
        """Build the vision decoder and prefix queries from the inference architecture."""
        from a2meas.models.world_model import build_latent_action_model
        camera_order = [self.config.lawam_primary_camera_key, *self.config.lawam_multi_view_camera_keys]
        lam = build_latent_action_model(self.config.lawam_config_path, self.config.lawam_vision_path, camera_order)
        self.lawam_lam = lam
        vlm_hidden = self._rynnbrain_hidden_size()
        num_q = int(self.config.lawam_num_action_queries)
        self.act_query = nn.Parameter(torch.randn(num_q, vlm_hidden) * 0.02)
        self.vlm_to_lam = VLMToLAMQFormer(vlm_hidden_dim=vlm_hidden, lam_code_dim=int(lam.code_dim), num_layers=1, num_heads=8, ffn_expansion_factor=4.0)
        self.lawam_prefix_adapter = LaWAMPrefixAdapter(vision_dim=int(lam.input_dim), output_dim=int(expert_width), num_tokens=self.config.lawam_num_prefix_tokens, num_heads=8, need_weights=False)
        for block in lam.decoder.dec_layers:
            block.mha_need_weights = False
        if not lam.decoder_multi_view:
            raise ValueError('RoboTwin requires a multi-view world decoder')
        self.lawam_camera_embedding = nn.Parameter(torch.zeros(int(lam.num_view_streams), int(expert_width)))
        self.lawam_vision_camera_embedding = nn.Parameter(torch.zeros(int(lam.num_view_streams), int(lam.input_dim)))
        self.lawam_kv_bank = LaWAMKVBank(
            num_layers=int(action_expert_config.depth), hidden_dim=int(expert_width),
            num_kv_heads=int(action_expert_config.num_kv_heads), head_dim=int(action_expert_config.head_dim),
            alternate_first='vlm', alternate_period=1,
        )

    def _lawam_current_frames(self, images, img_pad_masks):
        """Select the latest observed frame in the configured camera order."""
        keys = list(self.config.image_features)
        cameras = [self.config.lawam_primary_camera_key, *self.config.lawam_multi_view_camera_keys]
        frames, validity = ([], [])
        for key in cameras:
            index = keys.index(key)
            camera = images[index]
            frame = camera[:, -1] if camera.ndim == 5 else camera
            frames.append(frame)
            pad = None if img_pad_masks is None else img_pad_masks[index]
            if pad is None:
                valid = ~torch.all(frame.reshape(frame.shape[0], -1) == -1, dim=1)
            else:
                valid = ~(pad[:, -1] if pad.ndim == 2 else pad).bool().to(frame.device)
            validity.append(valid)
        auxiliary = torch.stack(frames[1:], dim=1) if len(frames) > 1 else None
        return (frames[0], auxiliary, torch.stack(validity, dim=1))

    def _lawam_build_videos(self, current, auxiliary):
        """Build DINO inputs from observations; future features come from the decoder."""
        resolution = self.config.lawam_image_resolution
        device = next(self.lawam_lam.parameters()).device
        video = prepare_lawam_rgb_batch(current, image_resolution=resolution).unsqueeze(1)
        video = video.to(device=device, non_blocking=device.type == 'cuda')
        views = None
        if auxiliary is not None:
            views = torch.stack([prepare_lawam_rgb_batch(auxiliary[:, i], image_resolution=resolution) for i in range(auxiliary.shape[1])], dim=1).unsqueeze(1)
            views = views.to(device=device, non_blocking=device.type == 'cuda')
        return (video, views)

    def _lawam_extract_vision_pack(self, video, *, multi_view_videos=None):
        """Keep DINO features; inference has no consumer for deterministic IDM targets."""
        lam = self.lawam_lam
        with torch.no_grad():
            return lam.extract_multi_view_features(video, multi_view_videos, n=lam.latent_layer_to_use)

    def _lawam_start_vision_prefetch(self, primary_current: Tensor, *, mv_current: Tensor | None=None) -> dict[str, Any] | None:
        """Extract DINO features on a side CUDA stream while the VLM runs."""
        lam_device = next(self.lawam_lam.parameters()).device
        if lam_device.type != 'cuda':
            return None
        video, multi_view_videos = self._lawam_build_videos(primary_current, mv_current)
        prep_done = torch.cuda.Event()
        prep_done.record()
        if self._lawam_vision_stream is None:
            self._lawam_vision_stream = torch.cuda.Stream(device=lam_device)
        side: torch.cuda.Stream = self._lawam_vision_stream
        side.wait_event(prep_done)
        box: dict[str, Any] = {'video': video, 'multi_view_videos': multi_view_videos}
        with torch.cuda.stream(side):
            box['features'] = self._lawam_extract_vision_pack(video, multi_view_videos=multi_view_videos)
            done = torch.cuda.Event()
            done.record(side)
        box['done_event'] = done
        return box

    def _lawam_wait_vision_prefetch(self, handle: dict[str, Any] | None) -> dict[str, Any] | None:
        if handle is None:
            return None
        torch.cuda.current_stream().wait_event(handle['done_event'])
        return handle

    def _compute_lawam_prefix_tokens(self, zhat: Tensor, primary_current: Tensor, *, vision_prefetch: dict[str, Any] | None=None, mv_current: Tensor | None=None, view_valid: Tensor | None=None):
        """Encode current and predicted future features as action-prefix tokens."""
        lam = self.lawam_lam
        device = zhat.device
        dtype = zhat.dtype
        lam_device = next(lam.parameters()).device
        if vision_prefetch is not None:
            features = vision_prefetch['features']
        else:
            video, multi_view_videos = self._lawam_build_videos(primary_current, mv_current)
            features = self._lawam_extract_vision_pack(video, multi_view_videos=multi_view_videos)
        h_t = features[:, 0].detach().clone()
        zhat_lam = zhat.to(device=lam_device, dtype=h_t.dtype)
        decoder_amp = torch.autocast('cuda', dtype=torch.bfloat16) if lam_device.type == 'cuda' else nullcontext()
        with decoder_amp:
            future = decode_future_tokens(lam, h_t, zhat_lam).to(device=device, dtype=dtype)
        prefix_amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with prefix_amp:
            self._lawam_cond_embs, self._lawam_cond_pad_mask, _ = encode_lawam_multi_view_visual_cond(
                self.lawam_prefix_adapter, h_t=h_t.detach().to(device=device, dtype=dtype), uhat=future,
                prefix_mode='compressed', inject_current=True, camera_embedding=self.lawam_camera_embedding,
                current_valid=view_valid, future_valid=view_valid, device=device, dtype=dtype,
                token_budget_mode='shared_total', vision_camera_embedding=self.lawam_vision_camera_embedding,
            )

    def _embed_lawam_prefix(self, images, *, img_pad_masks=None, raw_texts=None, state_history=None):
        current, mv_current, view_valid = self._lawam_current_frames(images, img_pad_masks)
        prefetch = self._lawam_start_vision_prefetch(current, mv_current=mv_current)
        inputs, pad, _ = self.get_rynnbrain_processor_inputs(images, img_pad_masks=img_pad_masks, raw_texts=raw_texts)
        la_query = torch.cat((self.act_query, self.trajectory_queries), dim=0)
        hidden, pad, att, result = self._run_feature_inject(inputs, pad, la_query=la_query)
        la_hidden = extract_la_hidden_from_inject_result(result)
        num_action_queries = int(self.config.lawam_num_action_queries)
        h_act = la_hidden[:, :num_action_queries]
        zhat = self.vlm_to_lam(h_act)
        self._compute_lawam_prefix_tokens(zhat, current, vision_prefetch=self._lawam_wait_vision_prefetch(prefetch), mv_current=mv_current, view_valid=view_valid)
        prefix = self.paligemma_with_expert.align_proj(hidden.to(dtype=self.paligemma_with_expert.align_proj.weight.dtype))
        prefix, pad, att = self._append_state_tokens_to_prefix(prefix, pad, att, state_history)
        return (prefix, pad, att)
