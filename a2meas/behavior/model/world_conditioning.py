"""LaWAM visual encoding, future-feature decoding and action-expert prefix assembly."""

from contextlib import nullcontext
from typing import Any
from a2meas.models.action_expert import append_prefix_tokens, extract_la_hidden_from_inject_result
import torch
from torch import Tensor, nn
from a2meas.models.world_conditioning import (
    VLMToLAMQFormer,
    LaWAMPrefixAdapter,
    LaWAMKVBank,
    lawam_inject_mode_is_alternate_kv,
    lawam_layer_uses_vlm_kv,
    prepare_lawam_rgb_batch,
    decode_future_tokens,
    encode_lawam_visual_cond,
    encode_lawam_multi_view_visual_cond,
)

from a2meas.behavior import CONFIG_ROOT

class WorldModelMixin:

    def _lawam_layer_uses_vlm_kv(self, layer_idx: int) -> bool:
        return lawam_layer_uses_vlm_kv(
            layer_idx,
            alternate_first=str(self.config.lawam_alternate_first),
            alternate_period=int(self.config.lawam_alternate_period),
        )

    def _lawam_alternate_kv_enabled(self) -> bool:
        if not self._lawam_enabled():
            return False
        return lawam_inject_mode_is_alternate_kv(self.config.lawam_inject_mode)

    def _lawam_enabled(self) -> bool:
        return bool(self.config.enable_lawam_stage2) and self.lawam_lam is not None

    def _init_lawam_stage2(self, *, expert_width: int, action_expert_config) -> None:
        """Load the frozen LAM and initialize latent-action query modules."""
        from a2meas.models.world_model import build_latent_action_model

        camera_order = [self.config.lawam_primary_camera_key, *self.config.lawam_multi_view_camera_keys]
        lam = build_latent_action_model(
            self.config.lawam_config_path, str(CONFIG_ROOT.parent / "shared/dinov3"), camera_order
        )
        lam.requires_grad_(False).eval()
        self.lawam_lam = lam
        vlm_hidden = self._rynnbrain_hidden_size()
        num_q = int(self.config.lawam_num_action_queries)
        self.act_query = nn.Parameter(torch.randn(num_q, vlm_hidden) * 0.02)
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden,
            lam_code_dim=int(lam.code_dim),
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=4.0,
        )
        self.lawam_prefix_adapter = LaWAMPrefixAdapter(
            vision_dim=int(lam.input_dim),
            output_dim=int(expert_width),
            num_tokens=self.config.lawam_num_prefix_tokens,
            num_heads=8,
            need_weights=bool(self.config.lawam_mha_need_weights),
        )
        decoder_need_weights = bool(self.config.lawam_mha_need_weights)
        for block in lam.decoder.dec_layers:
            block.mha_need_weights = decoder_need_weights
        prefix_view_mode = self.config.lawam_prefix_view_mode
        decoder_multi_view = bool(lam.decoder_multi_view)
        if prefix_view_mode == "all_views":
            if not decoder_multi_view:
                raise ValueError(
                    "lawam_prefix_view_mode=all_views requires a Stage-1 checkpoint trained with decoder_multi_view=true"
                )
            self.lawam_camera_embedding = nn.Parameter(
                torch.zeros(int(lam.num_view_streams), int(expert_width))
            )
            if self.config.lawam_prefix_token_budget_mode == "shared_total":
                self.lawam_vision_camera_embedding = nn.Parameter(
                    torch.zeros(int(lam.num_view_streams), int(lam.input_dim))
                )
        inject_mode = self.config.lawam_inject_mode
        if lawam_inject_mode_is_alternate_kv(inject_mode):
            self.lawam_kv_bank = LaWAMKVBank(
                num_layers=int(action_expert_config.depth),
                hidden_dim=int(expert_width),
                num_kv_heads=int(action_expert_config.num_kv_heads),
                head_dim=int(action_expert_config.head_dim),
                alternate_first=str(self.config.lawam_alternate_first),
                alternate_period=int(self.config.lawam_alternate_period),
            )
        else:
            self.lawam_kv_bank = None
        inject_ht = bool(self.config.lawam_inject_current_features)
        prefix_mode = self.config.lawam_visual_prefix_mode
        n_tok = int(self.config.lawam_num_prefix_tokens)
        prefix_view_multiplier = int(lam.num_view_streams) if prefix_view_mode == "all_views" else 1
        token_budget_mode = self.config.lawam_prefix_token_budget_mode
        if token_budget_mode == "shared_total":
            prefix_view_multiplier = 1
        if prefix_mode == "none":
            total_prefix = 0
        elif prefix_mode == "raw":
            total_prefix = -1
        else:
            total_prefix = n_tok * (2 if inject_ht else 1) * prefix_view_multiplier

    def _lawam_current_frames(self, images, img_pad_masks):
        camera_keys = list(self.config.image_features)
        primary_idx = camera_keys.index(self.config.lawam_primary_camera_key)
        auxiliary_indices = [camera_keys.index(key) for key in self.config.lawam_multi_view_camera_keys]
        current = images[primary_idx][:, -1]
        auxiliary = (
            torch.stack([images[index][:, -1] for index in auxiliary_indices], dim=1)
            if auxiliary_indices else None
        )
        valid = torch.stack(
            [~img_pad_masks[index][:, -1] for index in [primary_idx, *auxiliary_indices]], dim=1
        )
        return current, auxiliary, valid

    def _lawam_build_encoder_video(self, primary_current: Tensor) -> Tensor:
        current = prepare_lawam_rgb_batch(
            primary_current, image_resolution=int(self.config.lawam_image_resolution)
        )
        device = next(self.lawam_lam.parameters()).device
        return current.unsqueeze(1).to(device=device, non_blocking=device.type == "cuda")

    def _lawam_build_multi_view_encoder_videos(
        self, mv_current: Tensor | None, *, primary_video: Tensor
    ) -> Tensor | None:
        if mv_current is None:
            return None
        views = [
            prepare_lawam_rgb_batch(mv_current[:, index], image_resolution=int(self.config.lawam_image_resolution))
            for index in range(mv_current.shape[1])
        ]
        device = primary_video.device
        return torch.stack(views, dim=1).unsqueeze(1).to(
            device=device, non_blocking=device.type == "cuda"
        )

    def _lawam_extract_visual_features(
        self, video: Tensor, *, multi_view_videos: Tensor | None = None
    ) -> dict[str, Any]:
        lam = self.lawam_lam
        with torch.no_grad():
            if lam.decoder_multi_view:
                view_feats = lam.extract_multi_view_features(video, multi_view_videos, n=lam.latent_layer_to_use)
                return {"feats": view_feats[:, :, 0], "view_feats": view_feats}
            return {"feats": lam.extract_vision_features(video, n=lam.latent_layer_to_use)}

    def _lawam_start_encoder_prefetch(
        self,
        primary_current: Tensor,
        *,
        mv_current: Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Kick off encoder DINO(+IDM) on a side CUDA stream; returns a wait handle.

        Encoder vision depends only on RGB (not VLM), so it can run during inject.
        """
        assert self.lawam_lam is not None
        lam_device = next(self.lawam_lam.parameters()).device
        if lam_device.type != "cuda":
            return None
        video = self._lawam_build_encoder_video(primary_current)
        multi_view_videos = self._lawam_build_multi_view_encoder_videos(
            mv_current, primary_video=video
        )
        prep_done = torch.cuda.Event()
        prep_done.record()
        if getattr(self, "_lawam_encoder_stream", None) is None:
            self._lawam_encoder_stream = torch.cuda.Stream(device=lam_device)
        side: torch.cuda.Stream = self._lawam_encoder_stream
        side.wait_event(prep_done)
        box: dict[str, Any] = {"video": video}
        if multi_view_videos is not None:
            box["multi_view_videos"] = multi_view_videos
        with torch.cuda.stream(side):
            pack = self._lawam_extract_visual_features(
                video,
                multi_view_videos=multi_view_videos,
            )
            box.update(pack)
            done = torch.cuda.Event()
            done.record(side)
        box["done_event"] = done
        return box

    def _lawam_wait_encoder_prefetch(self, handle: dict[str, Any] | None) -> dict[str, Any] | None:
        if handle is None:
            return None
        stream = torch.cuda.current_stream()
        stream.wait_event(handle["done_event"])
        # The encoder creates these tensors on its side stream; the decoder owns their next use.
        for key in ("feats", "view_feats"):
            if key in handle:
                handle[key].record_stream(stream)
        return handle

    def _compute_lawam_prefix_tokens(
        self,
        zhat: Tensor,
        primary_current: Tensor,
        *,
        encoder_prefetch: dict[str, Any] | None = None,
        mv_current: Tensor | None = None,
        view_current_valid: Tensor | None = None,
    ):
        assert self.lawam_lam is not None and self.lawam_prefix_adapter is not None
        lam = self.lawam_lam
        device = zhat.device
        dtype = zhat.dtype
        lam_device = next(lam.parameters()).device
        if encoder_prefetch is not None:
            video = encoder_prefetch["video"]
            feats = encoder_prefetch["feats"]
            view_feats = encoder_prefetch.get("view_feats")
        else:
            video = self._lawam_build_encoder_video(primary_current)
            multi_view_videos = self._lawam_build_multi_view_encoder_videos(
                mv_current, primary_video=video
            )
            pack = self._lawam_extract_visual_features(video, multi_view_videos=multi_view_videos)
            feats = pack["feats"]
            view_feats = pack.get("view_feats")
        decoder_multi_view = bool(lam.decoder_multi_view)
        h_t = (view_feats[:, 0] if decoder_multi_view else feats[:, 0]).detach().clone()
        zhat_lam = zhat.to(device=lam_device, dtype=h_t.dtype)
        decoder_amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if lam_device.type == "cuda" and bool(self.config.lawam_decoder_bf16_autocast)
            else nullcontext()
        )
        with decoder_amp:
            uhat_for_prefix = decode_future_tokens(lam, h_t, zhat_lam).to(device=device, dtype=dtype).detach()
        prefix_mode = self.config.lawam_visual_prefix_mode
        inject_ht = bool(self.config.lawam_inject_current_features)
        alternate_kv = self._lawam_alternate_kv_enabled()
        lawam_tokens: Tensor | None = None
        cond_embs: Tensor | None = None
        cond_pad_mask: Tensor | None = None
        if prefix_mode != "none":
            prefix_amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and bool(self.config.lawam_prefix_bf16_autocast)
                else nullcontext()
            )
            prefix_view_mode = self.config.lawam_prefix_view_mode
            if decoder_multi_view and prefix_view_mode == "all_views":
                if self.lawam_camera_embedding is None:
                    raise RuntimeError("lawam_prefix_view_mode=all_views requires camera embedding")
                with prefix_amp:
                    (cond_embs, cond_pad_mask, _curr_len) = encode_lawam_multi_view_visual_cond(
                        self.lawam_prefix_adapter,
                        h_t=h_t.detach().to(device=device, dtype=dtype),
                        uhat=uhat_for_prefix,
                        prefix_mode=prefix_mode,
                        inject_current=inject_ht,
                        camera_embedding=self.lawam_camera_embedding,
                        current_valid=view_current_valid,
                        future_valid=view_current_valid,
                        device=device,
                        dtype=dtype,
                        token_budget_mode=str(self.config.lawam_prefix_token_budget_mode),
                        vision_camera_embedding=self.lawam_vision_camera_embedding,
                    )
            else:
                ht_for_cond = (
                    (h_t[:, 0] if decoder_multi_view else h_t).detach().to(device=device, dtype=dtype)
                )
                uhat_for_cond = uhat_for_prefix[:, 0] if decoder_multi_view else uhat_for_prefix
                with prefix_amp:
                    (cond_embs, cond_pad_mask, _curr_len) = encode_lawam_visual_cond(
                        self.lawam_prefix_adapter,
                        h_t=ht_for_cond,
                        uhat=uhat_for_cond,
                        prefix_mode=prefix_mode,
                        inject_current=inject_ht,
                        device=device,
                        dtype=dtype,
                    )
                if decoder_multi_view:
                    tokens_per_stream = int(cond_embs.shape[1]) // (2 if inject_ht else 1)
                    future_pad = view_current_valid[:, :1].expand(-1, tokens_per_stream)
                    if inject_ht:
                        current_pad = view_current_valid[:, :1].expand(-1, tokens_per_stream)
                        cond_pad_mask = torch.cat([current_pad, future_pad], dim=1)
                    else:
                        cond_pad_mask = future_pad
            if not alternate_kv:
                lawam_tokens = cond_embs
        self._lawam_cond_embs = cond_embs
        self._lawam_cond_pad_mask = cond_pad_mask
        return lawam_tokens

    def embed_prefix(
        self,
        images,
        *,
        img_pad_masks=None,
        raw_texts=None,
        state_history=None,
        short_term_memory_frame_age_seconds=None,
    ):
        current, mv_current, view_current_valid = self._lawam_current_frames(images, img_pad_masks)
        prefetch = self._lawam_start_encoder_prefetch(current, mv_current=mv_current)
        (inputs, pad) = self._prepare_rynnbrain_inputs(
            images,
            img_pad_masks=img_pad_masks,
            raw_texts=raw_texts,
            short_term_memory_frame_age_seconds=short_term_memory_frame_age_seconds,
        )
        trajectory_query_len = 0
        la_query = self.act_query
        if self.trajectory_2d_tokens is not None:
            trajectory_query_len = int(self.trajectory_2d_tokens.shape[0])
            la_query = torch.cat((la_query, self.trajectory_2d_tokens), dim=0)
        (hidden, pad, att, result) = self._run_feature_inject(
            inputs, pad, la_query=la_query
        )
        la_hidden = extract_la_hidden_from_inject_result(result)
        num_action_queries = int(self.config.lawam_num_action_queries)
        if la_hidden is None or la_hidden.shape[1] != num_action_queries + trajectory_query_len:
            raise RuntimeError("LaWAM action-query hidden length mismatch")
        h_act = la_hidden[:, :num_action_queries]
        zhat = self.vlm_to_lam(h_act)
        lawam_tokens = self._compute_lawam_prefix_tokens(
            zhat,
            current,
            encoder_prefetch=self._lawam_wait_encoder_prefetch(prefetch),
            mv_current=mv_current,
            view_current_valid=view_current_valid,
        )
        prefix = self.paligemma_with_expert.align_proj(
            hidden.to(dtype=self.paligemma_with_expert.align_proj.weight.dtype)
        )
        prefix, pad, att = append_prefix_tokens(
            prefix, pad, att, lawam_tokens, token_pad_mask=self._lawam_cond_pad_mask,
        )
        (prefix, pad, att) = self._append_state_tokens_to_prefix(prefix, pad, att, state_history)
        return (prefix, pad, att)
