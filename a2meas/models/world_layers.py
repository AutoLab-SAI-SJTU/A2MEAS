"""LaWAM decoder layers and checkpoint parameter definitions."""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn


class Fixed2DPositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, H: int, W: int, uniform_power: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.H, self.W = (H, W)
        self.uniform_power = uniform_power
        if not uniform_power:
            h_dim = embed_dim // 2
            w_dim = embed_dim - h_dim
        else:
            h_dim = w_dim = int(math.ceil(embed_dim / 4) * 2)
        self.register_buffer("pe_h", self._build_1d_pos_embed(h_dim, H), persistent=False)
        self.register_buffer("pe_w", self._build_1d_pos_embed(w_dim, W), persistent=False)
        pe_h_expand = self.pe_h[:, None, :]
        pe_w_expand = self.pe_w[None, :, :]
        pe_h_expand = nn.functional.pad(pe_h_expand, (0, embed_dim - h_dim))
        pe_w_expand = nn.functional.pad(pe_w_expand, (0, embed_dim - w_dim))
        self.register_buffer("pos_embed", pe_h_expand + pe_w_expand, persistent=False)

    def _build_1d_pos_embed(self, dim: int, length: int):
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            x = x.view(x.shape[0], self.H, self.W, self.embed_dim)
        x = x + self.pos_embed.to(x.device)
        return x.reshape(x.shape[0], -1, self.embed_dim)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """AdaLN-Zero attention and feed-forward modulation."""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_need_weights = True
        ffn_dim = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, D] visual tokens
            c: [B, D]    action conditioning vector
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=-1
        )
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=self.mha_need_weights)
        x = x + gate_msa * attn_out
        normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(normed)
        return x


class LAMDecoder(nn.Module):

    def __init__(
        self,
        context_dim,
        input_dim: int = 1024,
        num_queries: int = 1,
        num_layers=6,
        num_heads=16,
        dropout=0.1,
        grid_hw: Tuple[int, int] = (16, 16),
        ffn_expansion_factor=2,
        num_embodiments: int = 32,
        code_dim: Optional[int] = None,
        last_ln: bool = True,
    ):
        super().__init__()
        self.feature_dim = context_dim
        self.input_dim = input_dim
        self.grid_height = int(grid_hw[0])
        self.grid_width = int(grid_hw[1])
        self.num_queries = num_queries
        self.use_last_ln = bool(last_ln)
        self.dec_layers = nn.ModuleList(
            [
                AdaLNBlock(context_dim, num_heads, ffn_expansion_factor=ffn_expansion_factor, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.pos_embed = Fixed2DPositionalEncoding(context_dim, self.grid_height, self.grid_width)
        self.last_ln = nn.LayerNorm(input_dim) if self.use_last_ln else nn.Identity()
        if input_dim == context_dim:
            self.project_input = nn.Identity()
            self.project_output = nn.Identity()
        else:
            self.project_input = nn.Linear(input_dim, context_dim)
            self.project_output = nn.Linear(context_dim, input_dim)
        if code_dim is not None and code_dim != context_dim:
            self.action_in_proj = nn.Linear(code_dim, context_dim)
        else:
            self.action_in_proj = nn.Identity()

    def forward(self, features, actions):
        c = self.action_in_proj(actions)
        if c.dim() == 3:
            c = c.squeeze(1)
        features_tokens = self.project_input(features)
        if features_tokens.dim() == 4:
            features_tokens = features_tokens.squeeze(1)
        expected_tokens = int(self.grid_height * self.grid_width)
        if features_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Decoder token mismatch: got K={features_tokens.shape[1]}, expected K={expected_tokens} for grid_hw=({self.grid_height},{self.grid_width})."
            )
        x = self.pos_embed(features_tokens)
        for layer in self.dec_layers:
            x = layer(x, c)
        reconstructed_features = self.project_output(x)
        reconstructed_features = self.last_ln(reconstructed_features)
        return reconstructed_features.unsqueeze(1)


class MultiViewLAMDecoder(LAMDecoder):
    """Shared LAM decoder applied independently to an explicit camera axis.

    All inherited parameter names and shapes are intentionally identical to
    :class:`LAMDecoder`.  The only new checkpoint tensor is the zero-init
    ``view_embedding`` used to identify each camera in the action condition.
    """

    def __init__(self, *args, num_views: int, **kwargs):
        if int(num_views) < 2:
            raise ValueError(f"MultiViewLAMDecoder requires num_views >= 2, got {num_views}")
        super().__init__(*args, **kwargs)
        self.num_views = int(num_views)
        self.view_embedding = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,1,V,K,D]`` features with one shared decoder."""
        if features.dim() != 5 or int(features.shape[1]) != 1:
            raise ValueError(f"MultiViewLAMDecoder expects features [B,1,V,K,D], got {tuple(features.shape)}")
        bsz, _, num_views, num_tokens, feature_dim = features.shape
        if int(num_views) != self.num_views:
            raise ValueError(f"Decoder view mismatch: input V={num_views}, configured V={self.num_views}")
        if int(feature_dim) != self.input_dim:
            raise ValueError(
                f"Decoder feature dim mismatch: input D={feature_dim}, expected {self.input_dim}"
            )
        c = self.action_in_proj(actions)
        if c.dim() == 3:
            if int(c.shape[1]) != 1:
                raise ValueError(f"Expected one latent action query, got {tuple(c.shape)}")
            c = c.squeeze(1)
        if c.dim() != 2 or int(c.shape[0]) != int(bsz):
            raise ValueError(f"Expected actions [B,C] or [B,1,C], got {tuple(actions.shape)}")
        c = c[:, None, :] + self.view_embedding[None, :, :]
        c = c.reshape(int(bsz) * self.num_views, self.feature_dim)
        tokens = features[:, 0].reshape(int(bsz) * self.num_views, int(num_tokens), int(feature_dim))
        x = self.project_input(tokens)
        expected_tokens = int(self.grid_height * self.grid_width)
        if int(x.shape[1]) != expected_tokens:
            raise ValueError(
                f"Decoder token mismatch: got K={x.shape[1]}, expected K={expected_tokens} for grid_hw=({self.grid_height},{self.grid_width})."
            )
        x = self.pos_embed(x)
        for layer in self.dec_layers:
            x = layer(x, c)
        reconstructed = self.last_ln(self.project_output(x))
        return reconstructed.reshape(int(bsz), 1, self.num_views, int(num_tokens), int(feature_dim))


