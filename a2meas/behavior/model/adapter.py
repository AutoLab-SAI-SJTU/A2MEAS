"""Hidden-state conditioning after the ninth BEHAVIOR action-expert layer."""

import json

import numpy as np
import torch
from torch import nn


TEXT_LENGTH = 2048


def condition_arrays(task, context, history, history_mask, base):
    semantic = json.dumps(
        {
            "task": task,
            "instruction": context["instruction"],
            "direction": context["direction"],
            "grounding": context["grounding"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = semantic.encode("utf8")
    if len(encoded) > TEXT_LENGTH:
        raise ValueError("CorrectionContext exceeds frozen encoder capacity")
    tokens = np.zeros(TEXT_LENGTH, np.int64)
    tokens[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int64) + 1
    return dict(
        text_tokens=tokens,
        history=np.asarray(history, np.float32),
        history_mask=np.asarray(history_mask, bool),
        base_actions=np.asarray(base, np.float32),
    )


class HiddenAdapter(nn.Module):
    def __init__(self, width=256, hidden_dim=1024):
        super().__init__()
        self.width, self.hidden_dim = width, hidden_dim
        self.text_embedding = nn.Embedding(257, width, padding_idx=0)
        self.text_conv = nn.Sequential(
            nn.Conv1d(width, width, 5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=4, padding=2),
            nn.GELU(),
        )
        self.history_encoder = nn.GRU(46, width, batch_first=True)
        self.candidate_encoder = nn.GRU(23, width, batch_first=True)
        self.condition = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim + width, width)
        self.up = nn.Linear(width, hidden_dim)
        self.alpha = nn.Parameter(torch.full((hidden_dim,), 0.1))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.register_buffer("history_mean", torch.zeros(46))
        self.register_buffer("history_std", torch.ones(46))
        self.register_buffer("action_mean", torch.zeros(23))
        self.register_buffer("action_std", torch.ones(23))

    def encode(self, b):
        text = self.text_embedding(b["text_tokens"]).transpose(1, 2)
        text = self.text_conv(text).mean(-1)
        h = (b["history"].float() - self.history_mean) / self.history_std
        h = h * b["history_mask"][..., None]
        hs, _ = self.history_encoder(h)
        lengths = b["history_mask"].sum(-1)
        history = hs[torch.arange(len(h), device=h.device), (lengths - 1).clamp_min(0)]
        history = history * (lengths > 0)[:, None]
        _, candidate = self.candidate_encoder(
            (b["base_actions"].float() - self.action_mean) / self.action_std
        )
        return self.condition(torch.cat([text, history, candidate[-1]], -1))

    def inject(self, hidden, code):
        residual = self.up(
            torch.nn.functional.silu(
                self.down(
                    torch.cat(
                        [
                            self.hidden_norm(hidden.float()),
                            code[:, None, :].expand(-1, hidden.shape[1], -1),
                        ],
                        -1,
                    )
                )
            )
        )
        return hidden + (self.alpha * residual).to(hidden.dtype)

    def forward(self, batch):
        return self.encode(batch)
