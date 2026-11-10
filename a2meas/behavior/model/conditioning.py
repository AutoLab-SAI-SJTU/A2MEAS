"""Image-independent task conditioning for the A2WLA action expert."""

from typing import TypedDict
import torch
from torch import Tensor, nn

GLOBAL_TASK_TEXT = "global_task_text"
GLOBAL_TASK_ID = "global_task_id"


class GlobalTaskInputs(TypedDict, total=False):
    texts: list[str]
    ids: Tensor


def clean_task_text(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def validate_task_mapping(mapping: dict[str, int], num_tasks: int) -> dict[str, int]:
    """Normalize labels without silently merging conflicting task identities."""
    result = {}
    for text, task_id in mapping.items():
        label = clean_task_text(text)
        if not label or isinstance(task_id, bool) or (not isinstance(task_id, int)):
            raise ValueError("Global task mapping requires nonempty text and integer IDs")
        if not 0 <= task_id < num_tasks:
            raise ValueError(f"Global task ID {task_id} is outside [0, {num_tasks})")
        if label in result and result[label] != task_id:
            raise ValueError(f"Conflicting global task IDs for {label!r}")
        result[label] = task_id
    if not result:
        raise ValueError("Global task mapping must not be empty")
    return result


class GlobalTaskConditioner(nn.Module):
    """Fuse goal language and task ID into an initially zero AdaRMS residual."""

    def __init__(
        self,
        language_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_tasks: int,
        mode: str = "language_task_id",
    ):
        super().__init__()
        if mode not in {"language", "task_id", "language_task_id"}:
            raise ValueError(f"Invalid global task conditioning mode: {mode}")
        self.use_language = mode != "task_id"
        self.use_task_id = mode != "language"
        self.num_tasks = num_tasks
        self.language_proj = (
            nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, hidden_dim), nn.SiLU())
            if self.use_language
            else None
        )
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim) if self.use_task_id else None
        self.id_norm = nn.LayerNorm(hidden_dim) if self.use_task_id else None
        fused_dim = hidden_dim * (int(self.use_language) + int(self.use_task_id))
        self.fusion = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.SiLU())
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, language: Tensor | None, mask: Tensor | None, task_ids: Tensor | None) -> Tensor:
        parts = []
        weight = self.output_proj.weight
        if self.use_language:
            if language is None or mask is None or language.ndim != 3 or (mask.shape != language.shape[:2]):
                raise ValueError("Global language requires embeddings [B,L,D] and mask [B,L]")
            mask = mask.to(device=language.device, dtype=torch.bool)
            if not bool(mask.any(dim=1).all()):
                raise ValueError("Global language contains an empty task")
            pooled = language.float().masked_fill(~mask[..., None], 0).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True)
            parts.append(self.language_proj(pooled.to(device=weight.device, dtype=weight.dtype)))
        if self.use_task_id:
            if task_ids is None or task_ids.ndim != 1 or task_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Global task IDs must be an integer tensor [B]")
            if bool(((task_ids < 0) | (task_ids >= self.num_tasks)).any()):
                raise ValueError(f"Global task IDs must be in [0, {self.num_tasks})")
            ids = task_ids.to(device=weight.device, dtype=torch.long)
            if parts and parts[0].shape[0] != ids.shape[0]:
                raise ValueError("Global language and task ID batch sizes differ")
            parts.append(self.id_norm(self.task_embedding(ids)))
        return self.output_proj(self.fusion(torch.cat(parts, dim=-1)))
