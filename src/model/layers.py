from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.config import ModelConfig


PATH_STATE_DIM = 16


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _transformer_layer(cfg: ModelConfig) -> nn.TransformerEncoderLayer:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.hidden_dim,
        nhead=cfg.heads,
        dim_feedforward=cfg.hidden_dim * 4,
        dropout=cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    for name, parameter in layer.named_parameters():
        if parameter.ndim > 1:
            nn.init.xavier_uniform_(parameter)
        elif name.endswith("weight"):
            nn.init.ones_(parameter)
        else:
            nn.init.zeros_(parameter)
    return layer


class TransformerStack(nn.Module):
    """Transformer layers initialized independently instead of cloned."""

    def __init__(self, cfg: ModelConfig, count: int):
        super().__init__()
        self.layers = nn.ModuleList([_transformer_layer(cfg) for _ in range(count)])

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, src_mask=mask, src_key_padding_mask=padding_mask)
        return tokens


class RayEncoder(nn.Module):
    CUR_ROLE = 0
    GOAL_ROLE = 1

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ray_token = _mlp(
            cfg.ray_dim + cfg.endpoint_dim + cfg.feature_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.dropout,
        )
        self.wall_token = _mlp(
            cfg.wall_dim + cfg.ray_wall_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.dropout,
        )
        self.role = nn.Embedding(2, cfg.hidden_dim)
        self.encoder = TransformerStack(cfg, cfg.ray_layers)
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        ray: torch.Tensor,
        origin: torch.Tensor,
        target: torch.Tensor,
        wall: torch.Tensor,
        edge: torch.Tensor,
        wall_mask: torch.Tensor,
        role: int,
    ) -> torch.Tensor:
        batch, count, _ = ray.shape
        wall_count = wall.shape[2]
        ray_h = self.ray_token(torch.cat([ray, origin, target], dim=-1)).reshape(
            batch * count, 1, -1
        )
        wall_h = self.wall_token(torch.cat([wall, edge], dim=-1)).reshape(
            batch * count, wall_count, -1
        )
        tokens = torch.cat([ray_h, wall_h], dim=1)

        role_id = torch.full(
            (batch * count,), role, dtype=torch.long, device=ray.device
        )
        tokens = tokens + self.role(role_id).unsqueeze(1)
        padding_mask = torch.cat(
            [
                torch.zeros((batch * count, 1), dtype=torch.bool, device=ray.device),
                wall_mask.reshape(batch * count, wall_count),
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, padding_mask=padding_mask)
        return self.final_norm(encoded[:, 0]).reshape(batch, count, -1)


def numpy_batch_to_torch(
    features: dict, device: torch.device
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        tensor = torch.from_numpy(value)
        if key in {"building", "neighbor"}:
            tensor = tensor.long()
        elif np.issubdtype(value.dtype, np.bool_):
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        out[key] = tensor.to(device, non_blocking=True)
    return out
