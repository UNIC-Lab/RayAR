from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PathPhysicalBias(nn.Module):
    FEATURE_DIM = 9

    def __init__(self, hidden_dim: int):
        super().__init__()
        inner = max(hidden_dim // 2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        vertex_xyz_m: torch.Tensor,
        prefix_ids: torch.Tensor,
        tx_xyz_m: torch.Tensor,
        rx_xyz_m: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = prefix_ids.shape
        safe_ids = prefix_ids.clamp(min=0)
        path = torch.gather(
            vertex_xyz_m,
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, 3),
        )
        previous = torch.cat([tx_xyz_m.unsqueeze(1), path[:, :-1]], dim=1)
        incoming = path - previous
        segment = torch.linalg.vector_norm(incoming, dim=-1)
        accumulated = torch.cumsum(segment, dim=1)
        incoming_xy = F.normalize(incoming[..., :2], dim=-1, eps=1e-6)

        following = torch.cat([path[:, 1:], torch.zeros_like(path[:, :1])], dim=1)
        has_following = torch.zeros(batch, length, dtype=torch.bool, device=path.device)
        has_following[:, :-1] = prefix_ids[:, 1:].ge(0)
        outgoing_xy = F.normalize(following[..., :2] - path[..., :2], dim=-1, eps=1e-6)
        outgoing_xy = outgoing_xy * has_following.unsqueeze(-1)

        horizontal = torch.linalg.vector_norm(incoming[..., :2], dim=-1)
        elevation = torch.atan2(incoming[..., 2], horizontal.clamp_min(1e-6))
        to_rx = torch.linalg.vector_norm(rx_xyz_m.unsqueeze(1) - path, dim=-1)
        log_scale = math.log1p(256.0)
        features = torch.cat(
            [
                (torch.log1p(segment) / log_scale).unsqueeze(-1),
                (torch.log1p(accumulated) / log_scale).unsqueeze(-1),
                incoming_xy,
                outgoing_xy,
                torch.sin(elevation).unsqueeze(-1),
                torch.cos(elevation).unsqueeze(-1),
                (torch.log1p(to_rx) / log_scale).unsqueeze(-1),
            ],
            dim=-1,
        )
        valid = prefix_ids.ge(0).unsqueeze(-1).to(features.dtype)
        bias = self.mlp(features) * valid
        start = torch.zeros(
            batch, 1, bias.shape[-1], dtype=bias.dtype, device=bias.device
        )
        return torch.cat([start, bias], dim=1)
