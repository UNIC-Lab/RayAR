from __future__ import annotations

import math

import torch
from torch import nn

from src.config import ModelConfig


class PropagationRankHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = max(cfg.hidden_dim, 64)
        in_dim = cfg.hidden_dim * 4 + 1
        self.delta = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.delta[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        candidate: torch.Tensor,
        goal_los: torch.Tensor,
    ) -> torch.Tensor:
        query = query.expand_as(candidate)
        dot = torch.sum(query * candidate, dim=-1) / math.sqrt(candidate.shape[-1])
        pair = torch.cat(
            [
                query,
                candidate,
                query * candidate,
                torch.abs(query - candidate),
                goal_los.to(candidate.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        return dot + self.delta(pair).squeeze(-1)
