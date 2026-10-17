from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.config import ModelConfig
from src.model.layers import PATH_STATE_DIM, RayEncoder, TransformerStack, _mlp


HOP_OFFSET = 11


def _predicted_los_features(
    path_state: torch.Tensor,
    cur_los: torch.Tensor,
    goal_los: torch.Tensor,
) -> torch.Tensor:
    """Return model-predicted LOS features; no label or mask is consumed."""
    nonroot = 1.0 - path_state[:, HOP_OFFSET : HOP_OFFSET + 1]
    cur = nonroot * torch.tanh(cur_los.detach() / 2.0)
    goal = torch.tanh(goal_los.detach() / 2.0)
    return torch.stack([cur, goal], dim=-1)


class DynamicTopologyBias(nn.Module):
    """Candidate topology plus query-relative geometry attention bias."""

    def __init__(self, heads: int, state_dim: int):
        super().__init__()
        self.heads = heads
        self.same_building = nn.Parameter(torch.full((heads,), 0.05))
        self.polygon_neighbor = nn.Parameter(torch.full((heads,), 0.10))
        self.distance_scale = nn.Parameter(torch.full((heads,), -3.0))
        self.normal_alignment = nn.Parameter(torch.full((heads,), 0.02))

        hidden = max(32, heads * 4)
        self.query_bias = nn.Sequential(
            nn.Linear(state_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, heads * 2),
        )
        nn.init.zeros_(self.query_bias[-1].weight)
        nn.init.zeros_(self.query_bias[-1].bias)

    def forward(
        self,
        vertex: torch.Tensor,
        state: torch.Tensor,
        path_state: torch.Tensor,
        building: torch.Tensor,
        neighbor: torch.Tensor,
        cur_los: torch.Tensor,
        goal_los: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, count, _ = vertex.shape
        valid_building = building.ge(0)
        same = building.unsqueeze(2).eq(building.unsqueeze(1))
        same = same & valid_building.unsqueeze(2) & valid_building.unsqueeze(1)

        vertex_id = torch.arange(count, device=vertex.device).view(1, 1, count, 1)
        adjacent = neighbor.unsqueeze(2).eq(vertex_id).any(dim=-1)
        adjacent = adjacent | adjacent.transpose(1, 2)

        position = vertex[..., :3]
        distance = torch.cdist(position, position)
        normal = F.normalize(vertex[..., 4:6], dim=-1, eps=1e-6)
        normal_dot = torch.matmul(normal, normal.transpose(1, 2))

        candidate_bias = (
            same.unsqueeze(1) * self.same_building.view(1, self.heads, 1, 1)
            + adjacent.unsqueeze(1) * self.polygon_neighbor.view(1, self.heads, 1, 1)
            - distance.unsqueeze(1)
            * F.softplus(self.distance_scale).view(1, self.heads, 1, 1)
            + normal_dot.unsqueeze(1) * self.normal_alignment.view(1, self.heads, 1, 1)
        )
        full_bias = F.pad(candidate_bias, (1, 0, 1, 0))

        los = _predicted_los_features(path_state, cur_los, goal_los)
        relation = self.query_bias(torch.cat([state, los], dim=-1))
        relation = relation.view(batch, count, 2, self.heads)
        query_to_candidate = relation[:, :, 0].permute(0, 2, 1)
        candidate_to_query = relation[:, :, 1].permute(0, 2, 1)
        full_bias[:, :, 0, 1:] = query_to_candidate
        full_bias[:, :, 1:, 0] = candidate_to_query
        return full_bias.to(dtype=dtype).reshape(
            batch * self.heads, count + 1, count + 1
        )


class HybridRankHead(nn.Module):
    """Refine the global dot pointer with a lightweight nonlinear ranker."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = max(cfg.hidden_dim // 2, 64)
        in_dim = cfg.hidden_dim * 4 + cfg.state_dim + 2
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
        state: torch.Tensor,
        path_state: torch.Tensor,
        cur_los: torch.Tensor,
        goal_los: torch.Tensor,
    ) -> torch.Tensor:
        query = query.expand_as(candidate)
        dot = torch.sum(query * candidate, dim=-1) / math.sqrt(candidate.shape[-1])
        los = _predicted_los_features(path_state, cur_los, goal_los)
        pair = torch.cat(
            [
                query,
                candidate,
                query * candidate,
                torch.abs(query - candidate),
                state,
                los,
            ],
            dim=-1,
        )
        return dot + self.delta(pair).squeeze(-1)


class PathBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        self.ray_encoder = RayEncoder(cfg)
        pair_dim = cfg.hidden_dim * 4
        self.ray_gate = nn.Linear(pair_dim, cfg.hidden_dim)
        self.ray_mix = _mlp(pair_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.geom_encoder = _mlp(
            cfg.feature_dim + cfg.state_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.dropout,
        )
        self.visited_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.candidate_norm = nn.LayerNorm(cfg.hidden_dim)

        self.query = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.query_encoder = _mlp(
            PATH_STATE_DIM, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout
        )
        self.pointer = TransformerStack(cfg, cfg.pointer_layers)
        self.topology_bias = DynamicTopologyBias(cfg.heads, cfg.state_dim)
        self.query_norm = nn.LayerNorm(cfg.hidden_dim)
        self.pointer_candidate_norm = nn.LayerNorm(cfg.hidden_dim)
        self.rank_head = HybridRankHead(cfg)

        self.cur_los_head = nn.Linear(cfg.hidden_dim, 1)
        self.goal_los_head = nn.Linear(cfg.hidden_dim, 1)

    def _encode_rays(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self.ray_encoder(
            batch["cur_ray"],
            batch["cur_origin"],
            batch["cur_target"],
            batch["cur_wall"],
            batch["cur_edge"],
            batch["cur_wall_mask"],
            RayEncoder.CUR_ROLE,
        )
        goal = self.ray_encoder(
            batch["goal_ray"],
            batch["goal_origin"],
            batch["goal_target"],
            batch["goal_wall"],
            batch["goal_edge"],
            batch["goal_wall_mask"],
            RayEncoder.GOAL_ROLE,
        )
        return cur, goal

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        with_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        output = self.forward_with_aux(batch)
        return output if with_aux else output["pointer"]


__all__ = ["PathBackbone"]
