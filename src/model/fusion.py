from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers import _mlp
from src.data.los import LOS_CONTEXT_DIM


class GroupedFeatureFusion(nn.Module):
    BRANCH_DIMS = (None, None, 19, 25, LOS_CONTEXT_DIM, None)

    def __init__(
        self,
        hidden_dim: int,
        branch_dim: int,
        gate_groups: int,
        dropout: float,
    ):
        super().__init__()
        if branch_dim % gate_groups:
            raise ValueError("branch_dim must be divisible by gate_groups")
        self.hidden_dim = hidden_dim
        self.branch_dim = branch_dim
        self.gate_groups = gate_groups
        input_dims = tuple(
            hidden_dim if dim is None else dim for dim in self.BRANCH_DIMS
        )
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, branch_dim),
                nn.SiLU(),
                nn.LayerNorm(branch_dim),
            )
            for input_dim in input_dims
        )
        self.type_embedding = nn.Parameter(torch.empty(len(input_dims), branch_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.query_projection = nn.Linear(hidden_dim, branch_dim)
        self.gate_norm = nn.LayerNorm(branch_dim)
        self.gate = nn.Linear(branch_dim, gate_groups)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        self.mix = nn.Linear(len(input_dims) * branch_dim, hidden_dim)
        self.identity = _mlp(44, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        current_ray: torch.Tensor,
        goal_ray: torch.Tensor,
        vertex: torch.Tensor,
        route: torch.Tensor,
        los: torch.Tensor,
        future: torch.Tensor,
        query: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        values = (current_ray, goal_ray, vertex, route, los, future)
        branches = torch.stack(
            [encoder(value) for encoder, value in zip(self.projections, values)],
            dim=2,
        )
        branches = branches + self.type_embedding.view(
            1, 1, len(values), self.branch_dim
        )
        query_h = self.query_projection(query.squeeze(1))[:, None, None, :]
        gates = torch.sigmoid(self.gate(F.silu(self.gate_norm(branches + query_h))))
        group_width = self.branch_dim // self.gate_groups
        gated = branches.view(*branches.shape[:-1], self.gate_groups, group_width)
        gated = (gated * gates.unsqueeze(-1)).flatten(3)
        fused = self.mix(gated.flatten(2))
        identity = self.identity(torch.cat([vertex, route], dim=-1))
        output = self.output_norm(identity + fused)
        return output * valid.unsqueeze(-1).to(output.dtype)
