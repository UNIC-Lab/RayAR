from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.config import ModelConfig


class ReverseLOSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("LOS graph hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, heads),
        )
        self.edge_value = nn.Linear(edge_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = value.shape
        return value.view(batch, nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def _edge_terms(self, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.edge_bias(edge).permute(2, 0, 1).unsqueeze(0)
        value = (
            self.edge_value(edge)
            .view(edge.shape[0], edge.shape[1], self.heads, self.head_dim)
            .permute(2, 0, 1, 3)
        )
        return bias, value

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        query = self._heads(self.query(node))
        key = self._heads(self.key(node))
        value = self._heads(self.value(node))
        edge_bias, edge_value = self._edge_terms(edge)
        score = torch.einsum("bhid,bhjd->bhij", query, key)
        score = score / math.sqrt(self.head_dim) + edge_bias
        mask = adjacency.bool().unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        weight = F.softmax(score, dim=-1)
        pooled = torch.einsum("bhij,bhjd->bhid", weight, value)
        pooled = pooled + torch.einsum("bhij,hijd->bhid", weight, edge_value)
        pooled = pooled.permute(0, 2, 1, 3).flatten(2)
        node = self.norm1(node + self.dropout(self.output(pooled)))
        return self.norm2(node + self.dropout(self.ffn(node)))


class LOSGraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        node_dim = cfg.feature_dim + 7 + 6
        hidden = cfg.los_graph_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            ReverseLOSBlock(
                hidden,
                cfg.los_graph_heads,
                5,
                cfg.dropout,
            )
            for _ in range(cfg.los_graph_layers)
        )
        self.output = nn.Linear(hidden, cfg.hidden_dim)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        adjacency: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        node = self.node_encoder(node)
        for layer in self.layers:
            node = layer(node, edge, adjacency)
        rows = torch.arange(node.shape[0], device=node.device).unsqueeze(1)
        candidate = node[rows, candidate_ids.long()]
        return self.output(candidate)
