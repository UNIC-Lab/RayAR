from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig


class CandidateAttentionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.hidden_dim)
        self.attention = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
        )
        nn.init.normal_(self.attention.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.normal_(self.ffn[-1].weight, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(candidate)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        candidate = candidate + attended
        candidate = candidate + self.ffn(self.ffn_norm(candidate))
        return candidate * valid.unsqueeze(-1).to(candidate.dtype)


class CandidateAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            CandidateAttentionBlock(cfg) for _ in range(cfg.candidate_layers)
        )

    def forward(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            candidate = layer(candidate, valid)
        return candidate
