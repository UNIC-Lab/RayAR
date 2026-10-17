from __future__ import annotations

import torch
from torch.nn import functional as F


def dominant_path_loss(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    positive = positive.bool()
    if not bool(positive.sum(dim=1).eq(1).all().item()):
        raise ValueError("each sparse state must contain exactly one dominant target")
    target = positive.to(dtype=torch.int64).argmax(dim=1)
    rows = torch.arange(target.shape[0], device=target.device)
    if not bool(legal.bool()[rows, target].all().item()):
        raise ValueError("dominant next hop is not LOS-legal")
    return F.cross_entropy(logits, target, reduction="none")


def weighted_state_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(values.dtype)
    return torch.sum(values * weights) / weights.sum().clamp_min(1e-6)


__all__ = [
    "dominant_path_loss",
    "weighted_state_mean",
]
