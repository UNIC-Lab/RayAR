from __future__ import annotations

import torch


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (a != b),
                )
            )
        previous = current
    return previous[-1]


def next_node_hits(logits, positive_mask):
    top1 = torch.topk(logits, k=1, dim=1).indices
    top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
    hit1 = positive_mask.gather(1, top1).any(dim=1).sum().item()
    hit5 = positive_mask.gather(1, top5).any(dim=1).sum().item()
    return int(hit1), int(hit5), int(logits.shape[0])


__all__ = ["edit_distance", "next_node_hits"]
