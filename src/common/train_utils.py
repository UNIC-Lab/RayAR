from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os

import torch


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def init_dist(device_name):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        return DistInfo(False, 0, 0, 1, device)
    if device_name is not None:
        raise ValueError("torchrun 多卡训练时不要传 --gpu 或 --device")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    return DistInfo(
        True,
        rank,
        local_rank,
        world_size,
        torch.device("cuda", local_rank),
    )


def split_batches(batches, rank, world_size):
    if world_size <= 1:
        return batches
    steps = math.ceil(len(batches) / world_size)
    total = steps * world_size
    padded = batches + batches[: total - len(batches)]
    return padded[rank:total:world_size]


def balance_items(items, costs, world_size):
    parts = [[] for _ in range(world_size)]
    loads = [0] * world_size
    order = sorted(range(len(items)), key=lambda index: (-costs[index], index))
    for index in order:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        parts[rank].append((index, items[index]))
        loads[rank] += costs[index]
    return [[item for _, item in sorted(part)] for part in parts]


def chunks(iterator, size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_scheduler(optimizer, train_cfg, total_optimizer_steps):
    if train_cfg.scheduler == "none":
        return None
    if train_cfg.scheduler != "linear":
        raise ValueError(f"unsupported scheduler: {train_cfg.scheduler}")
    min_factor = min(
        max(float(train_cfg.min_lr) / max(float(train_cfg.lr), 1e-12), 0.0),
        1.0,
    )
    total_steps = max(int(total_optimizer_steps), 1)

    def lr_factor(step_index):
        progress = min(max(float(step_index), 0.0) / float(total_steps), 1.0)
        return (1.0 - progress) + min_factor * progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def map_vertices(cfg, map_id):
    path = cfg.data.cache_root / f"map_{map_id}" / "meta.json"
    return int(json.loads(path.read_text(encoding="utf-8"))["n_vertex"])


def candidate_cost(cfg, vertices, samples=1):
    ray_tokens = 2 * (cfg.model.max_wall_tokens + 1)
    return samples * vertices * (vertices + ray_tokens)


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


__all__ = [
    "DistInfo",
    "balance_items",
    "build_scheduler",
    "candidate_cost",
    "chunks",
    "init_dist",
    "map_vertices",
    "split_batches",
    "to_device",
]
