from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from src.config import experiment_maps, AppConfig, ModelConfig
from src.data.cache import load_map_cache
from src.common.train_utils import (
    balance_items as _balance_items,
    build_scheduler,
    candidate_cost as _candidate_cost,
    chunks as _chunks,
    init_dist as _init_dist,
    map_vertices as _map_vertices,
    split_batches as _split_batches,
    to_device as _to_device,
)
from src.data.features import PAFeatureBuilder, collate_sparse_features
from src.data.los import SourceLOS, load_source_los
from src.eval.predict import AutoregressivePredictor
from src.data.dataset import (
    dominant_chain,
    make_sample,
    split_slots,
    state_samples,
    valid_slots,
)
from src.data.overfit import select_pools
from src.model.network import IRTPathModel, MAX_PREFIX_LEN
from src.train.loss import dominant_path_loss, weighted_state_mean


_POOL_CACHE: dict[tuple, tuple[tuple[int, ...], ...]] = {}
_OVERFIT_BATCH_CACHE: dict[tuple, tuple[tuple[int, tuple], ...]] = {}
_SOURCE_CACHE: dict[tuple[str, int], SourceLOS] = {}


def _source_los(cfg: AppConfig, cache) -> SourceLOS:
    key = (str(cfg.data.los_root), int(cache.map_id))
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_source_los(
            cfg.data.los_root,
            cache.map_id,
            cache.tx_xyz,
            np.asarray(cache.vertex_xyz, dtype=np.float32)[:, :2],
        )
    return _SOURCE_CACHE[key]


def _vertex_decision_items(source, cache, items):
    return [
        item
        for item in items
        if not source.direct(cache.tx_xyz[item[0]], cache.rx_xyz[item[0], item[1]])
    ]


def _map_pools(
    cfg: AppConfig,
    map_id: int,
    split: str,
) -> tuple[tuple[int, ...], ...]:
    if cfg.overfit.enabled:
        if map_id != cfg.overfit.map_id:
            raise ValueError(f"overfit map is {cfg.overfit.map_id}, got {map_id}")
        key = (
            str(cfg.data.cache_root),
            map_id,
            "overfit",
            cfg.overfit.tx_count,
            cfg.overfit.rx_per_tx,
            cfg.train.seed,
        )
        if key not in _POOL_CACHE:
            cache = load_map_cache(cfg.data.cache_root, map_id)
            _POOL_CACHE[key] = select_pools(
                cache,
                cfg.overfit.tx_count,
                cfg.overfit.rx_per_tx,
                cfg.train.seed,
            )
        return _POOL_CACHE[key]

    key = (
        str(cfg.data.cache_root),
        map_id,
        split,
        cfg.experiment.split_mode,
        cfg.experiment.train_ratio,
        cfg.train.seed,
    )
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]

    cache = load_map_cache(cfg.data.cache_root, map_id)
    pools = []
    for tx_row in range(cache.n_tx):
        if cfg.experiment.split_mode == "within_map":
            train, val = split_slots(
                cache,
                tx_row,
                cfg.experiment.train_ratio,
                cfg.train.seed,
            )
            slots = train if split == "train" else val
        else:
            slots = tuple(range(int(cache.rx_id.shape[1])))
        pools.append(valid_slots(cache, tx_row, slots))
    result = tuple(pools)
    _POOL_CACHE[key] = result
    return result


class FeatureBatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: AppConfig,
        batches: list[tuple[int, tuple]],
    ):
        self.cfg = cfg
        self.batches = batches
        self.maps = {}

    def __len__(self) -> int:
        return len(self.batches)

    def _load_map(self, map_id: int):
        if map_id not in self.maps:
            cfg = self.cfg
            cache = load_map_cache(cfg.data.cache_root, map_id)
            builder = PAFeatureBuilder(
                cache,
                cfg.data.building_root,
                cfg.data.building_height,
                cfg.model.k_pre,
                cfg.model.max_wall_tokens,
                cfg.data.los_root,
            )
            self.maps[map_id] = (cache, builder)
        return self.maps[map_id]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        map_id, items = self.batches[index]
        cache, builder = self._load_map(map_id)
        samples = []
        if self.cfg.overfit.enabled:
            for tx_row, rx_slot in items:
                chain = builder.route_chain(
                    cache.tx_xyz[tx_row],
                    cache.rx_xyz[tx_row, rx_slot],
                    dominant_chain(cache, tx_row, rx_slot),
                )
                samples.extend(state_samples(cache, tx_row, rx_slot, chain))
        else:
            for tx_row, rx_slot, epoch in items:
                chain = builder.route_chain(
                    cache.tx_xyz[tx_row],
                    cache.rx_xyz[tx_row, rx_slot],
                    dominant_chain(cache, tx_row, rx_slot),
                )
                samples.append(
                    make_sample(
                        cache,
                        tx_row,
                        rx_slot,
                        epoch,
                        self.cfg.train.seed,
                        chain,
                    )
                )
        features = []
        for sample in samples:
            candidate_ids = builder.legal_ids(sample.tx_xyz, sample.prefix)
            item = builder.build_sparse(sample, candidate_ids)
            # Autoregressive decoder: supply prefix vertex ids as tokens
            prefix_padded = np.full(MAX_PREFIX_LEN, -1, dtype=np.int64)
            prefix_arr = np.array(sample.prefix, dtype=np.int64)
            prefix_padded[: len(prefix_arr)] = prefix_arr
            item["prefix_ids"] = prefix_padded
            item["tx_xyz"] = sample.tx_xyz
            item["rx_xyz"] = sample.rx_xyz
            item["state_weight"] = np.float32(
                1.0 / len(sample.truth_chain) if self.cfg.overfit.enabled else 1.0
            )
            features.append(item)
        batch = collate_sparse_features(features)
        if any(len(sample.prefix) == 0 for sample in samples):
            batch["los_graph_mask"] = builder.los_graph_mask
            batch["los_graph_edge"] = builder.los_graph_edge
        return batch


def _loader(dataset, cfg: AppConfig, device: torch.device):
    workers = max(int(cfg.train.num_workers), 0)
    kwargs = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["prefetch_factor"] = max(int(cfg.train.prefetch_factor), 1)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _train_batches(cfg: AppConfig, epoch: int) -> list[tuple[int, tuple]]:
    if cfg.overfit.enabled:
        batches = _overfit_batches(cfg)
        random.Random(cfg.train.seed + epoch).shuffle(batches)
        return batches

    batches = []
    for map_id in experiment_maps(cfg, "train"):
        cache = load_map_cache(cfg.data.cache_root, map_id)
        items = [
            (tx_row, rx_slot, epoch)
            for tx_row, slots in enumerate(_map_pools(cfg, map_id, "train"))
            for rx_slot in slots
        ]
        items = _vertex_decision_items(_source_los(cfg, cache), cache, items)
        batches.extend(
            (map_id, tuple(group)) for group in _chunks(items, cfg.train.batch_size)
        )
    random.Random(cfg.train.seed + epoch).shuffle(batches)
    return batches


def _val_batches(
    cfg: AppConfig,
    rx_limit: int,
    epoch: int,
) -> list[tuple[int, tuple]]:
    if cfg.overfit.enabled:
        return _overfit_batches(cfg)

    batches = []
    for map_id in experiment_maps(cfg, "val"):
        cache = load_map_cache(cfg.data.cache_root, map_id)
        items = [
            (tx_row, rx_slot, 0)
            for tx_row, slots in enumerate(_map_pools(cfg, map_id, "val"))
            for rx_slot in slots[: min(rx_limit, len(slots))]
        ]
        items = _vertex_decision_items(_source_los(cfg, cache), cache, items)
        batches.extend(
            (map_id, tuple(group)) for group in _chunks(items, cfg.train.batch_size)
        )
    random.Random(cfg.train.seed + 100_000 + epoch).shuffle(batches)
    return batches


def _overfit_batches(cfg: AppConfig) -> list[tuple[int, tuple]]:
    key = (
        str(cfg.data.cache_root),
        cfg.overfit.map_id,
        cfg.overfit.tx_count,
        cfg.overfit.rx_per_tx,
        cfg.train.seed,
        cfg.train.batch_size,
    )
    if key not in _OVERFIT_BATCH_CACHE:
        map_id = cfg.overfit.map_id
        cache = load_map_cache(cfg.data.cache_root, map_id)
        pools = _map_pools(cfg, map_id, "train")
        items = [
            (tx_row, rx_slot) for tx_row, slots in enumerate(pools) for rx_slot in slots
        ]
        items = _vertex_decision_items(_source_los(cfg, cache), cache, items)
        _OVERFIT_BATCH_CACHE[key] = tuple(
            (map_id, tuple(group)) for group in _chunks(items, cfg.train.batch_size)
        )
    return list(_OVERFIT_BATCH_CACHE[key])


def _loss_output(model, batch, train_cfg, backward: bool):
    output = model(batch, with_aux=True)
    state_loss = dominant_path_loss(
        output["pointer"], batch["positive"], output["legal"]
    )
    loss = weighted_state_mean(state_loss, batch["state_weight"])
    if backward:
        loss.backward()
    return loss, output


def update_early_stopping(
    best: float,
    bad_epochs: int,
    score: float,
    min_delta: float,
    patience: int,
) -> tuple[float, int, bool, bool]:
    improved = float(score) > float(best) + float(min_delta)
    if improved:
        best = float(score)
        bad_epochs = 0
    else:
        bad_epochs += 1
    stop = int(patience) > 0 and bad_epochs >= int(patience)
    return best, bad_epochs, stop, improved


@torch.no_grad()
def _chain_metrics(model, cfg, device, epoch, dist):
    tasks = [
        (map_id, tx_row, rx_slot)
        for map_id in experiment_maps(cfg, "val")
        for tx_row, slots in enumerate(_map_pools(cfg, map_id, "val"))
        for rx_slot in slots[: cfg.train.decode_rx_limit]
    ]
    vertex_counts = {
        map_id: _map_vertices(cfg, map_id) for map_id in {task[0] for task in tasks}
    }
    costs = [_candidate_cost(cfg, vertex_counts[map_id]) for map_id, _, _ in tasks]
    tasks = _balance_items(tasks, costs, dist.world_size)[dist.rank]

    exact = complete = dead_end = count = 0
    tasks_by_map: dict[int, list[tuple[int, int]]] = {}
    for map_id, tx_row, rx_slot in tasks:
        tasks_by_map.setdefault(map_id, []).append((tx_row, rx_slot))
    decode_batch_size = max(int(cfg.train.batch_size), 1)
    with tqdm(
        total=len(tasks),
        desc=f"val dominant chain epoch {epoch}",
        unit="case",
        dynamic_ncols=True,
        leave=False,
        disable=dist.rank != 0,
    ) as progress:
        for map_id, map_tasks in tasks_by_map.items():
            cache = load_map_cache(cfg.data.cache_root, map_id)
            builder = PAFeatureBuilder(
                cache,
                cfg.data.building_root,
                cfg.data.building_height,
                cfg.model.k_pre,
                cfg.model.max_wall_tokens,
                cfg.data.los_root,
            )
            predictor = AutoregressivePredictor(model, builder, device)
            for group in _chunks(map_tasks, decode_batch_size):
                tx = np.stack([cache.tx_xyz[tx_row] for tx_row, _ in group]).astype(
                    np.float32, copy=False
                )
                rx = np.stack(
                    [cache.rx_xyz[tx_row, rx_slot] for tx_row, rx_slot in group]
                ).astype(np.float32, copy=False)
                predictions = predictor.greedy_batch(tx, rx)

                for row, ((tx_row, rx_slot), prediction) in enumerate(
                    zip(group, predictions)
                ):
                    truth = builder.route_chain(
                        tx[row],
                        rx[row],
                        dominant_chain(cache, tx_row, rx_slot),
                    )
                    exact += int(prediction.complete and prediction.chain == truth)
                    complete += int(prediction.complete)
                    if prediction.complete:
                        continue
                    legal = builder.legal_candidates(tx[row], prediction.chain).copy()
                    if prediction.chain:
                        legal[np.asarray(prediction.chain, dtype=np.int64)] = False
                    dead_end += int(not legal.any())

                count += len(group)
                progress.update(len(group))
                progress.set_postfix(
                    chain_acc=f"{exact / count:.4f}",
                    complete=f"{complete / count:.4f}",
                )

    if dist.enabled:
        stats = torch.tensor(
            [exact, complete, dead_end, count],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        exact, complete, dead_end, count = stats.tolist()
    return {
        "chain_acc": exact / max(count, 1),
        "complete": complete / max(count, 1),
        "dead_end": dead_end / max(count, 1),
    }


@torch.no_grad()
def validation_metrics(model, cfg, device, rx_limit, epoch, dist):
    model.eval()
    total = loss_weight = next_count = hit1 = hit5 = 0
    legal_total = legal_weight = 0.0
    depth_hit = [0.0] * 4
    depth_count = [0.0] * 4
    all_batches = _val_batches(cfg, rx_limit, epoch)
    vertex_counts = {
        map_id: _map_vertices(cfg, map_id)
        for map_id in {batch[0] for batch in all_batches}
    }
    costs = [
        _candidate_cost(cfg, vertex_counts[map_id], len(items))
        for map_id, items in all_batches
    ]
    batches = _balance_items(all_batches, costs, dist.world_size)[dist.rank]
    dataset = FeatureBatchDataset(cfg, batches)

    with tqdm(
        total=len(dataset),
        desc=f"val epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=dist.rank != 0,
    ) as progress:
        for batch_index, features in enumerate(_loader(dataset, cfg, device)):
            batch = _to_device(features, device)
            loss, output = _loss_output(model, batch, cfg.train, False)
            weight = float(batch["state_weight"].sum().item())
            total += float(loss.item()) * weight
            loss_weight += weight
            target = batch["positive"].long().argmax(dim=1)
            state_weight = batch["state_weight"].to(output["pointer"].dtype)
            predicted = output["pointer"].argmax(dim=1)
            top5 = (
                output["pointer"]
                .topk(min(5, output["pointer"].shape[1]), dim=1)
                .indices
            )
            hit1 += float((predicted.eq(target) * state_weight).sum().item())
            hit5 += float(
                (top5.eq(target.unsqueeze(1)).any(dim=1) * state_weight).sum().item()
            )
            next_count += weight
            legal_count = output["legal"].sum(dim=1).to(state_weight.dtype)
            legal_total += float((legal_count * state_weight).sum().item())
            legal_weight += weight
            depth = batch["depth"].long()
            for index in range(4):
                mask = depth.eq(index)
                depth_hit[index] += float((predicted.eq(target) & mask).sum().item())
                depth_count[index] += float(mask.sum().item())
            progress.update(1)
            progress.set_postfix(
                map=batches[batch_index][0],
                loss=f"{total / max(loss_weight, 1):.4f}",
                next_acc=f"{hit1 / max(next_count, 1):.4f}",
            )

    if dist.enabled:
        stats = torch.tensor(
            [
                total,
                loss_weight,
                next_count,
                hit1,
                hit5,
                legal_total,
                legal_weight,
                *depth_hit,
                *depth_count,
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        (
            total,
            loss_weight,
            next_count,
            hit1,
            hit5,
            legal_total,
            legal_weight,
            *depth_values,
        ) = stats.tolist()
        depth_hit = depth_values[:4]
        depth_count = depth_values[4:]
    decode_interval = max(int(cfg.train.decode_interval), 1)
    decode_now = epoch % decode_interval == 0 or epoch == cfg.train.epochs
    chain = (
        _chain_metrics(model, cfg, device, epoch, dist)
        if decode_now
        else {"chain_acc": None, "complete": None, "dead_end": None}
    )
    return {
        "loss": total / max(loss_weight, 1),
        "next_acc": hit1 / max(next_count, 1),
        "next_hit5": hit5 / max(next_count, 1),
        "chain_acc": chain["chain_acc"],
        "complete": chain["complete"],
        "dead_end": chain["dead_end"],
        "los_candidates": legal_total / max(legal_weight, 1),
        "next_acc_by_depth": {
            str(index): depth_hit[index] / max(depth_count[index], 1)
            for index in range(4)
        },
    }


def _warm_start(model, checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    source = payload["model"]
    compatible = {
        key: value
        for key, value in source.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = [key for key in source if key not in compatible]
    result = model.load_state_dict(compatible, strict=False)
    return list(result.missing_keys), skipped


def _resume_path(
    output_dir: Path,
    resume: bool | str | Path | None,
) -> Path | None:
    if not resume:
        return None
    if resume is True or str(resume) == "auto":
        last_path = output_dir / "last.pt"
        if last_path.exists():
            return last_path
        best_path = output_dir / "best.pt"
        if best_path.exists():
            return best_path
        raise FileNotFoundError(f"no last.pt or best.pt found in {output_dir}")
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _set_scheduler_step(scheduler, optimizer, step: int) -> None:
    if scheduler is None:
        return
    scheduler.last_epoch = int(step)
    values = [
        base_lr * lr_lambda(int(step))
        for base_lr, lr_lambda in zip(
            scheduler.base_lrs,
            scheduler.lr_lambdas,
        )
    ]
    for group, value in zip(optimizer.param_groups, values):
        group["lr"] = value
    scheduler._last_lr = values


def _history(
    history_path: Path,
    payload: dict | None,
    start_epoch: int,
) -> list[dict]:
    if payload is not None and "history" in payload:
        rows = list(payload["history"])
    elif history_path.exists():
        rows = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        rows = []
    return [row for row in rows if int(row.get("epoch", 0)) <= start_epoch]


def _resume_best_chain(
    history: list[dict],
    payload: dict | None,
) -> float:
    if payload is not None:
        if "best_chain" in payload:
            return float(payload["best_chain"])
        if "val_chain_acc" in payload:
            return float(payload["val_chain_acc"])
    scores = [
        float(row["val_chain_acc"])
        for row in history
        if row.get("val_chain_acc") is not None
    ]
    return max(scores + [-1.0])


def train(
    cfg: AppConfig,
    output_dir: str | Path,
    val_rx_limit: int = 200,
    device_name: str | None = None,
    init_checkpoint: str | Path | None = None,
    resume: bool | str | Path | None = None,
) -> Path:
    dist = _init_dist(device_name)
    try:
        torch.manual_seed(cfg.train.seed + dist.rank)
        np.random.seed(cfg.train.seed + dist.rank)
        device = dist.device
        output_dir = Path(output_dir)
        if dist.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if dist.enabled:
            torch.distributed.barrier()

        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"
        history_path = output_dir / "history.json"
        resume_path = _resume_path(output_dir, resume)
        resume_payload = (
            torch.load(
                resume_path,
                map_location=device,
                weights_only=False,
            )
            if resume_path is not None
            else None
        )
        model = IRTPathModel(cfg.model).to(device)

        if resume_payload is not None:
            model.load_state_dict(resume_payload["model"])
        elif init_checkpoint is not None:
            missing, unexpected = _warm_start(model, init_checkpoint, device)
            if dist.rank == 0:
                print(
                    json.dumps(
                        {
                            "warm_start": str(init_checkpoint),
                            "new_parameters": missing,
                            "unused_parameters": unexpected,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        train_model = (
            DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.local_rank,
                find_unused_parameters=True,
            )
            if dist.enabled
            else model
        )
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        train_steps = math.ceil(len(_train_batches(cfg, 0)) / dist.world_size)
        scheduler = build_scheduler(
            optimizer, cfg.train, train_steps * cfg.train.epochs
        )

        start_epoch = (
            int(resume_payload.get("epoch", 0)) if resume_payload is not None else 0
        )
        history = _history(
            history_path,
            resume_payload,
            start_epoch,
        )
        best_chain = _resume_best_chain(history, resume_payload)
        bad_epochs = (
            int(resume_payload.get("early_stopping_bad_epochs", 0))
            if resume_payload is not None
            else 0
        )

        resume_mode = None
        if resume_payload is not None:
            if "optimizer" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer"])
                resume_mode = "full"
            else:
                resume_mode = "weights_only"

            same_steps = (
                int(resume_payload.get("train_steps", train_steps)) == train_steps
            )
            if (
                scheduler is not None
                and "scheduler" in resume_payload
                and resume_payload["scheduler"] is not None
                and same_steps
            ):
                scheduler.load_state_dict(resume_payload["scheduler"])
            else:
                _set_scheduler_step(
                    scheduler,
                    optimizer,
                    start_epoch * train_steps,
                )
            if dist.rank == 0:
                print(
                    json.dumps(
                        {
                            "resume": str(resume_path),
                            "resume_mode": resume_mode,
                            "start_epoch": start_epoch + 1,
                            "best_chain": best_chain,
                            "early_stopping_bad_epochs": bad_epochs,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        for epoch in range(start_epoch, cfg.train.epochs):
            train_model.train()
            train_total = 0.0
            train_count = 0
            batches = _split_batches(
                _train_batches(cfg, epoch),
                dist.rank,
                dist.world_size,
            )
            dataset = FeatureBatchDataset(cfg, batches)
            with tqdm(
                total=len(dataset),
                desc=f"train epoch {epoch + 1}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                disable=dist.rank != 0,
            ) as progress:
                for batch_index, features in enumerate(_loader(dataset, cfg, device)):
                    batch = _to_device(features, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = _loss_output(train_model, batch, cfg.train, True)
                    torch.nn.utils.clip_grad_norm_(
                        train_model.parameters(),
                        cfg.train.grad_clip,
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    weight = float(batch["state_weight"].sum().item())
                    train_total += float(loss.item()) * weight
                    train_count += weight
                    progress.update(1)
                    progress.set_postfix(
                        map=batches[batch_index][0],
                        loss=f"{train_total / max(train_count, 1):.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            if dist.enabled:
                stats = torch.tensor(
                    [train_total, train_count],
                    dtype=torch.float64,
                    device=device,
                )
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                train_total, train_count = stats.tolist()

            val = validation_metrics(
                model,
                cfg,
                device,
                val_rx_limit,
                epoch + 1,
                dist,
            )
            if val["chain_acc"] is None:
                should_stop = improved = False
            else:
                (
                    best_chain,
                    bad_epochs,
                    should_stop,
                    improved,
                ) = update_early_stopping(
                    best_chain,
                    bad_epochs,
                    val["chain_acc"],
                    cfg.train.early_stopping_min_delta,
                    cfg.train.early_stopping_patience,
                )
            if dist.rank == 0:
                row = {
                    "epoch": epoch + 1,
                    "train_loss": train_total / max(train_count, 1),
                    "val_loss": val["loss"],
                    "val_next_acc": val["next_acc"],
                    "val_next_hit5": val["next_hit5"],
                    "val_chain_acc": val["chain_acc"],
                    "val_complete": val["complete"],
                    "val_dead_end": val["dead_end"],
                    "val_los_candidates": val["los_candidates"],
                    **{
                        f"val_next_acc_depth{depth}": value
                        for depth, value in val["next_acc_by_depth"].items()
                    },
                }
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "model_config": asdict(cfg.model),
                    "train_config": asdict(cfg.train),
                    "epoch": epoch + 1,
                    "best_chain": best_chain,
                    "early_stopping_bad_epochs": bad_epochs,
                    "history": history,
                    "world_size": dist.world_size,
                    "train_steps": train_steps,
                    **row,
                }
                if improved:
                    torch.save(checkpoint_payload, best_path)
                torch.save(checkpoint_payload, last_path)
                if should_stop:
                    print(
                        json.dumps(
                            {
                                "early_stop": True,
                                "epoch": epoch + 1,
                                "best_chain": best_chain,
                                "patience": (cfg.train.early_stopping_patience),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if should_stop:
                break
        return best_path
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def load_checkpoint(path: str | Path, device: torch.device) -> IRTPathModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = IRTPathModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


__all__ = ["load_checkpoint", "train", "update_early_stopping"]
