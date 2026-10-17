from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import experiment_maps, AppConfig
from src.data.cache import load_map_cache
from src.common.metrics import edit_distance, next_node_hits
from src.data.dataset import (
    PathSample,
    chain_loss_db,
    dominant_chain,
    eval_cases,
    prefix_targets,
)
from src.data.features import PAFeatureBuilder, collate_sparse_features
from src.data.overfit import select_pools
from src.model.network import MAX_PREFIX_LEN, numpy_batch_to_torch
from src.train.runner import load_checkpoint
from src.eval.path_loss import compute_path_loss_db
from src.eval.predict import AutoregressivePredictor


LIGHT_SPEED_M_S = 299_792_458.0


def _sample_rate(sample_count: int, seconds: float) -> float:
    return sample_count / seconds if sample_count and seconds > 0.0 else 0.0


def _timed_decode(
    predictor,
    tx: np.ndarray,
    rx: np.ndarray,
    device,
    clock=time.perf_counter,
):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = clock()
    predictions = predictor.greedy_batch(tx, rx)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return predictions, clock() - started


def _mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else 0.0


def _percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def _summary(rows: list[dict]) -> dict:
    keys = (
        "exact",
        "greedy_recall",
        "hop_correct",
        "edit",
        "greedy_complete",
        "greedy_failure",
        "dead_end",
        "segment_valid",
        "energy_error",
        "energy_error_pct",
        "dod_error_deg",
        "doa_error_deg",
        "delay_error_ns",
        "delay_error_pct",
    )
    energy = np.asarray([row["energy_error"] for row in rows], dtype=np.float64)
    dod = np.asarray([row["dod_error_deg"] for row in rows])
    doa = np.asarray([row["doa_error_deg"] for row in rows])
    delay = np.asarray([row["delay_error_ns"] for row in rows])
    energy_pct = np.asarray([row["energy_error_pct"] for row in rows])
    delay_pct = np.asarray([row["delay_error_pct"] for row in rows])
    distribution_stats = {
        "energy_error_p90_db": (
            float(np.percentile(energy, 90)) if len(energy) else 0.0
        ),
        "energy_error_p95_db": (
            float(np.percentile(energy, 95)) if len(energy) else 0.0
        ),
        "energy_within_3db": (float(np.mean(energy <= 3.0)) if len(energy) else 0.0),
        "energy_error_p90_pct": _percentile(energy_pct, 90),
        "energy_error_p95_pct": _percentile(energy_pct, 95),
        "dod_error_p90_deg": (float(np.percentile(dod, 90)) if len(dod) else 0.0),
        "dod_error_p95_deg": (float(np.percentile(dod, 95)) if len(dod) else 0.0),
        "doa_error_p90_deg": (float(np.percentile(doa, 90)) if len(doa) else 0.0),
        "doa_error_p95_deg": (float(np.percentile(doa, 95)) if len(doa) else 0.0),
        "delay_error_p90_ns": (float(np.percentile(delay, 90)) if len(delay) else 0.0),
        "delay_error_p95_ns": (float(np.percentile(delay, 95)) if len(delay) else 0.0),
        "delay_error_p90_pct": _percentile(delay_pct, 90),
        "delay_error_p95_pct": _percentile(delay_pct, 95),
    }
    return (
        {key: _mean(rows, key) for key in keys}
        | distribution_stats
        | {"count": len(rows)}
    )


def _direction_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    sine = np.linalg.norm(np.cross(left, right))
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _relative_pct(error: float, truth: float) -> float:
    if abs(truth) > 1e-12:
        return abs(error) / abs(truth) * 100.0
    return 0.0 if abs(error) <= 1e-12 else float("nan")


def _path_observables(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    chain: tuple[int, ...],
    vertex_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return BFS-convention DoD, propagation-direction DoA, and delay."""
    middle = np.asarray(vertex_xyz[list(chain)], dtype=np.float64)
    points = np.concatenate(
        (
            np.asarray(tx_xyz, dtype=np.float64)[None, :],
            middle.reshape(-1, 3),
            np.asarray(rx_xyz, dtype=np.float64)[None, :],
        ),
        axis=0,
    )
    segments = np.diff(points, axis=0)
    delay_ns = float(np.linalg.norm(segments, axis=1).sum() / LIGHT_SPEED_M_S * 1e9)
    return segments[0], segments[-1], delay_ns


def _path_errors(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    truth: tuple[int, ...],
    predicted: tuple[int, ...],
    vertex_xyz: np.ndarray,
) -> dict[str, float]:
    """Compare endpoint directions in 3-D and geometric path delays."""
    truth_dod, truth_doa, truth_delay = _path_observables(
        tx_xyz, rx_xyz, truth, vertex_xyz
    )
    pred_dod, pred_doa, pred_delay = _path_observables(
        tx_xyz, rx_xyz, predicted, vertex_xyz
    )
    dod_error = _direction_error_deg(pred_dod, truth_dod)
    doa_error = _direction_error_deg(pred_doa, truth_doa)
    delay_error = abs(pred_delay - truth_delay)
    return {
        "dod_error_deg": dod_error,
        "doa_error_deg": doa_error,
        "delay_error_ns": delay_error,
        "delay_error_pct": _relative_pct(delay_error, truth_delay),
    }


def _balanced_hop_cases(cases, builder, count: int, seed: int):
    hops = (1, 2, 3)
    groups = {hop: [] for hop in hops}
    for case in cases:
        truth = builder.route_chain(case.tx_xyz, case.rx_xyz, case.truth_chain)
        if len(truth) in groups:
            groups[len(truth)].append(case)

    quotas = {
        hop: count // len(hops) + int(index < count % len(hops))
        for index, hop in enumerate(hops)
    }
    rng = np.random.default_rng(seed)
    selected = []
    for hop in hops:
        if len(groups[hop]) < quotas[hop]:
            raise ValueError(
                f"hop {hop} only has {len(groups[hop])} validation cases, "
                f"but {quotas[hop]} are required"
            )
        indices = rng.permutation(len(groups[hop]))[: quotas[hop]]
        selected.extend(groups[hop][int(index)] for index in indices)
    rng.shuffle(selected)
    return selected


def _energy_errors(
    cache,
    tx_row: int,
    rx_slot: int,
    truth: tuple[int, ...],
    predicted: tuple[int, ...],
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    vertex_xyz: np.ndarray,
    vertex_normal: np.ndarray,
) -> tuple[float, float]:
    """Return dominant-path excess loss in dB and relative error.

    Uses the cache when **both** chains are present (exact C++ ray trace).
    When either chain is missing, falls back to Python UTD for **both**
    so that systematic errors cancel in the relative comparison.
    """
    truth_cached = chain_loss_db(cache, tx_row, rx_slot, truth)
    pred_cached = chain_loss_db(cache, tx_row, rx_slot, predicted)

    if truth_cached is not None and pred_cached is not None:
        truth_loss, pred_loss = truth_cached, pred_cached
    else:
        # Consistent comparison: use Python UTD for both
        truth_loss = compute_path_loss_db(
            tx_xyz,
            rx_xyz,
            truth,
            vertex_xyz,
            vertex_normal,
        )
        pred_loss = compute_path_loss_db(
            tx_xyz,
            rx_xyz,
            predicted,
            vertex_xyz,
            vertex_normal,
        )
    absolute_error = abs(pred_loss - truth_loss)
    return (
        max(0.0, pred_loss - truth_loss),
        _relative_pct(absolute_error, truth_loss),
    )


def _segment_validity(
    builder: PAFeatureBuilder,
    tx: np.ndarray,
    rx: np.ndarray,
    chain: tuple[int, ...],
) -> float:
    if not chain:
        return float(builder.direct_los(tx, rx))
    valid = int(builder.source_los(tx)[chain[0]])
    valid += sum(
        int(builder.los_vv[left, right]) for left, right in zip(chain[:-1], chain[1:])
    )
    valid += int(builder.endpoint_los(rx)[chain[-1]])
    return valid / (len(chain) + 1)


def _add_prefix_ids(features: dict, prefix: tuple[int, ...]) -> None:
    """Mutate *features* in-place to include padded prefix vertex ids."""
    padded = np.full(MAX_PREFIX_LEN, -1, dtype=np.int64)
    arr = np.array(prefix, dtype=np.int64)
    padded[: len(arr)] = arr
    features["prefix_ids"] = padded[np.newaxis, :]


def _add_eval_context(features: dict, prefix: tuple[int, ...], builder) -> None:
    _add_prefix_ids(features, prefix)
    if not prefix:
        features["los_graph_mask"] = builder.los_graph_mask
        features["los_graph_edge"] = builder.los_graph_edge


def _prefix_metrics(model, builder, cases, truths, device) -> dict:
    items = []
    prefixes = []
    depths = []
    for case, truth in zip(cases, truths):
        for prefix, positives in prefix_targets(truth):
            sample = PathSample(
                case.map_id,
                case.tx_row,
                case.rx_slot,
                case.tx_xyz,
                case.rx_xyz,
                prefix,
                positives,
                truth,
            )
            item = builder.build_sparse(
                sample,
                builder.legal_ids(sample.tx_xyz, sample.prefix),
            )
            item["tx_xyz"] = sample.tx_xyz
            item["rx_xyz"] = sample.rx_xyz
            items.append(item)
            padded = np.full(MAX_PREFIX_LEN, -1, dtype=np.int64)
            padded[: len(prefix)] = np.asarray(prefix, dtype=np.int64)
            prefixes.append(padded)
            depths.append(min(len(prefix), 3))

    if not items:
        return {
            "next_hit1": 0,
            "next_hit5": 0,
            "next_count": 0,
            "legal_total": 0,
            "depth_hit": [0] * 4,
            "depth_count": [0] * 4,
        }

    features = collate_sparse_features(items)
    features["prefix_ids"] = np.stack(prefixes, axis=0)
    if any(depth == 0 for depth in depths):
        features["los_graph_mask"] = builder.los_graph_mask
        features["los_graph_edge"] = builder.los_graph_edge
    batch = numpy_batch_to_torch(features, device)
    output = model(batch, with_aux=True)
    logits = output["pointer"]
    hit1, hit5, count = next_node_hits(logits, batch["positive"])
    predicted = logits.argmax(dim=1)
    target = batch["positive"].long().argmax(dim=1)
    correct = predicted.eq(target)
    depth_tensor = torch.as_tensor(depths, device=device)
    return {
        "next_hit1": hit1,
        "next_hit5": hit5,
        "next_count": count,
        "legal_total": int(output["legal"].sum().item()),
        "depth_hit": [
            int((correct & depth_tensor.eq(depth)).sum().item()) for depth in range(4)
        ],
        "depth_count": [int(depth_tensor.eq(depth).sum().item()) for depth in range(4)],
    }


def _overfit_cases(cfg: AppConfig, cache, rx_limit: int | None):
    pools = select_pools(
        cache,
        cfg.overfit.tx_count,
        cfg.overfit.rx_per_tx,
        cfg.train.seed,
    )
    for tx_row, slots in enumerate(pools):
        if rx_limit is not None:
            slots = slots[: min(int(rx_limit), len(slots))]
        for rx_slot in slots:
            yield PathSample(
                cache.map_id,
                tx_row,
                rx_slot,
                np.asarray(cache.tx_xyz[tx_row], dtype=np.float32).copy(),
                np.asarray(cache.rx_xyz[tx_row, rx_slot], dtype=np.float32).copy(),
                (),
                (),
                dominant_chain(cache, tx_row, rx_slot),
            )


@torch.no_grad()
def evaluate(
    cfg: AppConfig,
    checkpoint: str | Path,
    output_path: str | Path,
    rx_limit: int | None = None,
    device_name: str | None = None,
    map_id: int | None = None,
    sample_count: int | None = None,
    balanced_hops: bool = False,
) -> dict:
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_checkpoint(checkpoint, device)
    rows = []
    exact_total = 0
    next_hit1 = 0
    next_hit5 = 0
    next_count = 0
    legal_total = 0
    depth_hit = [0] * 4
    depth_count = [0] * 4
    split = cfg.experiment.eval_split
    map_ids = (
        (int(map_id),)
        if map_id is not None
        else (
            (cfg.overfit.map_id,)
            if cfg.overfit.enabled
            else experiment_maps(cfg, split)
        )
    )
    within_map = cfg.experiment.split_mode == "within_map"
    batch_size = max(int(cfg.train.batch_size), 1)
    decode_count = 0
    decode_seconds = 0.0

    for map_id in map_ids:
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
        cases = list(
            _overfit_cases(cfg, cache, rx_limit)
            if cfg.overfit.enabled
            else eval_cases(
                cache,
                rx_limit,
                within_map,
                cfg.experiment.train_ratio,
                cfg.train.seed,
            )
        )
        if balanced_hops:
            if sample_count is None:
                raise ValueError("--balanced-hops requires --sample-count")
            cases = _balanced_hop_cases(cases, builder, sample_count, cfg.train.seed)
        elif sample_count is not None:
            cases = cases[:sample_count]
        progress = tqdm(
            total=len(cases),
            desc=f"eval map {map_id}",
            unit="case",
            dynamic_ncols=True,
        )
        for start in range(0, len(cases), batch_size):
            group = cases[start : start + batch_size]
            tx = np.stack([case.tx_xyz for case in group])
            rx = np.stack([case.rx_xyz for case in group])
            truths = [
                builder.route_chain(case.tx_xyz, case.rx_xyz, case.truth_chain)
                for case in group
            ]
            predictions, elapsed = _timed_decode(predictor, tx, rx, device)
            decode_count += len(group)
            decode_seconds += elapsed
            metrics = _prefix_metrics(model, builder, group, truths, device)
            next_hit1 += metrics["next_hit1"]
            next_hit5 += metrics["next_hit5"]
            next_count += metrics["next_count"]
            legal_total += metrics["legal_total"]
            depth_hit = [
                left + right for left, right in zip(depth_hit, metrics["depth_hit"])
            ]
            depth_count = [
                left + right for left, right in zip(depth_count, metrics["depth_count"])
            ]

            for case, truth, prediction in zip(group, truths, predictions):
                predicted = tuple(prediction.chain)
                is_exact = prediction.complete and predicted == truth
                exact_total += int(is_exact)
                legal = builder.legal_candidates(case.tx_xyz, predicted).copy()
                if predicted:
                    legal[np.asarray(predicted, dtype=np.int64)] = False
                energy_error, energy_error_pct = _energy_errors(
                    cache,
                    case.tx_row,
                    case.rx_slot,
                    truth,
                    predicted,
                    case.tx_xyz,
                    case.rx_xyz,
                    cache.vertex_xyz,
                    cache.vertex_normal,
                )
                rows.append(
                    {
                        "map_id": map_id,
                        "tx_row": case.tx_row,
                        "hop": len(truth),
                        "exact": float(is_exact),
                        "greedy_recall": float(is_exact),
                        "hop_correct": float(len(predicted) == len(truth)),
                        "edit": edit_distance(predicted, truth)
                        / max(len(predicted), len(truth), 1),
                        "greedy_complete": float(prediction.complete),
                        "greedy_failure": float(not prediction.complete),
                        "dead_end": float(not prediction.complete and not legal.any()),
                        "segment_valid": _segment_validity(
                            builder,
                            case.tx_xyz,
                            case.rx_xyz,
                            predicted,
                        ),
                        "energy_error": energy_error,
                        "energy_error_pct": energy_error_pct,
                        **_path_errors(
                            case.tx_xyz,
                            case.rx_xyz,
                            truth,
                            predicted,
                            cache.vertex_xyz,
                        ),
                    }
                )
            progress.update(len(group))
            progress.set_postfix(
                chain_acc=f"{exact_total / max(len(rows), 1):.4f}",
                next_acc=f"{next_hit1 / max(next_count, 1):.4f}",
                sample_s=(f"{_sample_rate(decode_count, decode_seconds):.2f}"),
            )
        progress.close()

    by_hop = {
        str(hop): _summary([row for row in rows if row["hop"] == hop])
        for hop in (0, 1, 2, 3)
    }
    by_map = [
        _summary([row for row in rows if row["map_id"] == map_id]) for map_id in map_ids
    ]
    tx_groups = defaultdict(list)
    for row in rows:
        tx_groups[(row["map_id"], row["tx_row"])].append(row)

    summary_keys = [key for key in _summary(rows) if key != "count"]
    result = {
        "rx_limit": rx_limit,
        "map_ids": [int(value) for value in map_ids],
        "sample_count": sample_count,
        "balanced_hops": bool(balanced_hops),
        "full_overfit": bool(cfg.overfit.enabled and rx_limit is None),
        "decode_samples_per_s": _sample_rate(decode_count, decode_seconds),
        "micro": _summary(rows),
        "map_macro": {key: _mean(by_map, key) for key in summary_keys},
        "tx_macro": {
            key: float(np.mean([_summary(group)[key] for group in tx_groups.values()]))
            if tx_groups
            else 0.0
            for key in summary_keys
        },
        "next_top1": next_hit1 / max(next_count, 1),
        "next_top5": next_hit5 / max(next_count, 1),
        "los_candidates": legal_total / max(next_count, 1),
        "next_top1_by_depth": {
            str(depth): depth_hit[depth] / max(depth_count[depth], 1)
            for depth in range(4)
        },
        "by_hop": by_hop,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


__all__ = ["evaluate"]
