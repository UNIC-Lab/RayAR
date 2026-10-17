from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.data.cache import MapCache


@dataclass(frozen=True)
class PathSample:
    map_id: int
    tx_row: int
    rx_slot: int
    tx_xyz: np.ndarray
    rx_xyz: np.ndarray
    prefix: tuple[int, ...]
    positives: tuple[int, ...]
    truth_chain: tuple[int, ...]
    q_ids: tuple[int, ...] = ()
    q_regret_db: tuple[float, ...] = ()


def _chain(cache: MapCache, path_id: int) -> tuple[int, ...]:
    length = int(cache.chain_len[path_id])
    return tuple(int(vertex) for vertex in cache.chain_ids[path_id, :length])


def dominant_chain(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
) -> tuple[int, ...]:
    paths = cache.paths(tx_row, rx_slot)

    tx = np.asarray(cache.tx_xyz[tx_row], dtype=np.float32)
    rx = np.asarray(cache.rx_xyz[tx_row, rx_slot], dtype=np.float32)

    def tie_key(path_id: int) -> tuple:
        path_chain = _chain(cache, path_id)
        points = (
            [tx]
            + [
                np.asarray(
                    cache.vertex_xyz[vertex],
                    dtype=np.float32,
                )
                for vertex in path_chain
            ]
            + [rx]
        )
        distance = sum(
            float(np.linalg.norm(points[index + 1] - points[index]))
            for index in range(len(points) - 1)
        )
        coordinates = tuple(
            float(value) for vertex in path_chain for value in cache.vertex_xyz[vertex]
        )
        return distance, len(path_chain), coordinates

    path_id = min(
        paths,
        key=lambda index: (
            float(cache.loss_db[index]),
            *tie_key(int(index)),
        ),
    )
    return _chain(cache, int(path_id))


def cost_to_go_targets(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    prefix: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    depth = len(prefix)
    best_by_action: dict[int, float] = {}
    for path_id in cache.paths(tx_row, rx_slot):
        path_chain = _chain(cache, int(path_id))
        if len(path_chain) <= depth or path_chain[:depth] != prefix:
            continue
        action = path_chain[depth]
        loss_db = float(cache.loss_db[path_id])
        best_by_action[action] = min(
            loss_db,
            best_by_action.get(action, float("inf")),
        )
    if not best_by_action:
        raise ValueError(f"no cached continuation for prefix {prefix}")
    floor = min(best_by_action.values())
    actions = tuple(sorted(best_by_action))
    regret = tuple(best_by_action[action] - floor for action in actions)
    return actions, regret


def cost_to_go_arrays(
    sample: PathSample,
    n_vertex: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(n_vertex, dtype=np.float32)
    mask = np.zeros(n_vertex, dtype=bool)
    if sample.q_ids:
        ids = np.asarray(sample.q_ids, dtype=np.int64)
        target[ids] = np.asarray(sample.q_regret_db, dtype=np.float32)
        mask[ids] = True
    return target, mask


def prefix_targets(
    chain: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return tuple((chain[:depth], (vertex,)) for depth, vertex in enumerate(chain))


def canonical_chain(
    chain: tuple[int, ...],
    direct_visible: bool,
    goal_visible: np.ndarray,
) -> tuple[int, ...]:
    if direct_visible:
        return ()
    for depth, vertex in enumerate(chain, start=1):
        if goal_visible[vertex]:
            return chain[:depth]
    return chain


def _sample_state(
    states: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    digest: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(states) == 1 or digest[0] % 2 == 0:
        return states[0]
    index = 1 + int.from_bytes(digest[1:9], "little") % (len(states) - 1)
    return states[index]


def make_sample(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    epoch: int,
    seed: int = 2026,
    chain: tuple[int, ...] | None = None,
) -> PathSample:
    chain = dominant_chain(cache, tx_row, rx_slot) if chain is None else chain
    states = prefix_targets(chain)
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{cache.map_id}:{tx_row}:{rx_slot}".encode("ascii")
    ).digest()
    prefix, positives = _sample_state(states, digest)
    return PathSample(
        cache.map_id,
        tx_row,
        rx_slot,
        np.array(cache.tx_xyz[tx_row], dtype=np.float32, copy=True),
        np.array(cache.rx_xyz[tx_row, rx_slot], dtype=np.float32, copy=True),
        prefix,
        positives,
        chain,
    )


def state_samples(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    chain: tuple[int, ...] | None = None,
) -> tuple[PathSample, ...]:
    chain = dominant_chain(cache, tx_row, rx_slot) if chain is None else chain
    tx_xyz = np.asarray(
        cache.tx_xyz[tx_row],
        dtype=np.float32,
    )
    rx_xyz = np.asarray(
        cache.rx_xyz[tx_row, rx_slot],
        dtype=np.float32,
    )
    samples = []
    for prefix, positives in prefix_targets(chain):
        samples.append(
            PathSample(
                cache.map_id,
                tx_row,
                rx_slot,
                tx_xyz.copy(),
                rx_xyz.copy(),
                prefix,
                positives,
                chain,
            )
        )
    return tuple(samples)


def _slot_rank(cache: MapCache, tx_row: int, slot: int, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}:{cache.map_id}:{int(cache.tx_id[tx_row])}:"
        f"{int(cache.rx_id[tx_row, slot])}".encode("ascii")
    ).digest()


def hash_split_slots(
    cache: MapCache,
    tx_row: int,
    train_ratio: float = 0.8,
    seed: int = 2026,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    slots = list(range(int(cache.rx_id.shape[1])))
    slots.sort(key=lambda slot: _slot_rank(cache, tx_row, slot, seed))
    cut = int(round(len(slots) * train_ratio))
    return tuple(slots[:cut]), tuple(slots[cut:])


def _dominant_lengths(cache: MapCache, tx_row: int) -> np.ndarray:
    width = int(cache.rx_id.shape[1])
    first = tx_row * width
    starts = np.asarray(cache.path_ptr[first : first + width], dtype=np.int64)
    ends = np.asarray(cache.path_ptr[first + 1 : first + width + 1], dtype=np.int64)
    lengths = np.empty(width, dtype=np.int64)
    for slot, (start, end) in enumerate(zip(starts, ends)):
        best = int(start + np.argmin(cache.loss_db[start:end]))
        lengths[slot] = int(cache.chain_len[best])
    return lengths


def stratified_split_slots(
    cache: MapCache,
    tx_row: int,
    train_ratio: float = 0.8,
    seed: int = 2026,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = int(cache.rx_id.shape[1])
    target_train = int(round(count * train_ratio))
    target_val = count - target_train

    tx = np.asarray(cache.tx_xyz[tx_row], dtype=np.float32)
    rx = np.asarray(cache.rx_xyz[tx_row], dtype=np.float32)
    distance = np.linalg.norm(rx - tx, axis=1)
    edges = np.quantile(distance, (0.25, 0.5, 0.75))
    distance_bin = np.searchsorted(edges, distance, side="right")
    chain_length = _dominant_lengths(cache, tx_row)

    strata: dict[tuple[int, int], list[int]] = {}
    for slot in range(count):
        key = (int(chain_length[slot]), int(distance_bin[slot]))
        strata.setdefault(key, []).append(slot)

    val_ratio = target_val / count
    quota = {
        key: int(np.floor(len(slots) * val_ratio)) for key, slots in strata.items()
    }
    remaining = target_val - sum(quota.values())
    order = sorted(
        strata,
        key=lambda key: (
            -(len(strata[key]) * val_ratio - quota[key]),
            hashlib.sha256(f"{seed}:{key}".encode("ascii")).digest(),
        ),
    )
    for key in order[:remaining]:
        quota[key] += 1

    train = []
    val = []
    for key, slots in strata.items():
        ranked = sorted(
            slots,
            key=lambda slot: _slot_rank(cache, tx_row, slot, seed),
        )
        split = quota[key]
        val.extend(ranked[:split])
        train.extend(ranked[split:])
    train.sort(key=lambda slot: _slot_rank(cache, tx_row, slot, seed))
    val.sort(key=lambda slot: _slot_rank(cache, tx_row, slot, seed))
    return tuple(train), tuple(val)


def split_slots(
    cache: MapCache,
    tx_row: int,
    train_ratio: float = 0.8,
    seed: int = 2026,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return stratified_split_slots(cache, tx_row, train_ratio, seed)


def valid_slots(
    cache: MapCache,
    tx_row: int,
    slots: Iterable[int],
) -> tuple[int, ...]:
    return tuple(int(slot) for slot in slots)


def eval_cases(
    cache: MapCache,
    rx_limit: int | None,
    within_map: bool,
    train_ratio: float,
    seed: int,
):
    for tx_row in range(cache.n_tx):
        pool = (
            split_slots(cache, tx_row, train_ratio, seed)[1]
            if within_map
            else tuple(range(int(cache.rx_id.shape[1])))
        )
        slots = valid_slots(cache, tx_row, pool)
        if rx_limit is not None:
            slots = slots[: min(int(rx_limit), len(slots))]
        for rx_slot in slots:
            chain = dominant_chain(cache, tx_row, rx_slot)
            yield PathSample(
                cache.map_id,
                tx_row,
                rx_slot,
                np.array(cache.tx_xyz[tx_row], dtype=np.float32, copy=True),
                np.array(
                    cache.rx_xyz[tx_row, rx_slot],
                    dtype=np.float32,
                    copy=True,
                ),
                (),
                (),
                chain,
            )


def chain_loss_db(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    chain: tuple[int, ...],
) -> float | None:
    """Return the ray-traced loss_db of *chain*, or None if not cached."""
    for path_id in cache.paths(tx_row, rx_slot):
        if _chain(cache, int(path_id)) == chain:
            return float(cache.loss_db[path_id])
    return None


def energy_error_db(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    truth: tuple[int, ...],
    predicted: tuple[int, ...],
) -> tuple[float, bool]:
    """Return (energy_error_dB, is_exact).

    If *predicted* is not in the cache, the returned error is a
    conservative **lower bound**: the error is *at least* the gap
    between the dominant loss and the worst cached loss.
    ``is_exact`` is True only when the error is an exact measurement.
    """
    truth_loss = chain_loss_db(cache, tx_row, rx_slot, truth)
    if truth_loss is None:
        return 0.0, False  # should never happen

    pred_loss = chain_loss_db(cache, tx_row, rx_slot, predicted)
    if pred_loss is not None:
        return max(0.0, pred_loss - truth_loss), True

    # Lower bound: predicted chain is not in the Top-K cache,
    # so its loss must be >= the largest cached loss.
    cached = [float(cache.loss_db[int(pid)]) for pid in cache.paths(tx_row, rx_slot)]
    max_cached = max(cached) if cached else truth_loss
    return max(0.0, max_cached - truth_loss), False


__all__ = [
    "PathSample",
    "chain_loss_db",
    "energy_error_db",
    "cost_to_go_arrays",
    "cost_to_go_targets",
    "dominant_chain",
    "canonical_chain",
    "eval_cases",
    "make_sample",
    "prefix_targets",
    "split_slots",
    "state_samples",
    "valid_slots",
]
