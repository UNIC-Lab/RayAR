from __future__ import annotations

import hashlib

from src.data.cache import MapCache
from src.data.dataset import valid_slots


def _rx_key(
    cache: MapCache,
    tx_row: int,
    rx_slot: int,
    seed: int,
) -> bytes:
    text = (
        f"{seed}:{cache.map_id}:"
        f"{int(cache.tx_id[tx_row])}:"
        f"{int(cache.rx_id[tx_row, rx_slot])}"
    )
    return hashlib.sha256(text.encode("ascii")).digest()


def select_pools(
    cache: MapCache,
    tx_count: int,
    rx_per_tx: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if cache.n_tx < tx_count:
        raise ValueError(f"cache has {cache.n_tx} Tx rows, needs {tx_count}")

    pools = []
    all_slots = tuple(range(int(cache.rx_id.shape[1])))
    for tx_row in range(tx_count):
        slots = valid_slots(cache, tx_row, all_slots)
        if len(slots) < rx_per_tx:
            raise ValueError(
                f"Tx row {tx_row} has {len(slots)} valid Rx slots, needs {rx_per_tx}"
            )
        ordered = sorted(
            slots,
            key=lambda slot: _rx_key(
                cache,
                tx_row,
                slot,
                seed,
            ),
        )
        pools.append(tuple(ordered[:rx_per_tx]))
    return tuple(pools)


__all__ = ["select_pools"]
