from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARRAY_NAMES = (
    "vertex_xyz",
    "vertex_normal",
    "vertex_building",
    "polygon_ptr",
    "tx_id",
    "tx_xyz",
    "rx_id",
    "rx_xyz",
    "path_ptr",
    "chain_len",
    "chain_ids",
    "path_rank",
    "loss_db",
    "dist_m",
)


@dataclass
class MapCache:
    map_id: int
    split: str
    arrays: dict[str, np.ndarray]

    def __getattr__(self, name: str):
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)

    @property
    def n_tx(self) -> int:
        return int(self.tx_id.shape[0])

    @property
    def n_vertex(self) -> int:
        return int(self.vertex_xyz.shape[0])

    def paths(self, tx_row: int, rx_slot: int) -> np.ndarray:
        flat = tx_row * self.rx_id.shape[1] + rx_slot
        return np.arange(self.path_ptr[flat], self.path_ptr[flat + 1], dtype=np.int64)


def save_map_cache(
    root: str | Path, map_id: int, arrays: dict[str, np.ndarray], meta: dict
) -> None:
    directory = Path(root) / f"map_{map_id}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        np.save(directory / f"{name}.npy", arrays[name], allow_pickle=False)
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_map_cache(root: str | Path, map_id: int, mmap: bool = True) -> MapCache:
    directory = Path(root) / f"map_{map_id}"
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    mode = "r" if mmap else None
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode=mode) for name in ARRAY_NAMES
    }
    return MapCache(map_id, meta["split"], arrays)


GEOMETRY_ARRAYS = (
    "vertex_xyz",
    "vertex_normal",
    "vertex_building",
    "polygon_ptr",
)


def load_geometry_cache(root: str | Path, map_id: int) -> MapCache:
    directory = Path(root) / f"map_{map_id}"
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in GEOMETRY_ARRAYS
    }
    return MapCache(map_id, meta["split"], arrays)
