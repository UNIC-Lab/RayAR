from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.config import DataConfig, TEST_MAPS, TRAIN_MAPS, VAL_MAPS, split_for_map
from src.data.cache import ARRAY_NAMES, save_map_cache
from src.data.prop import PropPath, PropReceiver, read_prop
from src.geom.scene import load_scene


SOURCE_RE = re.compile(r"source_(\d+)\.propbin(?:\.gz)?$")


def _source_id(path: Path) -> int:
    match = SOURCE_RE.match(path.name)
    if not match:
        raise ValueError(f"invalid PROP filename: {path.name}")
    return int(match.group(1))


def _rank_key(seed: int, map_id: int, tx_id: int, rx_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{map_id}:{tx_id}:{rx_id}".encode("ascii")).digest()


def _spatial_sample(
    eligible: list[tuple[PropReceiver, tuple[PropPath, ...]]],
    cfg: DataConfig,
    map_id: int,
    tx_id: int,
) -> list[tuple[PropReceiver, tuple[PropPath, ...]]]:
    bins = int(cfg.spatial_bins)
    buckets: dict[int, list[tuple[PropReceiver, tuple[PropPath, ...]]]] = {}
    for item in eligible:
        receiver = item[0]
        x = min(max(float(receiver.xyz[0]), 0.0), cfg.map_size)
        y = min(max(float(receiver.xyz[1]), 0.0), cfg.map_size)
        ix = min(int(x / cfg.map_size * bins), bins - 1)
        iy = min(int(y / cfg.map_size * bins), bins - 1)
        buckets.setdefault(iy * bins + ix, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: _rank_key(cfg.seed, map_id, tx_id, item[0].rx_id))

    selected: list[tuple[PropReceiver, tuple[PropPath, ...]]] = []
    offsets = {cell: 0 for cell in buckets}
    round_id = 0
    while len(selected) < cfg.rx_per_tx:
        active = [
            cell for cell, bucket in buckets.items() if offsets[cell] < len(bucket)
        ]
        if not active:
            break
        active.sort(
            key=lambda cell: hashlib.sha256(
                f"{cfg.seed}:{map_id}:{tx_id}:{round_id}:{cell}".encode("ascii")
            ).digest()
        )
        for cell in active:
            selected.append(buckets[cell][offsets[cell]])
            offsets[cell] += 1
            if len(selected) == cfg.rx_per_tx:
                break
        round_id += 1
    return selected


def _valid_paths(receiver: PropReceiver, n_vertex: int) -> tuple[PropPath, ...]:
    if any(path.path_type == 0 for path in receiver.paths):
        return ()
    valid = []
    for path in receiver.paths:
        if path.path_type <= 0:
            continue
        if path.path_type != len(path.chain_ids) or not 1 <= len(path.chain_ids) <= 3:
            continue
        if any(vertex < 0 or vertex >= n_vertex for vertex in path.chain_ids):
            continue
        if len(set(path.chain_ids)) != len(path.chain_ids):
            continue
        valid.append(path)
    return tuple(valid)


def prop_files(cfg: DataConfig, map_id: int) -> list[Path]:
    directory = cfg.prop_root / f"map_{map_id}" / "special_points_propbin_6GHz"
    files = list(directory.glob("source_*.propbin")) + list(
        directory.glob("source_*.propbin.gz")
    )
    return sorted(files, key=_source_id)


def _cached_meta(cfg: DataConfig, map_id: int) -> dict | None:
    directory = cfg.cache_root / f"map_{map_id}"
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    matches = (
        meta.get("map_id") == map_id
        and meta.get("rx_per_tx") == cfg.rx_per_tx
        and meta.get("spatial_bins") == cfg.spatial_bins
        and meta.get("sampling") == "spatial_round_robin"
        and meta.get("seed", cfg.seed) == cfg.seed
        and all((directory / f"{name}.npy").exists() for name in ARRAY_NAMES)
    )
    return meta if matches else None


def build_map(cfg: DataConfig, map_id: int, progress: bool = False) -> dict:
    split = split_for_map(map_id)
    scene = load_scene(cfg.building_root / f"{map_id}.json", cfg.building_height)
    files = prop_files(cfg, map_id)
    if not files:
        raise FileNotFoundError(f"map_{map_id}: no PROP files")
    tx_ids: list[int] = []
    tx_xyz: list[tuple[float, float, float]] = []
    rx_ids: list[list[int]] = []
    rx_xyz: list[list[tuple[float, float, float]]] = []
    path_ptr = [0]
    chain_len: list[int] = []
    chain_ids: list[list[int]] = []
    path_rank: list[int] = []
    loss_db: list[float] = []
    dist_m: list[float] = []
    counts = {"los": 0, "rooftop_only": 0, "invalid": 0, "eligible": 0, "sampled": 0}

    source_iter = tqdm(
        files,
        desc=f"map_{map_id} sources",
        unit="Tx",
        leave=False,
        disable=not progress,
    )
    for file_path in source_iter:
        tx_id = _source_id(file_path)
        try:
            prop = read_prop(file_path)
        except Exception as exc:
            raise RuntimeError(
                f"map_{map_id}/{file_path.name}: failed to read PROP"
            ) from exc
        eligible: list[tuple[PropReceiver, tuple[PropPath, ...]]] = []
        for receiver in prop.receivers:
            if any(path.path_type == 0 for path in receiver.paths):
                counts["los"] += 1
                continue
            valid = _valid_paths(receiver, scene.vertex_xyz.shape[0])
            if valid:
                eligible.append((receiver, valid))
            elif all(path.path_type == -1 for path in receiver.paths):
                counts["rooftop_only"] += 1
            else:
                counts["invalid"] += 1
        counts["eligible"] += len(eligible)
        if len(eligible) < cfg.rx_per_tx:
            raise ValueError(
                f"map_{map_id}/source_{tx_id}: only {len(eligible)} eligible Rx"
            )
        selected = _spatial_sample(eligible, cfg, map_id, tx_id)
        counts["sampled"] += len(selected)
        tx_ids.append(tx_id)
        tx_xyz.append(prop.source_xyz)
        rx_ids.append([receiver.rx_id for receiver, _ in selected])
        rx_xyz.append([receiver.xyz for receiver, _ in selected])
        for _, paths in selected:
            for path in paths:
                row = list(path.chain_ids) + [cfg.pad_vertex_id] * (
                    3 - len(path.chain_ids)
                )
                chain_len.append(len(path.chain_ids))
                chain_ids.append(row)
                path_rank.append(path.rank)
                loss_db.append(path.loss_db)
                dist_m.append(path.dist_m)
            path_ptr.append(len(chain_len))

    arrays = {
        "vertex_xyz": scene.vertex_xyz.astype(np.float32),
        "vertex_normal": scene.vertex_normal.astype(np.float32),
        "vertex_building": scene.vertex_building.astype(np.int32),
        "polygon_ptr": scene.polygon_ptr.astype(np.int32),
        "tx_id": np.asarray(tx_ids, dtype=np.uint16),
        "tx_xyz": np.asarray(tx_xyz, dtype=np.float32),
        "rx_id": np.asarray(rx_ids, dtype=np.int32),
        "rx_xyz": np.asarray(rx_xyz, dtype=np.float32),
        "path_ptr": np.asarray(path_ptr, dtype=np.int64),
        "chain_len": np.asarray(chain_len, dtype=np.uint8),
        "chain_ids": np.asarray(chain_ids, dtype=np.uint16).reshape(-1, 3),
        "path_rank": np.asarray(path_rank, dtype=np.uint8),
        "loss_db": np.asarray(loss_db, dtype=np.float32),
        "dist_m": np.asarray(dist_m, dtype=np.float32),
    }
    meta = {
        "map_id": map_id,
        "split": split,
        "n_tx": len(files),
        "rx_per_tx": cfg.rx_per_tx,
        "spatial_bins": cfg.spatial_bins,
        "sampling": "spatial_round_robin",
        "seed": cfg.seed,
        "n_vertex": int(scene.vertex_xyz.shape[0]),
        "n_path": len(chain_len),
        "counts": counts,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    save_map_cache(cfg.cache_root, map_id, arrays, meta)
    return meta


def _build_map_job(args: tuple[DataConfig, int]) -> dict:
    cfg, map_id = args
    try:
        return build_map(cfg, map_id, progress=False)
    except Exception as exc:
        raise RuntimeError(f"map_{map_id}: cache build failed") from exc


def build_dataset(
    cfg: DataConfig,
    map_ids: list[int] | tuple[int, ...] | None = None,
    workers: int = 1,
    rebuild: bool = False,
) -> dict:
    approved = TRAIN_MAPS + VAL_MAPS + TEST_MAPS
    selected = tuple(approved if map_ids is None else map_ids)
    cached = (
        {}
        if rebuild
        else {
            map_id: meta
            for map_id in selected
            if (meta := _cached_meta(cfg, map_id)) is not None
        }
    )
    pending = tuple(map_id for map_id in selected if map_id not in cached)
    if workers <= 1:
        map_iter = tqdm(
            pending, desc="maps", unit="map", total=len(selected), initial=len(cached)
        )
        built = [build_map(cfg, map_id, progress=True) for map_id in map_iter]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = pool.map(_build_map_job, ((cfg, map_id) for map_id in pending))
            built = list(
                tqdm(
                    jobs,
                    total=len(selected),
                    initial=len(cached),
                    desc="maps",
                    unit="map",
                )
            )
    maps = list(cached.values()) + built
    maps.sort(key=lambda item: item["map_id"])
    manifest = {
        "version": 1,
        "seed": cfg.seed,
        "rx_per_tx": cfg.rx_per_tx,
        "spatial_bins": cfg.spatial_bins,
        "sampling": "spatial_round_robin",
        "building_height_m": cfg.building_height,
        "map_size_m": cfg.map_size,
        "train_maps": list(TRAIN_MAPS),
        "val_maps": list(VAL_MAPS),
        "test_maps": list(TEST_MAPS),
        "maps": maps,
    }
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    (cfg.cache_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
