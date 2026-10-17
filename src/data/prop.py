from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = 0x50524F50
HEADER_SIZE = 32
RECEIVER_SIZE = 32
PATH_SIZE = 24


@dataclass(frozen=True)
class PropPath:
    rank: int
    path_type: int
    chain_ids: tuple[int, ...]
    loss_db: float
    dist_m: float


@dataclass(frozen=True)
class PropReceiver:
    rx_id: int
    xyz: tuple[float, float, float]
    paths: tuple[PropPath, ...]


@dataclass(frozen=True)
class PropFile:
    source_xyz: tuple[float, float, float]
    receivers: tuple[PropReceiver, ...]


def _half(value: int) -> float:
    return float(np.asarray(value, dtype=np.uint16).view(np.float16))


def _read_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def read_prop(path: str | Path) -> PropFile:
    path = Path(path)
    data = _read_bytes(path)
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MAGIC or version != 2:
        raise ValueError(f"{path}: expected PROP v2")
    source_xyz = struct.unpack_from("<3f", data, 8)
    n_receivers, n_paths, n_chain = struct.unpack_from("<III", data, 20)
    receiver_base = HEADER_SIZE
    path_base = receiver_base + n_receivers * RECEIVER_SIZE
    chain_base = path_base + n_paths * PATH_SIZE
    expected = chain_base + n_chain * 2
    if expected != len(data):
        raise ValueError(f"{path}: size mismatch, expected {expected}, got {len(data)}")

    path_rows: list[tuple[int, int, int, float, float]] = []
    for index in range(n_paths):
        offset = path_base + index * PATH_SIZE
        values = struct.unpack_from("<HHHHHHHHbBHI", data, offset)
        loss, dist = _half(values[0]), _half(values[1])
        chain_len, path_type, chain_offset = values[7], values[8], values[11]
        path_rows.append((chain_len, path_type, chain_offset, loss, dist))

    chain_values = np.frombuffer(data, dtype="<u2", count=n_chain, offset=chain_base)
    receivers: list[PropReceiver] = []
    for index in range(n_receivers):
        offset = receiver_base + index * RECEIVER_SIZE
        rx_id, x, y, z, _, _, count, _, path_offset = struct.unpack_from(
            "<i3f2fHHI", data, offset
        )
        paths: list[PropPath] = []
        for rank in range(count):
            chain_len, path_type, chain_offset, loss, dist = path_rows[
                path_offset + rank
            ]
            if path_type == -1:
                chain_ids: tuple[int, ...] = ()
            else:
                end = chain_offset + chain_len
                if end > len(chain_values):
                    raise ValueError(f"{path}: chain offset out of range")
                chain_ids = tuple(int(v) for v in chain_values[chain_offset:end])
            paths.append(PropPath(rank, path_type, chain_ids, loss, dist))
        receivers.append(PropReceiver(rx_id, (x, y, z), tuple(paths)))
    return PropFile(tuple(float(v) for v in source_xyz), tuple(receivers))
