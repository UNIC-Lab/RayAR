from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _key(xyz: np.ndarray) -> tuple[float, float]:
    xy = np.asarray(xyz, dtype=np.float32)[:2]
    return float(xy[0]), float(xy[1])


def _edge_features(distance, azimuth, elevation):
    distance = np.maximum(np.asarray(distance, dtype=np.float32), 0.0)
    azimuth = np.asarray(azimuth, dtype=np.float32)
    elevation = np.asarray(elevation, dtype=np.float32)
    return np.stack(
        [
            np.log1p(distance) / np.log1p(np.float32(256.0)),
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation),
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass(frozen=True)
class GeometrySource:
    vertex_grid: np.ndarray
    vertex_edge: np.ndarray
    vertex_distance: np.ndarray
    street_distance: np.ndarray
    street_angle: np.ndarray

    def goal_geometry(self, rx_xyz):
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return _edge_features(
            self.street_distance[:, x, y],
            self.street_angle[:, x, y, 0],
            self.street_angle[:, x, y, 1],
        )

    def goal_distance(self, rx_xyz):
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return np.asarray(self.street_distance[:, x, y], dtype=np.float32)


def load_geometry_source(root, map_id, vertex_xy):
    directory = Path(root) / f"map_{map_id}" / "fundamental"
    vertex_distance = np.load(
        directory / f"vertex_distance_mask_{map_id}.npy", mmap_mode="r"
    )
    vertex_angle = np.load(directory / f"vertex_angle_mask_{map_id}.npy", mmap_mode="r")
    street_distance = np.load(
        directory / f"street_distance_mask_{map_id}.npy", mmap_mode="r"
    )
    street_angle = np.load(directory / f"street_angle_mask_{map_id}.npy", mmap_mode="r")
    vertex_grid = np.clip(
        np.asarray(vertex_xy, dtype=np.float32).astype(np.int64), 0, 256
    )
    distance = vertex_distance[:, vertex_grid[:, 0], vertex_grid[:, 1]]
    angle = vertex_angle[:, vertex_grid[:, 0], vertex_grid[:, 1]]
    edge = _edge_features(
        distance,
        angle,
        np.zeros_like(angle, dtype=np.float32),
    )
    return GeometrySource(
        vertex_grid,
        edge,
        vertex_distance,
        street_distance,
        street_angle,
    )


class OnDemandSource:
    def __init__(self, base, backend):
        self.backend = backend
        self.vertex_edge = base.vertex_edge
        self.vertex_distance = base.vertex_distance
        self.street_distance = base.street_distance
        self.street_angle = base.street_angle
        self._vertex_los: dict[tuple[float, float], np.ndarray] = {}
        self._root_edge: dict[tuple[float, float], np.ndarray] = {}
        self._direct: dict[tuple[float, float], np.ndarray] = {}
        self._rx_index: dict[tuple[float, float], int] = {}

    def prepare(self, tx_xyz: np.ndarray, rx_xyz: np.ndarray) -> None:
        tx_xyz = np.asarray(tx_xyz, dtype=np.float32).reshape(-1, 3)
        rx_xyz = np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3)
        unique = {}
        for tx in tx_xyz:
            unique.setdefault(_key(tx), tx[:2])
        tx_xy = np.asarray(list(unique.values()), dtype=np.float32)
        rx_xy = np.asarray(rx_xyz[:, :2], dtype=np.float32)
        result = self.backend.query(tx_xy, rx_xy)
        self._vertex_los.clear()
        self._root_edge.clear()
        self._direct.clear()
        self._rx_index = {_key(rx): row for row, rx in enumerate(rx_xyz)}
        for row, key in enumerate(unique):
            self._vertex_los[key] = np.asarray(result["vertex_los"][row], dtype=bool)
            self._root_edge[key] = np.asarray(
                result["root_edge"][row], dtype=np.float32
            )
            self._direct[key] = np.asarray(result["rx_los"][row], dtype=bool)

    def vertices(self, tx_xyz: np.ndarray) -> np.ndarray:
        return self._vertex_los[_key(tx_xyz)]

    def direct(self, tx_xyz: np.ndarray, rx_xyz: np.ndarray) -> bool:
        return bool(self._direct[_key(tx_xyz)][self._rx_index[_key(rx_xyz)]])

    def root_geometry(self, tx_xyz: np.ndarray) -> np.ndarray:
        return self._root_edge[_key(tx_xyz)]

    def goal_geometry(self, rx_xyz: np.ndarray) -> np.ndarray:
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        distance = np.maximum(self.street_distance[:, x, y], 0.0)
        angle = self.street_angle[:, x, y]
        return np.stack(
            [
                np.log1p(distance) / np.log1p(np.float32(256.0)),
                np.sin(angle[:, 0]),
                np.cos(angle[:, 0]),
                np.sin(angle[:, 1]),
                np.cos(angle[:, 1]),
            ],
            axis=-1,
        ).astype(np.float32)

    def goal_distance(self, rx_xyz: np.ndarray) -> np.ndarray:
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return np.asarray(self.street_distance[:, x, y], dtype=np.float32)


__all__ = ["GeometrySource", "OnDemandSource", "load_geometry_source"]
