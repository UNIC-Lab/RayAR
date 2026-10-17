from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Building:
    building_id: int
    polygon: np.ndarray
    height: float

    @property
    def min_x(self) -> float:
        return float(np.min(self.polygon[:, 0]))

    @property
    def max_x(self) -> float:
        return float(np.max(self.polygon[:, 0]))

    @property
    def min_y(self) -> float:
        return float(np.min(self.polygon[:, 1]))

    @property
    def max_y(self) -> float:
        return float(np.max(self.polygon[:, 1]))


@dataclass
class SceneBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.x_min + self.x_max),
                0.5 * (self.y_min + self.y_max),
                0.5 * (self.z_min + self.z_max),
            ],
            dtype=np.float32,
        )

    @property
    def extent(self) -> np.ndarray:
        return np.array(
            [self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min],
            dtype=np.float32,
        )

    @property
    def scale(self) -> float:
        return float(max(np.max(self.extent), 1e-6))

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))


@dataclass(frozen=True)
class SceneData:
    polygons: tuple[np.ndarray, ...]
    vertex_xyz: np.ndarray
    vertex_normal: np.ndarray
    vertex_building: np.ndarray
    vertex_local: np.ndarray
    polygon_ptr: np.ndarray


def _clean_polygon(points: list, eps: float = 1e-6) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    raw = raw[:, :2]
    if len(raw) >= 2 and np.linalg.norm(raw[0] - raw[-1]) < eps:
        raw = raw[:-1]
    clean: list[np.ndarray] = []
    for point in raw:
        if not clean or np.linalg.norm(clean[-1] - point) >= eps:
            clean.append(point)
    out = np.asarray(clean, dtype=np.float32).reshape(-1, 2)
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) < eps:
        out = out[:-1]
    return out


def load_scene(path: str | Path, height: float = 20.0) -> SceneData:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    polygons: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    building_ids: list[int] = []
    local_ids: list[int] = []
    ptr = [0]

    for building_id, points in enumerate(raw):
        polygon = _clean_polygon(points)
        if len(polygon) < 3:
            continue
        polygons.append(polygon)
        for local_id, current in enumerate(polygon):
            previous = polygon[(local_id - 1) % len(polygon)]
            direction = previous - current
            length = float(np.linalg.norm(direction))
            if length <= 1e-12:
                direction = polygon[(local_id + 1) % len(polygon)] - current
                length = float(np.linalg.norm(direction))
            normal = direction / max(length, 1e-12)
            vertices.append(
                np.asarray([current[0], current[1], height], dtype=np.float32)
            )
            normals.append(normal.astype(np.float32))
            building_ids.append(building_id)
            local_ids.append(local_id)
        ptr.append(len(vertices))

    return SceneData(
        polygons=tuple(polygons),
        vertex_xyz=np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        vertex_normal=np.asarray(normals, dtype=np.float32).reshape(-1, 2),
        vertex_building=np.asarray(building_ids, dtype=np.int32),
        vertex_local=np.asarray(local_ids, dtype=np.int32),
        polygon_ptr=np.asarray(ptr, dtype=np.int32),
    )
