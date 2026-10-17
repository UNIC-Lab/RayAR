from __future__ import annotations

import numpy as np

from src.geom.scene import SceneData


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _on_segment(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-6
) -> bool:
    return (
        abs(_cross(point - a, b - a)) <= eps
        and np.all(point >= np.minimum(a, b) - eps)
        and np.all(point <= np.maximum(a, b) + eps)
    )


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-6
) -> bool:
    ab = b - a
    cd = d - c
    denominator = _cross(ab, cd)
    if abs(denominator) < 1e-8:
        return False
    delta = c - a
    t = _cross(delta, cd) / denominator
    s = _cross(delta, ab) / denominator
    return 1e-10 < t < 1.0 - 1e-10 and 1e-10 < s < 1.0 - 1e-10


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        if _on_segment(point, a, b):
            return False
        crosses = (a[1] > y) != (b[1] > y)
        if crosses:
            x_hit = float((b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
            if x < x_hit:
                inside = not inside
    return inside


class Visibility2D:
    def __init__(self, scene: SceneData):
        self.scene = scene

    def visible(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        ignore_building: int = -1,
        start_vertex: int = -1,
        end_vertex: int = -1,
    ) -> bool:
        start = np.asarray(start_xyz, dtype=np.float32)[:2]
        end = np.asarray(end_xyz, dtype=np.float32)[:2]
        if float(np.linalg.norm(end - start)) < 1e-10:
            return True

        if start_vertex >= 0 and end_vertex >= 0:
            b1 = int(self.scene.vertex_building[start_vertex])
            b2 = int(self.scene.vertex_building[end_vertex])
            if b1 == b2:
                l1 = int(self.scene.vertex_local[start_vertex])
                l2 = int(self.scene.vertex_local[end_vertex])
                polygon = self._polygon_for_building(b1)
                diff = abs(l1 - l2)
                if diff == 1 or diff == len(polygon) - 1:
                    return True
                if _point_in_polygon(0.5 * (start + end), polygon):
                    return False

        for building_id, polygon in self._buildings():
            if building_id == ignore_building:
                continue
            for index, a in enumerate(polygon):
                if _segments_intersect(
                    start, end, a, polygon[(index + 1) % len(polygon)]
                ):
                    return False

        distance = float(np.linalg.norm(end - start))
        steps = int(np.ceil(distance / 0.5))
        force_interior = (
            int(self.scene.vertex_building[start_vertex]) if start_vertex >= 0 else -1
        )
        for step in range(1, steps):
            point = start + (step / steps) * (end - start)
            for building_id, polygon in self._buildings():
                if building_id == ignore_building and building_id != force_interior:
                    continue
                if building_id != ignore_building and any(
                    _on_segment(point, a, polygon[(index + 1) % len(polygon)])
                    for index, a in enumerate(polygon)
                ):
                    return False
                if _point_in_polygon(point, polygon):
                    return False
        return True

    def _buildings(self):
        for index, polygon in enumerate(self.scene.polygons):
            vertex = int(self.scene.polygon_ptr[index])
            yield int(self.scene.vertex_building[vertex]), polygon

    def _polygon_for_building(self, building_id: int) -> np.ndarray:
        for bid, polygon in self._buildings():
            if bid == building_id:
                return polygon
        raise KeyError(building_id)
