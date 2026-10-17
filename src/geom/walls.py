"""Static wall/feature scene graph from building footprints and feature points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.geom.config import GraphConfig
from src.geom.scene import Building, SceneBounds


@dataclass
class SceneGraph:
    wall_x: np.ndarray
    wall_pos: np.ndarray
    wall_a: np.ndarray
    wall_b: np.ndarray
    wall_normal: np.ndarray
    wall_height: np.ndarray
    wall_width: np.ndarray
    wall_building_id: np.ndarray
    wall_edge_index: np.ndarray
    wall_edge_attr: np.ndarray
    feature_wall_edge_index: np.ndarray
    feature_wall_edge_attr: np.ndarray
    feature_wall: np.ndarray


def _clean_polygon(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32)
    if poly.shape[0] > 1 and np.linalg.norm(poly[0] - poly[-1]) <= 1e-6:
        poly = poly[:-1]
    return poly


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _wall_feature_array(
    buildings: list[Building],
    bounds: SceneBounds,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    wall_a: list[np.ndarray] = []
    wall_b: list[np.ndarray] = []
    wall_normal: list[np.ndarray] = []
    wall_height: list[float] = []
    wall_width: list[float] = []
    wall_building_id: list[int] = []
    wall_order: list[int] = []
    for building in buildings:
        poly = _clean_polygon(building.polygon)
        if poly.shape[0] < 3:
            continue
        ccw = _signed_area(poly) >= 0.0
        for i in range(poly.shape[0]):
            a = poly[i]
            b = poly[(i + 1) % poly.shape[0]]
            edge = b - a
            width = float(np.linalg.norm(edge))
            if width <= 1e-6:
                continue
            normal = (
                np.array([edge[1], -edge[0]], dtype=np.float32)
                if ccw
                else np.array([-edge[1], edge[0]], dtype=np.float32)
            )
            normal = normal / width
            wall_a.append(a.astype(np.float32))
            wall_b.append(b.astype(np.float32))
            wall_normal.append(normal.astype(np.float32))
            wall_height.append(float(building.height))
            wall_width.append(width)
            wall_building_id.append(int(building.building_id))
            wall_order.append(i)
    a_arr = np.asarray(wall_a, dtype=np.float32).reshape(-1, 2)
    b_arr = np.asarray(wall_b, dtype=np.float32).reshape(-1, 2)
    normal_arr = np.asarray(wall_normal, dtype=np.float32).reshape(-1, 2)
    height_arr = np.asarray(wall_height, dtype=np.float32)
    width_arr = np.asarray(wall_width, dtype=np.float32)
    bid_arr = np.asarray(wall_building_id, dtype=np.int32)
    order_arr = np.asarray(wall_order, dtype=np.int32)
    centroid_xy = 0.5 * (a_arr + b_arr)
    centroid_z = 0.5 * height_arr[:, None]
    wall_pos = np.concatenate([centroid_xy, centroid_z], axis=1).astype(np.float32)
    edge_unit = (b_arr - a_arr) / np.maximum(width_arr[:, None], 1e-6)
    n_buildings = max((b.building_id for b in buildings), default=0) + 1
    wall_x = np.concatenate(
        [
            ((wall_pos - bounds.center[None, :]) / bounds.scale).astype(np.float32),
            np.concatenate(
                [normal_arr, np.zeros((normal_arr.shape[0], 1), dtype=np.float32)],
                axis=1,
            ),
            (height_arr[:, None] / max(bounds.z_max, 1e-6)).astype(np.float32),
            (width_arr[:, None] / max(bounds.scale, 1e-6)).astype(np.float32),
            edge_unit.astype(np.float32),
            (bid_arr[:, None].astype(np.float32) / max(float(n_buildings - 1), 1.0)),
            (
                height_arr[:, None]
                * width_arr[:, None]
                / max(bounds.scale * bounds.scale, 1e-6)
            ).astype(np.float32),
            np.sin(order_arr[:, None].astype(np.float32)),
            np.cos(order_arr[:, None].astype(np.float32)),
            np.ones((height_arr.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    return wall_x, wall_pos, a_arr, b_arr, normal_arr, height_arr, width_arr, bid_arr


def _dedupe(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    valid = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes) & (src != dst)
    if not np.any(valid):
        return np.zeros((2, 0), dtype=np.int64)
    keys = src[valid] * np.int64(n_nodes) + dst[valid]
    keys = np.unique(keys)
    return np.stack([keys // n_nodes, keys % n_nodes], axis=0).astype(np.int64)


def _wall_edge_attr(
    wall_pos: np.ndarray,
    wall_normal: np.ndarray,
    wall_height: np.ndarray,
    wall_building_id: np.ndarray,
    edge_index: np.ndarray,
    bounds: SceneBounds,
) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((0, 10), dtype=np.float32)
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    delta = wall_pos[dst] - wall_pos[src]
    dist = np.linalg.norm(delta, axis=1).astype(np.float32)
    unit = delta / np.maximum(dist[:, None], 1e-6)
    az = np.arctan2(unit[:, 1], unit[:, 0]).astype(np.float32)
    normal_dot = np.sum(wall_normal[src] * wall_normal[dst], axis=1).astype(np.float32)
    same = (wall_building_id[src] == wall_building_id[dst]).astype(np.float32)
    hdiff = (wall_height[dst] - wall_height[src]).astype(np.float32) / max(
        bounds.z_max, 1e-6
    )
    return np.concatenate(
        [
            (delta / bounds.scale).astype(np.float32),
            (dist[:, None] / bounds.scale).astype(np.float32),
            np.log1p(dist)[:, None].astype(np.float32),
            normal_dot[:, None],
            same[:, None],
            hdiff[:, None],
            np.sin(az)[:, None],
            np.cos(az)[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def _build_wall_edges(
    wall_pos: np.ndarray,
    wall_building_id: np.ndarray,
    cfg: GraphConfig,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for bid in np.unique(wall_building_id):
        ids = np.flatnonzero(wall_building_id == bid).astype(np.int64)
        if ids.size <= 1:
            continue
        src = ids
        dst = np.roll(ids, -1)
        parts.append(np.stack([src, dst], axis=0))
        if ids.size > 2:
            parts.append(np.stack([src, np.roll(ids, 1)], axis=0))
    n = int(wall_pos.shape[0])
    if n > 1 and cfg.k_wall_neighbor > 0:
        kk = min(cfg.k_wall_neighbor + 1, n)
        _, idx = cKDTree(wall_pos).query(wall_pos, k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]
        src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1] - 1)
        dst = idx[:, 1:].reshape(-1).astype(np.int64)
        parts.append(np.stack([src, dst], axis=0))
    if not parts:
        return np.zeros((2, 0), dtype=np.int64)
    edge = np.concatenate(parts, axis=1)
    edge = np.concatenate([edge, edge[::-1]], axis=1)
    return _dedupe(edge, n)


def _assign_features_to_walls(
    feature_xyz: np.ndarray,
    feature_normals: np.ndarray,
    wall_a: np.ndarray,
    wall_b: np.ndarray,
    wall_pos: np.ndarray,
    wall_normal: np.ndarray,
    wall_height: np.ndarray,
    wall_width: np.ndarray,
    bounds: SceneBounds,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = int(feature_xyz.shape[0])
    n_walls = int(wall_a.shape[0])
    if n_features == 0 or n_walls == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 8), dtype=np.float32)
    edge = wall_b - wall_a
    denom = np.maximum(np.sum(edge * edge, axis=1), 1e-6)
    assigned = np.zeros((n_features,), dtype=np.int64)
    attrs = np.zeros((n_features, 8), dtype=np.float32)
    normals = (
        feature_normals
        if feature_normals.shape[0] == n_features
        else np.zeros((n_features, 2), dtype=np.float32)
    )
    for i, p in enumerate(feature_xyz.astype(np.float32)):
        rel = p[None, :2] - wall_a
        q = np.clip(np.sum(rel * edge, axis=1) / denom, 0.0, 1.0)
        closest = wall_a + q[:, None] * edge
        dxy = np.linalg.norm(p[None, :2] - closest, axis=1)
        z_penalty = np.maximum(p[2] - wall_height, 0.0) / max(bounds.scale, 1e-6)
        score = dxy / max(bounds.scale, 1e-6) + z_penalty
        wid = int(np.argmin(score))
        assigned[i] = wid
        rel3 = p - wall_pos[wid]
        signed_normal_dist = float(np.dot(p[:2] - wall_pos[wid, :2], wall_normal[wid]))
        feature_normal_dot = float(np.dot(normals[i], wall_normal[wid]))
        attrs[i] = np.asarray(
            [
                rel3[0] / bounds.scale,
                rel3[1] / bounds.scale,
                rel3[2] / bounds.scale,
                q[wid],
                p[2] / max(float(wall_height[wid]), 1e-6),
                signed_normal_dist / max(bounds.scale, 1e-6),
                feature_normal_dot,
                1.0,
            ],
            dtype=np.float32,
        )
    edge_index = np.stack([assigned, np.arange(n_features, dtype=np.int64)], axis=0)
    return edge_index, attrs


def build_scene_graph(
    buildings: list[Building],
    feature_xyz: np.ndarray,
    feature_normals: np.ndarray,
    bounds: SceneBounds,
    cfg: GraphConfig,
) -> SceneGraph:
    wall_x, wall_pos, wall_a, wall_b, wall_normal, wall_height, wall_width, wall_bid = (
        _wall_feature_array(buildings, bounds)
    )
    wall_edge_index = _build_wall_edges(wall_pos, wall_bid, cfg)
    wall_edge_attr = _wall_edge_attr(
        wall_pos, wall_normal, wall_height, wall_bid, wall_edge_index, bounds
    )
    feature_wall_edge_index, feature_wall_edge_attr = _assign_features_to_walls(
        feature_xyz,
        feature_normals,
        wall_a,
        wall_b,
        wall_pos,
        wall_normal,
        wall_height,
        wall_width,
        bounds,
    )
    feature_wall = (
        feature_wall_edge_index[0].astype(np.int64)
        if feature_wall_edge_index.size
        else np.full((feature_xyz.shape[0],), -1, dtype=np.int64)
    )
    return SceneGraph(
        wall_x=wall_x.astype(np.float32),
        wall_pos=wall_pos.astype(np.float32),
        wall_a=wall_a.astype(np.float32),
        wall_b=wall_b.astype(np.float32),
        wall_normal=wall_normal.astype(np.float32),
        wall_height=wall_height.astype(np.float32),
        wall_width=wall_width.astype(np.float32),
        wall_building_id=wall_bid.astype(np.int32),
        wall_edge_index=wall_edge_index.astype(np.int64),
        wall_edge_attr=wall_edge_attr.astype(np.float32),
        feature_wall_edge_index=feature_wall_edge_index.astype(np.int64),
        feature_wall_edge_attr=feature_wall_edge_attr.astype(np.float32),
        feature_wall=feature_wall.astype(np.int64),
    )
