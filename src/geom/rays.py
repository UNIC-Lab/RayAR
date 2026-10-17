from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geom.config import RayGraphConfig
from src.geom.scene import SceneBounds
from src.geom.walls import SceneGraph


@dataclass
class RayWallGraph:
    occ_edge_index: np.ndarray
    occ_edge_attr: np.ndarray
    post_edge_index: np.ndarray
    post_edge_attr: np.ndarray


SPAN_RAW_EDGE_DIM = 34


def ray_wall_edge_dim(cfg: RayGraphConfig) -> int:
    return SPAN_RAW_EDGE_DIM if cfg.wall_candidate_mode == "span_raw" else 15


def _edge_attrs(
    tx: np.ndarray,
    unit: np.ndarray,
    target_dist: float,
    wall_ids: np.ndarray,
    t: np.ndarray,
    rho: np.ndarray,
    angle: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
    cfg: RayGraphConfig,
    is_post: bool,
) -> np.ndarray:
    if wall_ids.size == 0:
        return np.zeros((0, 15), dtype=np.float32)
    wall_pos = scene_graph.wall_pos[wall_ids]
    wall_normal = scene_graph.wall_normal[wall_ids]
    wall_height = scene_graph.wall_height[wall_ids]
    wall_width = scene_graph.wall_width[wall_ids]
    z_at_t = tx[2] + unit[2] * t
    xy_unit = unit[:2] / max(float(np.linalg.norm(unit[:2])), 1e-6)
    normal_dot = np.sum(wall_normal * xy_unit[None, :], axis=1)
    dyn_rho = cfg.rho0 + cfg.alpha * np.maximum(t, 0.0)
    return np.stack(
        [
            t / max(target_dist, 1e-6),
            rho / max(bounds.scale, 1e-6),
            rho / np.maximum(dyn_rho, 1e-6),
            angle,
            np.log1p(np.maximum(t, 0.0)),
            wall_height / max(bounds.z_max, 1e-6),
            (wall_height - z_at_t) / max(bounds.scale, 1e-6),
            wall_width / max(bounds.scale, 1e-6),
            normal_dot,
            np.abs(normal_dot),
            np.full(wall_ids.shape[0], 1.0 if is_post else 0.0, dtype=np.float32),
            np.full(
                wall_ids.shape[0], cfg.rho0 / max(bounds.scale, 1e-6), dtype=np.float32
            ),
            np.full(wall_ids.shape[0], cfg.alpha, dtype=np.float32),
            np.full(
                wall_ids.shape[0],
                np.deg2rad(cfg.max_angle_deg) / np.pi,
                dtype=np.float32,
            ),
            np.ones(wall_ids.shape[0], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _ray_frame(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy_norm = float(np.linalg.norm(unit[:2]))
    if xy_norm > 1e-6:
        side = np.asarray(
            [-unit[1] / xy_norm, unit[0] / xy_norm, 0.0], dtype=np.float32
        )
    else:
        side = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.cross(side, unit).astype(np.float32)
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    return side, up


def _ray_frame_many(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(unit, dtype=np.float32)
    xy_norm = np.linalg.norm(unit[:, :2], axis=1).astype(np.float32)
    side = np.zeros_like(unit, dtype=np.float32)
    valid = xy_norm > 1e-6
    side[valid, 0] = -unit[valid, 1] / xy_norm[valid]
    side[valid, 1] = unit[valid, 0] / xy_norm[valid]
    side[~valid, 0] = 1.0
    up = np.cross(side, unit).astype(np.float32)
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-6)
    return side, up


def _point_ray_frame_coords(
    point: np.ndarray,
    tx: np.ndarray,
    unit: np.ndarray,
    side: np.ndarray,
    up: np.ndarray,
    target_dist: float,
    bounds: SceneBounds,
) -> np.ndarray:
    rel = point - tx[None, :]
    s = rel @ unit / max(float(target_dist), 1e-6)
    lat = rel @ side / max(bounds.scale, 1e-6)
    vert = rel @ up / max(bounds.scale, 1e-6)
    return np.stack([s, lat, vert], axis=1).astype(np.float32)


def _span_raw_attrs(
    tx: np.ndarray,
    unit: np.ndarray,
    target_dist: float,
    wall_ids: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
    is_post: bool,
) -> np.ndarray:
    if wall_ids.size == 0:
        return np.zeros((0, SPAN_RAW_EDGE_DIM), dtype=np.float32)
    side, up = _ray_frame(unit)
    a0 = np.concatenate(
        [
            scene_graph.wall_a[wall_ids],
            np.zeros((wall_ids.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    b0 = np.concatenate(
        [
            scene_graph.wall_b[wall_ids],
            np.zeros((wall_ids.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    a1 = a0.copy()
    b1 = b0.copy()
    a1[:, 2] = scene_graph.wall_height[wall_ids]
    b1[:, 2] = scene_graph.wall_height[wall_ids]
    c = scene_graph.wall_pos[wall_ids]
    normal3 = np.concatenate(
        [
            scene_graph.wall_normal[wall_ids],
            np.zeros((wall_ids.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    normal_frame = np.stack(
        [normal3 @ unit, normal3 @ side, normal3 @ up], axis=1
    ).astype(np.float32)
    point_coords = [
        _point_ray_frame_coords(a0, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(b0, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(a1, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(b1, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(c, tx, unit, side, up, target_dist, bounds),
    ]
    coord_stack = np.stack(point_coords, axis=1)
    s_vals = coord_stack[:, :, 0]
    lat_vals = coord_stack[:, :, 1]
    vert_vals = coord_stack[:, :, 2]
    s_min = np.min(s_vals, axis=1)
    s_max = np.max(s_vals, axis=1)
    s_mid = 0.5 * (s_min + s_max)
    s_span = s_max - s_min
    lat_min_abs = np.min(np.abs(lat_vals), axis=1)
    lat_mean_abs = np.mean(np.abs(lat_vals), axis=1)
    lat_span = np.max(lat_vals, axis=1) - np.min(lat_vals, axis=1)
    vert_min_abs = np.min(np.abs(vert_vals), axis=1)
    vert_mean_abs = np.mean(np.abs(vert_vals), axis=1)
    vert_span = np.max(vert_vals, axis=1) - np.min(vert_vals, axis=1)
    before_gap = np.maximum(np.maximum(-s_max, s_min - 1.0), 0.0)
    after_gap = np.maximum(1.0 - s_max, 0.0)
    center_s_minus_target = point_coords[-1][:, 0] - 1.0
    summary = np.stack(
        [
            s_min,
            s_max,
            s_mid,
            s_span,
            lat_min_abs,
            lat_mean_abs,
            lat_span,
            vert_min_abs,
            vert_mean_abs,
            vert_span,
            before_gap,
            after_gap,
            center_s_minus_target,
            normal_frame[:, 0],
        ],
        axis=1,
    ).astype(np.float32)
    parts = [
        *point_coords,
        normal_frame,
        summary,
        np.full((wall_ids.shape[0], 1), 1.0 if is_post else 0.0, dtype=np.float32),
        np.ones((wall_ids.shape[0], 1), dtype=np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _span_raw_score(
    tx: np.ndarray,
    unit: np.ndarray,
    target_dist: float,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side, up = _ray_frame(unit)
    a0 = np.concatenate(
        [
            scene_graph.wall_a,
            np.zeros((scene_graph.wall_a.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    b0 = np.concatenate(
        [
            scene_graph.wall_b,
            np.zeros((scene_graph.wall_b.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    a1 = a0.copy()
    b1 = b0.copy()
    a1[:, 2] = scene_graph.wall_height
    b1[:, 2] = scene_graph.wall_height
    coords = [
        _point_ray_frame_coords(a0, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(b0, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(a1, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(b1, tx, unit, side, up, target_dist, bounds),
        _point_ray_frame_coords(
            scene_graph.wall_pos, tx, unit, side, up, target_dist, bounds
        ),
    ]
    s_vals = np.stack([c[:, 0] for c in coords], axis=1)
    lat_vals = np.stack([c[:, 1] for c in coords], axis=1)
    vert_vals = np.stack([c[:, 2] for c in coords], axis=1)
    s_min = np.min(s_vals, axis=1)
    s_max = np.max(s_vals, axis=1)
    lat_min_abs = np.min(np.abs(lat_vals), axis=1)
    vert_min_abs = np.min(np.abs(vert_vals), axis=1)
    before_gap = np.maximum(np.maximum(-s_max, s_min - 1.0), 0.0)
    after_gap = np.maximum(1.0 - s_max, 0.0)
    score_base = lat_min_abs + 0.25 * vert_min_abs
    return score_base + before_gap, score_base + after_gap, s_min


def _build_span_raw_ray_wall_graph(
    tx_xyz_per_ray: np.ndarray,
    unit_dir: np.ndarray,
    target_dist: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
    cfg: RayGraphConfig,
) -> RayWallGraph:
    tx_xyz_per_ray = np.asarray(tx_xyz_per_ray, dtype=np.float32)
    unit_dir = np.asarray(unit_dir, dtype=np.float32)
    target_dist = np.asarray(target_dist, dtype=np.float32)
    n_rays = int(tx_xyz_per_ray.shape[0])
    n_walls = int(scene_graph.wall_pos.shape[0])
    if n_rays == 0 or n_walls == 0:
        empty_idx = np.zeros((2, 0), dtype=np.int64)
        empty_attr = np.zeros((0, SPAN_RAW_EDGE_DIM), dtype=np.float32)
        return RayWallGraph(empty_idx, empty_attr, empty_idx.copy(), empty_attr.copy())
    wall_all = np.arange(n_walls, dtype=np.int64)
    occ_src: list[np.ndarray] = []
    occ_dst: list[np.ndarray] = []
    occ_attr: list[np.ndarray] = []
    post_src: list[np.ndarray] = []
    post_dst: list[np.ndarray] = []
    post_attr: list[np.ndarray] = []
    for rid in range(n_rays):
        occ_score, post_score, s_min = _span_raw_score(
            tx_xyz_per_ray[rid],
            unit_dir[rid],
            float(target_dist[rid]),
            scene_graph,
            bounds,
        )
        if int(cfg.k_pre) <= 0 or int(cfg.k_pre) >= n_walls:
            occ_ids = wall_all
        else:
            occ_ids = np.argsort(occ_score, kind="stable")[: int(cfg.k_pre)].astype(
                np.int64
            )
        occ_src.append(occ_ids)
        occ_dst.append(np.full(occ_ids.shape, rid, dtype=np.int64))
        occ_attr.append(
            _span_raw_attrs(
                tx_xyz_per_ray[rid],
                unit_dir[rid],
                float(target_dist[rid]),
                occ_ids,
                scene_graph,
                bounds,
                is_post=False,
            )
        )
        if int(cfg.k_post) <= 0:
            continue
        post_pool = np.flatnonzero(s_min > 0.75).astype(np.int64)
        if post_pool.size == 0:
            continue
        order = np.argsort(post_score[post_pool], kind="stable")[: int(cfg.k_post)]
        post_ids = post_pool[order]
        post_src.append(post_ids)
        post_dst.append(np.full(post_ids.shape, rid, dtype=np.int64))
        post_attr.append(
            _span_raw_attrs(
                tx_xyz_per_ray[rid],
                unit_dir[rid],
                float(target_dist[rid]),
                post_ids,
                scene_graph,
                bounds,
                is_post=True,
            )
        )

    def pack(
        src_parts: list[np.ndarray],
        dst_parts: list[np.ndarray],
        attr_parts: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not src_parts:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(
                (0, SPAN_RAW_EDGE_DIM), dtype=np.float32
            )
        return (
            np.stack(
                [np.concatenate(src_parts), np.concatenate(dst_parts)], axis=0
            ).astype(np.int64),
            np.concatenate(attr_parts, axis=0).astype(np.float32),
        )

    occ_idx, occ_a = pack(occ_src, occ_dst, occ_attr)
    post_idx, post_a = pack(post_src, post_dst, post_attr)
    return RayWallGraph(occ_idx, occ_a, post_idx, post_a)


def _span_raw_coords_many(
    tx: np.ndarray,
    unit: np.ndarray,
    target_dist: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
) -> tuple[np.ndarray, np.ndarray]:
    side, up = _ray_frame_many(unit)
    n_walls = int(scene_graph.wall_a.shape[0])
    zeros = np.zeros((n_walls, 1), dtype=np.float32)
    a0 = np.concatenate([scene_graph.wall_a, zeros], axis=1)
    b0 = np.concatenate([scene_graph.wall_b, zeros], axis=1)
    a1 = a0.copy()
    b1 = b0.copy()
    a1[:, 2] = scene_graph.wall_height
    b1[:, 2] = scene_graph.wall_height
    points = [a0, b0, a1, b1, scene_graph.wall_pos]

    coords: list[np.ndarray] = []
    denom = np.maximum(target_dist[:, None], 1e-6)
    scale = max(bounds.scale, 1e-6)
    for point in points:
        rel = point[None, :, :] - tx[:, None, :]
        s = np.einsum("rwd,rd->rw", rel, unit).astype(np.float32) / denom
        lat = np.einsum("rwd,rd->rw", rel, side).astype(np.float32) / scale
        vert = np.einsum("rwd,rd->rw", rel, up).astype(np.float32) / scale
        coords.append(np.stack([s, lat, vert], axis=2))
    coord_stack = np.stack(coords, axis=2).astype(np.float32)

    normal3 = np.concatenate([scene_graph.wall_normal, zeros], axis=1)
    normal_frame = np.stack(
        [
            np.einsum("wd,rd->rw", normal3, unit),
            np.einsum("wd,rd->rw", normal3, side),
            np.einsum("wd,rd->rw", normal3, up),
        ],
        axis=2,
    ).astype(np.float32)
    return coord_stack, normal_frame


def _span_raw_occ_attrs_from_coords(
    coord_sel: np.ndarray, normal_sel: np.ndarray
) -> np.ndarray:
    if coord_sel.size == 0:
        return np.zeros((0, SPAN_RAW_EDGE_DIM), dtype=np.float32)
    s_vals = coord_sel[:, :, 0]
    lat_vals = coord_sel[:, :, 1]
    vert_vals = coord_sel[:, :, 2]
    s_min = np.min(s_vals, axis=1)
    s_max = np.max(s_vals, axis=1)
    s_mid = 0.5 * (s_min + s_max)
    summary = np.stack(
        [
            s_min,
            s_max,
            s_mid,
            s_max - s_min,
            np.min(np.abs(lat_vals), axis=1),
            np.mean(np.abs(lat_vals), axis=1),
            np.max(lat_vals, axis=1) - np.min(lat_vals, axis=1),
            np.min(np.abs(vert_vals), axis=1),
            np.mean(np.abs(vert_vals), axis=1),
            np.max(vert_vals, axis=1) - np.min(vert_vals, axis=1),
            np.maximum(np.maximum(-s_max, s_min - 1.0), 0.0),
            np.maximum(1.0 - s_max, 0.0),
            coord_sel[:, 4, 0] - 1.0,
            normal_sel[:, 0],
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        [
            coord_sel[:, 0],
            coord_sel[:, 1],
            coord_sel[:, 2],
            coord_sel[:, 3],
            coord_sel[:, 4],
            normal_sel,
            summary,
            np.zeros((coord_sel.shape[0], 1), dtype=np.float32),
            np.ones((coord_sel.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def build_span_raw_occ_graph(
    tx_xyz_per_ray: np.ndarray,
    unit_dir: np.ndarray,
    target_dist: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
    cfg: RayGraphConfig,
) -> RayWallGraph:
    """Build only Graphormer-required span_raw occlusion edges, in ray chunks."""
    if cfg.wall_candidate_mode != "span_raw":
        full = build_ray_wall_graph(
            tx_xyz_per_ray, unit_dir, target_dist, scene_graph, bounds, cfg
        )
        empty_post = np.zeros((2, 0), dtype=np.int64)
        empty_attr = np.zeros((0, ray_wall_edge_dim(cfg)), dtype=np.float32)
        return RayWallGraph(
            full.occ_edge_index, full.occ_edge_attr, empty_post, empty_attr
        )

    tx_xyz_per_ray = np.asarray(tx_xyz_per_ray, dtype=np.float32)
    unit_dir = np.asarray(unit_dir, dtype=np.float32)
    target_dist = np.asarray(target_dist, dtype=np.float32)
    n_rays = int(tx_xyz_per_ray.shape[0])
    n_walls = int(scene_graph.wall_pos.shape[0])
    empty_idx = np.zeros((2, 0), dtype=np.int64)
    empty_attr = np.zeros((0, SPAN_RAW_EDGE_DIM), dtype=np.float32)
    if n_rays == 0 or n_walls == 0:
        return RayWallGraph(empty_idx, empty_attr, empty_idx.copy(), empty_attr.copy())

    k = n_walls if int(cfg.k_pre) <= 0 else min(int(cfg.k_pre), n_walls)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    attr_parts: list[np.ndarray] = []
    step = max(int(cfg.chunk_rays), 1)
    for start in range(0, n_rays, step):
        end = min(start + step, n_rays)
        tx = tx_xyz_per_ray[start:end]
        unit = unit_dir[start:end]
        dist = target_dist[start:end]
        coord_stack, normal_frame = _span_raw_coords_many(
            tx, unit, dist, scene_graph, bounds
        )

        s_vals = coord_stack[:, :, :, 0]
        lat_vals = coord_stack[:, :, :, 1]
        vert_vals = coord_stack[:, :, :, 2]
        s_min = np.min(s_vals, axis=2)
        s_max = np.max(s_vals, axis=2)
        score = np.min(np.abs(lat_vals), axis=2) + 0.25 * np.min(
            np.abs(vert_vals), axis=2
        )
        score = score + np.maximum(np.maximum(-s_max, s_min - 1.0), 0.0)
        if k >= n_walls:
            cand = np.broadcast_to(
                np.arange(n_walls, dtype=np.int64), (end - start, n_walls)
            )
        else:
            cand = np.argsort(score, axis=1, kind="stable")[:, :k].astype(np.int64)

        rows = np.arange(end - start, dtype=np.int64)[:, None]
        coord_sel = coord_stack[rows, cand].reshape(-1, 5, 3)
        normal_sel = normal_frame[rows, cand].reshape(-1, 3)
        src_parts.append(cand.reshape(-1).astype(np.int64))
        dst_parts.append(np.repeat(np.arange(start, end, dtype=np.int64), k))
        attr_parts.append(_span_raw_occ_attrs_from_coords(coord_sel, normal_sel))

    occ_idx = np.stack(
        [np.concatenate(src_parts), np.concatenate(dst_parts)], axis=0
    ).astype(np.int64)
    occ_attr = np.concatenate(attr_parts, axis=0).astype(np.float32)
    return RayWallGraph(occ_idx, occ_attr, empty_idx.copy(), empty_attr.copy())


def build_ray_wall_graph(
    tx_xyz_per_ray: np.ndarray,
    unit_dir: np.ndarray,
    target_dist: np.ndarray,
    scene_graph: SceneGraph,
    bounds: SceneBounds,
    cfg: RayGraphConfig,
) -> RayWallGraph:
    """Build coarse feature-wall occlusion edges.

    This uses wall centroids, along-ray projection, angular deviation, and lateral
    distance only. It deliberately does not compute ray-wall intersections or first
    hit/blocker information.
    """
    if cfg.wall_candidate_mode == "span_raw":
        return _build_span_raw_ray_wall_graph(
            tx_xyz_per_ray, unit_dir, target_dist, scene_graph, bounds, cfg
        )
    tx_xyz_per_ray = np.asarray(tx_xyz_per_ray, dtype=np.float32)
    unit_dir = np.asarray(unit_dir, dtype=np.float32)
    target_dist = np.asarray(target_dist, dtype=np.float32)
    n_rays = int(tx_xyz_per_ray.shape[0])
    n_walls = int(scene_graph.wall_pos.shape[0])
    if n_rays == 0 or n_walls == 0:
        return RayWallGraph(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 15), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 15), dtype=np.float32),
        )
    theta = np.deg2rad(cfg.max_angle_deg)
    occ_src: list[np.ndarray] = []
    occ_dst: list[np.ndarray] = []
    occ_attr: list[np.ndarray] = []
    post_src: list[np.ndarray] = []
    post_dst: list[np.ndarray] = []
    post_attr: list[np.ndarray] = []

    for start in range(0, n_rays, max(int(cfg.chunk_rays), 1)):
        end = min(start + max(int(cfg.chunk_rays), 1), n_rays)
        tx = tx_xyz_per_ray[start:end]
        unit = unit_dir[start:end]
        dist = target_dist[start:end]
        rel = scene_graph.wall_pos[None, :, :] - tx[:, None, :]
        t = np.einsum("rwd,rd->rw", rel, unit).astype(np.float32)
        perp = rel - t[:, :, None] * unit[:, None, :]
        rho = np.linalg.norm(perp, axis=2).astype(np.float32)
        angle = np.arctan2(rho, np.maximum(t, 1e-6)).astype(np.float32)
        dyn_rho = cfg.rho0 + cfg.alpha * np.maximum(t, 0.0)

        for local in range(end - start):
            rid = start + local
            before = (
                (t[local] > 0.0)
                & (t[local] < dist[local])
                & (angle[local] <= theta)
                & (rho[local] <= dyn_rho[local])
            )
            ids = np.flatnonzero(before).astype(np.int64)
            if ids.size:
                score = angle[local, ids] / max(theta, 1e-6)
                score += rho[local, ids] / np.maximum(dyn_rho[local, ids], 1e-6)
                score += np.abs(t[local, ids] - 0.5 * dist[local]) / max(
                    float(dist[local]), 1e-6
                )
                order = np.argsort(score, kind="stable")[: cfg.k_pre]
                ids = ids[order]
                occ_src.append(ids)
                occ_dst.append(np.full(ids.shape, rid, dtype=np.int64))
                occ_attr.append(
                    _edge_attrs(
                        tx[local],
                        unit[local],
                        float(dist[local]),
                        ids,
                        t[local, ids],
                        rho[local, ids],
                        angle[local, ids],
                        scene_graph,
                        bounds,
                        cfg,
                        is_post=False,
                    )
                )

            after = (
                (t[local] > dist[local])
                & (t[local] < bounds.diagonal * 1.5)
                & (angle[local] <= theta)
                & (rho[local] <= dyn_rho[local])
            )
            ids = np.flatnonzero(after).astype(np.int64)
            if ids.size:
                score = angle[local, ids] / max(theta, 1e-6)
                score += rho[local, ids] / np.maximum(dyn_rho[local, ids], 1e-6)
                score += (t[local, ids] - dist[local]) / max(bounds.scale, 1e-6)
                order = np.argsort(score, kind="stable")[: cfg.k_post]
                ids = ids[order]
                post_src.append(ids)
                post_dst.append(np.full(ids.shape, rid, dtype=np.int64))
                post_attr.append(
                    _edge_attrs(
                        tx[local],
                        unit[local],
                        float(dist[local]),
                        ids,
                        t[local, ids],
                        rho[local, ids],
                        angle[local, ids],
                        scene_graph,
                        bounds,
                        cfg,
                        is_post=True,
                    )
                )

    def pack(
        src_parts: list[np.ndarray],
        dst_parts: list[np.ndarray],
        attr_parts: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not src_parts:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 15), dtype=np.float32)
        return (
            np.stack(
                [np.concatenate(src_parts), np.concatenate(dst_parts)], axis=0
            ).astype(np.int64),
            np.concatenate(attr_parts, axis=0).astype(np.float32),
        )

    occ_idx, occ_a = pack(occ_src, occ_dst, occ_attr)
    post_idx, post_a = pack(post_src, post_dst, post_attr)
    return RayWallGraph(occ_idx, occ_a, post_idx, post_a)
