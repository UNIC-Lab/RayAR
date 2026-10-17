from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TWO_HOP_DIM = 13
LOS_CONTEXT_DIM = 24
_LOG_DISTANCE_SCALE = np.log1p(np.float32(256.0))


def _edge_features(
    distance: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
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


def two_hop_context(
    vertex_los: np.ndarray,
    vertex_distance: np.ndarray,
    vertex_edge: np.ndarray,
    goal_los: np.ndarray,
    goal_distance: np.ndarray,
    goal_edge: np.ndarray,
    row_ids: np.ndarray | None = None,
    blocked_ids: np.ndarray | None = None,
    max_relays: int = 1,
) -> np.ndarray:
    """Summarize the shortest candidate-to-relay(s)-to-Rx continuation."""
    row_count, vertex_count = vertex_los.shape
    rows_global = (
        np.arange(row_count, dtype=np.int64)
        if row_ids is None
        else np.asarray(row_ids, dtype=np.int64)
    )
    out = np.zeros((row_count, TWO_HOP_DIM), dtype=np.float32)
    relay_allowed = np.ones(vertex_count, dtype=bool)
    if blocked_ids is not None:
        relay_allowed[np.asarray(blocked_ids, dtype=np.int64)] = False
    out[:, 0] = np.asarray(vertex_los[:, relay_allowed], dtype=bool).sum(axis=1) / max(
        int(relay_allowed.sum()), 1
    )

    goal_ids = np.flatnonzero(relay_allowed & np.asarray(goal_los, dtype=bool))
    if goal_ids.size == 0 or max_relays < 1:
        return out

    tail_cost = np.full(vertex_count, np.inf, dtype=np.float32)
    tail_goal = np.full(vertex_count, -1, dtype=np.int64)
    tail_cost[goal_ids] = np.maximum(goal_distance[goal_ids], 0.0)
    tail_goal[goal_ids] = goal_ids

    if max_relays >= 2:
        if row_count != vertex_count or not np.array_equal(
            rows_global, np.arange(vertex_count, dtype=np.int64)
        ):
            raise ValueError("two-relay context requires the full vertex graph")
        via_goal = (
            np.asarray(vertex_distance[:, goal_ids], dtype=np.float32)
            + np.maximum(goal_distance[goal_ids], 0.0)[None, :]
        )
        via_goal = np.where(vertex_los[:, goal_ids], via_goal, np.inf)
        best_goal_local = np.argmin(via_goal, axis=1)
        best_goal_cost = via_goal[np.arange(vertex_count), best_goal_local]
        improve = relay_allowed & (best_goal_cost < tail_cost)
        tail_cost[improve] = best_goal_cost[improve]
        tail_goal[improve] = goal_ids[best_goal_local[improve]]

    relay_ids = np.flatnonzero(relay_allowed & np.isfinite(tail_cost))
    relay_los = np.asarray(vertex_los[:, relay_ids], dtype=bool)
    cost = np.asarray(vertex_distance[:, relay_ids], dtype=np.float32)
    cost = cost + tail_cost[relay_ids][None, :]
    cost = np.where(relay_los, cost, np.inf)
    usable = ~np.asarray(goal_los, dtype=bool)[rows_global] & relay_los.any(axis=1)
    rows = np.flatnonzero(usable)
    if rows.size == 0:
        return out

    best_local = np.argmin(cost[rows], axis=1)
    best_relay = relay_ids[best_local]
    best_goal = tail_goal[best_relay]
    out[rows, 1] = 1.0
    out[rows, 2] = np.log1p(cost[rows, best_local]) / _LOG_DISTANCE_SCALE
    out[rows, 3:8] = vertex_edge[rows, best_relay]
    out[rows, 8:13] = goal_edge[best_goal]
    return out


@dataclass(frozen=True)
class SourceLOS:
    source_xy: np.ndarray
    source_index: dict[tuple[float, float], int]
    vertex_grid: np.ndarray
    vertex_mask: np.ndarray
    street_mask: np.ndarray
    root_edge: np.ndarray
    vertex_edge: np.ndarray
    vertex_distance: np.ndarray
    street_distance: np.ndarray
    street_angle: np.ndarray

    def _row(self, tx_xyz: np.ndarray) -> int:
        xy = np.asarray(tx_xyz, dtype=np.float32)[:2]
        key = (float(xy[0]), float(xy[1]))
        if key not in self.source_index:
            raise ValueError(f"Tx {xy.tolist()} is not a preprocessed BFS source")
        return self.source_index[key]

    def vertices(self, tx_xyz: np.ndarray) -> np.ndarray:
        row = self._row(tx_xyz)
        return np.asarray(
            self.vertex_mask[row, self.vertex_grid[:, 1], self.vertex_grid[:, 0]] == 1,
            dtype=bool,
        )

    def direct(self, tx_xyz: np.ndarray, rx_xyz: np.ndarray) -> bool:
        row = self._row(tx_xyz)
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return bool(self.street_mask[row, y, x] == 1)

    def root_geometry(self, tx_xyz: np.ndarray) -> np.ndarray:
        return self.root_edge[self._row(tx_xyz)]

    def goal_geometry(self, rx_xyz: np.ndarray) -> np.ndarray:
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return _edge_features(
            self.street_distance[:, x, y],
            self.street_angle[:, x, y, 0],
            self.street_angle[:, x, y, 1],
        )

    def goal_distance(self, rx_xyz: np.ndarray) -> np.ndarray:
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return np.asarray(self.street_distance[:, x, y], dtype=np.float32)


def load_source_los(
    root: str | Path,
    map_id: int,
    cache_tx_xyz: np.ndarray,
    vertex_xy: np.ndarray,
) -> SourceLOS:
    map_root = Path(root) / f"map_{map_id}"
    source_xy = np.load(
        map_root / f"selected_source_points_{map_id}.npy", mmap_mode="r"
    )
    vertex_mask = np.load(
        map_root / f"selected_visibility_vertex_mask_{map_id}.npy",
        mmap_mode="r",
    )
    street_mask = np.load(
        map_root / f"selected_visibility_street_mask_{map_id}.npy",
        mmap_mode="r",
    )
    all_source_xy = np.load(map_root / f"source_points_{map_id}.npy", mmap_mode="r")
    root_distance_all = np.load(
        map_root / f"source_to_vertex_distance_matrix_{map_id}.npy",
        mmap_mode="r",
    )
    root_angle_all = np.load(map_root / f"angle_matrix_{map_id}.npy", mmap_mode="r")
    cache_xy = np.asarray(cache_tx_xyz, dtype=np.float32)[:, :2]
    if source_xy.shape != cache_xy.shape or not np.array_equal(source_xy, cache_xy):
        raise ValueError(f"map_{map_id}: BFS selected sources do not match cache Tx")
    expected = (len(source_xy), 257, 257)
    if vertex_mask.shape != expected or street_mask.shape != expected:
        raise ValueError(f"map_{map_id}: invalid BFS selected LOS mask shape")
    vertex_count = int(np.asarray(vertex_xy).shape[0])
    if root_distance_all.shape != (len(all_source_xy), vertex_count):
        raise ValueError(f"map_{map_id}: invalid source distance shape")
    if root_angle_all.shape != (len(all_source_xy), vertex_count, 2):
        raise ValueError(f"map_{map_id}: invalid source angle shape")

    all_source_index = {
        (float(x), float(y)): row for row, (x, y) in enumerate(all_source_xy)
    }
    selected_rows = []
    for x, y in source_xy:
        key = (float(x), float(y))
        if key not in all_source_index:
            raise ValueError(
                f"map_{map_id}: selected source missing from source_points"
            )
        selected_rows.append(all_source_index[key])

    fundamental = map_root / "fundamental"
    vertex_distance = np.load(
        fundamental / f"vertex_distance_mask_{map_id}.npy", mmap_mode="r"
    )
    vertex_angle = np.load(
        fundamental / f"vertex_angle_mask_{map_id}.npy", mmap_mode="r"
    )
    street_distance = np.load(
        fundamental / f"street_distance_mask_{map_id}.npy", mmap_mode="r"
    )
    street_angle = np.load(
        fundamental / f"street_angle_mask_{map_id}.npy", mmap_mode="r"
    )
    grid_shape = (vertex_count, 257, 257)
    if (
        vertex_distance.shape != grid_shape
        or vertex_angle.shape != grid_shape
        or street_distance.shape != grid_shape
        or street_angle.shape != (*grid_shape, 2)
    ):
        raise ValueError(f"map_{map_id}: invalid BFS geometry mask shape")
    source_index = {(float(x), float(y)): row for row, (x, y) in enumerate(source_xy)}
    vertex_grid = np.clip(
        np.asarray(vertex_xy, dtype=np.float32).astype(np.int64), 0, 256
    )
    rows = np.asarray(selected_rows, dtype=np.int64)
    root_edge = _edge_features(
        root_distance_all[rows],
        root_angle_all[rows, :, 0],
        root_angle_all[rows, :, 1],
    )
    vv_distance = vertex_distance[:, vertex_grid[:, 0], vertex_grid[:, 1]]
    vv_angle = vertex_angle[:, vertex_grid[:, 0], vertex_grid[:, 1]]
    vertex_edge = _edge_features(
        vv_distance,
        vv_angle,
        np.zeros_like(vv_angle, dtype=np.float32),
    )
    return SourceLOS(
        source_xy,
        source_index,
        vertex_grid,
        vertex_mask,
        street_mask,
        root_edge,
        vertex_edge,
        vertex_distance,
        street_distance,
        street_angle,
    )


__all__ = [
    "LOS_CONTEXT_DIM",
    "SourceLOS",
    "load_source_los",
    "two_hop_context",
]
