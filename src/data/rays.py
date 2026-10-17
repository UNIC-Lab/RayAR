from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geom.features import (
    FeatureMeta,
    feature_node_features,
    ray_geometry_features,
    tx_node_features,
    FEATURE_TYPE_VERTICAL_CORNER,
)
from src.geom.rays import build_span_raw_occ_graph
from src.geom.scene import Building, SceneBounds, SceneData, load_scene
from src.geom.walls import build_scene_graph
from src.geom.config import GraphConfig, RayGraphConfig
from src.data.cache import MapCache
from src.data.visibility import LosLabels, load_los_labels
from src.data.dataset import PathSample


@dataclass
class MapFeatures:
    scene: SceneData
    bounds: SceneBounds
    vertex_x: np.ndarray
    graph: object


class FeatureBuilder:
    def __init__(
        self,
        cache: MapCache,
        building_root: str | Path,
        height: float = 20.0,
        k_pre: int = 64,
        max_wall_tokens: int = 32,
        los_root: str | Path | None = None,
    ):
        self.cache = cache
        self.max_wall_tokens = int(max_wall_tokens)
        scene = load_scene(Path(building_root) / f"{cache.map_id}.json", height)
        bounds = SceneBounds(0.0, 256.0, 0.0, 256.0, 0.0, height)
        buildings = [
            Building(bid, poly, height) for bid, poly in self._buildings(scene)
        ]
        meta = FeatureMeta(
            building_id=scene.vertex_building,
            feature_type=np.full(
                scene.vertex_xyz.shape[0],
                FEATURE_TYPE_VERTICAL_CORNER,
                dtype=np.int8,
            ),
            height_norm=np.ones(scene.vertex_xyz.shape[0], dtype=np.float32),
            building_height=np.full(
                scene.vertex_xyz.shape[0], height, dtype=np.float32
            ),
        )
        vertex_x = feature_node_features(
            scene.vertex_xyz,
            scene.vertex_normal,
            meta,
            bounds,
        )
        graph = build_scene_graph(
            buildings,
            scene.vertex_xyz,
            scene.vertex_normal,
            bounds,
            GraphConfig(),
        )
        self.map = MapFeatures(scene, bounds, vertex_x, graph)
        self.los: LosLabels | None = (
            load_los_labels(los_root, cache.map_id, scene.vertex_xyz[:, :2])
            if los_root is not None
            else None
        )
        self.ray_cfg = RayGraphConfig(
            wall_candidate_mode="span_raw",
            k_pre=k_pre,
            k_post=0,
            chunk_rays=256,
        )
        self.building = np.asarray(scene.vertex_building, dtype=np.int64)
        self.neighbor = self._neighbors(scene.polygon_ptr, len(scene.vertex_xyz))

    @staticmethod
    def _buildings(scene: SceneData):
        for index, polygon in enumerate(scene.polygons):
            vertex = int(scene.polygon_ptr[index])
            yield int(scene.vertex_building[vertex]), polygon

    @staticmethod
    def _neighbors(polygon_ptr: np.ndarray, n_vertex: int) -> np.ndarray:
        neighbor = np.empty((n_vertex, 2), dtype=np.int64)
        for start, end in zip(polygon_ptr[:-1], polygon_ptr[1:]):
            ids = np.arange(int(start), int(end), dtype=np.int64)
            neighbor[ids, 0] = np.roll(ids, 1)
            neighbor[ids, 1] = np.roll(ids, -1)
        return neighbor

    def _endpoint_vertex_x(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float32)
        pos = (xyz - self.map.bounds.center) / self.map.bounds.scale
        z_norm = xyz[2] / self.map.bounds.z_max
        return np.asarray(
            [
                pos[0],
                pos[1],
                pos[2],
                z_norm,
                0.0,
                0.0,
                0.0,
                z_norm,
                1.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def _ray_context(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        target_normals: np.ndarray,
        target_x: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ray_x, unit, dist, _ = ray_geometry_features(
            origins,
            targets,
            target_normals,
            self.map.bounds,
        )
        origin_x = tx_node_features(origins, self.map.bounds)
        wall_graph = build_span_raw_occ_graph(
            origins,
            unit,
            dist,
            self.map.graph,
            self.map.bounds,
            self.ray_cfg,
        )
        n_ray = origins.shape[0]
        wall_dim = self.map.graph.wall_x.shape[1]
        edge_dim = (
            wall_graph.occ_edge_attr.shape[1]
            if wall_graph.occ_edge_attr.ndim == 2
            else 34
        )
        wall_x = np.zeros((n_ray, self.max_wall_tokens, wall_dim), dtype=np.float32)
        edge_x = np.zeros((n_ray, self.max_wall_tokens, edge_dim), dtype=np.float32)
        wall_mask = np.ones((n_ray, self.max_wall_tokens), dtype=bool)
        if wall_graph.occ_edge_index.size:
            per_ray = wall_graph.occ_edge_index.shape[1] // n_ray
            take = min(per_ray, self.max_wall_tokens)
            wall_ids = wall_graph.occ_edge_index[0].reshape(n_ray, per_ray)[:, :take]
            attrs = wall_graph.occ_edge_attr.reshape(n_ray, per_ray, edge_dim)[:, :take]
            wall_x[:, :take] = self.map.graph.wall_x[wall_ids]
            edge_x[:, :take] = attrs
            wall_mask[:, :take] = False
        return {
            "ray": ray_x,
            "origin": origin_x,
            "target": target_x,
            "wall": wall_x,
            "edge": edge_x,
            "wall_mask": wall_mask,
        }

    def _path_state(
        self,
        sample: PathSample,
        current: np.ndarray,
        vertex_xyz: np.ndarray,
    ) -> np.ndarray:
        scale = self.map.bounds.scale
        center = self.map.bounds.center
        tx = (sample.tx_xyz - center) / scale
        cur = (current - center) / scale
        rx = (sample.rx_xyz - center) / scale
        if sample.prefix:
            previous = (
                sample.tx_xyz
                if len(sample.prefix) == 1
                else vertex_xyz[sample.prefix[-2]]
            )
            incoming = current[:2] - previous[:2]
            norm = float(np.linalg.norm(incoming))
            incoming = incoming / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)
        else:
            incoming = np.zeros(2, dtype=np.float32)
        hop = np.zeros(4, dtype=np.float32)
        hop[min(len(sample.prefix), 3)] = 1.0
        points = [sample.tx_xyz] + [vertex_xyz[v] for v in sample.prefix]
        cumulative = (
            sum(
                float(np.linalg.norm(points[index + 1] - points[index]))
                for index in range(len(points) - 1)
            )
            / scale
        )
        return np.concatenate(
            [
                tx,
                cur,
                rx,
                incoming,
                hop,
                np.asarray([cumulative], dtype=np.float32),
            ]
        ).astype(np.float32)

    def build(
        self,
        sample: PathSample,
        with_los_labels: bool = False,
    ) -> dict[str, np.ndarray]:
        vertex_xyz = self.map.scene.vertex_xyz
        n_vertex = len(vertex_xyz)
        current = sample.tx_xyz if not sample.prefix else vertex_xyz[sample.prefix[-1]]
        current_all = np.broadcast_to(current, (n_vertex, 3)).copy()
        current_ctx = self._ray_context(
            current_all,
            vertex_xyz,
            self.map.scene.vertex_normal,
            self.map.vertex_x,
        )

        rx_all = np.broadcast_to(sample.rx_xyz, (n_vertex, 3)).copy()
        rx_target_x = np.broadcast_to(
            self._endpoint_vertex_x(sample.rx_xyz),
            (n_vertex, 12),
        ).copy()
        goal_ctx = self._ray_context(
            vertex_xyz,
            rx_all,
            np.zeros((n_vertex, 2), dtype=np.float32),
            rx_target_x,
        )

        scale = self.map.bounds.scale
        rel_tx = (vertex_xyz - sample.tx_xyz) / scale
        rel_cur = (vertex_xyz - current) / scale
        rel_rx = (sample.rx_xyz - vertex_xyz) / scale
        d_tx = np.linalg.norm(vertex_xyz - sample.tx_xyz, axis=1)
        d_cur = np.linalg.norm(vertex_xyz - current, axis=1)
        d_rx = np.linalg.norm(sample.rx_xyz - vertex_xyz, axis=1)
        current_rx = max(float(np.linalg.norm(sample.rx_xyz - current)), 1e-6)
        progress = (current_rx - d_rx) / scale
        detour = (d_cur + d_rx) / current_rx
        incoming = (
            current - sample.tx_xyz
            if not sample.prefix
            else current
            - (
                sample.tx_xyz
                if len(sample.prefix) == 1
                else vertex_xyz[sample.prefix[-2]]
            )
        )
        outgoing = vertex_xyz - current
        in_norm = max(float(np.linalg.norm(incoming)), 1e-6)
        out_norm = np.maximum(np.linalg.norm(outgoing, axis=1), 1e-6)
        turn_cos = np.sum(outgoing * incoming[None, :], axis=1) / (in_norm * out_norm)
        turn_sin = np.cross(
            np.broadcast_to(incoming[:2], (n_vertex, 2)),
            outgoing[:, :2],
        ) / (in_norm * out_norm)
        hop = np.zeros((n_vertex, 4), dtype=np.float32)
        hop[:, min(len(sample.prefix), 3)] = 1.0
        points = [sample.tx_xyz] + [vertex_xyz[v] for v in sample.prefix]
        cumulative = (
            sum(
                float(np.linalg.norm(points[index + 1] - points[index]))
                for index in range(len(points) - 1)
            )
            / scale
        )
        state = np.concatenate(
            [
                rel_tx,
                rel_cur,
                rel_rx,
                (d_tx / scale)[:, None],
                (d_cur / scale)[:, None],
                (d_rx / scale)[:, None],
                progress[:, None],
                detour[:, None],
                turn_cos[:, None],
                turn_sin[:, None],
                hop,
                np.full((n_vertex, 1), cumulative, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        visited = np.zeros(n_vertex, dtype=bool)
        if sample.prefix:
            visited[np.asarray(sample.prefix, dtype=np.int64)] = True
        positive = np.zeros(n_vertex, dtype=bool)
        if sample.positives:
            positive[np.asarray(sample.positives, dtype=np.int64)] = True
        output = {
            "vertex": self.map.vertex_x,
            "state": state,
            "path_state": self._path_state(sample, current, vertex_xyz),
            "building": self.building,
            "neighbor": self.neighbor,
            "visited": visited,
            "positive": positive,
            **{f"cur_{key}": value for key, value in current_ctx.items()},
            **{f"goal_{key}": value for key, value in goal_ctx.items()},
        }
        if with_los_labels:
            if self.los is None:
                raise ValueError(
                    "LOS labels are only available in a training FeatureBuilder"
                )
            output["cur_los"] = (
                self.los.current(sample.prefix[-1])
                if sample.prefix
                else np.zeros(n_vertex, dtype=bool)
            )
            output["cur_los_valid"] = np.full(
                n_vertex,
                bool(sample.prefix),
                dtype=bool,
            )
            output["goal_los"] = self.los.goal(sample.rx_xyz)
        return output


def collate_features(
    items: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("empty feature batch")
    return {key: np.stack([item[key] for item in items], axis=0) for key in items[0]}
