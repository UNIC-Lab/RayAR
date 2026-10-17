from __future__ import annotations

import copy

import numpy as np

from src.data.rays import FeatureBuilder
from src.data.features import PAFeatureBuilder, canonical_wedges
from src.geom.source import load_geometry_source


class CalibrationFeatureBuilder(PAFeatureBuilder):
    def __init__(
        self,
        cache,
        building_root,
        height=20.0,
        k_pre=64,
        max_wall_tokens=32,
        los_root=None,
    ):
        FeatureBuilder.__init__(
            self,
            cache,
            building_root,
            height,
            k_pre,
            max_wall_tokens,
            los_root,
        )
        scene = self.map.scene
        checks = (
            ("vertex coordinates", self.cache.vertex_xyz, scene.vertex_xyz),
            ("vertex normals", self.cache.vertex_normal, scene.vertex_normal),
            (
                "vertex buildings",
                self.cache.vertex_building,
                scene.vertex_building,
            ),
            ("polygon pointers", self.cache.polygon_ptr, scene.polygon_ptr),
        )
        for name, cached, loaded in checks:
            cached = np.asarray(cached)
            loaded = np.asarray(loaded)
            if cached.shape != loaded.shape or not np.array_equal(cached, loaded):
                raise ValueError(f"cache/scene {name} do not match")
        scene_xyz = np.asarray(scene.vertex_xyz, dtype=np.float32)
        self.wedge = canonical_wedges(scene_xyz, scene.polygon_ptr)
        if self.los is None:
            raise ValueError("LOS graph input requires los_root")
        self.source = load_geometry_source(
            los_root,
            cache.map_id,
            scene_xyz[:, :2],
        )
        grid = self.source.vertex_grid
        self.vv_distance = np.asarray(
            self.source.vertex_distance[:, grid[:, 0], grid[:, 1]],
            dtype=np.float32,
        )
        self.los_vv = np.asarray(
            (self.los.vertex_mask[:, grid[:, 0], grid[:, 1]] == 1)
            & (self.vv_distance > 1e-3),
            dtype=bool,
        )
        self.los_graph_mask = self.los_vv.copy()
        np.fill_diagonal(self.los_graph_mask, True)
        self.los_graph_edge = np.asarray(
            self.source.vertex_edge, dtype=np.float32
        ).copy()
        self.los_graph_edge[~self.los_graph_mask] = 0.0
        diagonal = np.arange(len(self.los_graph_edge))
        self.los_graph_edge[diagonal, diagonal] = 0.0
        self._goal_key = None
        self._goal_value = None
        self._goal_ray_key = None
        self._goal_ray_value = None
        self._sparse_ids = None
        self._ray_cache = {}
        self._goal_ray_cache = {}
        self.ray_cache_hits = 0

    def clear_query_cache(self) -> None:
        self._ray_cache.clear()

    def fork(self) -> "CalibrationFeatureBuilder":
        builder = copy.copy(self)
        builder._ray_cache = {}
        builder.ray_cache_hits = 0
        builder._goal_key = None
        builder._goal_value = None
        builder._goal_ray_key = None
        builder._goal_ray_value = None
        builder._sparse_ids = None
        return builder

    def precompute_goal_rays(self, rx_xyz: np.ndarray) -> None:
        vertex = np.asarray(self.map.scene.vertex_xyz, dtype=np.float32)
        for rx in np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3):
            key = self._xyz_key(rx)
            if key in self._goal_ray_cache:
                continue
            targets = np.broadcast_to(rx, vertex.shape).copy()
            target_x = np.broadcast_to(
                self._endpoint_vertex_x(rx),
                (len(vertex), 12),
            ).copy()
            self._goal_ray_cache[key] = FeatureBuilder._ray_context(
                self,
                vertex,
                targets,
                np.zeros((len(vertex), 2), dtype=np.float32),
                target_x,
            )

    def _ray_context(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        target_normals: np.ndarray,
        target_x: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ids = self._sparse_ids
        vertex = np.asarray(self.map.scene.vertex_xyz, dtype=np.float32)
        is_goal = (
            origins.shape == vertex.shape
            and np.array_equal(origins, vertex)
            and bool(np.array_equal(targets, targets[:1]))
        )
        if is_goal:
            key = self._xyz_key(targets[0])
            if key not in self._goal_ray_cache:
                self.precompute_goal_rays(targets[:1])
            value = self._goal_ray_cache[key]
            if ids is None:
                return value
            return {name: array[ids] for name, array in value.items()}

        if ids is not None:
            origins = origins[ids]
            targets = targets[ids]
            target_normals = target_normals[ids]
            target_x = target_x[ids]
            id_key = np.asarray(ids, dtype=np.int32).tobytes()
        else:
            id_key = b"dense"
        origin_constant = bool(np.array_equal(origins, origins[:1]))
        target_constant = bool(np.array_equal(targets, targets[:1]))
        key = (
            id_key,
            origins[0].tobytes() if origin_constant else b"vertex-origin",
            targets[0].tobytes() if target_constant else b"vertex-target",
        )
        cached = self._ray_cache.get(key)
        if cached is not None:
            self.ray_cache_hits += 1
            return cached
        value = FeatureBuilder._ray_context(
            self, origins, targets, target_normals, target_x
        )
        self._ray_cache[key] = value
        return value
