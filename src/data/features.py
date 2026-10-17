from __future__ import annotations

import numpy as np

from src.data.rays import FeatureBuilder
from src.data.dataset import PathSample, canonical_chain
from src.data.los import TWO_HOP_DIM, load_source_los, two_hop_context


_CANDIDATE_KEYS = frozenset(
    {
        "vertex",
        "state",
        "visited",
        "positive",
        "cur_ray",
        "cur_origin",
        "cur_target",
        "cur_wall",
        "cur_edge",
        "cur_wall_mask",
        "goal_ray",
        "goal_origin",
        "goal_target",
        "goal_wall",
        "goal_edge",
        "goal_wall_mask",
        "transition",
        "los_current",
        "los_goal",
        "los_context",
        "candidate_mask",
        "candidate_ids",
    }
)
_PAD_TRUE_KEYS = frozenset({"cur_wall_mask", "goal_wall_mask"})
_SHARED_GRAPH_KEYS = frozenset({"los_graph_mask", "los_graph_edge"})


def collate_sparse_features(
    items: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    counts = [len(item["vertex"]) for item in items]
    width = max(counts)
    batch: dict[str, np.ndarray] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if key in _SHARED_GRAPH_KEYS:
            batch[key] = values[0]
            continue
        if key not in _CANDIDATE_KEYS:
            batch[key] = np.stack(values, axis=0)
            continue

        shape = (len(items), width, *values[0].shape[1:])
        fill = key in _PAD_TRUE_KEYS
        padded = np.full(shape, fill, dtype=values[0].dtype)
        for row, value in enumerate(values):
            padded[row, : len(value)] = value
        batch[key] = padded

    candidate_mask = np.zeros((len(items), width), dtype=bool)
    for row, count in enumerate(counts):
        candidate_mask[row, :count] = True
    batch["candidate_mask"] = candidate_mask
    target_rx = np.zeros(len(items), dtype=bool)
    batch["positive"] = np.concatenate([batch["positive"], target_rx[:, None]], axis=1)
    return batch


def _unit_xy(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.divide(
        delta,
        norm,
        out=np.zeros_like(delta, dtype=np.float32),
        where=norm > 1e-6,
    )


def canonical_wedges(
    vertex_xyz: np.ndarray,
    polygon_ptr: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(vertex_xyz, dtype=np.float32)
    out = np.zeros((len(xyz), 7), dtype=np.float32)
    for start, end in zip(polygon_ptr[:-1], polygon_ptr[1:]):
        ids = np.arange(int(start), int(end), dtype=np.int64)
        xy = xyz[ids, :2]
        area2 = np.sum(
            xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)
        )
        if area2 >= 0.0:
            prev = np.roll(ids, 1)
            following = np.roll(ids, -1)
        else:
            prev = np.roll(ids, -1)
            following = np.roll(ids, 1)

        edge_prev = _unit_xy(xyz[prev, :2] - xyz[ids, :2])
        edge_next = _unit_xy(xyz[following, :2] - xyz[ids, :2])
        incoming = -edge_prev
        turn = np.arctan2(
            incoming[:, 0] * edge_next[:, 1] - incoming[:, 1] * edge_next[:, 0],
            np.sum(incoming * edge_next, axis=1),
        )
        angle = np.pi - turn
        out[ids, :2] = edge_prev
        out[ids, 2:4] = edge_next
        out[ids, 4] = np.sin(angle)
        out[ids, 5] = np.cos(angle)
        out[ids, 6] = angle > np.pi
    return out


class PAFeatureBuilder(FeatureBuilder):
    def __init__(
        self,
        cache,
        building_root,
        height: float = 20.0,
        k_pre: int = 64,
        max_wall_tokens: int = 32,
        los_root=None,
    ):
        super().__init__(
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
            ("vertex buildings", self.cache.vertex_building, scene.vertex_building),
            ("polygon pointers", self.cache.polygon_ptr, scene.polygon_ptr),
        )
        for name, cached, loaded in checks:
            cached = np.asarray(cached)
            loaded = np.asarray(loaded)
            if cached.shape != loaded.shape or not np.array_equal(cached, loaded):
                raise ValueError(f"cache/scene {name} do not match")
        scene_xyz = np.asarray(scene.vertex_xyz, dtype=np.float32)
        self.wedge = canonical_wedges(
            scene_xyz,
            scene.polygon_ptr,
        )
        if self.los is None:
            raise ValueError("LOS graph input requires los_root")
        self.source = load_source_los(
            los_root,
            cache.map_id,
            cache.tx_xyz,
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
        self._goal_key: tuple[float, ...] | None = None
        self._goal_value: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._goal_ray_key: tuple[float, ...] | None = None
        self._goal_ray_value: dict[str, np.ndarray] | None = None
        self._sparse_ids: np.ndarray | None = None

    @staticmethod
    def _xyz_key(xyz: np.ndarray) -> tuple[float, ...]:
        return tuple(np.asarray(xyz, dtype=np.float32).tolist())

    def _goal_context(
        self,
        xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = self._xyz_key(xyz)
        if key != self._goal_key:
            goal_los = self.los.goal(xyz)
            goal_edge = self.source.goal_geometry(xyz).copy()
            goal_edge[~goal_los] = 0.0
            self._goal_key = key
            self._goal_value = (
                goal_los,
                goal_edge,
                self.source.goal_distance(xyz),
            )
        return self._goal_value

    def endpoint_los(self, xyz: np.ndarray) -> np.ndarray:
        return self._goal_context(xyz)[0]

    def _ray_context(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        target_normals: np.ndarray,
        target_x: np.ndarray,
    ) -> dict[str, np.ndarray]:
        vertex = np.asarray(self.map.scene.vertex_xyz, dtype=np.float32)
        is_goal = origins.shape == vertex.shape and np.array_equal(origins, vertex)
        if self._sparse_ids is not None:
            ids = self._sparse_ids
            return super()._ray_context(
                origins[ids],
                targets[ids],
                target_normals[ids],
                target_x[ids],
            )
        if not is_goal:
            return super()._ray_context(origins, targets, target_normals, target_x)

        key = self._xyz_key(targets[0])
        if key != self._goal_ray_key:
            self._goal_ray_key = key
            self._goal_ray_value = super()._ray_context(
                origins, targets, target_normals, target_x
            )
        return self._goal_ray_value

    def source_los(self, tx_xyz: np.ndarray) -> np.ndarray:
        return self.source.vertices(tx_xyz)

    def direct_los(self, tx_xyz: np.ndarray, rx_xyz: np.ndarray) -> bool:
        return self.source.direct(tx_xyz, rx_xyz)

    def legal_candidates(
        self,
        tx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> np.ndarray:
        return self.source_los(tx_xyz) if not prefix else self.los_vv[prefix[-1]]

    def legal_ids(
        self,
        tx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> np.ndarray:
        legal = self.legal_candidates(tx_xyz, prefix).copy()
        if prefix:
            legal[np.asarray(prefix, dtype=np.int64)] = False
        return np.flatnonzero(legal)

    def visible_to_rx(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> bool:
        if not prefix:
            return self.direct_los(tx_xyz, rx_xyz)
        return bool(self.endpoint_los(rx_xyz)[prefix[-1]])

    def route_chain(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        chain: tuple[int, ...],
    ) -> tuple[int, ...]:
        return canonical_chain(
            chain,
            self.direct_los(tx_xyz, rx_xyz),
            self.endpoint_los(rx_xyz),
        )

    def _los_features(
        self,
        sample: PathSample,
        candidate_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        all_current = self.legal_candidates(sample.tx_xyz, sample.prefix)
        ids = (
            np.arange(len(all_current), dtype=np.int64)
            if candidate_ids is None
            else np.asarray(candidate_ids, dtype=np.int64)
        )
        los_current = all_current[ids]
        los_rx, all_goal_edge, goal_distance = self._goal_context(sample.rx_xyz)
        root_edge = self.source.root_geometry(sample.tx_xyz)[ids]
        step_edge = (
            root_edge
            if not sample.prefix
            else self.source.vertex_edge[sample.prefix[-1], ids]
        ).copy()
        step_edge[~los_current] = 0.0
        depth = len(sample.prefix)
        if depth >= 2:
            continuation = np.zeros((len(ids), TWO_HOP_DIM), dtype=np.float32)
        elif depth == 0:
            full_continuation = two_hop_context(
                self.los_vv,
                self.vv_distance,
                self.source.vertex_edge,
                los_rx,
                goal_distance,
                all_goal_edge,
                max_relays=2,
            )
            continuation = full_continuation[ids]
        else:
            continuation = two_hop_context(
                self.los_vv[ids],
                self.vv_distance[ids],
                self.source.vertex_edge[ids],
                los_rx,
                goal_distance,
                all_goal_edge,
                ids,
                np.asarray(sample.prefix, dtype=np.int64),
            )
        los_context = np.concatenate(
            [
                step_edge,
                los_rx[ids, None].astype(np.float32),
                all_goal_edge[ids],
                continuation,
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "los_current": los_current.astype(bool, copy=False),
            "los_goal": los_rx[ids].astype(bool, copy=False),
            "los_context": los_context,
        }

    def _geometry(
        self,
        sample: PathSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vertex = np.asarray(self.map.scene.vertex_xyz, dtype=np.float32)
        center = np.asarray(self.map.bounds.center, dtype=np.float32)
        scale = float(self.map.bounds.scale)
        depth = len(sample.prefix)
        current = sample.tx_xyz if depth == 0 else vertex[sample.prefix[-1]]
        previous = sample.tx_xyz if depth <= 1 else vertex[sample.prefix[-2]]

        incoming = _unit_xy((current - previous)[:2])
        depth_oh = np.zeros(4, dtype=np.float32)
        depth_oh[depth] = 1.0
        points = [sample.tx_xyz] + [vertex[index] for index in sample.prefix]
        cumulative = sum(
            float(np.linalg.norm(points[index + 1] - points[index]))
            for index in range(len(points) - 1)
        )
        current_wedge = (
            self.wedge[sample.prefix[-1]] if depth else np.zeros(7, dtype=np.float32)
        )
        query = np.concatenate(
            [
                (sample.tx_xyz - center) / scale,
                (sample.rx_xyz - center) / scale,
                (current - center) / scale,
                (previous - center) / scale,
                incoming,
                depth_oh,
                np.asarray([cumulative / scale], dtype=np.float32),
                current_wedge,
            ]
        ).astype(np.float32)

        from_current = vertex - current
        to_rx = sample.rx_xyz - vertex
        current_dir = _unit_xy(from_current[:, :2])
        rx_dir = _unit_xy(to_rx[:, :2])
        turn_sin = incoming[0] * current_dir[:, 1] - incoming[1] * current_dir[:, 0]
        turn_cos = np.sum(current_dir * incoming[None, :], axis=1)
        d_cur = np.linalg.norm(from_current, axis=1)
        d_rx = np.linalg.norm(to_rx, axis=1)
        log_scale = np.log1p(scale)
        transition = np.concatenate(
            [
                (vertex - center) / scale,
                from_current / scale,
                to_rx / scale,
                (np.log1p(d_cur) / log_scale)[:, None],
                (np.log1p(d_rx) / log_scale)[:, None],
                current_dir,
                rx_dir,
                turn_sin[:, None],
                turn_cos[:, None],
                self.wedge,
            ],
            axis=1,
        ).astype(np.float32)
        return query, transition, current, previous

    def build(
        self,
        sample: PathSample,
        with_los_labels: bool = False,
    ) -> dict[str, np.ndarray]:
        out = super().build(sample, with_los_labels=False)
        query, transition, _, _ = self._geometry(sample)
        depth = len(sample.prefix)
        tail = np.full(2, -1, dtype=np.int64)
        if depth == 1:
            tail[1] = sample.prefix[-1]
        elif depth >= 2:
            tail[:] = sample.prefix[-2:]

        out.update(
            query_geom=query,
            transition=transition,
            depth=np.int64(depth),
            tail_ids=tail,
            vertex_xyz_m=np.asarray(
                self.map.scene.vertex_xyz,
                dtype=np.float32,
            ),
            rx_available=np.bool_(
                self.visible_to_rx(sample.tx_xyz, sample.rx_xyz, sample.prefix)
            ),
            **self._los_features(sample),
        )
        return out

    def build_sparse(
        self,
        sample: PathSample,
        candidate_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(candidate_ids, dtype=np.int64)
        self._sparse_ids = ids
        try:
            out = super().build(sample, with_los_labels=False)
        finally:
            self._sparse_ids = None

        for key in (
            "vertex",
            "state",
            "visited",
            "positive",
        ):
            out[key] = out[key][ids]
        out.pop("building")
        out.pop("neighbor")

        query, transition, _, _ = self._geometry(sample)
        depth = len(sample.prefix)
        goal_los, goal_edge, _ = self._goal_context(sample.rx_xyz)
        out.update(
            query_geom=query,
            transition=transition[ids],
            depth=np.int64(depth),
            candidate_mask=np.ones(len(ids), dtype=bool),
            candidate_ids=ids,
            path_vertex=np.concatenate([self.map.vertex_x, self.wedge], axis=1).astype(
                np.float32
            ),
            graph_goal=np.concatenate(
                [goal_los[:, None].astype(np.float32), goal_edge], axis=1
            ).astype(np.float32),
            vertex_xyz_m=np.asarray(self.map.scene.vertex_xyz, dtype=np.float32),
            rx_available=np.bool_(
                self.visible_to_rx(sample.tx_xyz, sample.rx_xyz, sample.prefix)
            ),
            **self._los_features(sample, ids),
        )
        return out


__all__ = [
    "PAFeatureBuilder",
    "canonical_wedges",
    "collate_sparse_features",
]
