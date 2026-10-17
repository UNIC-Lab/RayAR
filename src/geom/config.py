from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GraphMode = Literal["knn_topology", "knn_topology_los"]

WallCandidateMode = Literal["centroid", "span_raw"]


@dataclass
class GraphConfig:
    mode: GraphMode = "knn_topology_los"
    k_scene: int = 8
    topology_k_same_building: int = 4
    k_wall_neighbor: int = 4
    edge_attr_scale_eps: float = 1e-6


@dataclass
class RayGraphConfig:
    wall_candidate_mode: WallCandidateMode = "centroid"
    k_pre: int = 32
    k_post: int = 32
    k_ray: int = 8
    max_angle_deg: float = 12.0
    ray_smoothing_angle_deg: float = 5.0
    ray_smoothing_dist_ratio: float = 2.0
    rho0: float = 2.0
    alpha: float = 0.08
    chunk_rays: int = 256
    use_ray_smoothing: bool = True
    use_first_hit_aux: bool = True
