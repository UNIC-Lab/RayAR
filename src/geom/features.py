from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geom.scene import SceneBounds


FEATURE_TYPE_VERTICAL_CORNER = 2


@dataclass
class FeatureMeta:
    building_id: np.ndarray
    feature_type: np.ndarray
    height_norm: np.ndarray
    building_height: np.ndarray


def feature_node_features(
    feature_xyz: np.ndarray,
    normals: np.ndarray,
    meta: FeatureMeta,
    bounds: SceneBounds,
) -> np.ndarray:
    xyz = np.asarray(feature_xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape[0] != xyz.shape[0]:
        normals = np.zeros((xyz.shape[0], 2), dtype=np.float32)
    pos_norm = (xyz - bounds.center[None, :]) / bounds.scale
    z_over_scene = xyz[:, 2:3] / max(bounds.z_max, 1e-6)
    normal_len = np.linalg.norm(normals, axis=1, keepdims=True).astype(np.float32)
    type_oh = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    type_idx = np.clip(meta.feature_type.astype(np.int64), 0, 2)
    type_oh[np.arange(xyz.shape[0]), type_idx] = 1.0
    bias = np.ones((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate(
        [
            pos_norm.astype(np.float32),
            z_over_scene.astype(np.float32),
            normals.astype(np.float32),
            normal_len,
            meta.height_norm[:, None].astype(np.float32),
            type_oh,
            bias,
        ],
        axis=1,
    )


def ray_geometry_features(
    tx_xyz: np.ndarray,
    target_xyz: np.ndarray,
    target_normals: np.ndarray,
    bounds: SceneBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = target_xyz - tx_xyz
    dist = np.linalg.norm(delta, axis=1).astype(np.float32)
    unit = delta / np.maximum(dist[:, None], 1e-6)
    elev = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)).astype(np.float32)
    azim = np.arctan2(unit[:, 1], unit[:, 0]).astype(np.float32)
    xy_unit = unit[:, :2] / np.maximum(
        np.linalg.norm(unit[:, :2], axis=1, keepdims=True), 1e-6
    )
    normal_dot = np.sum(target_normals * xy_unit, axis=1, keepdims=True).astype(
        np.float32
    )
    ground_dist = np.zeros((target_xyz.shape[0], 1), dtype=np.float32)
    downward = unit[:, 2] < -1e-6
    ground_dist[downward, 0] = np.maximum(
        (0.0 - target_xyz[downward, 2]) / unit[downward, 2], 0.0
    )
    bbox_diag = bounds.diagonal
    bounds_dist = np.full((target_xyz.shape[0], 1), bbox_diag, dtype=np.float32)
    feats = np.concatenate(
        [
            (delta / bounds.scale).astype(np.float32),
            (dist[:, None] / bounds.scale).astype(np.float32),
            np.log1p(dist)[:, None].astype(np.float32),
            unit.astype(np.float32),
            np.sin(azim)[:, None],
            np.cos(azim)[:, None],
            np.sin(elev)[:, None],
            np.cos(elev)[:, None],
            normal_dot,
            np.abs(normal_dot),
            np.log1p(ground_dist),
            np.log1p(bounds_dist),
            np.ones((target_xyz.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    return (
        feats.astype(np.float32),
        unit.astype(np.float32),
        dist,
        np.stack([dist, elev, azim], axis=1),
    )


def tx_node_features(tx_xyz: np.ndarray, bounds: SceneBounds) -> np.ndarray:
    tx_xyz = np.asarray(tx_xyz, dtype=np.float32)
    if tx_xyz.ndim == 1:
        tx_xyz = tx_xyz[None, :]
    pos_norm = (tx_xyz - bounds.center[None, :]) / max(bounds.scale, 1e-6)
    z_over_scene = tx_xyz[:, 2:3] / max(bounds.z_max, 1e-6)
    bias = np.ones((tx_xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate(
        [pos_norm.astype(np.float32), z_over_scene.astype(np.float32), bias], axis=1
    )
