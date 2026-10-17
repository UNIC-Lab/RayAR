from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LosLabels:
    coords: np.ndarray
    vertex_mask: np.ndarray
    street_mask: np.ndarray

    def current(self, vertex: int) -> np.ndarray:
        return np.asarray(
            self.vertex_mask[vertex, self.coords[:, 0], self.coords[:, 1]] == 1,
            dtype=bool,
        )

    def goal(self, rx_xyz: np.ndarray) -> np.ndarray:
        x = min(max(int(float(rx_xyz[0])), 0), 256)
        y = min(max(int(float(rx_xyz[1])), 0), 256)
        return np.asarray(self.street_mask[:, x, y] == 1, dtype=bool)


def load_los_labels(
    root: str | Path,
    map_id: int,
    vertex_xy: np.ndarray,
) -> LosLabels:
    root = Path(root) / f"map_{map_id}"
    los_vertex = np.load(root / f"vertex_points_{map_id}.npy", mmap_mode="r")
    if los_vertex.shape != vertex_xy.shape or not np.allclose(
        los_vertex, vertex_xy, atol=1e-6
    ):
        raise ValueError(
            f"map_{map_id}: LOS vertex_points do not match PROP vertex IDs"
        )

    fundamental = root / "fundamental"
    vertex_mask = np.load(
        fundamental / f"vertex_visibility_mask_{map_id}.npy", mmap_mode="r"
    )
    street_mask = np.load(
        fundamental / f"street_visibility_mask_{map_id}.npy", mmap_mode="r"
    )
    expected = (vertex_xy.shape[0], 257, 257)
    if vertex_mask.shape != expected or street_mask.shape != expected:
        raise ValueError(f"map_{map_id}: invalid fundamental LOS mask shape")

    coords = np.clip(np.rint(vertex_xy).astype(np.int64), 0, 256)
    return LosLabels(coords, vertex_mask, street_mask)
