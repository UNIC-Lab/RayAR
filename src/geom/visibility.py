from __future__ import annotations

import numpy as np
import torch


_DEDUP_EPS = np.float32(1e-6)
_EDGE_EPS = np.float32(1e-6)
_PARALLEL_EPS = np.float32(1e-8)
_RAY_EPS = np.float32(1e-10)
_SAMPLE_STEP = np.float32(0.5)


def _clean_polygon(points: np.ndarray) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    raw = raw[:, :2]
    if len(raw) >= 2 and np.linalg.norm(raw[0] - raw[-1]) < _DEDUP_EPS:
        raw = raw[:-1]
    clean = []
    for point in raw:
        if not clean or np.linalg.norm(clean[-1] - point) >= _DEDUP_EPS:
            clean.append(point)
    polygon = np.asarray(clean, dtype=np.float32).reshape(-1, 2)
    if len(polygon) >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) < _DEDUP_EPS:
        polygon = polygon[:-1]
    return np.ascontiguousarray(polygon)


def _points_on_boundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    edge_start = polygon
    edge_delta = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - edge_start[None, :, :]
    cross = (
        relative[..., 1] * edge_delta[None, :, 0]
        - relative[..., 0] * edge_delta[None, :, 1]
    )
    dot = np.sum(relative * edge_delta[None, :, :], axis=-1)
    length_sq = np.sum(edge_delta * edge_delta, axis=-1)
    regular = length_sq >= np.float32(1e-10)
    parameter = np.divide(
        dot,
        length_sq[None, :],
        out=np.zeros_like(dot),
        where=regular[None, :],
    )
    on_regular = regular[None, :] & (parameter >= 0.0) & (parameter <= 1.0)
    on_degenerate = (~regular)[None, :] & (
        np.linalg.norm(relative, axis=-1) < _EDGE_EPS
    )
    return np.any((np.abs(cross) <= _EDGE_EPS) & (on_regular | on_degenerate), axis=1)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        crosses = ((first[1] <= points[:, 1]) & (points[:, 1] < second[1])) | (
            (second[1] <= points[:, 1]) & (points[:, 1] < first[1])
        )
        if first[1] == second[1]:
            continue
        x_intersect = first[0] + (
            (points[:, 1] - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
        )
        inside ^= crosses & (points[:, 0] < x_intersect)
    return inside


class PythonVisibility:
    def __init__(
        self,
        polygons: tuple[np.ndarray, ...],
        vertex_height: float,
        device_height: float,
    ):
        self.polygons = tuple(_clean_polygon(value) for value in polygons)
        valid = tuple(value for value in self.polygons if len(value) >= 3)
        self.vertices = (
            np.concatenate(valid, axis=0)
            if valid
            else np.empty((0, 2), dtype=np.float32)
        )
        edge_start = []
        edge_end = []
        bounds = []
        for polygon in valid:
            edge_start.append(polygon)
            edge_end.append(np.roll(polygon, -1, axis=0))
            bounds.append(
                np.array(
                    [
                        polygon[:, 0].min(),
                        polygon[:, 0].max(),
                        polygon[:, 1].min(),
                        polygon[:, 1].max(),
                    ],
                    dtype=np.float32,
                )
            )
        self.valid_polygons = valid
        self.edge_start = (
            np.concatenate(edge_start, axis=0)
            if edge_start
            else np.empty((0, 2), dtype=np.float32)
        )
        self.edge_delta = (
            np.concatenate(edge_end, axis=0) - self.edge_start
            if edge_end
            else np.empty((0, 2), dtype=np.float32)
        )
        self.bounds = np.stack(bounds) if bounds else np.empty((0, 4), dtype=np.float32)
        self.height_delta = np.float32(vertex_height - device_height)

    def query(self, tx_xy: np.ndarray, rx_xy: np.ndarray) -> dict[str, np.ndarray]:
        tx = np.asarray(tx_xy, dtype=np.float32)
        rx = np.asarray(rx_xy, dtype=np.float32)
        if tx.ndim != 2 or tx.shape[1] != 2 or rx.ndim != 2 or rx.shape[1] != 2:
            raise ValueError("tx_xy and rx_xy must have shape [N, 2]")
        tx = np.ascontiguousarray(tx)
        rx = np.ascontiguousarray(rx)

        return {
            "vertex_los": self._visibility(tx, self.vertices),
            "root_edge": self._root_geometry(tx),
            "rx_los": self._visibility(tx, rx),
        }

    def _root_geometry(self, tx: np.ndarray) -> np.ndarray:
        delta = self.vertices[None, :, :] - tx[:, None, :]
        horizontal = np.sqrt(np.sum(delta * delta, axis=-1))
        distance = np.sqrt(horizontal * horizontal + self.height_delta**2)
        azimuth = np.arctan2(delta[..., 1], delta[..., 0])
        elevation = np.arctan2(self.height_delta, horizontal)
        return np.stack(
            [
                np.log1p(distance) / np.log1p(np.float32(256.0)),
                np.sin(azimuth),
                np.cos(azimuth),
                np.sin(elevation),
                np.cos(elevation),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def _visibility(self, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        output = np.empty((len(sources), len(targets)), dtype=np.uint8)
        for row, source in enumerate(sources):
            output[row] = self._visible_from(source, targets)
        return output

    def _visible_from(self, source: np.ndarray, targets: np.ndarray) -> np.ndarray:
        delta = targets - source
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        visible = distance < _RAY_EPS
        pending = ~visible
        if not np.any(pending):
            return visible.astype(np.uint8)

        if len(self.edge_start):
            dx = delta[:, 0, None]
            dy = delta[:, 1, None]
            ex = self.edge_delta[None, :, 0]
            ey = self.edge_delta[None, :, 1]
            relative = self.edge_start - source
            denom = dx * ey - dy * ex
            valid = np.abs(denom) >= _PARALLEL_EPS
            t = np.divide(
                relative[None, :, 0] * ey - relative[None, :, 1] * ex,
                denom,
                out=np.zeros_like(denom),
                where=valid,
            )
            s = np.divide(
                relative[None, :, 0] * dy - relative[None, :, 1] * dx,
                denom,
                out=np.zeros_like(denom),
                where=valid,
            )
            intersects = (
                valid
                & (t > _RAY_EPS)
                & (t < np.float32(1.0) - _RAY_EPS)
                & (s > _RAY_EPS)
                & (s < np.float32(1.0) - _RAY_EPS)
            )
            pending &= ~np.any(intersects, axis=1)

        for target_index in np.flatnonzero(pending):
            if self._sample_path_pass(
                source, targets[target_index], distance[target_index]
            ):
                visible[target_index] = True
        return visible.astype(np.uint8)

    def _sample_path_pass(
        self,
        start: np.ndarray,
        end: np.ndarray,
        distance: np.float32,
    ) -> bool:
        if distance <= _SAMPLE_STEP or not self.valid_polygons:
            return True
        low = np.minimum(start, end) - _EDGE_EPS
        high = np.maximum(start, end) + _EDGE_EPS
        overlaps = (
            (self.bounds[:, 1] >= low[0])
            & (self.bounds[:, 0] <= high[0])
            & (self.bounds[:, 3] >= low[1])
            & (self.bounds[:, 2] <= high[1])
        )
        if not np.any(overlaps):
            return True

        step_count = int(np.ceil(distance / _SAMPLE_STEP))
        if step_count <= 1:
            return True
        fractions = np.arange(1, step_count, dtype=np.float32) / np.float32(step_count)
        samples = start + fractions[:, None] * (end - start)
        for polygon_index in np.flatnonzero(overlaps):
            polygon = self.valid_polygons[polygon_index]
            if np.any(_points_on_boundary(samples, polygon)):
                return False
            if np.any(_points_in_polygon(samples, polygon)):
                return False
        return True


class TorchVisibility(PythonVisibility):
    def __init__(
        self,
        polygons: tuple[np.ndarray, ...],
        vertex_height: float,
        device_height: float,
        device: str | torch.device,
    ):
        super().__init__(polygons, vertex_height, device_height)
        self.device = torch.device(device)
        self.t_vertices = torch.as_tensor(self.vertices, device=self.device)
        self.t_edge_start = torch.as_tensor(self.edge_start, device=self.device)
        self.t_edge_delta = torch.as_tensor(self.edge_delta, device=self.device)
        self.t_bounds = torch.as_tensor(self.bounds, device=self.device)
        self.t_polygons = tuple(
            torch.as_tensor(value, device=self.device) for value in self.valid_polygons
        )

    def query(self, tx_xy: np.ndarray, rx_xy: np.ndarray) -> dict[str, np.ndarray]:
        tx = np.asarray(tx_xy, dtype=np.float32)
        rx = np.asarray(rx_xy, dtype=np.float32)
        if tx.ndim != 2 or tx.shape[1] != 2 or rx.ndim != 2 or rx.shape[1] != 2:
            raise ValueError("tx_xy and rx_xy must have shape [N, 2]")

        sources = torch.as_tensor(np.ascontiguousarray(tx), device=self.device)
        receivers = torch.as_tensor(np.ascontiguousarray(rx), device=self.device)
        targets = torch.cat([self.t_vertices, receivers], dim=0)
        visibility = self._torch_visibility(sources, targets)
        vertex_count = len(self.vertices)
        return {
            "vertex_los": visibility[:, :vertex_count],
            "root_edge": self._torch_root_geometry(sources),
            "rx_los": visibility[:, vertex_count:],
        }

    def _torch_root_geometry(self, sources: torch.Tensor) -> np.ndarray:
        delta = self.t_vertices.unsqueeze(0) - sources.unsqueeze(1)
        horizontal = torch.sqrt(torch.sum(delta * delta, dim=-1))
        height_delta = torch.tensor(self.height_delta, device=self.device)
        distance = torch.sqrt(horizontal * horizontal + height_delta**2)
        azimuth = torch.atan2(delta[..., 1], delta[..., 0])
        elevation = torch.atan2(height_delta, horizontal)
        features = torch.stack(
            [
                torch.log1p(distance) / np.log1p(np.float32(256.0)),
                torch.sin(azimuth),
                torch.cos(azimuth),
                torch.sin(elevation),
                torch.cos(elevation),
            ],
            dim=-1,
        )
        return features.cpu().numpy().astype(np.float32, copy=False)

    def _torch_visibility(
        self,
        sources: torch.Tensor,
        targets: torch.Tensor,
    ) -> np.ndarray:
        delta = targets.unsqueeze(0) - sources.unsqueeze(1)
        distance = torch.sqrt(torch.sum(delta * delta, dim=-1))
        visible = distance < float(_RAY_EPS)
        pending = ~visible

        if len(self.edge_start):
            dx = delta[..., 0].unsqueeze(-1)
            dy = delta[..., 1].unsqueeze(-1)
            ex = self.t_edge_delta[:, 0].view(1, 1, -1)
            ey = self.t_edge_delta[:, 1].view(1, 1, -1)
            relative = self.t_edge_start.unsqueeze(0) - sources.unsqueeze(1)
            relative_x = relative[:, None, :, 0]
            relative_y = relative[:, None, :, 1]
            denom = dx * ey - dy * ex
            valid = torch.abs(denom) >= float(_PARALLEL_EPS)
            safe_denom = torch.where(valid, denom, torch.ones_like(denom))
            t = (relative_x * ey - relative_y * ex) / safe_denom
            s = (relative_x * dy - relative_y * dx) / safe_denom
            intersects = (
                valid
                & (t > float(_RAY_EPS))
                & (t < 1.0 - float(_RAY_EPS))
                & (s > float(_RAY_EPS))
                & (s < 1.0 - float(_RAY_EPS))
            )
            pending &= ~torch.any(intersects, dim=-1)

        visible |= pending & self._torch_sample_path_pass(
            sources, targets, distance, pending
        )
        return visible.to(dtype=torch.uint8).cpu().numpy()

    def _torch_sample_path_pass(
        self,
        sources: torch.Tensor,
        targets: torch.Tensor,
        distance: torch.Tensor,
        pending: torch.Tensor,
    ) -> torch.Tensor:
        passes = torch.ones_like(pending)
        rays = torch.nonzero(pending & (distance > float(_SAMPLE_STEP)), as_tuple=False)
        if not len(rays) or not self.t_polygons:
            return passes

        starts = sources[rays[:, 0]]
        ends = targets[rays[:, 1]]
        ray_distance = distance[rays[:, 0], rays[:, 1]]
        step_count = torch.ceil(ray_distance / float(_SAMPLE_STEP)).long()
        sample_index = torch.arange(
            1,
            int(step_count.max().item()),
            device=self.device,
        )
        sample_valid = sample_index.unsqueeze(0) < step_count.unsqueeze(1)
        fractions = sample_index / step_count.unsqueeze(1)
        samples = (
            starts[:, None, :] + fractions[..., None] * (ends - starts)[:, None, :]
        )
        low = torch.minimum(starts, ends) - float(_EDGE_EPS)
        high = torch.maximum(starts, ends) + float(_EDGE_EPS)

        ray_pass = torch.ones(len(rays), dtype=torch.bool, device=self.device)
        for polygon_index, polygon in enumerate(self.t_polygons):
            bounds = self.t_bounds[polygon_index]
            overlaps = (
                (bounds[1] >= low[:, 0])
                & (bounds[0] <= high[:, 0])
                & (bounds[3] >= low[:, 1])
                & (bounds[2] <= high[:, 1])
                & ray_pass
            )
            active = torch.nonzero(overlaps, as_tuple=False).squeeze(1)
            if not len(active):
                continue
            points = samples[active]
            valid_samples = sample_valid[active]
            edge_start = polygon
            edge_delta = torch.roll(polygon, -1, dims=0) - polygon
            relative = points[:, :, None, :] - edge_start[None, None, :, :]
            cross = (
                relative[..., 1] * edge_delta[None, None, :, 0]
                - relative[..., 0] * edge_delta[None, None, :, 1]
            )
            dot = torch.sum(relative * edge_delta[None, None, :, :], dim=-1)
            length_sq = torch.sum(edge_delta * edge_delta, dim=-1)
            regular = length_sq >= 1e-10
            safe_length = torch.where(regular, length_sq, torch.ones_like(length_sq))
            parameter = dot / safe_length[None, None, :]
            on_edge = (torch.abs(cross) <= float(_EDGE_EPS)) & (
                (regular[None, None, :] & (parameter >= 0.0) & (parameter <= 1.0))
                | (
                    (~regular)[None, None, :]
                    & (torch.linalg.vector_norm(relative, dim=-1) < float(_EDGE_EPS))
                )
            )

            first = polygon
            second = torch.roll(polygon, -1, dims=0)
            point_x = points[..., 0, None]
            point_y = points[..., 1, None]
            crosses = ((first[:, 1] <= point_y) & (point_y < second[:, 1])) | (
                (second[:, 1] <= point_y) & (point_y < first[:, 1])
            )
            height = second[:, 1] - first[:, 1]
            safe_height = torch.where(height != 0.0, height, torch.ones_like(height))
            x_intersect = first[:, 0] + (
                (point_y - first[:, 1]) * (second[:, 0] - first[:, 0]) / safe_height
            )
            inside = (
                torch.sum(crosses & (point_x < x_intersect), dim=-1).remainder(2).bool()
            )
            blocked = torch.any(
                valid_samples & (torch.any(on_edge, dim=-1) | inside),
                dim=1,
            )
            ray_pass[active[blocked]] = False

        failed = rays[~ray_pass]
        passes[failed[:, 0], failed[:, 1]] = False
        return passes


__all__ = ["PythonVisibility", "TorchVisibility"]
