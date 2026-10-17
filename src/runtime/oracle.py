from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.eval.path_loss import compute_path_loss_db
from src.eval.predict import AutoregressivePredictor
from src.checkpoint import load_checkpoint
from src.runtime.base import OracleOutput, RadioAROracle, chain_geometry
from src.data.cache import load_geometry_cache
from src.geom.source import OnDemandSource
from src.geom.visibility import TorchVisibility
from src.runtime.features import CalibrationFeatureBuilder
from src.runtime.predict import ParallelAutoregressivePredictor


_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}
_RUNTIME_CACHE: dict[tuple, tuple[object, TorchVisibility]] = {}


def _model(checkpoint: str, device: torch.device):
    key = str(Path(checkpoint).resolve()), str(device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_checkpoint(checkpoint, device)
    return _MODEL_CACHE[key]


def _runtime(
    cfg,
    config_path: str,
    cache_root: str,
    map_id: int,
    device: torch.device,
):
    key = (
        str(Path(config_path).resolve()),
        str(Path(cache_root).resolve()),
        int(map_id),
        str(device),
    )
    if key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE.clear()
        cache = load_geometry_cache(cache_root, map_id)
        builder = CalibrationFeatureBuilder(
            cache,
            cfg.data.building_root,
            cfg.data.building_height,
            cfg.model.k_pre,
            cfg.model.max_wall_tokens,
            cfg.data.los_root,
        )
        backend = TorchVisibility(
            builder.map.scene.polygons,
            cfg.data.building_height,
            1.5,
            device,
        )
        _RUNTIME_CACHE[key] = builder, backend
    return _RUNTIME_CACHE[key]


class ModelOracle(RadioAROracle):
    """Frozen RayAR chain prediction followed by deterministic observables."""

    def __init__(
        self,
        checkpoint: str,
        config_path: str,
        map_id: int,
        rx_xyz: np.ndarray,
        device: str,
        cache_root: str = "assets/cache",
        position_batch: int = 48,
        feature_workers: int = 8,
        amp: bool = True,
    ):
        cfg = load_config(config_path)
        self.device = torch.device(device)
        template, backend = _runtime(
            cfg,
            config_path,
            cache_root,
            map_id,
            self.device,
        )
        self.builder = template.fork()
        self.builder._goal_ray_cache = {}
        self.cache = self.builder.cache
        self.source = OnDemandSource(self.builder.source, backend)
        self.builder.source = self.source
        self.builder.precompute_goal_rays(rx_xyz)
        self.builders = [self.builder]
        for _ in range(max(1, int(feature_workers)) - 1):
            worker = self.builder.fork()
            worker.source = self.source
            self.builders.append(worker)
        self.model = _model(checkpoint, self.device)
        self.predictor = (
            AutoregressivePredictor(self.model, self.builder, self.device)
            if len(self.builders) == 1
            else ParallelAutoregressivePredictor(self.model, self.builders, self.device)
        )
        self.rx_xyz = np.asarray(rx_xyz, dtype=np.float32).reshape(-1, 3)
        self.position_batch = int(position_batch)
        self.amp = bool(amp and self.device.type == "cuda")
        self.path_height = float(cfg.data.building_height)
        self.path_vertices = np.asarray(self.cache.vertex_xyz).copy()
        self.path_vertices[:, 2] = self.path_height

    def measure_chain(self, tx_xyz, rx_xyz, chain):
        loss = compute_path_loss_db(
            tx_xyz,
            rx_xyz,
            chain,
            self.path_vertices,
            self.cache.vertex_normal,
            vertex_height=self.path_height,
        )
        distance, _ = chain_geometry(tx_xyz, rx_xyz, chain, self.path_vertices)
        _, doa = chain_geometry(tx_xyz, rx_xyz, chain, self.path_vertices)
        return loss, distance, doa

    def _predict_chunk(self, tx_xy):
        output = RadioAROracle._predict_chunk(self, tx_xy)
        tx_xyz = np.column_stack([tx_xy, np.full(len(tx_xy), 1.5)]).astype(np.float32)
        loss = np.empty_like(output.loss_db)
        distance = np.empty_like(output.distance_m)
        doa = np.empty_like(output.doa_unit)
        for tx_row, chains in enumerate(output.chains):
            for rx_row, chain in enumerate(chains):
                (
                    loss[tx_row, rx_row],
                    distance[tx_row, rx_row],
                    doa[tx_row, rx_row],
                ) = self.measure_chain(tx_xyz[tx_row], self.rx_xyz[rx_row], chain)
        return OracleOutput(
            loss,
            distance,
            doa,
            output.complete,
            output.chains,
        )

    def close(self):
        parallel = getattr(self.predictor, "parallel_builder", None)
        if parallel is not None:
            parallel.close()
        self.predictor = None
        self.model = None
        self.builders = []
        self.builder = None
        self.source = None
        self.cache = None
        _MODEL_CACHE.clear()
        _RUNTIME_CACHE.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = ["ModelOracle"]
