from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config
from src.runtime.oracle import ModelOracle


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class RayAR:
    def __init__(
        self,
        map_id,
        rx_xyz,
        device="cuda:0",
        checkpoint=None,
        config=None,
        cache_root=None,
        position_batch=48,
        feature_workers=8,
    ):
        checkpoint = (
            PACKAGE_ROOT / "models" / "last.pt"
            if checkpoint is None
            else Path(checkpoint)
        )
        config = (
            PACKAGE_ROOT / "configs" / "inference.json"
            if config is None
            else Path(config)
        )
        cfg = load_config(config)
        cache_root = cfg.data.cache_root if cache_root is None else cache_root
        rx = np.asarray(rx_xyz, dtype=np.float32)
        if rx.ndim != 2 or rx.shape[1] != 3:
            raise ValueError("rx_xyz must have shape [N, 3]")
        if not np.allclose(rx[:, 2], 1.5):
            raise ValueError("RayAR requires Rx height z=1.5 m")
        self.oracle = ModelOracle(
            checkpoint,
            config,
            int(map_id),
            rx,
            device,
            cache_root=cache_root,
            position_batch=position_batch,
            feature_workers=feature_workers,
        )

    def predict(self, tx_xy):
        tx = np.asarray(tx_xy, dtype=np.float32)
        if tx.ndim == 1:
            tx = tx[None, :]
        if tx.ndim != 2 or tx.shape[1] != 2:
            raise ValueError("tx_xy must have shape [2] or [B, 2]")
        return self.oracle.predict(tx)

    def close(self):
        if self.oracle is not None:
            self.oracle.close()
            self.oracle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["RayAR"]
