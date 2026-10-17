from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.api import RayAR


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--input", required=True)
    value.add_argument("--checkpoint", default=str(PACKAGE_ROOT / "models" / "last.pt"))
    value.add_argument(
        "--config", default=str(PACKAGE_ROOT / "configs" / "inference.json")
    )
    value.add_argument("--cache-root", default=str(PACKAGE_ROOT / "assets" / "cache"))
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--position-batch", type=int, default=48)
    value.add_argument("--feature-workers", type=int, default=8)
    value.add_argument("--output", required=True)
    return value


def run(args):
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tx_xyz = np.asarray(request["tx_xyz"], dtype=np.float32)
    rx_xyz = np.asarray(request["rx_xyz"], dtype=np.float32)
    if tx_xyz.shape != (3,):
        raise ValueError("tx_xyz must have shape [3]")
    if not np.isclose(tx_xyz[2], 1.5):
        raise ValueError("RayAR requires Tx height z=1.5 m")
    started = time.perf_counter()
    with RayAR(
        map_id=request["map_id"],
        rx_xyz=rx_xyz,
        device=f"cuda:{args.gpu}",
        checkpoint=args.checkpoint,
        config=args.config,
        cache_root=args.cache_root,
        position_batch=args.position_batch,
        feature_workers=args.feature_workers,
    ) as model:
        prediction = model.predict(tx_xyz[:2])
    elapsed = time.perf_counter() - started
    result = {
        "map_id": int(request["map_id"]),
        "tx_xyz": tx_xyz.tolist(),
        "rx_xyz": rx_xyz.tolist(),
        "chains": [list(chain) for chain in prediction.chains[0]],
        "complete": prediction.complete[0].tolist(),
        "loss_db": prediction.loss_db[0].tolist(),
        "distance_m": prediction.distance_m[0].tolist(),
        "delay_ns": (prediction.distance_m[0] / 0.299792458).tolist(),
        "doa_unit": prediction.doa_unit[0].tolist(),
        "inference_seconds": elapsed,
        "link_per_second": len(rx_xyz) / max(elapsed, 1e-12),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False))


__all__ = ["main", "parser", "run"]
