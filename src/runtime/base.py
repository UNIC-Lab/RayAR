from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.eval.path_loss import compute_path_loss_db


def chain_geometry(tx_xyz, rx_xyz, chain, vertex_xyz):
    middle = np.asarray(vertex_xyz[list(chain)], dtype=np.float64)
    points = np.concatenate(
        [
            np.asarray(tx_xyz, dtype=np.float64)[None, :],
            middle.reshape(-1, 3),
            np.asarray(rx_xyz, dtype=np.float64)[None, :],
        ]
    )
    segments = np.diff(points, axis=0)
    distance = float(np.linalg.norm(segments, axis=1).sum())
    doa = segments[-1] / max(float(np.linalg.norm(segments[-1])), 1e-12)
    return distance, doa


@dataclass(frozen=True)
class ChannelObservation:
    loss_db: np.ndarray
    distance_m: np.ndarray
    doa_unit: np.ndarray


@dataclass(frozen=True)
class OracleOutput:
    loss_db: np.ndarray
    distance_m: np.ndarray
    doa_unit: np.ndarray
    complete: np.ndarray
    chains: tuple[tuple[tuple[int, ...], ...], ...]

    def observation(self, row):
        return ChannelObservation(
            self.loss_db[row], self.distance_m[row], self.doa_unit[row]
        )


class RadioAROracle:
    def predict(self, tx_xy):
        tx_xy = np.asarray(tx_xy, dtype=np.float32).reshape(-1, 2)
        for builder in self.builders:
            builder.clear_query_cache()
        outputs = [
            self._predict_chunk(tx_xy[start : start + self.position_batch])
            for start in range(0, len(tx_xy), self.position_batch)
        ]
        return OracleOutput(
            np.concatenate([value.loss_db for value in outputs], axis=0),
            np.concatenate([value.distance_m for value in outputs], axis=0),
            np.concatenate([value.doa_unit for value in outputs], axis=0),
            np.concatenate([value.complete for value in outputs], axis=0),
            tuple(chain for value in outputs for chain in value.chains),
        )

    def _predict_chunk(self, tx_xy):
        tx_xyz = np.concatenate(
            [
                tx_xy,
                np.full((len(tx_xy), 1), 1.5, dtype=np.float32),
            ],
            axis=1,
        )
        self.source.prepare(tx_xyz, self.rx_xyz)
        receiver_count = len(self.rx_xyz)
        expanded_tx = np.repeat(tx_xyz, receiver_count, axis=0)
        expanded_rx = np.tile(self.rx_xyz, (len(tx_xyz), 1))
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            predictions = self.predictor.greedy_batch(expanded_tx, expanded_rx)

        loss = np.empty((len(tx_xyz), receiver_count), dtype=np.float64)
        distance = np.empty_like(loss)
        doa = np.empty((len(tx_xyz), receiver_count, 3), dtype=np.float64)
        complete = np.empty((len(tx_xyz), receiver_count), dtype=bool)
        rows = []
        vertices = self.cache.vertex_xyz
        normals = self.cache.vertex_normal
        for tx_row in range(len(tx_xyz)):
            chains = []
            for rx_row in range(receiver_count):
                flat = tx_row * receiver_count + rx_row
                prediction = predictions[flat]
                chain = tuple(prediction.chain)
                chains.append(chain)
                complete[tx_row, rx_row] = prediction.complete
                loss[tx_row, rx_row] = compute_path_loss_db(
                    tx_xyz[tx_row],
                    self.rx_xyz[rx_row],
                    chain,
                    vertices,
                    normals,
                )
                distance[tx_row, rx_row], doa[tx_row, rx_row] = chain_geometry(
                    tx_xyz[tx_row],
                    self.rx_xyz[rx_row],
                    chain,
                    vertices,
                )
            rows.append(tuple(chains))
        return OracleOutput(
            loss,
            distance,
            doa,
            complete,
            tuple(rows),
        )


__all__ = ["OracleOutput", "RadioAROracle", "chain_geometry"]
