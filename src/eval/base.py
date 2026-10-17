from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.geom.segments import Visibility2D
from src.data.dataset import PathSample
from src.data.rays import FeatureBuilder, collate_features
from src.model.layers import numpy_batch_to_torch


@dataclass(frozen=True)
class Prediction:
    chain: tuple[int, ...]
    score: float
    complete: bool


class PathPredictor:
    def __init__(self, model, builder: FeatureBuilder, device: torch.device):
        self.model = model
        self.builder = builder
        self.device = device
        self.visibility = Visibility2D(builder.map.scene)

    def _visible_to_rx(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> bool:
        if not prefix:
            return self.visibility.visible(tx_xyz, rx_xyz)
        vertex = prefix[-1]
        return self.visibility.visible(
            self.builder.map.scene.vertex_xyz[vertex],
            rx_xyz,
            ignore_building=int(self.builder.map.scene.vertex_building[vertex]),
            start_vertex=vertex,
        )

    @torch.no_grad()
    def logits(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> torch.Tensor:
        sample = PathSample(
            self.builder.cache.map_id,
            0,
            0,
            tx_xyz,
            rx_xyz,
            prefix,
            (),
            (),
        )
        features = collate_features([self.builder.build(sample)])
        batch = numpy_batch_to_torch(features, self.device)
        return self.model(batch)[0]

    @torch.no_grad()
    def greedy(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        max_hops: int = 3,
    ) -> Prediction:
        prefix: tuple[int, ...] = ()
        if self._visible_to_rx(tx_xyz, rx_xyz, prefix):
            return Prediction(prefix, 0.0, True)
        score = 0.0
        for _ in range(max_hops):
            log_prob = torch.log_softmax(self.logits(tx_xyz, rx_xyz, prefix), dim=0)
            vertex = int(torch.argmax(log_prob).item())
            score += float(log_prob[vertex].item())
            prefix += (vertex,)
            if self._visible_to_rx(tx_xyz, rx_xyz, prefix):
                return Prediction(prefix, score, True)
        return Prediction(prefix, score, False)

    @torch.no_grad()
    def beam(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        width: int = 5,
        max_hops: int = 3,
    ) -> Prediction:
        candidates = self.beam_candidates(tx_xyz, rx_xyz, width, max_hops)
        complete = [item for item in candidates if item.complete]
        return max(complete or candidates, key=lambda item: item.score)

    @torch.no_grad()
    def beam_candidates(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        width: int = 5,
        max_hops: int = 3,
    ) -> list[Prediction]:
        if self._visible_to_rx(tx_xyz, rx_xyz, ()):
            return [Prediction((), 0.0, True)]
        beams = [Prediction((), 0.0, False)]
        complete: list[Prediction] = []
        for _ in range(max_hops):
            expanded: list[Prediction] = []
            for beam in beams:
                log_prob = torch.log_softmax(
                    self.logits(tx_xyz, rx_xyz, beam.chain), dim=0
                )
                values, indices = torch.topk(log_prob, k=min(width, log_prob.numel()))
                for value, index in zip(values.tolist(), indices.tolist()):
                    chain = beam.chain + (int(index),)
                    item = Prediction(
                        chain,
                        beam.score + float(value),
                        self._visible_to_rx(tx_xyz, rx_xyz, chain),
                    )
                    (complete if item.complete else expanded).append(item)
            expanded.sort(key=lambda item: item.score, reverse=True)
            beams = expanded[:width]
            if not beams:
                break
        candidates = complete + beams
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:width] or [Prediction((), float("-inf"), False)]
