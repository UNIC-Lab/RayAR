from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.data.dataset import PathSample
from src.data.features import collate_sparse_features
from src.eval.predict import AutoregressivePredictor
from src.model.network import MAX_PREFIX_LEN, numpy_batch_to_torch


class ParallelSparseBuilder:
    def __init__(self, builders):
        self.builders = list(builders)
        self.pool = ThreadPoolExecutor(max_workers=len(self.builders))

    @staticmethod
    def _key(state) -> tuple[bytes, tuple[int, ...], bytes]:
        tx_xyz, _, prefix, candidate_ids = state
        return (
            np.asarray(tx_xyz, dtype=np.float32).tobytes(),
            tuple(prefix),
            np.asarray(candidate_ids, dtype=np.int64).tobytes(),
        )

    def _partitions(self, states) -> list[list[int]]:
        grouped = defaultdict(list)
        for index, state in enumerate(states):
            grouped[self._key(state)].append(index)

        partitions = [[] for _ in self.builders]
        loads = [0 for _ in self.builders]
        for indices in sorted(grouped.values(), key=len, reverse=True):
            worker = int(np.argmin(loads))
            partitions[worker].extend(indices)
            loads[worker] += len(indices)
        return partitions

    @staticmethod
    def _build_partition(builder, states, indices):
        output = []
        for index in indices:
            tx_xyz, rx_xyz, prefix, candidate_ids = states[index]
            sample = PathSample(
                builder.cache.map_id,
                0,
                0,
                tx_xyz,
                rx_xyz,
                prefix,
                (),
                (),
            )
            output.append((index, builder.build_sparse(sample, candidate_ids)))
        return output

    def build(self, states):
        partitions = self._partitions(states)
        futures = [
            self.pool.submit(
                self._build_partition,
                builder,
                states,
                indices,
            )
            for builder, indices in zip(self.builders, partitions)
            if indices
        ]
        items = [None] * len(states)
        for future in futures:
            for index, item in future.result():
                items[index] = item
        return items

    def close(self) -> None:
        self.pool.shutdown(wait=True)


class ParallelAutoregressivePredictor(AutoregressivePredictor):
    def __init__(self, model, builders, device):
        super().__init__(model, builders[0], device)
        self.parallel_builder = ParallelSparseBuilder(builders)

    @torch.no_grad()
    def _sparse_pointer_batch(self, states) -> torch.Tensor:
        items = self.parallel_builder.build(states)
        features = collate_sparse_features(items)
        if any(not state[2] for state in states):
            features["los_graph_mask"] = self.builder.los_graph_mask
            features["los_graph_edge"] = self.builder.los_graph_edge

        prefixes = np.full((len(states), MAX_PREFIX_LEN), -1, dtype=np.int64)
        for row, state in enumerate(states):
            prefix = state[2]
            prefixes[row, : len(prefix)] = np.asarray(prefix, dtype=np.int64)
        features["prefix_ids"] = prefixes
        features["tx_xyz"] = np.stack([state[0] for state in states], axis=0)
        features["rx_xyz"] = np.stack([state[1] for state in states], axis=0)
        batch = numpy_batch_to_torch(features, self.device)
        return self.model(batch)


__all__ = ["ParallelAutoregressivePredictor", "ParallelSparseBuilder"]
