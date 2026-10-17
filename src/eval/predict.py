from __future__ import annotations

import numpy as np
import torch

from src.eval.base import PathPredictor, Prediction
from src.data.dataset import PathSample
from src.data.features import collate_sparse_features
from src.model.network import MAX_PREFIX_LEN, numpy_batch_to_torch


class AutoregressivePredictor(PathPredictor):
    @torch.no_grad()
    def _sparse_pointer_batch(
        self,
        states: list[
            tuple[
                np.ndarray,
                np.ndarray,
                tuple[int, ...],
                np.ndarray,
            ]
        ],
    ) -> torch.Tensor:
        items = []
        prefixes = []
        tx_values = []
        rx_values = []
        for tx_xyz, rx_xyz, prefix, candidate_ids in states:
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
            items.append(self.builder.build_sparse(sample, candidate_ids))
            padded = np.full(MAX_PREFIX_LEN, -1, dtype=np.int64)
            padded[: len(prefix)] = np.asarray(prefix, dtype=np.int64)
            prefixes.append(padded)
            tx_values.append(tx_xyz)
            rx_values.append(rx_xyz)

        features = collate_sparse_features(items)
        if any(not state[2] for state in states):
            features["los_graph_mask"] = self.builder.los_graph_mask
            features["los_graph_edge"] = self.builder.los_graph_edge
        features["prefix_ids"] = np.stack(prefixes, axis=0)
        features["tx_xyz"] = np.stack(tx_values, axis=0)
        features["rx_xyz"] = np.stack(rx_values, axis=0)
        batch = numpy_batch_to_torch(features, self.device)
        return self.model(batch)

    @torch.no_grad()
    def _sparse_logits_batch(
        self,
        states: list[
            tuple[
                np.ndarray,
                np.ndarray,
                tuple[int, ...],
                np.ndarray,
            ]
        ],
    ) -> list[torch.Tensor]:
        pointer = self._sparse_pointer_batch(states)
        return [
            torch.cat(
                [pointer[row, : len(state[3])], pointer[row, -1:]],
                dim=0,
            )
            for row, state in enumerate(states)
        ]

    @torch.no_grad()
    def _sparse_logits(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        prefix: tuple[int, ...],
        candidate_ids: np.ndarray,
    ) -> torch.Tensor:
        return self._sparse_logits_batch([(tx_xyz, rx_xyz, prefix, candidate_ids)])[0]

    @torch.no_grad()
    def logits(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        prefix: tuple[int, ...],
    ) -> torch.Tensor:
        legal = self.builder.legal_candidates(tx_xyz, prefix)
        candidate_ids = self.builder.legal_ids(tx_xyz, prefix)
        sparse = self._sparse_logits(tx_xyz, rx_xyz, prefix, candidate_ids)
        full = torch.full(
            (len(legal) + 1,),
            torch.finfo(sparse.dtype).min,
            dtype=sparse.dtype,
            device=sparse.device,
        )
        full[torch.as_tensor(candidate_ids, device=sparse.device)] = sparse[:-1]
        full[-1] = sparse[-1]
        return full

    @torch.no_grad()
    def greedy(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        max_hops: int = 3,
    ) -> Prediction:
        prefix: tuple[int, ...] = ()
        score = 0.0
        for _ in range(max_hops + 1):
            candidate_ids = self.builder.legal_ids(tx_xyz, prefix)
            if candidate_ids.size == 0:
                available = torch.tensor(
                    [self.builder.visible_to_rx(tx_xyz, rx_xyz, prefix)],
                    dtype=torch.bool,
                    device=self.device,
                )
                output = self.model.rx_only_action(available)
                return Prediction(
                    prefix,
                    score,
                    bool(output["legal"][0, 0].item()),
                )
            log_prob = torch.log_softmax(
                self._sparse_logits(tx_xyz, rx_xyz, prefix, candidate_ids),
                dim=0,
            )
            local = int(torch.argmax(log_prob).item())
            if local == len(candidate_ids):
                score += float(log_prob[local].item())
                return Prediction(prefix, score, True)
            if len(prefix) >= max_hops:
                return Prediction(prefix, score, False)
            vertex = int(candidate_ids[local])
            score += float(log_prob[local].item())
            prefix += (vertex,)
        return Prediction(prefix, score, False)

    @torch.no_grad()
    def greedy_batch(
        self,
        tx_xyz: np.ndarray,
        rx_xyz: np.ndarray,
        max_hops: int = 3,
    ) -> list[Prediction]:
        prefixes = [tuple() for _ in range(len(tx_xyz))]
        scores = np.zeros(len(tx_xyz), dtype=np.float64)
        complete = np.zeros(len(tx_xyz), dtype=bool)
        done = np.zeros(len(tx_xyz), dtype=bool)

        while not done.all():
            active = []
            states = []
            for row in np.flatnonzero(~done):
                prefix = prefixes[row]
                if self.builder.visible_to_rx(tx_xyz[row], rx_xyz[row], prefix):
                    complete[row] = True
                    done[row] = True
                    continue
                if len(prefix) >= max_hops:
                    done[row] = True
                    continue
                candidate_ids = self.builder.legal_ids(tx_xyz[row], prefix)
                if candidate_ids.size == 0:
                    done[row] = True
                    continue
                active.append((int(row), candidate_ids))
                states.append(
                    (
                        tx_xyz[row],
                        rx_xyz[row],
                        prefix,
                        candidate_ids,
                    )
                )

            if not states:
                break
            pointer = self._sparse_pointer_batch(states)
            log_prob = torch.log_softmax(pointer, dim=1)
            selected = torch.argmax(log_prob, dim=1)
            selected_score = torch.gather(log_prob, 1, selected.unsqueeze(1)).squeeze(1)
            selected = selected.cpu().tolist()
            selected_score = selected_score.cpu().tolist()
            rx_slot = pointer.shape[1] - 1
            for (row, candidate_ids), local, value in zip(
                active, selected, selected_score
            ):
                scores[row] += float(value)
                if local == rx_slot:
                    complete[row] = True
                    done[row] = True
                    continue
                prefixes[row] += (int(candidate_ids[local]),)

        return [
            Prediction(prefixes[row], float(scores[row]), bool(complete[row]))
            for row in range(len(prefixes))
        ]


__all__ = ["AutoregressivePredictor"]
