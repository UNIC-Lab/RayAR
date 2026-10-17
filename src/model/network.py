from __future__ import annotations

import torch
from torch import nn

from src.model.layers import _mlp, numpy_batch_to_torch
from src.model.backbone import PathBackbone
from src.config import ModelConfig
from src.model.history import PathPhysicalBias
from src.model.rank import PropagationRankHead
from src.model.fusion import GroupedFeatureFusion
from src.model.attention import CandidateAttention
from src.model.graph import LOSGraphEncoder


MAX_PREFIX_LEN = 4

MAX_HOPS = 3


class IRTPathModel(PathBackbone):
    """Transformer path decoder with propagation-aware candidate scoring."""

    def __init__(self, cfg: ModelConfig | None = None):
        cfg = cfg or ModelConfig()
        super().__init__(cfg)
        del self.query
        del self.topology_bias
        del self.pointer_candidate_norm
        del self.cur_los_head
        del self.goal_los_head
        del self.ray_gate
        del self.ray_mix
        del self.geom_encoder
        del self.visited_embedding
        del self.candidate_norm
        self.start_token = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.start_token, std=0.02)
        self.path_pos_embedding = nn.Embedding(MAX_PREFIX_LEN + 1, cfg.hidden_dim)
        nn.init.normal_(self.path_pos_embedding.weight, std=0.02)
        self.path_phys_bias = PathPhysicalBias(cfg.hidden_dim)
        self.query_geom_encoder = _mlp(26, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.path_vertex_encoder = _mlp(
            cfg.feature_dim + 7,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.dropout,
        )
        self.candidate_attention = CandidateAttention(cfg)
        self.los_graph = LOSGraphEncoder(cfg)
        self.feature_fusion = GroupedFeatureFusion(
            cfg.hidden_dim,
            cfg.fusion_dim,
            cfg.fusion_gate_groups,
            cfg.dropout,
        )
        self.rank_head = PropagationRankHead(cfg)

    def rx_only_action(self, rx_available: torch.Tensor) -> dict[str, torch.Tensor]:
        legal = rx_available.bool().reshape(-1, 1)
        pointer = torch.zeros(
            legal.shape,
            dtype=self.start_token.dtype,
            device=legal.device,
        ).masked_fill(~legal, torch.finfo(self.start_token.dtype).min)
        return {"pointer": pointer, "legal": legal}

    def _root_los_future(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_ids = batch["candidate_ids"]
        shape = (
            candidate_ids.shape[0],
            candidate_ids.shape[1],
            self.los_graph.output.out_features,
        )
        future = batch["path_vertex"].new_zeros(shape)
        roots = torch.nonzero(batch["depth"].eq(0), as_tuple=False).flatten()
        if roots.numel() == 0:
            return future
        graph_node = torch.cat(
            [batch["path_vertex"][roots], batch["graph_goal"][roots]],
            dim=-1,
        )
        root_future = self.los_graph(
            graph_node,
            batch["los_graph_edge"],
            batch["los_graph_mask"],
            candidate_ids[roots],
        )
        return future.index_copy(0, roots, root_future.to(future.dtype))

    def _path_context(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch["vertex"].shape[0]
        prefix_ids = batch.get("prefix_ids")
        if prefix_ids is None:
            prefix_ids = torch.full(
                (batch_size, MAX_PREFIX_LEN),
                -1,
                dtype=torch.long,
                device=batch["vertex"].device,
            )
        else:
            prefix_ids = prefix_ids.long()
        prefix_mask = prefix_ids.ge(0)

        start = self.start_token.expand(batch_size, -1, -1)
        start = start + self.query_encoder(batch["path_state"]).unsqueeze(1)
        start = start + self.query_geom_encoder(batch["query_geom"]).unsqueeze(1)

        static_vertex = batch.get("path_vertex")
        if static_vertex is None:
            static_vertex = torch.cat(
                [batch["vertex"], batch["transition"][..., -7:]],
                dim=-1,
            )
        safe_ids = prefix_ids.clamp(min=0)
        path = torch.gather(
            static_vertex,
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, static_vertex.shape[-1]),
        )
        path = self.path_vertex_encoder(path)
        positions = torch.arange(1, prefix_ids.shape[1] + 1, device=path.device)
        path = path + self.path_pos_embedding(positions).unsqueeze(0)
        path = path * prefix_mask.unsqueeze(-1).to(path.dtype)
        tokens = torch.cat([start, path], dim=1)
        tokens = tokens + self.path_phys_bias(
            batch["vertex_xyz_m"],
            prefix_ids,
            batch["tx_xyz"],
            batch["rx_xyz"],
        )

        valid = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=path.device),
                prefix_mask,
            ],
            dim=1,
        )
        length = tokens.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=path.device),
            diagonal=1,
        )
        encoded = self.pointer(tokens, mask=causal, padding_mask=~valid)
        last = valid.sum(dim=1) - 1
        query = self.query_norm(
            encoded[torch.arange(batch_size, device=path.device), last]
        ).unsqueeze(1)
        return encoded, valid, query

    def forward_with_aux(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        cur, goal = self._encode_rays(batch)
        future = self._root_los_future(batch)
        _, _, query = self._path_context(batch)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is None:
            candidate_mask = batch["los_current"].bool()
        vertex = torch.cat([batch["vertex"], batch["transition"][..., -7:]], dim=-1)
        route = torch.cat([batch["state"], batch["transition"][..., 11:15]], dim=-1)
        candidate_h = self.feature_fusion(
            cur,
            goal,
            vertex,
            route,
            batch["los_context"],
            future,
            query,
            candidate_mask.bool(),
        )
        candidate_h = self.candidate_attention(
            candidate_h,
            candidate_mask.bool(),
        )

        policy = self.rank_head(
            query,
            candidate_h,
            batch["los_goal"],
        )
        logits = policy
        rx_available = batch["rx_available"].bool()
        vertex_legal = (
            candidate_mask.bool()
            & ~batch["visited"].bool()
            & ~rx_available.unsqueeze(1)
            & batch["depth"].lt(MAX_HOPS).unsqueeze(1)
        )
        minimum = torch.finfo(logits.dtype).min
        vertex_pointer = logits.masked_fill(~vertex_legal, minimum)
        rx_pointer = torch.zeros_like(logits[:, :1]).masked_fill(
            ~rx_available.unsqueeze(1), minimum
        )
        pointer = torch.cat([vertex_pointer, rx_pointer], dim=1)
        legal = torch.cat([vertex_legal, rx_available.unsqueeze(1)], dim=1)
        return {"pointer": pointer, "legal": legal}


__all__ = ["IRTPathModel", "MAX_PREFIX_LEN", "MAX_HOPS", "numpy_batch_to_torch"]
