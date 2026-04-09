from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from starry_unigraph.lib import load_bts_sampler_module


def is_bts_sampler_available() -> bool:
    try:
        load_bts_sampler_module()
    except Exception:
        return False
    return True


def build_temporal_neighbor_block(
    graph_name: str,
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    eid: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    timestamp: torch.Tensor | None = None,
    is_distinct: int = 0,
):
    sampler_mod = load_bts_sampler_module()
    return sampler_mod.get_neighbors(
        graph_name,
        row.contiguous(),
        col.contiguous(),
        int(num_nodes),
        int(is_distinct),
        eid.contiguous(),
        None,
        edge_weight.contiguous() if edge_weight is not None else None,
        timestamp.contiguous() if timestamp is not None else None,
    )


@dataclass
class BTSNativeSampler:
    tnb: Any
    num_nodes: int
    num_edges: int
    num_layers: int
    fanout: list[int]
    workers: int = 1
    policy: str = "uniform"
    local_part: int = -1
    edge_part: torch.Tensor | None = None
    node_part: torch.Tensor | None = None
    probability: float = 1.0

    def __post_init__(self) -> None:
        sampler_mod = load_bts_sampler_module()
        if self.edge_part is None:
            self.edge_part = torch.zeros(self.num_edges, dtype=torch.int32)
        if self.node_part is None:
            self.node_part = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._sampler = sampler_mod.ParallelSampler(
            self.tnb,
            int(self.num_nodes),
            int(self.num_edges),
            int(self.workers),
            list(self.fanout),
            int(self.num_layers),
            str(self.policy),
            int(self.local_part),
            self.edge_part.to(torch.int32),
            self.node_part.to(torch.int32),
            float(self.probability),
        )

    def sample_from_nodes(self, nodes: torch.Tensor, ts: torch.Tensor | None = None):
        timestamps = ts
        if timestamps is None:
            timestamps = torch.zeros(nodes.numel(), dtype=torch.int64, device=nodes.device)
        self._sampler.neighbor_sample_from_nodes(nodes.contiguous(), timestamps.to(torch.int64).contiguous(), None)
        return self._sampler.get_ret()

    def reset(self) -> None:
        self._sampler.reset()
