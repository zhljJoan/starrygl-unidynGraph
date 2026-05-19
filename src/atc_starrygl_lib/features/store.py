from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class FeatureStore:
    def __init__(self, node_features: Optional[Tensor] = None, edge_features: Optional[Tensor] = None) -> None:
        self.node_features = node_features
        self.edge_features = edge_features

    def node(self, node_ids: Tensor) -> Tensor:
        if self.node_features is None:
            return torch.empty((node_ids.numel(), 0), device=node_ids.device)
        return self.node_features.index_select(0, node_ids.long().to(self.node_features.device)).to(node_ids.device)

    def edge(self, edge_ids: Tensor) -> Tensor:
        if self.edge_features is None:
            return torch.empty((edge_ids.numel(), 0), device=edge_ids.device)
        return self.edge_features.index_select(0, edge_ids.long().to(self.edge_features.device)).to(edge_ids.device)
