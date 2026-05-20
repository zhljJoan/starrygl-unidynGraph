from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class FeatureStore:
    """Row-indexed feature storage with optional temporal snapshots."""

    def __init__(self, node_features: Optional[Tensor] = None, edge_features: Optional[Tensor] = None) -> None:
        self.node_features = node_features
        self.edge_features = edge_features

    @property
    def device(self) -> torch.device:
        if self.node_features is not None:
            return self.node_features.device
        if self.edge_features is not None:
            return self.edge_features.device
        return torch.device("cpu")

    def gather_rows(self, rows: Tensor, time_slices: Tensor | None = None) -> Tensor:
        return self.gather_node_rows(rows, time_slices=time_slices)

    def gather_node_rows(self, rows: Tensor, time_slices: Tensor | None = None) -> Tensor:
        if self.node_features is None:
            return torch.empty((rows.numel(), 0), device=rows.device)
        return _gather_feature_rows(self.node_features, rows, time_slices)

    def gather_edge_rows(self, rows: Tensor, time_slices: Tensor | None = None) -> Tensor:
        if self.edge_features is None:
            return torch.empty((rows.numel(), 0), device=rows.device)
        return _gather_feature_rows(self.edge_features, rows, time_slices)

    def node(self, node_ids: Tensor, time_slices: Tensor | None = None) -> Tensor:
        return self.gather_node_rows(node_ids, time_slices=time_slices)

    def edge(self, edge_ids: Tensor, time_slices: Tensor | None = None) -> Tensor:
        return self.gather_edge_rows(edge_ids, time_slices=time_slices)


def _gather_feature_rows(features: Tensor, rows: Tensor, time_slices: Tensor | None) -> Tensor:
    row = rows.long().to(features.device)
    if time_slices is None:
        return features.index_select(0, row).to(rows.device)
    if features.dim() < 3:
        return features.index_select(0, row).to(rows.device)
    ts = time_slices.long().to(features.device).reshape(-1)
    if ts.numel() != row.numel():
        raise ValueError("time_slices must match rows length")
    return features[ts, row].to(rows.device)
