from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class FeatureStore:
    """Row-indexed feature storage with optional temporal snapshots."""

    def __init__(
        self,
        node_features: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        *,
        node_row_map: Optional[Tensor] = None,
        edge_row_map: Optional[Tensor] = None,
        sort_mapped_gather: bool = False,
        pin_memory_transfer: bool = False,
    ) -> None:
        self.node_features = node_features
        self.edge_features = edge_features
        self.node_row_map = node_row_map
        self.edge_row_map = edge_row_map
        self.sort_mapped_gather = bool(sort_mapped_gather)
        self.pin_memory_transfer = bool(pin_memory_transfer)

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
        return _gather_feature_rows(
            self.node_features,
            rows,
            time_slices,
            row_map=self.node_row_map,
            sort_mapped_gather=self.sort_mapped_gather,
            pin_memory_transfer=self.pin_memory_transfer,
        )

    def gather_edge_rows(self, rows: Tensor, time_slices: Tensor | None = None) -> Tensor:
        if self.edge_features is None:
            return torch.empty((rows.numel(), 0), device=rows.device)
        return _gather_feature_rows(
            self.edge_features,
            rows,
            time_slices,
            row_map=self.edge_row_map,
            sort_mapped_gather=self.sort_mapped_gather,
            pin_memory_transfer=self.pin_memory_transfer,
        )

    def node(self, node_ids: Tensor, time_slices: Tensor | None = None) -> Tensor:
        return self.gather_node_rows(node_ids, time_slices=time_slices)

    def edge(self, edge_ids: Tensor, time_slices: Tensor | None = None) -> Tensor:
        return self.gather_edge_rows(edge_ids, time_slices=time_slices)


def _gather_feature_rows(
    features: Tensor,
    rows: Tensor,
    time_slices: Tensor | None,
    *,
    row_map: Tensor | None = None,
    sort_mapped_gather: bool = False,
    pin_memory_transfer: bool = False,
) -> Tensor:
    row = rows.long().to(features.device)
    if row_map is not None:
        row = row_map.to(features.device).index_select(0, row)
    restore = None
    if bool(sort_mapped_gather) and row.numel() > 1:
        order = torch.argsort(row, stable=True)
        restore = torch.empty_like(order)
        restore[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        row = row.index_select(0, order)
        if time_slices is not None:
            time_slices = time_slices.to(features.device).reshape(-1).index_select(0, order)
    if time_slices is None:
        out = features.index_select(0, row)
        out = out if restore is None else out.index_select(0, restore)
        return _to_request_device(out, rows, pin_memory_transfer=pin_memory_transfer)
    if features.dim() < 3:
        out = features.index_select(0, row)
        out = out if restore is None else out.index_select(0, restore)
        return _to_request_device(out, rows, pin_memory_transfer=pin_memory_transfer)
    ts = time_slices.long().to(features.device).reshape(-1)
    if ts.numel() != row.numel():
        raise ValueError("time_slices must match rows length")
    out = features[ts, row]
    out = out if restore is None else out.index_select(0, restore)
    return _to_request_device(out, rows, pin_memory_transfer=pin_memory_transfer)


def _to_request_device(out: Tensor, rows: Tensor, *, pin_memory_transfer: bool) -> Tensor:
    if out.device == rows.device:
        return out
    if bool(pin_memory_transfer) and out.device.type == "cpu" and rows.device.type == "cuda":
        out = out.pin_memory()
        return out.to(rows.device, non_blocking=True)
    return out.to(rows.device)
