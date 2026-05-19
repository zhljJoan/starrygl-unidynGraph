"""Compact embedding row lookup helpers shared by model heads."""

from __future__ import annotations

import torch
from torch import Tensor


def compact_rows(node_ids: Tensor, id_map_nodes: Tensor | None, embeddings: Tensor) -> Tensor:
    """Map global node ids to compact embedding rows.

    ``id_map_nodes`` is the ordered list of global node ids represented by
    rows in ``embeddings``.  If it is absent, node ids are already row ids.
    """
    if id_map_nodes is None:
        return node_ids.long().to(device=embeddings.device)
    id_map_nodes = id_map_nodes.to(device=node_ids.device).long().contiguous()
    node_ids = node_ids.long()
    if id_map_nodes.numel() == 0:
        if node_ids.numel() == 0:
            return node_ids.to(device=embeddings.device)
        missing = node_ids[:8].detach().cpu().tolist()
        raise KeyError(f"node ids missing from empty compact id_map_nodes: {missing}")
    rows = torch.searchsorted(id_map_nodes, node_ids)
    valid_rows = rows.clamp_max(id_map_nodes.numel() - 1)
    valid = (rows < id_map_nodes.numel()) & (id_map_nodes[valid_rows] == node_ids)
    if not bool(valid.all()):
        missing = node_ids[~valid][:8].detach().cpu().tolist()
        raise KeyError(f"node ids missing from compact id_map_nodes: {missing}")
    return rows.to(device=embeddings.device)
