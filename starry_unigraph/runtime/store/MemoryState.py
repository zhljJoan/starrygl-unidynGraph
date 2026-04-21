"""Per-node memory bank with K-slot mailbox for CTDG online training.

:class:`CTDGMemoryBank` is the central state container for TGN-style models.
It stores per-node hidden memory and a K-slot mailbox (edge-feature messages),
handles distributed async sync via ``all_to_all``, and integrates with
:class:`CTDGHistoricalCache` for change-detection filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from starry_unigraph.runtime.route.comm_plan import SpatialDeps, StateDeps
from starry_unigraph.types import DistributedContext



@dataclass
class _PendingMemorySync:
    id_work: Any
    payload_work: Any
    recv_ids: torch.Tensor
    recv_payload: torch.Tensor
    recv_counts: list[int]
    sent_remote_ids: torch.Tensor | None = None
    sent_remote_vals: torch.Tensor | None = None
    sent_remote_ts: torch.Tensor | None = None


@dataclass
class _PendingMailSync:
    id_work: Any
    payload_work: Any
    recv_ids: torch.Tensor
    recv_payload: torch.Tensor
    recv_counts: list[int]


@dataclass
class MemoryState:
    def __init__(self, node_nums, hidden_dim, memory_bank):
        self.node_nums = node_nums
        self.state_deps = StateDeps
        self.spatial_deps = SpatialDeps
        self.memory_bank = memory_bank
        self.memory = torch.zeros((node_nums, hidden_dim), device=memory_bank.device)
            
    def update(self, index, memory, mails=None, StateDeps=None):
        pass
    
    def gather(self, index, memory, mails=None, StateDeps=None):
        pass
    
@dataclass
class MemoryBank:
    def __init__(self, num_nodes, hidden_dim, edge_feat_dim=0, mailbox_slots=1, device="cpu", async_sync=False):
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.mailbox_slots = mailbox_slots
        self.device = device
        self.async_sync = async_sync
        self.mailbox = torch.zeros((num_nodes, mailbox_slots, 2*hidden_dim + edge_feat_dim), device=device)

    def describe(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_local_nodes": self.num_local_nodes,
            "hidden_dim": self.hidden_dim,
            "mailbox_slots": self.mailbox_slots,
            "memory_version": self.memory_version,
            "storage_device": str(self._storage_device),
        }
