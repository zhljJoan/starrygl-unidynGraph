"""Chunk communication route data structures.

Two orthogonal route types, organized by what is communicated:

SpatialRouteData — all-to-all for node *feature* exchange.
    Used in GNN aggregation when edges cross partition boundaries.
    A partition sends its local nodes' features to peers that need
    them as source (boundary) nodes, and receives remote src features.
    Precomputed from the edge structure; valid for the whole epoch.

MemoryRouteData — all-to-all for memory/state *cache* updates.
    Carries the latest temporal memory vector for each node that
    participated in events of a given time slice.

    Two-phase construction separates the graph-partition-independent
    part from the partition-dependent routing:

      Phase 1 (pre-partition, cheap to redo on repartition):
        unique_nodes  — deduplicated src∪dst, one entry per node
        cand_pos      — [D, K] candidate event positions, latest first

      Phase 2 (post-partition, lightweight O(D) per slice):
        send_ptr / recv_ptr / recv_node_ids — routing by node_owner

    When chunk ownership changes (rebalance), only Phase 2 needs
    recomputing.  Phase 1 output is reused as-is.

CPUMemoryLayout — row order of the CPU feature/memory store.
    Hot (replica) nodes first; cold k-hop neighbours after.
    global_to_local[g] = row index or -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Spatial feature route
# ---------------------------------------------------------------------------

@dataclass
class SpatialRouteData:
    """All-to-all routing for node feature fetch and exchange.

    Covers both:
      - Feature fetch: remote src node features pulled from owner partitions
      - Aggregation exchange: local dst node features pushed to peer partitions

    The same send_index / send_ptr / recv_ptr structure serves both
    directions because they are symmetric within one all-to-all call.

    Attributes:
        send_index:    [S] local node indices (into partition's node array),
                       sorted by destination partition.
        send_ptr:      [P+1] CSR of send_index by destination partition.
        recv_ptr:      [P+1] precomputed CSR of expected recv count per source.
        recv_node_ids: [recv_total] global IDs of nodes we receive.
    """

    send_index:    Tensor   # [S]
    send_ptr:      Tensor   # [P+1]
    recv_ptr:      Tensor   # [P+1]
    recv_node_ids: Tensor   # [recv_total]
    unique_index: Optional[Tensor] = None       # [S] packed DistIndex for sent rows
    recv_index: Optional[Tensor] = None         # [recv_total] packed DistIndex for received rows
    read_dist_index: Optional[Tensor] = None    # [recv_total] sampled CTDG remote read set
    master_dist_index: Optional[Tensor] = None  # optional debug/compat lookup

    @property
    def num_send(self) -> int:
        return int(self.send_index.numel())

    @property
    def num_recv(self) -> int:
        recv_ids = getattr(self, "recv_node_ids", None)
        recv_index = getattr(self, "recv_index", None)
        if recv_ids is not None:
            return int(recv_ids.numel())
        return 0 if recv_index is None else int(recv_index.numel())

    def pin_memory(self) -> SpatialRouteData:
        return SpatialRouteData(
            send_index    = self.send_index.pin_memory(),
            send_ptr      = self.send_ptr.pin_memory(),
            recv_ptr      = self.recv_ptr.pin_memory(),
            recv_node_ids = self.recv_node_ids.pin_memory(),
            unique_index  = None if getattr(self, "unique_index", None) is None else self.unique_index.pin_memory(),
            recv_index    = None if getattr(self, "recv_index", None) is None else self.recv_index.pin_memory(),
            read_dist_index = None if getattr(self, "read_dist_index", None) is None else self.read_dist_index.pin_memory(),
            master_dist_index = None if getattr(self, "master_dist_index", None) is None else self.master_dist_index.pin_memory(),
        )

    def to(self, device) -> SpatialRouteData:
        return SpatialRouteData(
            send_index    = self.send_index.to(device),
            send_ptr      = self.send_ptr.to(device),
            recv_ptr      = self.recv_ptr.to(device),
            recv_node_ids = self.recv_node_ids.to(device),
            unique_index  = None if getattr(self, "unique_index", None) is None else self.unique_index.to(device),
            recv_index    = None if getattr(self, "recv_index", None) is None else self.recv_index.to(device),
            read_dist_index = None if getattr(self, "read_dist_index", None) is None else self.read_dist_index.to(device),
            master_dist_index = None if getattr(self, "master_dist_index", None) is None else self.master_dist_index.to(device),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self, Path(path).expanduser().resolve())

    @classmethod
    def load(cls, path: str | Path) -> SpatialRouteData:
        return torch.load(Path(path).expanduser().resolve(), weights_only=False)


# ---------------------------------------------------------------------------
# Memory cache update route
# ---------------------------------------------------------------------------

@dataclass
class MemoryRouteData:
    """All-to-all routing for memory/state cache updates per time slice.

    Phase 1 fields (graph-partition independent):
        unique_nodes  [D] — node IDs appearing in this slice (src ∪ dst),
                            deduplicated, one entry per node.
        cand_pos      [D, K] — candidate event positions, sorted latest-first
                            per node; -1 = absent.  Column 0 is always the
                            latest event.  Used for training-time perturbation.

    Phase 2 fields (depends on partition assignment; cheap to recompute):
        send_ptr      [P+1] — CSR of unique_nodes grouped by owner partition.
        recv_ptr      [P+1] — precomputed: how many nodes we receive per source.
        recv_node_ids [recv_total] — global IDs of nodes we receive.

    Replica fields (hot nodes with cross-partition copies):
        replica_idx       [R] — indices into unique_nodes that are replicas.
        replica_send_ptr  [P+1] — CSR of replica nodes by owner.
        replica_recv_ptr  [P+1] — precomputed recv for replica all-gather.
    """

    # Phase 1 — stable across repartitioning
    unique_nodes: Tensor   # [D]
    cand_pos:     Tensor   # [D, K]

    # Phase 2 — partition-dependent
    send_ptr:      Tensor  # [P+1]
    recv_ptr:      Tensor  # [P+1]
    recv_node_ids: Tensor  # [recv_total]
    unique_index: Optional[Tensor] = None       # [D] packed DistIndex aligned with unique_nodes
    recv_index: Optional[Tensor] = None         # [recv_total] packed DistIndex aligned with recv_node_ids
    read_dist_index: Optional[Tensor] = None    # optional owner/local read rows
    master_dist_index: Optional[Tensor] = None  # optional debug/compat lookup

    # Replica / hot-node sync
    replica_idx:      Optional[Tensor] = None   # [R]
    replica_send_ptr: Optional[Tensor] = None   # [P+1]
    replica_recv_ptr: Optional[Tensor] = None   # [P+1]

    @property
    def num_unique(self) -> int:
        return int(self.unique_nodes.numel())

    @property
    def num_candidates(self) -> int:
        return int(self.cand_pos.size(1))

    @property
    def has_replicas(self) -> bool:
        return self.replica_idx is not None and self.replica_idx.numel() > 0

    def latest_pos(self) -> Tensor:
        """[D] position of the latest event per unique node."""
        return self.cand_pos[:, 0]

    def filter_updates(self, keep_mask: Tensor) -> MemoryRouteData:
        """Return a route containing only nodes selected by ``keep_mask``.

        ``unique_nodes`` are stored sorted by destination owner, so the CSR
        send pointer can be rebuilt by segment ids and ``bincount`` without a
        per-partition Python loop.  This is used by the MemShare-style
        change-rate check before memory communication.
        """
        if keep_mask.dtype != torch.bool:
            raise TypeError(f"keep_mask must be bool, got {keep_mask.dtype}")
        if keep_mask.numel() != self.unique_nodes.numel():
            raise ValueError("keep_mask length must match unique_nodes")

        keep_mask = keep_mask.to(device=self.unique_nodes.device)
        old_ptr = self.send_ptr.to(device=keep_mask.device)
        num_parts = int(old_ptr.numel()) - 1
        segment_sizes = old_ptr[1:] - old_ptr[:-1]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_parts, dtype=torch.long, device=old_ptr.device),
            segment_sizes,
        )
        kept_segments = segment_ids[keep_mask]
        send_counts = torch.bincount(kept_segments, minlength=num_parts)
        send_ptr = torch.zeros(num_parts + 1, dtype=torch.long, device=old_ptr.device)
        send_ptr[1:] = send_counts.cumsum(0)

        kept_nodes = self.unique_nodes[keep_mask]
        kept_cand = self.cand_pos[keep_mask]
        unique_index = None
        if getattr(self, "unique_index", None) is not None:
            unique_index = self.unique_index.to(keep_mask.device)[keep_mask]
        read_dist_index = None
        if getattr(self, "read_dist_index", None) is not None:
            read_dist_index = self.read_dist_index.to(keep_mask.device)[keep_mask]

        replica_idx = replica_send_ptr = replica_recv_ptr = None
        if self.replica_idx is not None and self.replica_idx.numel() > 0:
            old_to_new = torch.full(
                (keep_mask.numel(),),
                -1,
                dtype=torch.long,
                device=keep_mask.device,
            )
            old_to_new[keep_mask] = torch.arange(int(keep_mask.sum().item()), device=keep_mask.device)
            replica_idx = old_to_new[self.replica_idx.to(keep_mask.device)]
            replica_idx = replica_idx[replica_idx >= 0]
            if self.replica_send_ptr is not None:
                old_rep_ptr = self.replica_send_ptr.to(device=keep_mask.device)
                rep_keep = keep_mask[self.replica_idx.to(keep_mask.device)]
                rep_segment_sizes = old_rep_ptr[1:] - old_rep_ptr[:-1]
                rep_segment_ids = torch.repeat_interleave(
                    torch.arange(num_parts, dtype=torch.long, device=keep_mask.device),
                    rep_segment_sizes,
                )
                rep_counts = torch.bincount(rep_segment_ids[rep_keep], minlength=num_parts)
                replica_send_ptr = torch.zeros(num_parts + 1, dtype=torch.long, device=keep_mask.device)
                replica_send_ptr[1:] = rep_counts.cumsum(0)
            replica_recv_ptr = self.replica_recv_ptr

        return MemoryRouteData(
            unique_nodes=kept_nodes,
            cand_pos=kept_cand,
            send_ptr=send_ptr,
            recv_ptr=self.recv_ptr,
            recv_node_ids=self.recv_node_ids,
            unique_index=unique_index,
            recv_index=getattr(self, "recv_index", None),
            read_dist_index=read_dist_index,
            master_dist_index=getattr(self, "master_dist_index", None),
            replica_idx=replica_idx,
            replica_send_ptr=replica_send_ptr,
            replica_recv_ptr=replica_recv_ptr,
        )

    def sampled_pos(self, rng: Optional[torch.Generator] = None) -> Tensor:
        """[D] randomly sampled candidate event position (perturbation).

        Falls back to latest (col 0) for nodes with fewer than K events.
        """
        D, K = self.cand_pos.shape
        col = torch.randint(0, K, (D,), generator=rng, device=self.cand_pos.device)
        chosen = self.cand_pos[torch.arange(D, device=self.cand_pos.device), col]
        fallback = chosen < 0
        if fallback.any():
            chosen[fallback] = self.cand_pos[fallback, 0]
        return chosen

    def assign_partition_ptrs(
        self,
        node_owner: Tensor,
        num_parts:  int,
    ) -> MemoryRouteData:
        """Return a new MemoryRouteData with Phase 2 fields filled.

        Call this after (re)partitioning to update routing without
        re-running the expensive Phase 1 dedup.

        Args:
            node_owner: [num_nodes] owner partition per global node ID.
            num_parts:  total partition count.
        """
        owners = node_owner[self.unique_nodes]          # [D]
        sort_o = torch.argsort(owners, stable=True)     # [D]

        sorted_nodes  = self.unique_nodes[sort_o]
        sorted_owners = owners[sort_o]
        sorted_cand   = self.cand_pos[sort_o]

        send_counts = torch.bincount(sorted_owners, minlength=num_parts)
        send_ptr    = torch.zeros(num_parts + 1, dtype=torch.long)
        send_ptr[1:] = send_counts.cumsum(0)

        # replica
        replica_idx = replica_send_ptr = replica_recv_ptr = None
        if self.replica_idx is not None:
            # Map old indices through sort_o inverse
            inv = torch.empty_like(sort_o)
            inv[sort_o] = torch.arange(len(sort_o), device=sort_o.device)
            new_replica_idx = inv[self.replica_idx]
            rep_owners = sorted_owners[new_replica_idx]
            rep_cnt    = torch.bincount(rep_owners, minlength=num_parts)
            replica_send_ptr = torch.zeros(num_parts + 1, dtype=torch.long)
            replica_send_ptr[1:] = rep_cnt.cumsum(0)
            replica_idx = new_replica_idx

        return MemoryRouteData(
            unique_nodes      = sorted_nodes,
            cand_pos          = sorted_cand,
            send_ptr          = send_ptr,
            recv_ptr          = torch.zeros(num_parts + 1, dtype=torch.long),  # filled by fill_recv_ptrs
            recv_node_ids     = torch.zeros(0, dtype=torch.long),
            unique_index      = None,
            recv_index        = None,
            read_dist_index   = None,
            master_dist_index = getattr(self, "master_dist_index", None),
            replica_idx       = replica_idx,
            replica_send_ptr  = replica_send_ptr,
            replica_recv_ptr  = replica_recv_ptr,
        )

    def pin_memory(self) -> MemoryRouteData:
        def _p(t): return None if t is None else t.pin_memory()
        return MemoryRouteData(
            unique_nodes      = self.unique_nodes.pin_memory(),
            cand_pos          = self.cand_pos.pin_memory(),
            send_ptr          = self.send_ptr.pin_memory(),
            recv_ptr          = self.recv_ptr.pin_memory(),
            recv_node_ids     = self.recv_node_ids.pin_memory(),
            unique_index      = _p(getattr(self, "unique_index", None)),
            recv_index        = _p(getattr(self, "recv_index", None)),
            read_dist_index   = _p(getattr(self, "read_dist_index", None)),
            master_dist_index = _p(getattr(self, "master_dist_index", None)),
            replica_idx       = _p(self.replica_idx),
            replica_send_ptr  = _p(self.replica_send_ptr),
            replica_recv_ptr  = _p(self.replica_recv_ptr),
        )

    def to(self, device) -> MemoryRouteData:
        def _t(x): return None if x is None else x.to(device)
        return MemoryRouteData(
            unique_nodes      = self.unique_nodes.to(device),
            cand_pos          = self.cand_pos.to(device),
            send_ptr          = self.send_ptr.to(device),
            recv_ptr          = self.recv_ptr.to(device),
            recv_node_ids     = self.recv_node_ids.to(device),
            unique_index      = _t(getattr(self, "unique_index", None)),
            recv_index        = _t(getattr(self, "recv_index", None)),
            read_dist_index   = _t(getattr(self, "read_dist_index", None)),
            master_dist_index = _t(getattr(self, "master_dist_index", None)),
            replica_idx       = _t(self.replica_idx),
            replica_send_ptr  = _t(self.replica_send_ptr),
            replica_recv_ptr  = _t(self.replica_recv_ptr),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self, Path(path).expanduser().resolve())

    @classmethod
    def load(cls, path: str | Path) -> MemoryRouteData:
        return torch.load(Path(path).expanduser().resolve(), weights_only=False)


# ---------------------------------------------------------------------------
# CPU memory layout
# ---------------------------------------------------------------------------

@dataclass
class CPUMemoryLayout:
    """Hot/cold node layout for the CPU feature and memory store.

    layout_order[:hot_boundary]  → hot (replica) nodes, placed first
    layout_order[hot_boundary:]  → cold k-hop neighbours

    global_to_local[g] = row index in CPU store, or -1 if not cached.
    """

    layout_order:    Tensor   # [C]
    hot_boundary:    int
    global_to_local: Tensor   # [num_nodes]

    @property
    def num_cached(self) -> int:
        return int(self.layout_order.numel())

    @classmethod
    def build(cls, hot_nodes: Tensor, cold_nodes: Tensor, num_nodes: int) -> CPUMemoryLayout:
        layout_order    = torch.cat([hot_nodes, cold_nodes])
        global_to_local = torch.full((num_nodes,), -1, dtype=torch.long)
        global_to_local[layout_order] = torch.arange(layout_order.numel(), dtype=torch.long)
        return cls(layout_order=layout_order, hot_boundary=int(hot_nodes.numel()),
                   global_to_local=global_to_local)

    def save(self, path: str | Path) -> None:
        torch.save(self, Path(path).expanduser().resolve())

    @classmethod
    def load(cls, path: str | Path) -> CPUMemoryLayout:
        return torch.load(Path(path).expanduser().resolve(), weights_only=False)
