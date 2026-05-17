"""Chunk load statistics: Computation and data structures.

ChunkLoadStats accumulates per-chunk load metrics over all time windows,
used by the rebalancer to decide which chunks to migrate.

Metrics collected per chunk:
- edge_count: total edges touching this chunk's destination nodes
- active_node_count: unique src nodes appearing in this chunk's edges
- remote_edge_count: edges whose src owner != this chunk's current owner
- remote_node_count: unique remote src nodes
- sampling_load: estimate of per-event temporal-neighbor sampling cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class ChunkLoadStats:
    """Load metrics for a single chunk, aggregated across all time windows.

    Attributes:
        chunk_id: Global chunk identifier
        edge_count: Total edges whose dst belongs to this chunk
        active_node_count: Unique source nodes seen across all time windows
        remote_edge_count: Edges where src owner partition != this chunk's owner
        remote_node_count: Unique remote src nodes
        sampling_load: Estimated sampling cost (sum of neighbor counts)
        total_load: Composite scalar used for balancing (computed from above)
    """

    chunk_id: int
    edge_count: int = 0
    active_node_count: int = 0
    remote_edge_count: int = 0
    remote_node_count: int = 0
    sampling_load: float = 0.0
    total_load: float = 0.0

    def compute_total_load(
        self,
        edge_weight: float = 1.0,
        remote_edge_weight: float = 2.0,
        sampling_weight: float = 1.0,
    ) -> float:
        """Compute composite load scalar used for balancing.

        Remote edges are weighted more heavily because they incur
        cross-partition communication overhead.

        Args:
            edge_weight: Weight for local edge count
            remote_edge_weight: Weight for remote (cross-partition) edge count
            sampling_weight: Weight for sampling load

        Returns:
            Composite load value (stored in self.total_load)
        """
        local_edge_count = self.edge_count - self.remote_edge_count
        self.total_load = (
            local_edge_count * edge_weight
            + self.remote_edge_count * remote_edge_weight
            + self.sampling_load * sampling_weight
        )
        return self.total_load


def compute_chunk_load_stats(
    edge_src: Tensor,
    edge_dst: Tensor,
    node_to_chunk: Tensor,
    chunk_to_owner_partition: Tensor,
    node_to_partition: Tensor,
    edge_timestamps: Optional[Tensor] = None,
    num_time_windows: int = 1,
) -> Dict[int, ChunkLoadStats]:
    """Compute per-chunk load statistics from edge data (fully vectorised).

    All edge-level counting is done via scatter_add / torch.unique — no
    Python-level for loops over individual edges.

    Args:
        edge_src: [E] Source node global IDs
        edge_dst: [E] Destination node global IDs
        node_to_chunk: [num_nodes]
        chunk_to_owner_partition: [num_chunks]
        node_to_partition: [num_nodes] Partition for each node
        edge_timestamps: unused (reserved for windowed variant)
        num_time_windows: unused here

    Returns:
        Dict mapping chunk_id → ChunkLoadStats
    """
    num_chunks = int(chunk_to_owner_partition.numel())
    num_nodes = int(node_to_chunk.numel())

    dst_chunks = node_to_chunk[edge_dst]              # [E]
    dst_owners = chunk_to_owner_partition[dst_chunks]  # [E]
    src_parts = node_to_partition[edge_src]            # [E]

    # --- edge_count per chunk ---
    ones = torch.ones(dst_chunks.numel(), dtype=torch.long, device=dst_chunks.device)
    edge_count = torch.zeros(num_chunks, dtype=torch.long, device=dst_chunks.device)
    edge_count.scatter_add_(0, dst_chunks, ones)

    # --- remote_edge_count per chunk ---
    is_remote = (src_parts != dst_owners)              # [E] bool
    remote_edge_count = torch.zeros(num_chunks, dtype=torch.long, device=dst_chunks.device)
    remote_edge_count.scatter_add_(0, dst_chunks, is_remote.long())

    # --- active_node_count per chunk (unique src per chunk) ---
    # Encode (dst_chunk, src_node) as a single int64 key, then unique + scatter
    stride = num_nodes + 1
    combined = dst_chunks.long() * stride + edge_src.long()          # [E]
    unique_pairs = torch.unique(combined)                             # [U]
    unique_cids = (unique_pairs // stride).to(torch.long)            # [U]
    active_node_count = torch.zeros(num_chunks, dtype=torch.long, device=dst_chunks.device)
    active_node_count.scatter_add_(
        0, unique_cids, torch.ones_like(unique_cids)
    )

    # --- remote_node_count per chunk (unique remote src per chunk) ---
    remote_node_count = torch.zeros(num_chunks, dtype=torch.long, device=dst_chunks.device)
    if is_remote.any():
        remote_combined = combined[is_remote]
        unique_remote = torch.unique(remote_combined)
        unique_remote_cids = (unique_remote // stride).to(torch.long)
        remote_node_count.scatter_add_(
            0, unique_remote_cids, torch.ones_like(unique_remote_cids)
        )

    # --- Assemble ChunkLoadStats objects ---
    edge_count_cpu = edge_count.cpu().tolist()
    active_cpu = active_node_count.cpu().tolist()
    remote_e_cpu = remote_edge_count.cpu().tolist()
    remote_n_cpu = remote_node_count.cpu().tolist()

    stats: Dict[int, ChunkLoadStats] = {}
    for cid in range(num_chunks):
        s = ChunkLoadStats(
            chunk_id=cid,
            edge_count=edge_count_cpu[cid],
            active_node_count=active_cpu[cid],
            remote_edge_count=remote_e_cpu[cid],
            remote_node_count=remote_n_cpu[cid],
        )
        s.compute_total_load()
        stats[cid] = s

    return stats


def compute_chunk_load_stats_from_windows(
    window_edge_srcs: List[Tensor],
    window_edge_dsts: List[Tensor],
    node_to_chunk: Tensor,
    chunk_to_owner_partition: Tensor,
    node_to_partition: Tensor,
) -> Dict[int, ChunkLoadStats]:
    """Compute load stats by aggregating over multiple time windows.

    Each window contributes edge/node counts independently, then all
    windows are summed into a single ChunkLoadStats per chunk.

    Args:
        window_edge_srcs: List of [num_edges_w] tensors, one per time window
        window_edge_dsts: List of [num_edges_w] tensors, one per time window
        node_to_chunk: [num_nodes]
        chunk_to_owner_partition: [num_chunks]
        node_to_partition: [num_nodes]

    Returns:
        Dict mapping chunk_id → aggregated ChunkLoadStats
    """
    num_chunks = int(chunk_to_owner_partition.numel())
    aggregated: Dict[int, ChunkLoadStats] = {
        cid: ChunkLoadStats(chunk_id=cid) for cid in range(num_chunks)
    }

    for w_src, w_dst in zip(window_edge_srcs, window_edge_dsts):
        window_stats = compute_chunk_load_stats(
            w_src, w_dst, node_to_chunk, chunk_to_owner_partition, node_to_partition
        )
        for cid, wstat in window_stats.items():
            aggregated[cid].edge_count += wstat.edge_count
            aggregated[cid].remote_edge_count += wstat.remote_edge_count
            # active/remote node sets overlap across windows; take max as proxy
            aggregated[cid].active_node_count = max(
                aggregated[cid].active_node_count, wstat.active_node_count
            )
            aggregated[cid].remote_node_count = max(
                aggregated[cid].remote_node_count, wstat.remote_node_count
            )

    for cid in range(num_chunks):
        aggregated[cid].compute_total_load()

    return aggregated


def compute_chunk_load_by_slice(
    window_edge_srcs: List[Tensor],
    window_edge_dsts: List[Tensor],
    node_to_chunk: Tensor,
    chunk_to_owner_partition: Tensor,
    node_to_partition: Tensor,
) -> Tensor:
    """Compute a dense [num_slices, num_chunks] chunk load matrix.

    Each row is the composite load for one time slice, using the same scalar
    load formula as :class:`ChunkLoadStats`.  The matrix is intended for
    vector-aware chunk ownership assignment, where a chunk's load shape across
    time matters instead of only its aggregate sum.
    """

    num_slices = len(window_edge_srcs)
    num_chunks = int(chunk_to_owner_partition.numel())
    load = torch.zeros(num_slices, num_chunks, dtype=torch.float32)
    for t, (w_src, w_dst) in enumerate(zip(window_edge_srcs, window_edge_dsts)):
        window_stats = compute_chunk_load_stats(
            w_src, w_dst, node_to_chunk, chunk_to_owner_partition, node_to_partition
        )
        for cid, stat in window_stats.items():
            load[t, cid] = float(stat.total_load)
    return load
