"""Edge index conversion utilities for chunk PartitionData.

Provides functions to convert between:
- edge_index format: (edge_src, edge_dst, edge_timestamps, edge_ids)
- PartitionData format: (src_ids, dst_ids, edge_src, edge_dst, dst_chunk, ...)

Key operations:
1. from_edge_index: Build PartitionData from raw edge lists
2. to_edge_index: Reconstruct edge lists from PartitionData
3. edge_events: Extract temporal events for CTDG sampling

The key constraint: Edges are stored sorted by dst_chunk to ensure
edges within the same chunk are contiguous.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def compute_dst_chunks_for_edges(
    edge_dst: Tensor,
    node_to_chunk: Tensor,
) -> Tensor:
    """Compute the dst_chunk for each edge.

    Args:
        edge_dst: [num_edges] Global destination node IDs
        node_to_chunk: [num_nodes] Chunk ID for each node

    Returns:
        [num_edges] Chunk ID for destination of each edge
    """
    return node_to_chunk[edge_dst]


def sort_edges_by_dst_chunk(
    edge_src: Tensor,
    edge_dst: Tensor,
    edge_timestamps: Optional[Tensor] = None,
    edge_ids: Optional[Tensor] = None,
    node_to_chunk: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
    """Sort edges by destination chunk.

    This ensures edges within the same dst_chunk are contiguous in storage,
    facilitating efficient chunk-based sampling.

    Args:
        edge_src: [num_edges] Source node IDs
        edge_dst: [num_edges] Destination node IDs
        edge_timestamps: [num_edges] Optional timestamps
        edge_ids: [num_edges] Optional edge IDs
        node_to_chunk: [num_nodes] Chunk assignment

    Returns:
        (sorted_src, sorted_dst, sorted_ts, sorted_ids, dst_chunks)
    """
    if node_to_chunk is None:
        raise ValueError("node_to_chunk required to sort by dst_chunk")

    # Compute dst_chunk for each edge
    dst_chunks = compute_dst_chunks_for_edges(edge_dst, node_to_chunk)

    # Get sort order by dst_chunk
    num_nodes = num_nodes or (edge_dst.max().item() + 1)
    sort_key = dst_chunks * num_nodes + edge_dst
    sort_order = torch.argsort(sort_key, stable=True)

    # Apply sort order to all edge arrays
    sorted_src = edge_src[sort_order]
    sorted_dst = edge_dst[sort_order]
    sorted_ts = None if edge_timestamps is None else edge_timestamps[sort_order]
    sorted_ids = None if edge_ids is None else edge_ids[sort_order]
    sorted_chunks = dst_chunks[sort_order]

    return sorted_src, sorted_dst, sorted_ts, sorted_ids, sorted_chunks


def build_edge_csr(
    edge_src: Tensor,
    edge_dst: Tensor,
    local_dst_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Build CSR format (src_indices, dst_indices) for edges.

    Args:
        edge_src: [num_edges] Source indices (may be global or local)
        edge_dst: [num_edges] Destination indices (should be local to partition)
        local_dst_mask: [num_nodes] Which destination nodes are local

    Returns:
        (edge_src, edge_dst) already in CSR-like format (sorted by dst)
    """
    # Edges are already sorted by dst_chunk, so they should be fairly local-grouped
    return edge_src, edge_dst


def get_csc_local_ids(
    edge_src: Tensor,
    edge_dst: Tensor,
    local_node_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Get local IDs for CSC format (src_indices, dst_indices).

    Args:
        edge_src: [num_edges] Source indices (may be global or local)
        edge_dst: [num_edges] Destination indices (should be local to partition)
        local_node_ids: [num_local] Local node IDs (if provided, use as ground truth)

    Returns:
        (local_src_ids, local_dst_ids) for CSC format
    """

    # Get unique destination nodes (assumed to be local)
    unique_dst = torch.unique_consecutive(edge_dst)
    if local_node_ids is None:
        # Get unique source nodes
        unique_src = torch.unique(edge_src)
        src_is_local = torch.isin(unique_src, unique_dst)
        remote_src = unique_src[~src_is_local]
        return torch.cat([unique_dst, remote_src]), unique_dst
    else:
        unique_src = torch.unique(edge_src)
        remote_src = unique_src[~torch.isin(unique_src, local_node_ids)]
        return torch.cat([unique_dst, remote_src]), unique_dst


def snapshot_to_partitiondata_tensors(
    edge_src: Tensor,
    edge_dst: Tensor,
    edge_timestamps: Optional[Tensor] = None,
    edge_ids: Optional[Tensor] = None,
    node_to_chunk: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Convert a snapshot's edges to PartitionData tensor representation.

    Args:
        edge_src: [num_edges] Source nodes
        edge_dst: [num_edges] Destination nodes
        edge_timestamps: [num_edges] Optional timestamps
        edge_ids: [num_edges] Optional edge IDs
        node_to_chunk: [num_nodes] Chunk assignment for dst sorting
        num_nodes: Total number of nodes (for completeness check)

    Returns:
        Dict with keys: 'src_ids', 'dst_ids', 'edge_src', 'edge_dst', 'dst_chunk',
        'edge_ids', 'edge_ts'
    """
    # Sort edges by dst_chunk
    sorted_src, sorted_dst, sorted_ts, sorted_ids, dst_chunks = sort_edges_by_dst_chunk(
        edge_src, edge_dst, edge_timestamps, edge_ids, node_to_chunk
    )

    local_dst = torch.unique_consecutive(sorted_dst,return_counts=False)
    # Partition nodes into local (dst) and remote (src - dst).  The local
    # edge source indices are built against the full block source space
    # [dst_ids | remote_src_ids], while PartitionData stores only the remote
    # suffix in src_ids.
    all_src_ids, dst_ids = get_csc_local_ids(sorted_src, sorted_dst)

    # Build mappings from global ID to local index
    # src_ids space: [0..len(src_ids)-1]
    # dst_ids space: [0..len(dst_ids)-1]
    global_to_src_idx = {int(gid): idx for idx, gid in enumerate(all_src_ids)}
    global_to_dst_idx = {int(gid): idx for idx, gid in enumerate(dst_ids)}
    edge_src_local = 
    edge_dst_local = torch.arange(len(dst_ids), dtype=sorted_dst.dtype, device=sorted_dst.device)
    # Remap edge indices to local space
    # edge_src_local = torch.tensor(
    #     [global_to_src_idx[int(g)] for g in sorted_src],
    #     dtype=sorted_src.dtype,
    #     device=sorted_src.device,
    # )

    # edge_dst_local = torch.tensor(
    #     [global_to_dst_idx[int(g)] for g in sorted_dst],
    #     dtype=sorted_dst.dtype,
    #     device=sorted_dst.device,
    # )

    return {
        "src_ids": all_src_ids[dst_ids.numel() :],
        "dst_ids": dst_ids,
        "edge_src": edge_src_local,
        "edge_dst": edge_dst_local,
        "dst_chunk": dst_chunks,
        "edge_ids": sorted_ids,
        "edge_ts": sorted_ts,
    }


def reconstruct_edge_index_from_snapshot(
    src_ids: Tensor,
    dst_ids: Tensor,
    edge_src: Tensor,
    edge_dst: Tensor,
    global_ids: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct global edge_index from PartitionData snapshot.

    Args:
        src_ids: Remote source node IDs
        dst_ids: Local destination node IDs
        edge_src: Edge source indices (into src+dst combined space)
        edge_dst: Edge destination indices (into dst-only space)
        global_ids: If True, return global IDs; else local indices

    Returns:
        (edge_src_global, edge_dst_global)
    """
    # Combined src+dst space: [dst_ids | src_ids]
    all_src_nodes = torch.cat([dst_ids, src_ids], dim=0)

    # Map local indices to global
    edge_src_global = all_src_nodes[edge_src]
    edge_dst_global = dst_ids[edge_dst]

    return edge_src_global, edge_dst_global
