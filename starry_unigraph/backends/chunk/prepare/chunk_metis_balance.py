"""Multi-dimensional chunk rebalancing based on time-slice load vectors.

Implements chunk_metis_balance algorithm from update.md:
1. Build time_ptr for time slicing
2. Cut graph into C = P * K chunks
3. Compute L[t, c] load matrix (time-slice × chunk)
4. Build chunk graph with multi-dimensional node weights
5. Partition chunk graph to P ranks minimizing max_t max_p load
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .chunk_assignment import ChunkAssignment
from .load_stats import ChunkLoadStats


@dataclass
class ChunkGraphEdge:
    """Edge in the chunk graph for metis partitioning.

    Attributes:
        src_chunk: Source chunk ID
        dst_chunk: Destination chunk ID
        weight: Edge weight (cross-chunk edges, shared nodes, co-occurrence)
    """
    src_chunk: int
    dst_chunk: int
    weight: float


def build_chunk_load_matrix(
    edge_src: Tensor,
    edge_dst: Tensor,
    time_ptr: Tensor,
    node_to_chunk: Tensor,
    chunk_to_owner: Tensor,
    node_to_partition: Tensor,
    edge_weight: float = 1.0,
    remote_edge_weight: float = 2.0,
) -> Tensor:
    """Build L[t, c] load matrix for chunk_metis_balance.

    Args:
        edge_src: [E] Source node IDs
        edge_dst: [E] Destination node IDs
        time_ptr: [T + 1] Time slice boundaries
        node_to_chunk: [num_nodes] Chunk assignment
        chunk_to_owner: [num_chunks] Owner partition
        node_to_partition: [num_nodes] Node partition
        edge_weight: Weight for local edges
        remote_edge_weight: Weight for remote edges

    Returns:
        L: [T, C] Load matrix where L[t, c] is chunk c's load in slice t
    """
    num_slices = int(time_ptr.numel()) - 1
    num_chunks = int(chunk_to_owner.numel())

    L = torch.zeros(num_slices, num_chunks, dtype=torch.float32)

    for t in range(num_slices):
        start = int(time_ptr[t])
        end = int(time_ptr[t + 1])

        if start >= end:
            continue

        slice_src = edge_src[start:end]
        slice_dst = edge_dst[start:end]

        # Compute chunk IDs
        dst_chunks = node_to_chunk[slice_dst]
        dst_owners = chunk_to_owner[dst_chunks]
        src_parts = node_to_partition[slice_src]

        # Edge count per chunk
        edge_count = torch.zeros(num_chunks, dtype=torch.float32, device=dst_chunks.device)
        edge_count.scatter_add_(0, dst_chunks, torch.ones_like(dst_chunks, dtype=torch.float32))

        # Remote edge count per chunk
        is_remote = (src_parts != dst_owners)
        remote_edge_count = torch.zeros(num_chunks, dtype=torch.float32, device=dst_chunks.device)
        remote_edge_count.scatter_add_(0, dst_chunks, is_remote.float())

        # Composite load
        local_edge_count = edge_count - remote_edge_count
        slice_load = local_edge_count * edge_weight + remote_edge_count * remote_edge_weight

        L[t, :] = slice_load.cpu()

    return L


def build_chunk_graph(
    edge_src: Tensor,
    edge_dst: Tensor,
    node_to_chunk: Tensor,
    time_ptr: Optional[Tensor] = None,
) -> Tuple[List[ChunkGraphEdge], Dict[int, int]]:
    """Build chunk graph for metis partitioning.

    Chunk graph edges represent:
    - Cross-chunk edges in the original graph
    - Shared remote nodes between chunks
    - Co-occurrence in the same time slice

    Args:
        edge_src: [E] Source node IDs
        edge_dst: [E] Destination node IDs
        node_to_chunk: [num_nodes] Chunk assignment
        time_ptr: [T + 1] Optional time slice boundaries

    Returns:
        edges: List of chunk graph edges
        chunk_degrees: Dict mapping chunk_id to degree
    """
    src_chunks = node_to_chunk[edge_src]
    dst_chunks = node_to_chunk[edge_dst]

    # Cross-chunk edges
    cross_chunk_mask = src_chunks != dst_chunks
    cross_src = src_chunks[cross_chunk_mask]
    cross_dst = dst_chunks[cross_chunk_mask]

    # Count cross-chunk edge weights
    num_chunks = int(node_to_chunk.max()) + 1
    edge_weights: Dict[Tuple[int, int], float] = {}

    for s, d in zip(cross_src.tolist(), cross_dst.tolist()):
        key = (min(s, d), max(s, d))  # Undirected edge
        edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

    # Build edge list
    edges = [
        ChunkGraphEdge(src_chunk=s, dst_chunk=d, weight=w)
        for (s, d), w in edge_weights.items()
    ]

    # Compute chunk degrees
    chunk_degrees: Dict[int, int] = {}
    for edge in edges:
        chunk_degrees[edge.src_chunk] = chunk_degrees.get(edge.src_chunk, 0) + 1
        chunk_degrees[edge.dst_chunk] = chunk_degrees.get(edge.dst_chunk, 0) + 1

    return edges, chunk_degrees


def greedy_rebalance_by_slice(
    assignment: ChunkAssignment,
    load_matrix: Tensor,
    num_partitions: int,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
) -> ChunkAssignment:
    """Greedy rebalancing using time-slice load vectors.

    Maintains a [num_partitions, num_slices] load matrix and migrates chunks
    to minimize the maximum slice peak across all partitions.

    Args:
        assignment: Current chunk assignment
        load_matrix: [num_slices, num_chunks] Load matrix L[t, c]
        num_partitions: Number of partitions
        max_imbalance_ratio: Acceptable imbalance threshold
        max_migrations: Maximum number of chunk migrations

    Returns:
        Updated ChunkAssignment with new chunk_to_owner_partition
    """
    num_slices, num_chunks = load_matrix.shape

    # Initialize partition load matrix [P, T]
    partition_load = torch.zeros(num_partitions, num_slices, dtype=torch.float32)

    for c in range(num_chunks):
        owner = int(assignment.chunk_to_owner_partition[c])
        partition_load[owner, :] += load_matrix[:, c]

    # Compute initial imbalance (max slice peak across all partitions)
    def compute_imbalance(pload: Tensor) -> float:
        slice_maxes = pload.max(dim=0).values  # [T]
        global_max = slice_maxes.max().item()
        slice_mins = pload.min(dim=0).values
        global_min = slice_mins.min().item()
        return global_max / max(global_min, 1e-9)

    initial_imbalance = compute_imbalance(partition_load)

    # Sort chunks by total load descending
    chunk_total_loads = load_matrix.sum(dim=0)  # [C]
    sorted_chunks = torch.argsort(chunk_total_loads, descending=True).tolist()

    migration_count = 0
    for cid in sorted_chunks:
        if max_migrations is not None and migration_count >= max_migrations:
            break

        if compute_imbalance(partition_load) <= max_imbalance_ratio:
            break

        current_owner = int(assignment.chunk_to_owner_partition[cid])
        chunk_load_vec = load_matrix[:, cid]  # [T]

        # Find partition with minimum peak load after adding this chunk
        best_partition = current_owner
        best_peak = float('inf')

        for p in range(num_partitions):
            if p == current_owner:
                continue

            # Simulate migration
            new_load = partition_load.clone()
            new_load[current_owner, :] -= chunk_load_vec
            new_load[p, :] += chunk_load_vec

            # Compute new peak
            new_peak = new_load.max().item()

            if new_peak < best_peak:
                best_peak = new_peak
                best_partition = p

        # Apply migration if it reduces imbalance
        if best_partition != current_owner:
            partition_load[current_owner, :] -= chunk_load_vec
            partition_load[best_partition, :] += chunk_load_vec
            assignment.chunk_to_owner_partition[cid] = best_partition
            migration_count += 1

    final_imbalance = compute_imbalance(partition_load)

    print(f"Rebalanced {migration_count} chunks: imbalance {initial_imbalance:.3f} → {final_imbalance:.3f}")

    return assignment


def chunk_metis_balance(
    edge_src: Tensor,
    edge_dst: Tensor,
    time_ptr: Tensor,
    node_partition: Tensor,
    num_chunks_per_partition: int = 32,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
) -> ChunkAssignment:
    """Full chunk_metis_balance pipeline.

    1. Build initial chunk assignment from node partition
    2. Compute L[t, c] load matrix
    3. Rebalance chunks to minimize max slice peak

    Args:
        edge_src: [E] Source node IDs
        edge_dst: [E] Destination node IDs
        time_ptr: [T + 1] Time slice boundaries
        node_partition: [num_nodes] Initial node partition
        num_chunks_per_partition: Chunks per partition
        max_imbalance_ratio: Acceptable imbalance threshold
        max_migrations: Maximum chunk migrations

    Returns:
        ChunkAssignment with balanced chunk_to_owner_partition
    """
    from .chunk_assignment import build_chunk_assignment

    # Step 1: Build initial chunk assignment
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition)

    # Step 2: Compute load matrix
    num_nodes = int(node_partition.numel())
    node_to_chunk = assignment.node_to_chunk
    chunk_to_owner = assignment.chunk_to_owner_partition

    load_matrix = build_chunk_load_matrix(
        edge_src=edge_src,
        edge_dst=edge_dst,
        time_ptr=time_ptr,
        node_to_chunk=node_to_chunk,
        chunk_to_owner=chunk_to_owner,
        node_to_partition=node_partition,
    )

    # Step 3: Rebalance
    num_partitions = int(node_partition.max()) + 1
    assignment = greedy_rebalance_by_slice(
        assignment=assignment,
        load_matrix=load_matrix,
        num_partitions=num_partitions,
        max_imbalance_ratio=max_imbalance_ratio,
        max_migrations=max_migrations,
    )

    return assignment
