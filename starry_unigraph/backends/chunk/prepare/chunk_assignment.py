"""Chunk assignment and mapping: Core data structures for chunk-based partitioning.

Defines:
- ChunkAssignment: Tracks chunk membership (node_to_chunk, chunk_to_nodes, etc.)
- Functions to build chunk assignments from initial node partitions
- Support for chunk owner rebalancing (phase 3)

Key definitions:
- global_chunk_id = partition_id * num_chunks_per_partition + local_chunk_id
- chunk_to_initial_partition: Partition where chunk was initially assigned
- chunk_to_owner_partition: Current partition that owns this chunk (after rebalancing)
- node owner = owner(chunk(node)) = chunk_to_owner_partition[node_to_chunk[node]]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ChunkAssignment:
    """Chunk assignment and load tracking for a graph partition.

    Attributes:
        num_chunks_per_partition: Chunks per partition (default 32)
        node_to_chunk: [num_nodes] Global chunk ID for each node
        chunk_ptr: [num_chunks + 1] CSR pointer into chunk_nodes
        chunk_nodes: [num_nodes] Node IDs sorted by chunk assignment
        chunk_to_initial_partition: [num_chunks] Initial partition assignment
        chunk_to_owner_partition: [num_chunks] Current owner partition (may differ after rebalancing)
        chunk_load_stats: [num_chunks] Load statistics for each chunk (edge count, active nodes, etc.)
        total_nodes: Total number of nodes
        total_chunks: Total number of chunks
    """

    num_chunks_per_partition: int
    node_to_chunk: Tensor  # [num_nodes] → global chunk IDs
    chunk_ptr: Tensor  # [num_chunks + 1] → CSR pointer
    chunk_nodes: Tensor  # [num_nodes] → node IDs sorted by chunk
    chunk_to_initial_partition: Tensor  # [num_chunks] → partition IDs
    chunk_to_owner_partition: Tensor  # [num_chunks] → owner partition IDs (may change)
    chunk_load_stats: dict = field(default_factory=dict)  # {chunk_id: load_dict}
    total_nodes: int = 0
    total_chunks: int = 0

    def __post_init__(self) -> None:
        self.total_nodes = int(self.node_to_chunk.numel())
        self.total_chunks = int(self.chunk_to_initial_partition.numel())

        if int(self.chunk_ptr.numel()) != self.total_chunks + 1:
            raise ValueError(f"Mismatch: {self.total_chunks} chunks but chunk_ptr has {int(self.chunk_ptr.numel())} entries (expected {self.total_chunks + 1})")

        if int(self.chunk_nodes.numel()) != self.total_nodes:
            raise ValueError(f"Mismatch: {self.total_nodes} nodes but chunk_nodes has {int(self.chunk_nodes.numel())} entries")

        if int(self.chunk_to_owner_partition.numel()) != self.total_chunks:
            raise ValueError(f"Mismatch: {self.total_chunks} chunks but {int(self.chunk_to_owner_partition.numel())} owner entries")

    def get_node_owner(self, node_id: int) -> int:
        """Get the owning partition for a node.

        Returns: partition_id that owns chunk(node_id)
        """
        chunk_id = int(self.node_to_chunk[node_id])
        return int(self.chunk_to_owner_partition[chunk_id])

    def get_chunk_owner(self, chunk_id: int) -> int:
        """Get the owning partition for a chunk."""
        return int(self.chunk_to_owner_partition[chunk_id])

    def get_nodes_in_chunk(self, chunk_id: int) -> Tensor:
        """Get all nodes in a chunk (returns tensor slice from CSR)."""
        start = int(self.chunk_ptr[chunk_id])
        end = int(self.chunk_ptr[chunk_id + 1])
        return self.chunk_nodes[start:end]

    @property
    def chunk_to_nodes(self) -> list[list[int]]:
        """Legacy list view of chunk membership.

        New code should use ``chunk_ptr``/``chunk_nodes`` directly.  This view
        is kept for older tests and artifact consumers during the CSR migration.
        """
        return [self.get_nodes_in_chunk(i).tolist() for i in range(self.total_chunks)]

    def get_nodes_in_partition(self, partition_id: int) -> Tensor:
        """Get all nodes owned by a partition via chunk ownership (vectorised)."""
        owned_chunk_mask = self.chunk_to_owner_partition == partition_id   # [num_chunks]
        owned_chunk_ids = owned_chunk_mask.nonzero(as_tuple=True)[0]      # [k]
        # node_to_chunk gives us the chunk for each node; select nodes in owned chunks
        node_owned = torch.isin(self.node_to_chunk, owned_chunk_ids)
        return node_owned.nonzero(as_tuple=True)[0]

    def get_chunks_owned_by_partition(self, partition_id: int) -> Tensor:
        """Get all chunks owned by a partition (vectorised)."""
        mask = self.chunk_to_owner_partition == partition_id
        return mask.nonzero(as_tuple=True)[0]

    def save(self, path: Path | str) -> None:
        """Save ChunkAssignment to disk."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: Path | str) -> ChunkAssignment:
        """Load ChunkAssignment from disk."""
        loaded = torch.load(Path(path).expanduser().resolve(), weights_only=False)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded


def build_chunk_assignment(
    node_partition: Tensor,
    num_chunks_per_partition: int = 32,
) -> ChunkAssignment:
    """Build chunk assignment from initial node partition assignment (vectorised).

    Divides each partition's nodes into `num_chunks_per_partition` chunks.
    global_chunk_id = partition_id * num_chunks_per_partition + local_chunk_id

    Args:
        node_partition: [num_nodes] partition ID for each node
        num_chunks_per_partition: default 32

    Returns:
        ChunkAssignment with all mappings initialised
    """
    num_nodes = int(node_partition.numel())
    num_partitions = int(node_partition.max()) + 1
    num_chunks = num_partitions * num_chunks_per_partition

    node_to_chunk = torch.zeros(num_nodes, dtype=torch.long)

    for partition_id in range(num_partitions):
        # All node indices belonging to this partition
        part_nodes = (node_partition == partition_id).nonzero(as_tuple=True)[0]  # [P]
        n = part_nodes.numel()
        if n == 0:
            continue

        # Assign local chunk IDs uniformly: node rank within partition → chunk bucket
        # rank = 0..n-1; local_chunk_id = rank * K // n  (K = num_chunks_per_partition)
        rank = torch.arange(n, dtype=torch.long)
        local_cid = rank * num_chunks_per_partition // n              # [P]
        global_cid = partition_id * num_chunks_per_partition + local_cid  # [P]
        node_to_chunk[part_nodes] = global_cid

    # chunk_to_initial/owner partition: determined purely by chunk_id // K
    chunk_ids = torch.arange(num_chunks, dtype=torch.long)
    chunk_to_initial_partition = chunk_ids // num_chunks_per_partition
    chunk_to_owner_partition = chunk_to_initial_partition.clone()

    # Build CSR: sort nodes by chunk assignment
    sort_order = torch.argsort(node_to_chunk, stable=True)
    sorted_chunks = node_to_chunk[sort_order]

    # Counts per chunk using bincount
    chunk_counts = torch.bincount(sorted_chunks, minlength=num_chunks)  # [num_chunks]

    # Build CSR pointer
    chunk_ptr = torch.zeros(num_chunks + 1, dtype=torch.long)
    chunk_ptr[1:] = chunk_counts.cumsum(0)

    # chunk_nodes is the sorted node IDs
    chunk_nodes = sort_order.long()

    return ChunkAssignment(
        num_chunks_per_partition=num_chunks_per_partition,
        node_to_chunk=node_to_chunk,
        chunk_ptr=chunk_ptr,
        chunk_nodes=chunk_nodes,
        chunk_to_initial_partition=chunk_to_initial_partition,
        chunk_to_owner_partition=chunk_to_owner_partition,
    )


def rebalance_chunk_assignment(
    assignment: ChunkAssignment,
    chunk_load_stats: dict,
    target_partition_ids: Optional[List[int]] = None,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
) -> ChunkAssignment:
    """Rebalance chunk ownership across partitions.

    Accepts either raw dicts {chunk_id: {"edges": N, ...}} or
    ChunkLoadStats objects.  Delegates to the greedy rebalancer and
    stores updated stats back on the assignment.

    Args:
        assignment: Current chunk assignment (mutated in-place)
        chunk_load_stats: {chunk_id: ChunkLoadStats | dict}
        target_partition_ids: Ignored (all partitions used)
        max_imbalance_ratio: Acceptable imbalance threshold
        max_migrations: Hard cap on number of migrations (None = unlimited)

    Returns:
        Updated ChunkAssignment with new chunk_to_owner_partition
    """
    from .load_stats import ChunkLoadStats
    from .rebalancer import greedy_rebalance

    # Normalise to ChunkLoadStats objects
    normalised: dict = {}
    for cid, val in chunk_load_stats.items():
        if isinstance(val, ChunkLoadStats):
            normalised[cid] = val
        else:
            stat = ChunkLoadStats(
                chunk_id=cid,
                edge_count=val.get("edges", val.get("edge_count", 0)),
                active_node_count=val.get("active_nodes", val.get("active_node_count", 0)),
                remote_edge_count=val.get("remote_edges", val.get("remote_edge_count", 0)),
                remote_node_count=val.get("remote_nodes", val.get("remote_node_count", 0)),
            )
            stat.compute_total_load()
            normalised[cid] = stat

    num_partitions = int(assignment.chunk_to_initial_partition.max().item()) + 1
    updated, _ = greedy_rebalance(
        assignment=assignment,
        load_stats=normalised,
        num_partitions=num_partitions,
        max_imbalance_ratio=max_imbalance_ratio,
        max_migrations=max_migrations,
    )
    updated.chunk_load_stats = normalised
    return updated
