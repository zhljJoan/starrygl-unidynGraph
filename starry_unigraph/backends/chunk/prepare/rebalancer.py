"""Chunk rebalancer: Load-balanced chunk-to-partition assignment.

Given a ChunkAssignment and per-chunk load statistics, this module
decides which chunks to migrate to different owner partitions and
produces the final node_owner tensor plus manifests.

Algorithm: GreedyRebalancer
  1. Sort chunks by total_load descending.
  2. Maintain per-partition running load sums.
  3. For each chunk (heaviest first), assign it to the partition with
     the current minimum load, unless it is already there and the
     imbalance ratio is within tolerance.
  4. Only migrate a chunk if doing so reduces the imbalance; this
     produces small, targeted moves rather than a full reshuffle.

Outputs:
  - Updated chunk_to_owner_partition in ChunkAssignment
  - node_owner: [num_nodes] final partition for every node
  - ChunkReassignmentManifest: records every migration for logging/debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .chunk_assignment import ChunkAssignment
from .load_stats import ChunkLoadStats


# ---------------------------------------------------------------------------
# Manifest data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkMigration:
    """A single chunk migration record.

    Attributes:
        chunk_id: Global chunk ID that was moved
        from_partition: Previous owner partition
        to_partition: New owner partition
        load: Total load of this chunk (used to decide migration)
    """
    chunk_id: int
    from_partition: int
    to_partition: int
    load: float


@dataclass
class ChunkReassignmentManifest:
    """Complete record of a rebalancing run.

    Attributes:
        migrations: List of all chunk migrations performed
        partition_loads_before: Per-partition total load before rebalancing
        partition_loads_after: Per-partition total load after rebalancing
        imbalance_before: Max/min load ratio before
        imbalance_after: Max/min load ratio after
        num_partitions: Number of partitions
        num_chunks: Total chunks
    """

    migrations: List[ChunkMigration] = field(default_factory=list)
    partition_loads_before: Dict[int, float] = field(default_factory=dict)
    partition_loads_after: Dict[int, float] = field(default_factory=dict)
    imbalance_before: float = 1.0
    imbalance_after: float = 1.0
    num_partitions: int = 0
    num_chunks: int = 0

    def save(self, path: Path | str) -> None:
        import json
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "migrations": [
                {"chunk_id": m.chunk_id, "from": m.from_partition,
                 "to": m.to_partition, "load": m.load}
                for m in self.migrations
            ],
            "partition_loads_before": self.partition_loads_before,
            "partition_loads_after": self.partition_loads_after,
            "imbalance_before": self.imbalance_before,
            "imbalance_after": self.imbalance_after,
            "num_partitions": self.num_partitions,
            "num_chunks": self.num_chunks,
        }
        with open(path, "w") as f:
            import json
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Greedy rebalancer
# ---------------------------------------------------------------------------

def _imbalance_ratio(partition_loads: Dict[int, float]) -> float:
    """Max-load / min-load ratio across all partitions."""
    loads = [v for v in partition_loads.values() if v > 0]
    if len(loads) < 2:
        return 1.0
    return max(loads) / max(min(loads), 1e-9)


def greedy_rebalance(
    assignment: ChunkAssignment,
    load_stats: Dict[int, ChunkLoadStats],
    num_partitions: int,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
) -> Tuple[ChunkAssignment, ChunkReassignmentManifest]:
    """Greedy load-balanced chunk reassignment.

    Iterates chunks from heaviest to lightest. For each chunk, if moving
    it to the least-loaded partition would improve the imbalance ratio
    beyond the tolerance threshold, the migration is recorded and the
    chunk_to_owner_partition is updated.

    Chunk node membership never changes — only owner changes.

    Args:
        assignment: Current ChunkAssignment (mutated in-place)
        load_stats: {chunk_id: ChunkLoadStats} from compute_chunk_load_stats*
        num_partitions: Number of partitions
        max_imbalance_ratio: Stop migrating when imbalance drops below this
        max_migrations: Hard cap on number of migrations (None = unlimited)

    Returns:
        (updated assignment, manifest)
    """
    manifest = ChunkReassignmentManifest(
        num_partitions=num_partitions,
        num_chunks=assignment.total_chunks,
    )

    # Current per-partition loads (sum of owned chunk loads)
    partition_load: Dict[int, float] = {p: 0.0 for p in range(num_partitions)}
    for cid in range(assignment.total_chunks):
        owner = int(assignment.chunk_to_owner_partition[cid])
        partition_load[owner] += load_stats[cid].total_load if cid in load_stats else 0.0

    manifest.partition_loads_before = {p: partition_load[p] for p in range(num_partitions)}
    manifest.imbalance_before = _imbalance_ratio(partition_load)

    # Sort chunks by load descending (heaviest first)
    sorted_chunks = sorted(
        range(assignment.total_chunks),
        key=lambda cid: load_stats[cid].total_load if cid in load_stats else 0.0,
        reverse=True,
    )

    migration_count = 0
    for cid in sorted_chunks:
        if max_migrations is not None and migration_count >= max_migrations:
            break

        if _imbalance_ratio(partition_load) <= max_imbalance_ratio:
            break

        current_owner = int(assignment.chunk_to_owner_partition[cid])
        chunk_load = load_stats[cid].total_load if cid in load_stats else 0.0

        if chunk_load == 0.0:
            continue

        # Find the partition with minimum load (excluding current owner if it's heaviest)
        min_partition = min(partition_load, key=partition_load.__getitem__)

        if min_partition == current_owner:
            continue

        # Only migrate if it genuinely reduces imbalance
        new_load_from = partition_load[current_owner] - chunk_load
        new_load_to = partition_load[min_partition] + chunk_load

        new_partition_load = dict(partition_load)
        new_partition_load[current_owner] = new_load_from
        new_partition_load[min_partition] = new_load_to

        if _imbalance_ratio(new_partition_load) < _imbalance_ratio(partition_load):
            # Apply migration
            assignment.chunk_to_owner_partition[cid] = min_partition
            partition_load = new_partition_load
            manifest.migrations.append(ChunkMigration(
                chunk_id=cid,
                from_partition=current_owner,
                to_partition=min_partition,
                load=chunk_load,
            ))
            migration_count += 1

    manifest.partition_loads_after = {p: partition_load[p] for p in range(num_partitions)}
    manifest.imbalance_after = _imbalance_ratio(partition_load)

    return assignment, manifest


# ---------------------------------------------------------------------------
# Final node_owner derivation
# ---------------------------------------------------------------------------

def derive_node_owner(assignment: ChunkAssignment) -> Tensor:
    """Derive final node-to-partition mapping from chunk ownership.

    node_owner[node] = chunk_to_owner_partition[node_to_chunk[node]]

    Args:
        assignment: ChunkAssignment with (possibly rebalanced) chunk_to_owner_partition

    Returns:
        [num_nodes] LongTensor of owner partition IDs
    """
    return assignment.chunk_to_owner_partition[assignment.node_to_chunk]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def rebalance_chunks(
    assignment: ChunkAssignment,
    load_stats: Dict[int, ChunkLoadStats],
    num_partitions: int,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
) -> Tuple[ChunkAssignment, Tensor, ChunkReassignmentManifest]:
    """Full rebalancing pipeline: reassign chunks and derive node_owner.

    This is the main entry point for Phase 4. Call after computing
    load_stats (via compute_chunk_load_stats or compute_chunk_load_stats_from_windows).

    Args:
        assignment: ChunkAssignment from build_chunk_assignment
        load_stats: Per-chunk load metrics
        num_partitions: Total number of partitions
        max_imbalance_ratio: Acceptable imbalance threshold (1.0 = perfect)
        max_migrations: Hard cap on number of chunk migrations

    Returns:
        (updated_assignment, node_owner, manifest)
        - updated_assignment: ChunkAssignment with new chunk_to_owner_partition
        - node_owner: [num_nodes] LongTensor, final partition per node
        - manifest: ChunkReassignmentManifest with migration log
    """
    updated, manifest = greedy_rebalance(
        assignment=assignment,
        load_stats=load_stats,
        num_partitions=num_partitions,
        max_imbalance_ratio=max_imbalance_ratio,
        max_migrations=max_migrations,
    )
    node_owner = derive_node_owner(updated)
    return updated, node_owner, manifest
