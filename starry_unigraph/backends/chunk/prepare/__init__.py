"""Chunk preprocessing: Partitioning, chunk assignment, and load balancing.

Modules:
- chunk_assignment: Core chunk mapping and rebalancing logic
- load_stats: Per-chunk load statistics computation
- rebalancer: Greedy load-balanced chunk-to-partition reassignment
- route_builder: DTDGSnapshotRoute and CTDGSliceRoute construction
- adaptive: Legacy experimental adaptive chunk generator (not used by pipeline.prepare)
- generate_chunk: Legacy chunk generation (deprecated; not used by pipeline.prepare)
- time_split: Time window splitting for snapshots
"""

from .chunk_assignment import ChunkAssignment, build_chunk_assignment, rebalance_chunk_assignment
from .load_stats import (
    ChunkLoadStats,
    compute_chunk_load_by_slice,
    compute_chunk_load_stats,
    compute_chunk_load_stats_from_windows,
)
from .rebalancer import (
    ChunkMigration,
    ChunkReassignmentManifest,
    greedy_rebalance,
    greedy_rebalance_by_slice,
    derive_node_owner,
    rebalance_chunks,
)
from .route_builder import (
    build_cpu_memory_layout,
    build_memory_route_phase1,
    assign_memory_route_ptrs,
    build_spatial_routes,
)
from .propagation_builder import build_propagation_routes
from .pipeline import PrepareArtifacts, build_node_partition, prepare

__all__ = [
    "ChunkAssignment",
    "build_chunk_assignment",
    "rebalance_chunk_assignment",
    "ChunkLoadStats",
    "compute_chunk_load_by_slice",
    "compute_chunk_load_stats",
    "compute_chunk_load_stats_from_windows",
    "ChunkMigration",
    "ChunkReassignmentManifest",
    "greedy_rebalance",
    "greedy_rebalance_by_slice",
    "derive_node_owner",
    "rebalance_chunks",
    "build_cpu_memory_layout",
    "build_memory_route_phase1",
    "assign_memory_route_ptrs",
    "build_spatial_routes",
    "build_propagation_routes",
    "PrepareArtifacts",
    "build_node_partition",
    "prepare",
]
