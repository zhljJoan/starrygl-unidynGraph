"""Low-overhead API alignment objects for chunk-backed DTDG/CTDG modes.

These dataclasses deliberately align the outer scheduling and communication
contracts without forcing DTDG full-subgraph payloads and CTDG sampled-MFG
payloads into the same internal representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from torch import Tensor

GraphMode = Literal["dtdg", "ctdg"]


@dataclass(slots=True)
class PlacementView:
    """Versioned node placement tables shared by DTDG and CTDG views."""

    placement_version: int
    node_to_chunk: Tensor
    node_owner: Tensor
    node_master: Tensor
    replica_mask: Tensor
    master_dist_index: Optional[Tensor] = None
    read_dist_index: Optional[Tensor] = None


@dataclass(slots=True)
class TemporalIndexView:
    """Temporal adjacency index used by samplers, not an outer CTDG batch.

    All tensors are expected to be 1-D contiguous tensors.  The view owns no
    semantic copy of the graph: ``edge_ids`` points back to the canonical
    ``PartitionData`` edge storage.
    """

    indptr: Tensor
    indices: Tensor
    edge_ids: Tensor
    timestamps: Optional[Tensor]
    placement: PlacementView

    @property
    def num_nodes(self) -> int:
        return max(0, int(self.indptr.numel()) - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.numel())


CSCGraphView = TemporalIndexView


@dataclass(slots=True)
class SnapshotView:
    """DTDG payload view; payload remains the mode-native full subgraph/blob."""

    snapshot_id: int
    window_start: int
    window_end: int
    payload: Any
    node_ids: Optional[Tensor]
    edge_ids: Optional[Tensor]
    placement_version: int


@dataclass(slots=True)
class EventView:
    """CTDG event-batch view backed by a shared temporal sampling index."""

    batch_id: int
    time_slice_id: int
    batch_offset: int
    event_start: int
    event_end: int
    root_nodes: Tensor
    root_ts: Tensor
    temporal_index: TemporalIndexView
    placement_version: int
    event_indices: Optional[Tensor] = None

    @property
    def csc(self) -> TemporalIndexView:
        """Compatibility alias; new code should use ``temporal_index``."""
        return self.temporal_index


# Backward-compatible aliases for the temporary names used during API alignment.
DTDGInputView = SnapshotView
CTDGInputView = EventView


@dataclass(slots=True)
class CTDGSampleResult:
    """CTDG sampler output; MFGs remain backend-native for performance."""

    mfgs: list[Any]
    input_nodes: Tensor
    output_nodes: Tensor
    edge_ids: Tensor
    node_ts: Optional[Tensor] = None
    edge_ts: Optional[Tensor] = None
    memory_node_ids: Optional[Tensor] = None
    remote_node_ids: Optional[Tensor] = None
    local_node_ids: Optional[Tensor] = None
    remote_read_index: Optional[Tensor] = None
    local_read_index: Optional[Tensor] = None
    id_map_nodes: Optional[Tensor] = None


@dataclass(slots=True)
class BlockPlan:
    """Unified outer scheduling plan for DTDG snapshots and CTDG event batches."""

    mode: GraphMode
    block_id: int
    placement_version: int
    time_range: Optional[tuple[float, float]] = None
    snapshot_range: Optional[tuple[int, int]] = None
    event_range: Optional[tuple[int, int]] = None
    root_nodes: Optional[Tensor] = None
    edge_ids: Optional[Tensor] = None
    load_hint: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FetchPlan:
    """Feature and remote-memory read plan for the batch-prepare stage."""

    block_id: int
    placement_version: int
    feature_node_ids: Tensor
    feature_owners: Tensor
    remote_read_index: Optional[Tensor] = None
    local_read_index: Optional[Tensor] = None
    remote_node_ids: Optional[Tensor] = None
    local_node_ids: Optional[Tensor] = None
    memory_node_ids: Optional[Tensor] = None
    memory_owners: Optional[Tensor] = None
    memory_read_index: Optional[Tensor] = None
    cache_policy: str = "none"


@dataclass(slots=True)
class PropagationPlan:
    """Autograd-capable model-layer propagation route bundle."""

    block_id: int
    placement_version: int
    layer_routes: list[Any]
    autograd_enabled: bool = True


@dataclass(slots=True)
class StateSyncPlan:
    """Memory/state/cache update plan for the batch-commit stage."""

    block_id: int
    placement_version: int
    update_node_ids: Tensor
    update_owners: Tensor
    update_index: Optional[Tensor] = None
    replica_node_ids: Optional[Tensor] = None
    replica_owners: Optional[Tensor] = None
    replica_index: Optional[Tensor] = None
    sync_policy: str = "owner_write"
    change_threshold: float = 0.0
    change_metric: str = "cos"


@dataclass(slots=True)
class CommPlanBundle:
    """Aligned communication plan wrapper used by both graph modes."""

    fetch: Optional[FetchPlan] = None
    propagation: Optional[PropagationPlan] = None
    state_sync: Optional[StateSyncPlan] = None


PlanBundle = CommPlanBundle


@dataclass(slots=True)
class ExecutionUnit:
    """Mode-aligned wrapper around native DTDG/CTDG training payloads."""

    mode: GraphMode
    block_id: int
    placement_version: int
    payload: Any
    comm_plan: CommPlanBundle = field(default_factory=CommPlanBundle)
    profile_hint: dict[str, float] = field(default_factory=dict)


GraphBatchEnvelope = ExecutionUnit


@dataclass(slots=True)
class ProfileRecord:
    """Unified profiling record consumed by chunk placement replanning."""

    mode: GraphMode
    block_id: int
    placement_version: int
    sample_ms: float = 0.0
    fetch_ms: float = 0.0
    propagation_ms: float = 0.0
    state_sync_ms: float = 0.0
    compute_ms: float = 0.0
    num_nodes: int = 0
    num_edges: int = 0
    num_remote_nodes: int = 0
    num_memory_nodes: int = 0
    peak_mem_bytes: int = 0


@dataclass(slots=True)
class ChunkPlacement:
    """Concrete placement tables used for a full hyper-parameter trial."""

    placement_version: int
    node_to_chunk: Tensor
    node_owner: Tensor
    node_master: Tensor
    replica_mask: Tensor
    master_dist_index: Optional[Tensor] = None
    read_dist_index: Optional[Tensor] = None

    def view(self) -> PlacementView:
        return PlacementView(
            placement_version=self.placement_version,
            node_to_chunk=self.node_to_chunk,
            node_owner=self.node_owner,
            node_master=self.node_master,
            replica_mask=self.replica_mask,
            master_dist_index=self.master_dist_index,
            read_dist_index=self.read_dist_index,
        )


@dataclass(slots=True)
class PlacementDelta:
    """Partial migration descriptor applied only between tuning trials."""

    old_version: int
    new_version: int
    moved_chunks: Tensor
    affected_nodes: Tensor
    affected_blocks: Tensor
    affected_routes: Tensor
