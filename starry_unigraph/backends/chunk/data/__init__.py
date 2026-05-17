"""Chunk data layer: Core data structures and interfaces.

Exports:
- PartitionData, TensorData, RouteData: Partition graph containers
- BatchData: Unified batch container
- FeatureStore, FeatureStoreConfig: CPU feature management
- MemoryStore: State management interface
- SpatialRouteData: All-to-all for node feature exchange
- MemoryRouteData: All-to-all for memory/state cache updates
- CPUMemoryLayout: Hot/cold node layout for CPU store
- CommPipeline, SpatialResult, MemoryResult: Async communication
"""

from .partition import PartitionData, RouteData, TensorData
from .batch import BatchData
from .feature_store import FeatureStore, FeatureStoreConfig
from .memory_store import MemoryStore, DummyMemoryStore
from .route import SpatialRouteData, MemoryRouteData, CPUMemoryLayout
from .comm import CommPipeline, SpatialResult, MemoryResult
from .graph_store import ChunkGraphStore
from .plans import (
    BlockPlan,
    CSCGraphView,
    EventView,
    ExecutionUnit,
    CTDGInputView,
    CTDGSampleResult,
    ChunkPlacement,
    CommPlanBundle,
    SnapshotView,
    DTDGInputView,
    FetchPlan,
    GraphBatchEnvelope,
    PlacementDelta,
    PlacementView,
    PlanBundle,
    ProfileRecord,
    PropagationPlan,
    StateSyncPlan,
    TemporalIndexView,
)
from .propagation_route import ChunkPropagationRoute

__all__ = [
    "PartitionData",
    "RouteData",
    "TensorData",
    "BatchData",
    "FeatureStore",
    "FeatureStoreConfig",
    "MemoryStore",
    "DummyMemoryStore",
    "SpatialRouteData",
    "MemoryRouteData",
    "CPUMemoryLayout",
    "CommPipeline",
    "SpatialResult",
    "MemoryResult",
    "ChunkGraphStore",
    "BlockPlan",
    "CSCGraphView",
    "EventView",
    "ExecutionUnit",
    "CTDGInputView",
    "CTDGSampleResult",
    "ChunkPlacement",
    "CommPlanBundle",
    "SnapshotView",
    "DTDGInputView",
    "FetchPlan",
    "GraphBatchEnvelope",
    "PlacementDelta",
    "PlacementView",
    "PlanBundle",
    "ProfileRecord",
    "PropagationPlan",
    "StateSyncPlan",
    "TemporalIndexView",
    "ChunkPropagationRoute",
]
