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
]
