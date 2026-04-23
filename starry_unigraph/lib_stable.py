"""Stable library APIs for external integrations. Stable API v0.1.0+

This module exports the explicit stable interface of StarryUniGraph. All exports here
are backward-compatible across patch versions and safe for external code to depend on.

Stability Guarantees:
- No breaking changes to public APIs until major version bump
- Function signatures remain compatible (only additions allowed)
- Behavior is well-tested and documented

The following are considered internal/unstable:
- Backend-specific implementations (backends/ctdg/*, backends/dtdg/*)
- Task-specific adapters without registry entry
- Runtime internals (memory managers, route implementations)
- Command-line interfaces (may change without notice)

See :doc:`docs/source/architecture/` for full technical reference.
"""

from __future__ import annotations

# === Data Structures (Unified) ===
from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.sample_config import SampleConfig
from starry_unigraph.data.partition import PartitionData, RouteData, TensorData
from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.data.raw_temporal import NodeTemporalFeatureTable, RawTemporalEvents

# === Protocols (Abstractions) ===
from starry_unigraph.runtime.backend import GraphBackend, StateManager
from starry_unigraph.registry.task_adapter import TaskAdapter
from starry_unigraph.models.base import TemporalModel

# === Models & Heads ===
from starry_unigraph.models.wrapped import WrappedModel
from starry_unigraph.models.task_head import (
    EdgePredictHead,
    NodeRegressHead,
    NodeClassifyHead,
)

# === Reusable Module Components ===
from starry_unigraph.runtime.modules import (
    TimeEncode,
    GCNStack,
    MatGRUCell,
    _LSTMCell,
)

# === Training & Dispatch ===
from starry_unigraph.session import SchedulerSession
from starry_unigraph.runtime.engine import PipelineEngine

# === Task Registry ===
from starry_unigraph.registry.task_registry import TaskRegistry

__version__ = "0.1.0"
__stability__ = "stable"

__all__ = [
    # Version
    "__version__",
    "__stability__",
    # Data structures
    "BatchData",
    "SampleConfig",
    "PartitionData",
    "RouteData",
    "TensorData",
    "ChunkAtomic",
    "NodeTemporalFeatureTable",
    "RawTemporalEvents",
    # Protocols
    "GraphBackend",
    "StateManager",
    "TaskAdapter",
    "TemporalModel",
    # Models
    "WrappedModel",
    "EdgePredictHead",
    "NodeRegressHead",
    "NodeClassifyHead",
    # Reusable components
    "TimeEncode",
    "GCNStack",
    "MatGRUCell",
    "_LSTMCell",
    # Training & dispatch
    "SchedulerSession",
    "PipelineEngine",
    # Registry
    "TaskRegistry",
]


def get_stability_info() -> str:
    """Return stability information for this release.

    Returns:
        Human-readable stability status.
    """
    return f"""
StarryUniGraph Library API v{__version__}
Status: {__stability__}

Stable Exports:
- Data structures: BatchData, PartitionData, RouteData, SampleConfig, etc.
- Protocols: GraphBackend, TaskAdapter, TemporalModel, StateManager
- Training: PipelineEngine, SchedulerSession
- Models: WrappedModel, task heads (EdgePredict, NodeRegress, NodeClassify)
- Components: TimeEncode, GCNStack, RNN cells

Unstable/Internal:
- Backend implementations (backends/ctdg/*, backends/dtdg/*)
- Command-line tools and scripts
- Preprocessing internals
- Route implementations (use via protocols)

For detailed documentation, see: docs/source/architecture/

Guarantees:
- No breaking API changes in patch releases (0.1.0 → 0.1.1)
- New features may be added (source-compatible)
- Bug fixes applied transparently
- Major version bump (0.x → 1.0) for breaking changes
"""


if __name__ == "__main__":
    print(get_stability_info())
