"""Model components: temporal GNN backbones, task heads, wrapped models."""
from .base import TemporalModel
from .id_map import compact_rows
from .task_head import EdgePredictHead, EdgeRegressHead, NodeRegressHead, NodeClassifyHead
from .wrapped import WrappedModel
from .layers import ChunkPropagationRoute, PropagationRoute

# Re-export reusable components for backward compatibility
from starry_unigraph.runtime.modules import TimeEncode, GCNStack

__all__ = [
    "TemporalModel",
    "compact_rows",
    "EdgePredictHead",
    "EdgeRegressHead",
    "NodeRegressHead",
    "NodeClassifyHead",
    "WrappedModel",
    "ChunkPropagationRoute",
    "PropagationRoute",
    # Reusable components (from runtime.modules)
    "TimeEncode",
    "GCNStack",
]
