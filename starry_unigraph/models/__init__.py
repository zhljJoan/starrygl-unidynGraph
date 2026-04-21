"""Model components: temporal GNN backbones, task heads, wrapped models."""
from .base import TemporalModel
from .task_head import EdgePredictHead, NodeRegressHead, NodeClassifyHead
from .wrapped import WrappedModel

# Re-export reusable components for backward compatibility
from starry_unigraph.runtime.modules import TimeEncode, GCNStack

__all__ = [
    "TemporalModel",
    "EdgePredictHead",
    "NodeRegressHead",
    "NodeClassifyHead",
    "WrappedModel",
    # Reusable components (from runtime.modules)
    "TimeEncode",
    "GCNStack",
]
