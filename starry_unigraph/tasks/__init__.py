from .base import BaseTaskAdapter
from .node_regression import NodeRegressionTaskAdapter
from .temporal_link_prediction import TemporalLinkPredictionTaskAdapter

__all__ = [
    "BaseTaskAdapter",
    "NodeRegressionTaskAdapter",
    "TemporalLinkPredictionTaskAdapter",
]
