from .base import BaseTaskAdapter
from .node_regression import NodeRegressionTaskAdapter
from .temporal_link_prediction import TemporalLinkPredictionTaskAdapter, EdgePredictAdapter
from .node_classify import NodeClassifyAdapter

__all__ = [
    "BaseTaskAdapter",
    "NodeRegressionTaskAdapter",
    "TemporalLinkPredictionTaskAdapter",
    "EdgePredictAdapter",
    "NodeClassifyAdapter",
]

