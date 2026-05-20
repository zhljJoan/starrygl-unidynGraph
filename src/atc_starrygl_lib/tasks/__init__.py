from .base import BaseTask, TaskSpec
from .edge_prediction import EdgePredictionTask
from .edge_label_prediction import EdgeLabelPredictionTask
from .edge_regression import EdgeRegressionTask
from .node_prediction import NodePredictionTask
from .node_regression import NodeRegressionTask

__all__ = [
    "BaseTask",
    "EdgeLabelPredictionTask",
    "EdgePredictionTask",
    "EdgeRegressionTask",
    "NodePredictionTask",
    "NodeRegressionTask",
    "TaskSpec",
    "register_builtin_tasks",
]


def register_builtin_tasks() -> None:
    from atc_starrygl_lib.core.errors import RegistryError
    from atc_starrygl_lib.core.registry import TaskRegistry

    for name, task in {
        "edge_prediction": EdgePredictionTask,
        "edge_predict": EdgePredictionTask,
        "link_prediction": EdgePredictionTask,
        "node_prediction": NodePredictionTask,
        "node_regression": NodeRegressionTask,
        "edge_regression": EdgeRegressionTask,
        "edge_label_prediction": EdgeLabelPredictionTask,
    }.items():
        try:
            TaskRegistry.register(name, task)
        except RegistryError:
            pass
