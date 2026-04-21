from __future__ import annotations

from starry_unigraph.tasks import (
    NodeRegressionTaskAdapter,
    TemporalLinkPredictionTaskAdapter,
    EdgePredictAdapter,
    NodeClassifyAdapter,
)


class TaskRegistry:
    _tasks: dict[str, type] = {
        # Edge prediction task (temporal link prediction)
        "temporal_link_prediction": TemporalLinkPredictionTaskAdapter,
        "edge_predict": EdgePredictAdapter,

        # Node regression task
        "snapshot_node_regression": NodeRegressionTaskAdapter,
        "node_regression": NodeRegressionTaskAdapter,

        # Node classification task
        "node_classification": NodeClassifyAdapter,
    }

    @classmethod
    def resolve(cls, task_type: str) -> type:
        if task_type not in cls._tasks:
            raise KeyError(f"Unknown task type: {task_type!r}")
        return cls._tasks[task_type]

    @classmethod
    def list_tasks(cls) -> list[str]:
        """Return all registered task types."""
        return list(cls._tasks.keys())

