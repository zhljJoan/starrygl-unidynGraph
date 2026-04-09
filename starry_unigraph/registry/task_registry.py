from __future__ import annotations

from starry_unigraph.tasks import NodeRegressionTaskAdapter, TemporalLinkPredictionTaskAdapter


class TaskRegistry:
    _tasks: dict[str, type] = {
        "temporal_link_prediction": TemporalLinkPredictionTaskAdapter,
        "snapshot_node_regression": NodeRegressionTaskAdapter,
    }

    @classmethod
    def resolve(cls, task_type: str) -> type:
        if task_type not in cls._tasks:
            raise KeyError(f"Unknown task type: {task_type!r}")
        return cls._tasks[task_type]
