from __future__ import annotations

from torch import Tensor

from atc_starrygl_lib.core.types import Batch, RegressionOutput
from .base import BaseTask, TaskSpec


class NodeRegressionTask(BaseTask):
    spec = TaskSpec(name="node_regression", task_type="node_regression")

    def compute_loss(self, output: RegressionOutput, batch: Batch) -> Tensor:
        if batch.labels is None:
            raise ValueError("node_regression requires batch.labels")
        return self.regression_loss(output.pred, batch.labels)

    def compute_metrics(self, output: RegressionOutput, batch: Batch) -> dict[str, float]:
        if batch.labels is None:
            return {"mse": 0.0, "mae": 0.0}
        return self.regression_metrics(output.pred, batch.labels)
