from __future__ import annotations

from torch import Tensor

from atc_starrygl_lib.core.types import Batch, ClassifyOutput
from .base import BaseTask, TaskSpec


class NodePredictionTask(BaseTask):
    spec = TaskSpec(name="node_prediction", task_type="node_classification")

    def compute_loss(self, output: ClassifyOutput, batch: Batch) -> Tensor:
        logits = output.logits
        if batch.labels is None:
            raise ValueError("node_prediction requires batch.labels")
        if logits.ndim == 1 or logits.size(-1) == 1:
            return self.binary_loss(logits, batch.labels)
        return self.multiclass_loss(logits, batch.labels)

    def compute_metrics(self, output: ClassifyOutput, batch: Batch) -> dict[str, float]:
        logits = output.logits
        if batch.labels is None:
            return {"accuracy": 0.0}
        if logits.ndim == 1 or logits.size(-1) == 1:
            return self.binary_metrics(logits, batch.labels)
        return self.classification_metrics(logits, batch.labels)
