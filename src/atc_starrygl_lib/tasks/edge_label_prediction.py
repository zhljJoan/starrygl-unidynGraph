from __future__ import annotations

from torch import Tensor

from atc_starrygl_lib.core.types import Batch, ClassifyOutput
from .base import BaseTask, TaskSpec


class EdgeLabelPredictionTask(BaseTask):
    spec = TaskSpec(name="edge_label_prediction", task_type="edge_classification")

    def compute_loss(self, output: ClassifyOutput, batch: Batch) -> Tensor:
        logits = output.logits
        labels = batch.labels
        if labels is None:
            raise ValueError("edge_label_prediction requires batch.labels")
        if logits.ndim == 1 or logits.size(-1) == 1:
            return self.binary_loss(logits, labels)
        return self.multiclass_loss(logits, labels)

    def compute_metrics(self, output: ClassifyOutput, batch: Batch) -> dict[str, float]:
        logits = output.logits
        labels = batch.labels
        if labels is None:
            return {"accuracy": 0.0}
        if logits.ndim == 1 or logits.size(-1) == 1:
            return self.binary_metrics(logits, labels)
        return self.classification_metrics(logits, labels)
