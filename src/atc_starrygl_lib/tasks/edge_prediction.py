from __future__ import annotations

import torch
from torch import Tensor

from atc_starrygl_lib.core.types import Batch, EdgePredOutput
from .base import BaseTask, TaskSpec


class EdgePredictionTask(BaseTask):
    spec = TaskSpec(name="edge_prediction", task_type="link_prediction")

    def compute_loss(self, output: EdgePredOutput, batch: Batch) -> Tensor:
        logits = torch.cat([output.pos_score.flatten(), output.neg_score.flatten()])
        labels = torch.cat([
            torch.ones_like(output.pos_score.flatten()),
            torch.zeros_like(output.neg_score.flatten()),
        ])
        return self.binary_loss(logits, labels)

    def compute_metrics(self, output: EdgePredOutput, batch: Batch) -> dict[str, float]:
        return self.link_prediction_metrics(output.pos_score, output.neg_score)
