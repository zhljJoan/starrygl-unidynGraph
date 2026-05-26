from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from atc_starrygl_lib.core.types import Batch, EdgePredOutput
from .base import BaseTask, TaskSpec


class EdgePredictionTask(BaseTask):
    spec = TaskSpec(name="edge_prediction", task_type="link_prediction")

    def compute_loss(self, output: EdgePredOutput, batch: Batch) -> Tensor:
        pos_score = output.pos_score.flatten()
        neg_score = output.neg_score.flatten()
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score.float(),
            torch.ones_like(pos_score, dtype=torch.float32),
        )
        if batch.neg_weight is None:
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score.float(),
                torch.zeros_like(neg_score, dtype=torch.float32),
            )
            return pos_loss + neg_loss
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score.float(),
            torch.zeros_like(neg_score, dtype=torch.float32),
            weight=batch.neg_weight.to(device=neg_score.device, dtype=torch.float32).flatten(),
        )
        return pos_loss + neg_loss

    def compute_metrics(self, output: EdgePredOutput, batch: Batch) -> dict[str, float]:
        return self.link_prediction_metrics(output.pos_score, output.neg_score)
