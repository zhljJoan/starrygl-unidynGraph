from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from atc_starrygl_lib.core.types import Batch


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str


class BaseTask:
    spec: TaskSpec

    @staticmethod
    def binary_loss(logits: Tensor, labels: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.float().flatten(), labels.float().flatten()
        )

    @staticmethod
    def multiclass_loss(logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits.float(), labels.long().flatten())

    @staticmethod
    def regression_loss(pred: Tensor, labels: Tensor) -> Tensor:
        return F.mse_loss(pred.float().reshape_as(labels.float()), labels.float())

    @staticmethod
    def binary_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
        if logits.numel() == 0:
            return {"accuracy": 0.0}
        probs = torch.sigmoid(logits.detach().float().flatten())
        target = labels.detach().float().flatten()
        pred = probs >= 0.5
        truth = target >= 0.5
        return {"accuracy": float((pred == truth).float().mean().item())}

    @staticmethod
    def classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
        if logits.numel() == 0:
            return {"accuracy": 0.0}
        pred = logits.detach().argmax(dim=-1)
        target = labels.detach().long().flatten()
        return {"accuracy": float((pred == target).float().mean().item())}

    @staticmethod
    def regression_metrics(pred: Tensor, labels: Tensor) -> dict[str, float]:
        if pred.numel() == 0:
            return {"mse": 0.0, "mae": 0.0}
        err = pred.detach().float().reshape_as(labels.float()) - labels.detach().float()
        return {
            "mse": float((err * err).mean().item()),
            "mae": float(err.abs().mean().item()),
        }

    @staticmethod
    def link_prediction_metrics(pos_score: Tensor, neg_score: Tensor) -> dict[str, float]:
        if pos_score.numel() == 0:
            return {"auc": 0.0, "ap": 0.0}
        pos = torch.sigmoid(pos_score.detach().float().flatten())
        neg = torch.sigmoid(neg_score.detach().float().flatten())
        try:
            import numpy as np
            from sklearn.metrics import average_precision_score, roc_auc_score
            scores = np.concatenate([pos.cpu().numpy(), neg.cpu().numpy()])
            labels = np.concatenate([
                np.ones(pos.numel(), dtype=np.float32),
                np.zeros(neg.numel(), dtype=np.float32),
            ])
            return {
                "auc": float(roc_auc_score(labels, scores)),
                "ap": float(average_precision_score(labels, scores)),
            }
        except ImportError:
            auc = float((pos[:, None] > neg[None, :]).float().mean().item())
            return {"auc": auc, "ap": auc}
