from __future__ import annotations

from .base import BaseTaskAdapter
import torch

class TemporalLinkPredictionTaskAdapter(BaseTaskAdapter):
    task_type = "temporal_link_prediction"
    
    def compute_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return f

    def format_prediction(self, output: dict[str, Any]) -> dict[str, Any]:
        return output