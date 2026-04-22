"""Node Regression Task Adapter."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.sample_config import SampleConfig
from .base import BaseTaskAdapter


class NodeRegressionTaskAdapter(BaseTaskAdapter):
    """Adapter for node regression task.

    Predicts continuous values (e.g., node features) for labeled nodes.
    Uses MSE loss and MAE/RMSE metrics.
    """
    task_type = "node_regression"

    def build_sample_config(
        self,
        chunk: Any,  # ChunkAtomic
        model: Any,  # nn.Module
        split: str,
    ) -> SampleConfig:
        """For node regression, sample neighbors of labeled nodes.

        Args:
            chunk: ChunkAtomic with node_set and optional labeled node info
            model: Model instance
            split: Data split

        Returns:
            SampleConfig with target_nodes/target_labels
        """
        # Placeholder: in practice, chunk would have labeled_node_ids and labels
        # For now, assume all nodes in chunk have labels
        target_nodes = chunk.node_set
        target_labels = None  # Would come from chunk in real scenario

        return SampleConfig(
            target_nodes=target_nodes,
            target_labels=target_labels,
            neg_strategy="none",  # No negative sampling for regression
            num_neighbors=[20, 10],
            num_layers=2,
        )

    def compute_loss(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Tensor:
        """Compute MSE loss for node regression.

        Args:
            model_output: {'node_pred': [N, feat_dim]}
            batch: BatchData with labels

        Returns:
            Scalar MSE loss
        """
        pred = model_output.get("node_pred")
        labels = batch.labels

        if pred is None or labels is None:
            return torch.tensor(0.0)

        return F.mse_loss(pred, labels)

    def compute_metrics(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, float]:
        """Compute MAE and RMSE metrics for node regression.

        Args:
            model_output: {'node_pred': [N, feat_dim]}
            batch: BatchData with labels

        Returns:
            {'mae': float, 'rmse': float}
        """
        pred = model_output.get("node_pred")
        labels = batch.labels

        if pred is None or labels is None:
            return {}

        mae = (pred - labels).abs().mean().item()
        rmse = ((pred - labels) ** 2).mean().sqrt().item()

        return {
            "mae": mae,
            "rmse": rmse,
        }

    def format_output(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, Any]:
        """Format node predictions."""
        return {
            "node_pred": model_output.get("node_pred", torch.tensor([])).detach().cpu(),
            "node_ids": batch.target_nodes.detach().cpu() if batch.target_nodes is not None else None,
        }