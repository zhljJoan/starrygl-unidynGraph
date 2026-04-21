"""Node Classification Task Adapter."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.sample_config import SampleConfig
from .base import BaseTaskAdapter


class NodeClassifyAdapter(BaseTaskAdapter):
    """Adapter for node classification task.

    Predicts discrete class labels for nodes.
    Uses cross-entropy loss and accuracy metrics.
    """
    task_type = "node_classification"

    def build_sample_config(
        self,
        chunk: Any,  # ChunkAtomic
        model: Any,  # nn.Module
        split: str,
    ) -> SampleConfig:
        """For node classification, sample neighbors of labeled nodes.

        Args:
            chunk: ChunkAtomic with node_set and optional labeled node info
            model: Model instance
            split: Data split

        Returns:
            SampleConfig with target_nodes/target_labels
        """
        target_nodes = chunk.node_set
        target_labels = None  # Would come from chunk in real scenario

        return SampleConfig(
            target_nodes=target_nodes,
            target_labels=target_labels,
            neg_strategy="none",
            num_neighbors=[20, 10],
            num_layers=2,
        )

    def compute_loss(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Tensor:
        """Compute cross-entropy loss for node classification.

        Args:
            model_output: {'logits': [N, num_classes]}
            batch: BatchData with labels (class indices)

        Returns:
            Scalar cross-entropy loss
        """
        logits = model_output.get("logits")
        labels = batch.labels

        if logits is None or labels is None:
            return torch.tensor(0.0)

        return F.cross_entropy(logits, labels.long())

    def compute_metrics(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, float]:
        """Compute accuracy and other metrics for node classification.

        Args:
            model_output: {'logits': [N, num_classes]}
            batch: BatchData with labels

        Returns:
            {'accuracy': float}
        """
        logits = model_output.get("logits")
        labels = batch.labels

        if logits is None or labels is None:
            return {}

        pred = logits.argmax(dim=1)
        accuracy = (pred == labels.long()).float().mean().item()

        return {"accuracy": accuracy}

    def format_output(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, Any]:
        """Format node classification predictions."""
        logits = model_output.get("logits")
        pred = logits.argmax(dim=1) if logits is not None else None

        return {
            "logits": logits.detach().cpu() if logits is not None else None,
            "pred": pred.detach().cpu() if pred is not None else None,
            "node_ids": batch.target_nodes.detach().cpu() if batch.target_nodes is not None else None,
        }
