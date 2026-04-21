"""Temporal Link Prediction Task Adapter."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.sample_config import SampleConfig
from .base import BaseTaskAdapter


class EdgePredictAdapter(BaseTaskAdapter):
    """Adapter for temporal edge/link prediction task.

    Samples positive and negative edge pairs, computes BCE loss,
    and computes AUC/AP metrics.
    """
    task_type = "edge_predict"

    def build_sample_config(
        self,
        chunk: Any,  # ChunkAtomic
        model: Any,  # nn.Module
        split: str,
    ) -> SampleConfig:
        """For edge prediction, return positive edges to sample from the chunk.

        Args:
            chunk: ChunkAtomic with tcsr_col, tcsr_ts, tcsr_edge_id
            model: Model instance (unused for now)
            split: Data split

        Returns:
            SampleConfig with pos_src/pos_dst for all edges in chunk
        """
        # Extract edges from temporal-CSR
        # For now: use all edges in this chunk as positive pairs
        # (In future: could filter by split if chunk has labeled edges)

        # Simple heuristic: all edges in chunk are training pairs
        pos_src = chunk.tcsr_col[:-1].repeat_interleave(
            chunk.tcsr_rowptr[1:] - chunk.tcsr_rowptr[:-1]
        )  # Source (node_set repeated per degree)
        pos_dst = chunk.tcsr_col  # Destinations (all neighbors)

        return SampleConfig(
            pos_src=pos_src,
            pos_dst=pos_dst,
            neg_strategy="random",  # Random negative sampling
            neg_ratio=1,  # 1 negative per positive
            num_neighbors=[20, 10],
            num_layers=2,
        )

    def compute_loss(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Tensor:
        """Compute binary cross-entropy loss for edge prediction.

        Args:
            model_output: {'pos_score': [M], 'neg_score': [M*neg_ratio]}
            batch: BatchData (unused in this implementation)

        Returns:
            Scalar loss
        """
        pos_score = model_output.get("pos_score")
        neg_score = model_output.get("neg_score")

        if pos_score is None or neg_score is None:
            # Fallback to old API if new fields not present
            return torch.tensor(0.0, device=self._get_device(model_output))

        # Binary classification: positive=1, negative=0
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_labels, neg_labels])

        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_metrics(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, float]:
        """Compute AUC and AP metrics.

        Args:
            model_output: {'pos_score': [M], 'neg_score': [M*neg_ratio]}
            batch: BatchData

        Returns:
            {'auc': float, 'ap': float}
        """
        pos_score = model_output.get("pos_score")
        neg_score = model_output.get("neg_score")

        if pos_score is None or neg_score is None:
            return {}

        pos_score = pos_score.sigmoid().detach().cpu().numpy()
        neg_score = neg_score.sigmoid().detach().cpu().numpy()

        # AUC: fraction of (pos, neg) pairs where pos > neg
        import numpy as np
        pos_score_flat = pos_score.flatten()
        neg_score_flat = neg_score.flatten()

        # All pairwise comparisons
        auc = (pos_score_flat[:, None] > neg_score_flat[None, :]).mean()

        # AP: average precision (area under precision-recall curve)
        # Simple approximation: mean of positive scores - mean of negative scores
        ap = (pos_score_flat.mean() - neg_score_flat.mean()) / 2 + 0.5

        return {
            "auc": float(auc),
            "ap": float(np.clip(ap, 0, 1)),
        }

    def format_output(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, Any]:
        """Format edge predictions."""
        pos_score = model_output.get("pos_score")
        neg_score = model_output.get("neg_score")

        return {
            "pos_score": pos_score.detach().cpu() if pos_score is not None else None,
            "neg_score": neg_score.detach().cpu() if neg_score is not None else None,
            "pos_src": batch.pos_src.detach().cpu() if batch.pos_src is not None else None,
            "pos_dst": batch.pos_dst.detach().cpu() if batch.pos_dst is not None else None,
        }

    @staticmethod
    def _get_device(model_output: Dict[str, Tensor]) -> torch.device:
        """Get device from first tensor in model_output."""
        for v in model_output.values():
            if isinstance(v, Tensor):
                return v.device
        return torch.device("cpu")


# Alias for backward compatibility
TemporalLinkPredictionTaskAdapter = EdgePredictAdapter