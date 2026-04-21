"""WrappedModel: Composes backbone (temporal GNN) with task-specific head."""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from torch import Tensor

from starry_unigraph.data.batch_data import BatchData


class WrappedModel(nn.Module):
    """Combines a temporal GNN backbone with a task-specific output head.

    This allows flexible composition:
    - Backbone: TGN, DyRep, MPNN-LSTM, etc. (task-agnostic)
    - Head: EdgePredictHead, NodeRegressHead, NodeClassifyHead (task-specific)
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        """Initialize wrapped model.

        Args:
            backbone: Temporal GNN module (TemporalModel protocol)
            head: Task-specific output head (EdgePredictHead, NodeRegressHead, etc.)
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def predict(
        self,
        state: Dict[str, Any],
        batch: BatchData,
    ) -> Dict[str, Tensor]:
        """Full forward pass: backbone → head.

        Args:
            state: Dictionary of RNN states from StateManager
            batch: BatchData with graph structure and labels

        Returns:
            Dictionary of task-specific predictions:
            - EdgePredict: {'pos_score': [...], 'neg_score': [...]}
            - NodeRegress: {'node_pred': [...]}
            - NodeClassify: {'logits': [...]}
        """
        # Backbone forward: compute node embeddings
        embeddings = self.backbone(batch.mfg, state)

        # Head forward: task-specific predictions
        output = self.head(embeddings, batch)

        return output

    def forward(self, *args, **kwargs):
        """Alias for predict() to support nn.Module calling conventions."""
        # When called as model(...), delegate to predict
        # This allows compatibility with both self.model.predict() and self.model()
        if len(args) == 2:
            return self.predict(args[0], args[1])
        else:
            return self.predict(**kwargs)
