"""Task-specific output heads for model predictions."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.data.batch_data import BatchData


class EdgePredictHead(nn.Module):
    """Output head for edge/link prediction task.

    Takes node embeddings and edge pairs, computes edge scores.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Simple inner product scoring
        self.scorer = nn.Identity()  # Or could be MLP for more expressiveness

    def forward(
        self,
        embeddings: Tensor,  # [N, embedding_dim]
        batch: BatchData,
    ) -> Dict[str, Tensor]:
        """Compute edge scores for positive and negative edge pairs.

        Args:
            embeddings: [N, embedding_dim] node embeddings
            batch: BatchData with pos_src, pos_dst, neg_src, neg_dst

        Returns:
            {
                'pos_score': [M_pos] scores for positive edges,
                'neg_score': [M_neg] scores for negative edges,
            }
        """
        # Positive edge scores: inner product of source and destination embeddings
        if batch.pos_src is not None and batch.pos_dst is not None:
            pos_src_emb = embeddings[batch.pos_src]
            pos_dst_emb = embeddings[batch.pos_dst]
            pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)
        else:
            pos_score = torch.tensor([])

        # Negative edge scores
        if batch.neg_src is not None and batch.neg_dst is not None:
            neg_src_emb = embeddings[batch.neg_src]
            neg_dst_emb = embeddings[batch.neg_dst]
            neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
        else:
            neg_score = torch.tensor([])

        return {
            "pos_score": pos_score,
            "neg_score": neg_score,
        }


class NodeRegressHead(nn.Module):
    """Output head for node regression task.

    Predicts continuous values for nodes (e.g., future node features).
    """

    def __init__(self, embedding_dim: int, output_dim: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.mlp = nn.Linear(embedding_dim, output_dim)

    def forward(
        self,
        embeddings: Tensor,  # [N, embedding_dim]
        batch: BatchData,
    ) -> Dict[str, Tensor]:
        """Predict regression targets for labeled nodes.

        Args:
            embeddings: [N, embedding_dim] node embeddings
            batch: BatchData with target_nodes (subset of nodes to predict)

        Returns:
            {
                'node_pred': [M, output_dim] predictions for target nodes,
            }
        """
        if batch.target_nodes is not None:
            target_embeddings = embeddings[batch.target_nodes]
        else:
            target_embeddings = embeddings

        pred = self.mlp(target_embeddings)

        return {"node_pred": pred}


class NodeClassifyHead(nn.Module):
    """Output head for node classification task.

    Predicts discrete class labels for nodes.
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.mlp = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        embeddings: Tensor,  # [N, embedding_dim]
        batch: BatchData,
    ) -> Dict[str, Tensor]:
        """Predict class logits for labeled nodes.

        Args:
            embeddings: [N, embedding_dim] node embeddings
            batch: BatchData with target_nodes (subset of nodes to classify)

        Returns:
            {
                'logits': [M, num_classes] class logits for target nodes,
            }
        """
        if batch.target_nodes is not None:
            target_embeddings = embeddings[batch.target_nodes]
        else:
            target_embeddings = embeddings

        logits = self.mlp(target_embeddings)

        return {"logits": logits}
