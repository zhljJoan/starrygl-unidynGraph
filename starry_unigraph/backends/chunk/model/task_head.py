"""Task output heads for chunk pipeline.

Copied from starry_unigraph.models.task_head and extended with:
- EdgeRegressHead: predicts a scalar per edge (e.g., weight, time-to-event)

All heads receive node embeddings + BatchData and return a dict of tensors.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData


class EdgePredictHead(nn.Module):
    """Edge/link existence prediction via inner-product scoring."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: Tensor, batch: BatchData) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        if batch.pos_src is not None and batch.pos_dst is not None:
            out["pos_score"] = (embeddings[batch.pos_src] * embeddings[batch.pos_dst]).sum(1)
        if batch.neg_src is not None and batch.neg_dst is not None:
            out["neg_score"] = (embeddings[batch.neg_src] * embeddings[batch.neg_dst]).sum(1)
        return out


class EdgeRegressHead(nn.Module):
    """Edge attribute regression (e.g., predict edge weight, timestamp delta)."""

    def __init__(self, embedding_dim: int, output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Linear(embedding_dim * 2, output_dim)

    def forward(self, embeddings: Tensor, batch: BatchData) -> Dict[str, Tensor]:
        if batch.pos_src is None or batch.pos_dst is None:
            return {}
        src_emb = embeddings[batch.pos_src]
        dst_emb = embeddings[batch.pos_dst]
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        return {"edge_pred": self.mlp(edge_emb)}


class NodeRegressHead(nn.Module):
    """Node regression (continuous target per node)."""

    def __init__(self, embedding_dim: int, output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Linear(embedding_dim, output_dim)

    def forward(self, embeddings: Tensor, batch: BatchData) -> Dict[str, Tensor]:
        idx = batch.target_nodes if batch.target_nodes is not None else slice(None)
        return {"node_pred": self.mlp(embeddings[idx])}


class NodeClassifyHead(nn.Module):
    """Node classification (discrete class label per node)."""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: Tensor, batch: BatchData) -> Dict[str, Tensor]:
        idx = batch.target_nodes if batch.target_nodes is not None else slice(None)
        return {"logits": self.mlp(embeddings[idx])}
