"""Unified prediction head for chunk tasks.

Aligns task interfaces for:
  - edge_predict
  - edge_regress
  - node_classify
  - node_regress
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData


class PredictionHead(nn.Module):
    """Task-aligned prediction head.

    Input: node embeddings from model backbone.
    Output: task-specific tensor dict with unified key conventions.
    """

    def __init__(
        self,
        task_type: str,
        embedding_dim: int,
        output_dim: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        if task_type == "edge_predict":
            self.scorer = nn.Identity()
        elif task_type == "edge_regress":
            self.edge_mlp = nn.Linear(embedding_dim * 2, output_dim)
        elif task_type == "node_classify":
            self.node_mlp = nn.Linear(embedding_dim, num_classes)
        elif task_type == "node_regress":
            self.node_mlp = nn.Linear(embedding_dim, output_dim)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def forward(self, embeddings: Tensor, batch: BatchData) -> Dict[str, Tensor]:
        if self.task_type == "edge_predict":
            out: Dict[str, Tensor] = {}
            if batch.pos_src is not None and batch.pos_dst is not None:
                out["pos_score"] = (embeddings[batch.pos_src] * embeddings[batch.pos_dst]).sum(dim=1)
            if batch.neg_src is not None and batch.neg_dst is not None:
                out["neg_score"] = (embeddings[batch.neg_src] * embeddings[batch.neg_dst]).sum(dim=1)
            return out

        if self.task_type == "edge_regress":
            if batch.pos_src is None or batch.pos_dst is None:
                return {}
            edge_emb = torch.cat([embeddings[batch.pos_src], embeddings[batch.pos_dst]], dim=1)
            return {"edge_pred": self.edge_mlp(edge_emb)}

        idx = batch.target_nodes if batch.target_nodes is not None else slice(None)
        node_out = self.node_mlp(embeddings[idx])
        if self.task_type == "node_classify":
            return {"logits": node_out}
        return {"node_pred": node_out}
