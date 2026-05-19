from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from atc_starrygl_lib.core.types import Batch, EdgePredOutput, ClassifyOutput, RegressionOutput


class EdgePredictHead(nn.Module):
    """Link prediction head: inner-product scorer on src/dst embeddings.

    Expects batch.pos_src, batch.pos_dst, batch.neg_dst (or batch.neg_src)
    to be compact row indices into the embedding matrix.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)

    def forward(self, embeddings: Tensor, batch: Batch) -> EdgePredOutput:
        assert batch.pos_src is not None and batch.pos_dst is not None

        h_src = self.src_fc(embeddings[batch.pos_src])
        h_pos_dst = self.dst_fc(embeddings[batch.pos_dst])
        pos_score = self.out_fc(F.relu(h_src + h_pos_dst)).flatten()

        if batch.neg_dst is not None:
            neg_count = batch.neg_dst.size(0)
            pos_count = batch.pos_src.size(0)
            neg_ratio = neg_count // pos_count
            h_neg_dst = self.dst_fc(embeddings[batch.neg_dst])
            h_neg_edge = F.relu(h_src.repeat_interleave(neg_ratio, dim=0) + h_neg_dst)
            neg_score = self.out_fc(h_neg_edge).flatten()
        elif batch.neg_src is not None:
            h_neg_src = self.src_fc(embeddings[batch.neg_src])
            h_neg_dst = self.dst_fc(embeddings[batch.neg_dst]) if batch.neg_dst is not None else h_pos_dst
            neg_score = self.out_fc(F.relu(h_neg_src + h_neg_dst)).flatten()
        else:
            neg_score = torch.empty(0, device=embeddings.device)

        return EdgePredOutput(pos_score=pos_score, neg_score=neg_score)


class EdgeLabelHead(nn.Module):
    """Edge classification/regression head for observed edges.

    Uses concatenated src+dst embeddings as edge representation.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Linear(dim * 2, num_classes)

    def forward(self, embeddings: Tensor, batch: Batch) -> ClassifyOutput:
        assert batch.pos_src is not None and batch.pos_dst is not None
        src_emb = embeddings[batch.pos_src]
        dst_emb = embeddings[batch.pos_dst]
        logits = self.mlp(torch.cat([src_emb, dst_emb], dim=-1))
        return ClassifyOutput(logits=logits)


class EdgeRegressHead(nn.Module):
    """Edge regression head for observed edges."""

    def __init__(self, dim: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Linear(dim * 2, out_dim)

    def forward(self, embeddings: Tensor, batch: Batch) -> RegressionOutput:
        assert batch.pos_src is not None and batch.pos_dst is not None
        src_emb = embeddings[batch.pos_src]
        dst_emb = embeddings[batch.pos_dst]
        pred = self.mlp(torch.cat([src_emb, dst_emb], dim=-1))
        return RegressionOutput(pred=pred)


class NodeClassifyHead(nn.Module):
    """Node classification head.

    If batch.node_ids is set, indexes into embeddings; otherwise uses all rows.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Linear(dim, num_classes)

    def forward(self, embeddings: Tensor, batch: Batch) -> ClassifyOutput:
        h = embeddings[batch.node_ids] if batch.node_ids is not None else embeddings
        return ClassifyOutput(logits=self.mlp(h))


class NodeRegressHead(nn.Module):
    """Node regression head."""

    def __init__(self, dim: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Linear(dim, out_dim)

    def forward(self, embeddings: Tensor, batch: Batch) -> RegressionOutput:
        h = embeddings[batch.node_ids] if batch.node_ids is not None else embeddings
        return RegressionOutput(pred=self.mlp(h))
