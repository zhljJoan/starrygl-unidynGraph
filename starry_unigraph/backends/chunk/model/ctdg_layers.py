"""CTDG model layers: TransformerAttentionLayer, IdentityNormLayer, JODIETimeEmbedding.

Direct port from MemShare-public/starrygl/module/layers.py, adapted for the
chunk backend's PartitionData / BatchData / packed DistIndex conventions.

Reference: ~/MemShare-public/MemShare/starrygl/module/layers.py
"""

from __future__ import annotations

import math
from typing import Optional

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from torch import Tensor

from starry_unigraph.runtime.modules.time_encode import TimeEncode


__all__ = [
    "TransformerAttentionLayer",
    "IdentityNormLayer",
    "JODIETimeEmbedding",
    "EdgePredictor",
]


class TransformerAttentionLayer(nn.Module):
    """Temporal transformer attention layer for CTDG.

    Computes attention over temporal neighbors using node features,
    edge features, and time encoding.

    Expected MFG srcdata keys:
        'h'   : node features / memory  [num_src, dim_node_feat]
        'f'   : edge features            [num_edges, dim_edge_feat]  (edata)
        'dt'  : time delta per edge      [num_edges]                 (edata)

    Args:
        dim_node_feat: Node feature dimension (0 if none)
        dim_edge_feat: Edge feature dimension (0 if none)
        dim_time: Time encoding dimension (0 to disable)
        num_head: Number of attention heads
        dropout: Dropout rate
        att_dropout: Attention dropout rate
        dim_out: Output dimension
        combined: Whether to use combined Q/K/V projections
    """

    def __init__(
        self,
        dim_node_feat: int,
        dim_edge_feat: int,
        dim_time: int,
        num_head: int,
        dropout: float,
        att_dropout: float,
        dim_out: int,
        combined: bool = False,
    ):
        super().__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.combined = combined

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.att_act = nn.LeakyReLU(0.2)

        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        if combined:
            if dim_node_feat > 0:
                self.w_q_n = nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = nn.Linear(dim_time, dim_out)
                self.w_k_t = nn.Linear(dim_time, dim_out)
                self.w_v_t = nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)

        self.w_out = nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, b: DGLBlock) -> Tensor:
        """Forward pass.

        Args:
            b: DGL block with srcdata['h'], edata['f'], edata['dt']

        Returns:
            dst_feat: [num_dst, dim_out]
        """
        assert self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0
        device = b.device

        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=device)

        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(
                torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=device)
            )

        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=device)
            K = torch.zeros((b.num_edges(), self.dim_out), device=device)
            V = torch.zeros((b.num_edges(), self.dim_out), device=device)

            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
        else:
            # Non-combined: concatenate features
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=device)
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.edges()[0]])
                V = self.w_v(b.srcdata['h'][b.edges()[0]])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(time_feat)
                V = self.w_v(time_feat)
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.edges()[0]], b.edata['f'], time_feat], dim=1))

        Q = Q.reshape(Q.shape[0], self.num_head, -1)
        K = K.reshape(K.shape[0], self.num_head, -1)
        V = V.reshape(V.shape[0], self.num_head, -1)

        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q * K, dim=2)))
        att = self.att_dropout(att)
        V = (V * att[:, :, None]).reshape(V.shape[0], -1)

        b.edata['v'] = V
        b.update_all(fn.copy_e('v', 'm'), fn.sum('m', 'h'))

        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']

        rst = self.w_out(rst)
        rst = F.relu(self.dropout(rst))
        return self.layer_norm(rst)


class IdentityNormLayer(nn.Module):
    """Identity layer with LayerNorm (used for JODIE-style models).

    Args:
        dim_out: Output dimension
    """

    def __init__(self, dim_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, b: DGLBlock) -> Tensor:
        return self.norm(b.srcdata['h'][:b.num_dst_nodes()])


class JODIETimeEmbedding(nn.Module):
    """JODIE-style time embedding for trajectory-based models.

    Modulates node embeddings by time delta: h * (1 + W * dt)

    Args:
        dim_out: Embedding dimension
    """

    def __init__(self, dim_out: int):
        super().__init__()
        self.dim_out = dim_out

        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)

    def forward(self, h: Tensor, mem_ts: Tensor, ts: Tensor) -> Tensor:
        """Apply time modulation.

        Args:
            h: [N, dim_out] Node embeddings
            mem_ts: [N] Last memory update timestamps
            ts: [N] Current timestamps

        Returns:
            h_modulated: [N, dim_out]
        """
        time_diff = (ts - mem_ts) / (ts + 1)
        return h * (1 + self.time_emb(time_diff.unsqueeze(1)))


class EdgePredictor(nn.Module):
    """Edge predictor for link prediction tasks.

    Computes edge scores from src/dst embeddings.

    Args:
        dim_in: Input embedding dimension
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in
        self.src_fc = nn.Linear(dim_in, dim_in)
        self.dst_fc = nn.Linear(dim_in, dim_in)
        self.out_fc = nn.Linear(dim_in, 1)

    def forward(
        self,
        h_pos_src: Tensor,
        h_pos_dst: Tensor,
        h_neg_src: Optional[Tensor] = None,
        h_neg_dst: Optional[Tensor] = None,
        neg_samples: int = 1,
        mode: str = "triplet",
    ):
        """Compute edge scores.

        Args:
            h_pos_src: [B, dim_in] Positive source embeddings
            h_pos_dst: [B, dim_in] Positive destination embeddings
            h_neg_src: Optional [B*neg, dim_in] Negative source embeddings
            h_neg_dst: [B*neg, dim_in] Negative destination embeddings
            neg_samples: Number of negatives per positive
            mode: "triplet" (share src) or "independent" (separate src/dst)

        Returns:
            (pos_scores, neg_scores): [B, 1], [B*neg, 1]
        """
        h_pos_src = self.src_fc(h_pos_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_neg_dst = self.dst_fc(h_neg_dst)

        if mode == "triplet":
            h_pos_edge = F.relu(h_pos_src + h_pos_dst)
            h_neg_edge = F.relu(h_pos_src.tile(neg_samples, 1) + h_neg_dst)
        else:
            h_neg_src = self.src_fc(h_neg_src)
            h_pos_edge = F.relu(h_pos_src + h_pos_dst)
            h_neg_edge = F.relu(h_neg_src + h_neg_dst)

        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
