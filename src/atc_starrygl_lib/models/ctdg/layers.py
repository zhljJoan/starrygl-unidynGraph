from __future__ import annotations

import math
import torch
import torch.nn as nn
import dgl
from torch import Tensor

from ..shared.time_encode import TimeEncode


class TransformerAttentionLayer(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.att_act = nn.LeakyReLU(0.2)
        self.combined = combined
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

    def forward(self, b) -> Tensor:
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
            src, dst = b.edges()
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=device)
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[dst]
                K = self.w_k(b.srcdata['h'][src])
                V = self.w_v(b.srcdata['h'][src])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[dst]
                K = self.w_k(torch.cat([b.srcdata['h'][src], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][src], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(zero_time_feat)[dst]
                K = self.w_k(time_feat)
                V = self.w_v(time_feat)
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[dst]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[dst]
                K = self.w_k(torch.cat([b.srcdata['h'][src], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][src], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[dst]
                K = self.w_k(torch.cat([b.srcdata['h'][src], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][src], b.edata['f'], time_feat], dim=1))

        Q = Q.reshape(Q.shape[0], self.num_head, -1)
        K = K.reshape(K.shape[0], self.num_head, -1)
        V = V.reshape(V.shape[0], self.num_head, -1)
        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q * K, dim=2)))
        att = self.att_dropout(att)
        V = (V * att[:, :, None]).reshape(V.shape[0], -1)
        b.edata['v'] = V
        b.update_all(dgl.function.copy_e('v', 'm'), dgl.function.sum('m', 'h'))

        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)


class IdentityNormLayer(nn.Module):
    def __init__(self, dim_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, b) -> Tensor:
        return self.norm(b.srcdata['h'])


class JODIETimeEmbedding(nn.Module):
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
        time_diff = (ts - mem_ts) / (ts + 1)
        return h * (1 + self.time_emb(time_diff.unsqueeze(1)))
