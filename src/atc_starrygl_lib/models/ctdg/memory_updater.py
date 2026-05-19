from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..shared.time_encode import TimeEncode


class GRUMemoryUpdater(nn.Module):
    def __init__(
        self,
        memory_param: dict,
        dim_in: int,
        dim_hid: int,
        dim_time: int,
        dim_node_feat: int,
    ):
        super().__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory: Tensor | None = None
        self.last_updated_ts: Tensor | None = None
        self.last_updated_nid: Tensor | None = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param.get('combine_node_feature') and dim_node_feat > 0 and dim_node_feat != dim_hid:
            self.node_feat_map = nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg, param=None) -> None:
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param.get('combine_node_feature'):
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] = b.srcdata['h'] + updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory


class RNNMemoryUpdater(nn.Module):
    def __init__(
        self,
        memory_param: dict,
        dim_in: int,
        dim_hid: int,
        dim_time: int,
        dim_node_feat: int,
    ):
        super().__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory: Tensor | None = None
        self.last_updated_ts: Tensor | None = None
        self.last_updated_nid: Tensor | None = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param.get('combine_node_feature') and dim_node_feat > 0 and dim_node_feat != dim_hid:
            self.node_feat_map = nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg, param=None) -> None:
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param.get('combine_node_feature'):
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] = b.srcdata['h'] + updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory


class TransformerMemoryUpdater(nn.Module):
    def __init__(
        self,
        memory_param: dict,
        dim_in: int,
        dim_out: int,
        dim_time: int,
        train_param: dict,
    ):
        super().__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = nn.Linear(dim_out, dim_out)
        self.w_k = nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.mlp = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(train_param['dropout'])
        self.att_dropout = nn.Dropout(train_param['att_dropout'])

    def forward(self, b, param=None) -> Tensor:
        Q = self.w_q(b.srcdata['mem']).reshape(b.num_src_nodes(), self.att_h, -1)
        mails = b.srcdata['mem_input'].reshape(
            b.num_src_nodes(), self.memory_param['mailbox_size'], -1
        )
        if self.dim_time > 0:
            time_feat = self.time_enc(
                b.srcdata['ts'][:, None] - b.srcdata['mail_ts']
            ).reshape(b.num_src_nodes(), self.memory_param['mailbox_size'], -1)
            mails = torch.cat([mails, time_feat], dim=2)
        K = self.w_k(mails).reshape(
            b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1
        )
        V = self.w_v(mails).reshape(
            b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1
        )
        att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.att_dropout(att)
        rst = (att[:, :, :, None] * V).sum(dim=1).reshape(b.num_src_nodes(), -1)
        rst = rst + b.srcdata['mem']
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        return torch.nn.functional.relu(rst)
