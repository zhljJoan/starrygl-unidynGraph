from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..shared.time_encode import TimeEncode


def _apply_memory_to_node_features(
    block,
    updated_memory: Tensor,
    memory_param: dict,
    *,
    dim_node_feat: int,
    dim_hid: int,
    node_feat_map: nn.Module | None = None,
) -> None:
    if memory_param.get('combine_node_feature'):
        if dim_node_feat > 0:
            if dim_node_feat == dim_hid:
                block.srcdata['h'] = block.srcdata['h'] + updated_memory
            else:
                assert node_feat_map is not None
                block.srcdata['h'] = updated_memory + node_feat_map(block.srcdata['h'])
        else:
            block.srcdata['h'] = updated_memory
    else:
        block.srcdata['h'] = updated_memory


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
        else:
            self.node_feat_map = None

    def forward(self, mfg, param=None) -> None:
        for b in mfg:
            self.forward_from_mfg(b)

    def forward_from_mfg(self, block) -> Tensor:
        if self.dim_time > 0:
            time_feat = self.time_enc(block.srcdata['ts'] - block.srcdata['mem_ts'])
            block.srcdata['mem_input'] = torch.cat([block.srcdata['mem_input'], time_feat], dim=1)
        updated_memory = self.updater(block.srcdata['mem_input'], block.srcdata['mem'])
        self.last_updated_ts = block.srcdata['ts'].detach().clone()
        self.last_updated_memory = updated_memory.detach().clone()
        self.last_updated_nid = block.srcdata['ID'].detach().clone()
        _apply_memory_to_node_features(
            block,
            updated_memory,
            self.memory_param,
            dim_node_feat=self.dim_node_feat,
            dim_hid=self.dim_hid,
            node_feat_map=self.node_feat_map,
        )
        return updated_memory


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
        else:
            self.node_feat_map = None

    def forward(self, mfg, param=None) -> None:
        for b in mfg:
            self.forward_from_mfg(b)

    def forward_from_mfg(self, block) -> Tensor:
        if self.dim_time > 0:
            time_feat = self.time_enc(block.srcdata['ts'] - block.srcdata['mem_ts'])
            block.srcdata['mem_input'] = torch.cat([block.srcdata['mem_input'], time_feat], dim=1)
        updated_memory = self.updater(block.srcdata['mem_input'], block.srcdata['mem'])
        self.last_updated_ts = block.srcdata['ts'].detach().clone()
        self.last_updated_memory = updated_memory.detach().clone()
        self.last_updated_nid = block.srcdata['ID'].detach().clone()
        _apply_memory_to_node_features(
            block,
            updated_memory,
            self.memory_param,
            dim_node_feat=self.dim_node_feat,
            dim_hid=self.dim_hid,
            node_feat_map=self.node_feat_map,
        )
        return updated_memory


class TransformerMemoryUpdater(nn.Module):
    def __init__(
        self,
        memory_param: dict,
        dim_in: int,
        dim_out: int,
        dim_time: int,
        train_param: dict,
        dim_node_feat: int = 0,
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
        self.dim_hid = dim_out
        self.dim_node_feat = int(dim_node_feat)
        self.last_updated_memory: Tensor | None = None
        self.last_updated_ts: Tensor | None = None
        self.last_updated_nid: Tensor | None = None
        self.node_feat_map: nn.Module | None = None
        if memory_param.get('combine_node_feature') and self.dim_node_feat > 0 and self.dim_node_feat != self.dim_hid:
            self.node_feat_map = nn.Linear(self.dim_node_feat, self.dim_hid)

    def forward(self, mfg, param=None) -> Tensor | None:
        updated = None
        blocks = mfg if isinstance(mfg, (list, tuple)) else [mfg]
        for block in blocks:
            updated = self.forward_from_mfg(block)
        return updated

    def forward_from_mfg(self, block) -> Tensor:
        Q = self.w_q(block.srcdata['mem']).reshape(block.num_src_nodes(), self.att_h, -1)
        mails = block.srcdata['mem_input'].reshape(
            block.num_src_nodes(), self.memory_param['mailbox_size'], -1
        )
        if self.dim_time > 0:
            time_feat = self.time_enc(
                block.srcdata['ts'][:, None] - block.srcdata['mail_ts']
            ).reshape(block.num_src_nodes(), self.memory_param['mailbox_size'], -1)
            mails = torch.cat([mails, time_feat], dim=2)
        K = self.w_k(mails).reshape(
            block.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1
        )
        V = self.w_v(mails).reshape(
            block.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1
        )
        att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.att_dropout(att)
        rst = (att[:, :, :, None] * V).sum(dim=1).reshape(block.num_src_nodes(), -1)
        rst = rst + block.srcdata['mem']
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        updated_memory = torch.nn.functional.relu(rst)
        self.last_updated_ts = block.srcdata['ts'].detach().clone()
        self.last_updated_memory = updated_memory.detach().clone()
        self.last_updated_nid = block.srcdata['ID'].detach().clone()
        _apply_memory_to_node_features(
            block,
            updated_memory,
            self.memory_param,
            dim_node_feat=self.dim_node_feat,
            dim_hid=self.dim_hid,
            node_feat_map=self.node_feat_map,
        )
        return updated_memory
