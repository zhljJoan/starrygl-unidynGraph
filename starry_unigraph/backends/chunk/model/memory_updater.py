"""Memory updaters for CTDG temporal graph learning.

Implements MemShare-style memory updaters that update node memory
through messages. Supports GRU, RNN, and Transformer updaters.

Reference: ~/MemShare-public/starrygl/module/memorys.py
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.runtime.modules.time_encode import TimeEncode


__all__ = [
    "GRUMemoryUpdater",
    "RNNMemoryUpdater",
    "TransformerMemoryUpdater",
]


class GRUMemoryUpdater(nn.Module):
    """Update node memory using GRU cell.

    Computes: new_memory = GRU(cat(message, time_enc(dt)), old_memory)

    Args:
        memory_dim: Dimension of memory vectors
        message_dim: Dimension of input messages
        time_dim: Dimension of time encoding
        node_feat_dim: Dimension of node features (0 if none)
        combine_node_feature: Whether to add node features to output
    """

    def __init__(
        self,
        memory_dim: int,
        message_dim: int,
        time_dim: int,
        node_feat_dim: int = 0,
        combine_node_feature: bool = True,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        self.node_feat_dim = node_feat_dim
        self.combine_node_feature = combine_node_feature

        self.gru = nn.GRUCell(message_dim + time_dim, memory_dim)

        if time_dim > 0:
            self.time_enc = TimeEncode(time_dim)

        if combine_node_feature and node_feat_dim > 0 and node_feat_dim != memory_dim:
            self.node_feat_map = nn.Linear(node_feat_dim, memory_dim)

        self.last_updated_memory: Optional[Tensor] = None
        self.last_updated_ts: Optional[Tensor] = None
        self.last_updated_nid: Optional[Tensor] = None

    def forward(
        self,
        memory: Tensor,
        messages: Tensor,
        timestamps: Tensor,
        memory_ts: Tensor,
        node_feats: Optional[Tensor] = None,
    ) -> Tensor:
        """Update memory.

        Args:
            memory: [num_nodes, memory_dim] Current memory
            messages: [num_nodes, message_dim] Aggregated messages
            timestamps: [num_nodes] Current timestamps
            memory_ts: [num_nodes] Last memory update timestamps
            node_feats: Optional [num_nodes, node_feat_dim] node features

        Returns:
            new_memory: [num_nodes, memory_dim] Updated memory
        """
        # Time encoding: encode time delta since last memory update
        if self.time_dim > 0:
            dt = timestamps - memory_ts
            time_feat = self.time_enc(dt)
            gru_input = torch.cat([messages, time_feat], dim=-1)
        else:
            gru_input = messages

        # GRU update
        new_memory = self.gru(gru_input, memory)

        # Combine with node features
        if self.combine_node_feature and node_feats is not None:
            if self.node_feat_dim == self.memory_dim:
                new_memory = new_memory + node_feats
            elif hasattr(self, 'node_feat_map'):
                new_memory = new_memory + self.node_feat_map(node_feats)

        # Store for external access
        self.last_updated_memory = new_memory.detach().clone()
        self.last_updated_ts = timestamps.detach().clone()

        return new_memory

    def forward_from_mfg(self, mfg, node_feats: Optional[Tensor] = None) -> Tensor:
        """Update memory from DGL MFG srcdata.

        Expects mfg.srcdata to contain:
            'mem': current memory
            'mem_input': aggregated messages
            'ts': current timestamps
            'mem_ts': last memory update timestamps
            'ID': node IDs

        Args:
            mfg: DGL block with srcdata
            node_feats: Optional node features

        Returns:
            new_memory: [num_src_nodes, memory_dim]
        """
        memory = mfg.srcdata['mem']
        messages = mfg.srcdata['mem_input']
        timestamps = mfg.srcdata['ts']
        memory_ts = mfg.srcdata['mem_ts']

        new_memory = self.forward(memory, messages, timestamps, memory_ts, node_feats)

        self.last_updated_nid = mfg.srcdata['ID'].detach().clone()

        # Write back to mfg
        if self.combine_node_feature:
            if 'h' in mfg.srcdata:
                if self.node_feat_dim == self.memory_dim:
                    mfg.srcdata['h'] = mfg.srcdata['h'] + new_memory
                else:
                    mfg.srcdata['h'] = new_memory
            else:
                mfg.srcdata['h'] = new_memory

        return new_memory


class RNNMemoryUpdater(nn.Module):
    """Update node memory using RNN cell.

    Same interface as GRUMemoryUpdater but uses RNNCell.
    """

    def __init__(
        self,
        memory_dim: int,
        message_dim: int,
        time_dim: int,
        node_feat_dim: int = 0,
        combine_node_feature: bool = True,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        self.node_feat_dim = node_feat_dim
        self.combine_node_feature = combine_node_feature

        self.rnn = nn.RNNCell(message_dim + time_dim, memory_dim)

        if time_dim > 0:
            self.time_enc = TimeEncode(time_dim)

        if combine_node_feature and node_feat_dim > 0 and node_feat_dim != memory_dim:
            self.node_feat_map = nn.Linear(node_feat_dim, memory_dim)

        self.last_updated_memory: Optional[Tensor] = None
        self.last_updated_ts: Optional[Tensor] = None
        self.last_updated_nid: Optional[Tensor] = None

    def forward(
        self,
        memory: Tensor,
        messages: Tensor,
        timestamps: Tensor,
        memory_ts: Tensor,
        node_feats: Optional[Tensor] = None,
    ) -> Tensor:
        if self.time_dim > 0:
            dt = timestamps - memory_ts
            time_feat = self.time_enc(dt)
            rnn_input = torch.cat([messages, time_feat], dim=-1)
        else:
            rnn_input = messages

        new_memory = self.rnn(rnn_input, memory)

        if self.combine_node_feature and node_feats is not None:
            if self.node_feat_dim == self.memory_dim:
                new_memory = new_memory + node_feats
            elif hasattr(self, 'node_feat_map'):
                new_memory = new_memory + self.node_feat_map(node_feats)

        self.last_updated_memory = new_memory.detach().clone()
        self.last_updated_ts = timestamps.detach().clone()

        return new_memory

    def forward_from_mfg(self, mfg, node_feats: Optional[Tensor] = None) -> Tensor:
        memory = mfg.srcdata['mem']
        messages = mfg.srcdata['mem_input']
        timestamps = mfg.srcdata['ts']
        memory_ts = mfg.srcdata['mem_ts']

        new_memory = self.forward(memory, messages, timestamps, memory_ts, node_feats)
        self.last_updated_nid = mfg.srcdata['ID'].detach().clone()

        if self.combine_node_feature:
            if 'h' in mfg.srcdata:
                mfg.srcdata['h'] = mfg.srcdata['h'] + new_memory if self.node_feat_dim == self.memory_dim else new_memory
            else:
                mfg.srcdata['h'] = new_memory

        return new_memory


class TransformerMemoryUpdater(nn.Module):
    """Update node memory using Transformer attention over mailbox messages.

    Attends over the mailbox_size most recent messages to compute new memory.

    Args:
        memory_dim: Dimension of memory vectors
        message_dim: Dimension of input messages
        time_dim: Dimension of time encoding
        mailbox_size: Number of messages to attend over
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        memory_dim: int,
        message_dim: int,
        time_dim: int,
        mailbox_size: int = 10,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        self.mailbox_size = mailbox_size
        self.num_heads = num_heads

        if time_dim > 0:
            self.time_enc = TimeEncode(time_dim)

        self.w_q = nn.Linear(memory_dim, memory_dim)
        self.w_k = nn.Linear(message_dim + time_dim, memory_dim)
        self.w_v = nn.Linear(message_dim + time_dim, memory_dim)
        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(memory_dim)
        self.mlp = nn.Linear(memory_dim, memory_dim)
        self.dropout = nn.Dropout(dropout)

        self.last_updated_memory: Optional[Tensor] = None
        self.last_updated_ts: Optional[Tensor] = None
        self.last_updated_nid: Optional[Tensor] = None

    def forward(
        self,
        memory: Tensor,
        mailbox_messages: Tensor,
        timestamps: Tensor,
        mail_timestamps: Tensor,
    ) -> Tensor:
        """Update memory using attention over mailbox.

        Args:
            memory: [N, memory_dim] Current memory
            mailbox_messages: [N, mailbox_size, message_dim] Historical messages
            timestamps: [N] Current timestamps
            mail_timestamps: [N, mailbox_size] Message timestamps

        Returns:
            new_memory: [N, memory_dim]
        """
        N = memory.size(0)

        # Query from current memory
        Q = self.w_q(memory).reshape(N, self.num_heads, -1)  # [N, H, d]

        # Key/Value from mailbox messages with time encoding
        if self.time_dim > 0:
            dt = timestamps.unsqueeze(1) - mail_timestamps  # [N, mailbox_size]
            time_feat = self.time_enc(dt.reshape(-1)).reshape(N, self.mailbox_size, -1)
            kv_input = torch.cat([mailbox_messages, time_feat], dim=-1)
        else:
            kv_input = mailbox_messages

        K = self.w_k(kv_input).reshape(N, self.mailbox_size, self.num_heads, -1)  # [N, M, H, d]
        V = self.w_v(kv_input).reshape(N, self.mailbox_size, self.num_heads, -1)

        # Attention
        att = self.att_act((Q.unsqueeze(1) * K).sum(dim=-1))  # [N, M, H]
        att = torch.softmax(att, dim=1)
        att = self.dropout(att)

        # Weighted sum
        out = (att.unsqueeze(-1) * V).sum(dim=1)  # [N, H, d]
        out = out.reshape(N, -1)  # [N, memory_dim]

        # Residual + norm
        out = out + memory
        out = self.layer_norm(out)
        out = self.mlp(out)
        out = self.dropout(out)
        out = torch.relu(out)

        self.last_updated_memory = out.detach().clone()
        self.last_updated_ts = timestamps.detach().clone()

        return out

    def forward_from_mfg(self, mfg) -> Tensor:
        """Update memory from DGL MFG srcdata.

        Expects mfg.srcdata to contain:
            'mem': current memory [N, memory_dim]
            'mem_input': mailbox messages [N, mailbox_size * message_dim]
            'ts': current timestamps [N]
            'mail_ts': mailbox timestamps [N, mailbox_size]
        """
        N = mfg.num_src_nodes()
        memory = mfg.srcdata['mem']
        messages = mfg.srcdata['mem_input'].reshape(N, self.mailbox_size, -1)
        timestamps = mfg.srcdata['ts']
        mail_ts = mfg.srcdata.get('mail_ts', torch.zeros(N, self.mailbox_size, device=memory.device))

        new_memory = self.forward(memory, messages, timestamps, mail_ts)
        self.last_updated_nid = mfg.srcdata['ID'].detach().clone()

        mfg.srcdata['h'] = new_memory
        return new_memory
