"""GeneralModel: Complete CTDG model integrating memory, mailbox, and GNN layers.

Implements MemShare-style GeneralModel adapted for the chunk backend.
Supports TGN, JODIE, DyRep architectures via config.

Reference: ~/MemShare-public/MemShare/starrygl/module/modules.py
           ~/MemShare-public/MemShare/starrygl/module/memorys.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from starry_unigraph.backends.chunk.model.ctdg_layers import (
    EdgePredictor,
    IdentityNormLayer,
    JODIETimeEmbedding,
    TransformerAttentionLayer,
)
from starry_unigraph.backends.chunk.model.memory_updater import (
    GRUMemoryUpdater,
    RNNMemoryUpdater,
    TransformerMemoryUpdater,
)
from starry_unigraph.backends.chunk.runtime.mailbox import Mailbox, MailboxConfig


__all__ = ["GeneralModel", "NodeClassificationModel"]


class GeneralModel(nn.Module):
    """Complete CTDG model: memory update + GNN embedding + edge prediction.

    Supports architectures:
    - TGN: memory (GRU/RNN/Transformer) + transformer_attention GNN
    - JODIE: memory (GRU) + identity + time_transform
    - DyRep: memory (GRU) + identity (dyrep=True uses memory as output)

    Args:
        dim_node: Node feature dimension
        dim_edge: Edge feature dimension
        sample_param: Sampling config dict (history, num_neighbors, ...)
        memory_param: Memory config dict (type, dim_out, memory_update, ...)
        gnn_param: GNN config dict (arch, layer, dim_time, att_head, dim_out, ...)
        train_param: Training config dict (dropout, att_dropout, ...)
        num_nodes: Total number of nodes
        mailbox: Optional Mailbox instance
        combined: Whether to use combined Q/K/V projections
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        sample_param: Dict[str, Any],
        memory_param: Dict[str, Any],
        gnn_param: Dict[str, Any],
        train_param: Dict[str, Any],
        num_nodes: Optional[int] = None,
        mailbox: Optional[Mailbox] = None,
        combined: bool = False,
    ):
        super().__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        self.gnn_param = gnn_param
        self.train_param = train_param

        if 'dim_out' not in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']

        # ── Memory updater ──────────────────────────────────────────────────
        if memory_param['type'] == 'node':
            dim_msg = 2 * memory_param['dim_out'] + dim_edge
            dim_mem = memory_param['dim_out']
            dim_time = memory_param.get('dim_time', 0)

            upd_type = memory_param['memory_update']
            if upd_type == 'gru':
                self.memory_updater = GRUMemoryUpdater(
                    memory_dim=dim_mem,
                    message_dim=dim_msg,
                    time_dim=dim_time,
                    node_feat_dim=dim_node,
                    combine_node_feature=memory_param.get('combine_node_feature', True),
                )
            elif upd_type == 'rnn':
                self.memory_updater = RNNMemoryUpdater(
                    memory_dim=dim_mem,
                    message_dim=dim_msg,
                    time_dim=dim_time,
                    node_feat_dim=dim_node,
                    combine_node_feature=memory_param.get('combine_node_feature', True),
                )
            elif upd_type == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(
                    memory_dim=dim_mem,
                    message_dim=dim_msg,
                    time_dim=dim_time,
                    mailbox_size=memory_param.get('mailbox_size', 10),
                    num_heads=memory_param.get('attention_head', 2),
                    dropout=train_param.get('dropout', 0.1),
                )
            else:
                raise NotImplementedError(f"Unknown memory_update: {upd_type}")

            self.dim_node_input = memory_param['dim_out']
        else:
            self.memory_updater = None

        self.mailbox = mailbox

        # ── GNN layers ───────────────────────────────────────────────────────
        self.layers = nn.ModuleDict()
        arch = gnn_param['arch']
        num_layers = gnn_param.get('layer', 1)
        num_history = sample_param.get('history', 1)
        dim_time = gnn_param.get('dim_time', 0)
        att_head = gnn_param.get('att_head', 2)
        dropout = train_param.get('dropout', 0.1)
        att_dropout = train_param.get('att_dropout', 0.1)
        dim_out = gnn_param['dim_out']

        if arch == 'transformer_attention':
            for h in range(num_history):
                self.layers[f'l0h{h}'] = TransformerAttentionLayer(
                    self.dim_node_input, dim_edge, dim_time,
                    att_head, dropout, att_dropout, dim_out,
                    combined=combined,
                )
            for l in range(1, num_layers):
                for h in range(num_history):
                    self.layers[f'l{l}h{h}'] = TransformerAttentionLayer(
                        dim_out, dim_edge, dim_time,
                        att_head, dropout, att_dropout, dim_out,
                        combined=False,
                    )

        elif arch == 'identity':
            gnn_param['layer'] = 1
            for h in range(num_history):
                self.layers[f'l0h{h}'] = IdentityNormLayer(self.dim_node_input)
                if gnn_param.get('time_transform') == 'JODIE':
                    self.layers[f'l0h{h}t'] = JODIETimeEmbedding(dim_out)
        else:
            raise NotImplementedError(f"Unknown GNN arch: {arch}")

        # ── Predictor ────────────────────────────────────────────────────────
        self.edge_predictor = EdgePredictor(dim_out)

        # Optional RNN combiner across history
        if gnn_param.get('combine') == 'rnn':
            self.combiner = nn.RNN(dim_out, dim_out)

        # Store last embedding for DyRep
        self.embedding: Optional[Tensor] = None

    def forward(
        self,
        mfgs: List[List[Any]],
        metadata: Dict[str, Tensor],
        neg_samples: int = 1,
        mode: str = "triplet",
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            mfgs: mfgs[layer][history] = DGL block
            metadata: Dict with keys:
                'src_pos_index': [B] positive source indices into output
                'dst_pos_index': [B] positive destination indices
                'dst_neg_index': [B*neg] negative destination indices
            neg_samples: Number of negatives per positive
            mode: "triplet" or "independent"

        Returns:
            (pos_scores, neg_scores): [B, 1], [B*neg, 1]
        """
        # ── Memory update ────────────────────────────────────────────────────
        if self.memory_updater is not None:
            self.memory_updater.forward_from_mfg(mfgs[0][0])

        # ── GNN embedding ────────────────────────────────────────────────────
        num_layers = self.gnn_param.get('layer', 1)
        num_history = self.sample_param.get('history', 1)
        out_list = []

        for l in range(num_layers):
            for h in range(num_history):
                rst = self.layers[f'l{l}h{h}'](mfgs[l][h])

                if self.gnn_param.get('time_transform') == 'JODIE':
                    b = mfgs[l][h]
                    rst = self.layers[f'l0h{h}t'](
                        rst,
                        b.srcdata['mem_ts'][:b.num_dst_nodes()],
                        b.srcdata['ts'][:b.num_dst_nodes()],
                    )

                if l < num_layers - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out_list.append(rst)

        # Combine history outputs
        if len(out_list) == 1:
            out = out_list[0]
        elif hasattr(self, 'combiner'):
            stacked = torch.stack(out_list, dim=0)  # [H, N, dim]
            out, _ = self.combiner(stacked)
            out = out[-1]
        else:
            out = out_list[0]

        self.embedding = out.detach().clone()

        # DyRep: use memory as output
        if self.gnn_param.get('dyrep', False) and self.memory_updater is not None:
            out = self.memory_updater.last_updated_memory

        # ── Edge prediction ──────────────────────────────────────────────────
        h_pos_src = out[metadata['src_pos_index']]
        h_pos_dst = out[metadata['dst_pos_index']]
        h_neg_dst = out[metadata['dst_neg_index']]

        return self.edge_predictor(
            h_pos_src, h_pos_dst, None, h_neg_dst,
            neg_samples=neg_samples, mode=mode,
        )

    @classmethod
    def from_config(
        cls,
        dim_node: int,
        dim_edge: int,
        num_nodes: int,
        config: Dict[str, Any],
        mailbox: Optional[Mailbox] = None,
    ) -> "GeneralModel":
        """Construct from a flat config dict.

        Expected config keys:
            sample.history, sample.num_neighbors
            memory.type, memory.dim_out, memory.memory_update,
                memory.dim_time, memory.combine_node_feature, memory.mailbox_size
            gnn.arch, gnn.layer, gnn.dim_time, gnn.att_head, gnn.dim_out,
                gnn.time_transform, gnn.dyrep, gnn.combine
            train.dropout, train.att_dropout
        """
        sample_param = config.get('sample', {'history': 1})
        memory_param = config.get('memory', {
            'type': 'node',
            'dim_out': 100,
            'memory_update': 'gru',
            'dim_time': 100,
            'combine_node_feature': True,
        })
        gnn_param = config.get('gnn', {
            'arch': 'transformer_attention',
            'layer': 1,
            'dim_time': 100,
            'att_head': 2,
            'dim_out': 100,
        })
        train_param = config.get('train', {'dropout': 0.1, 'att_dropout': 0.1})

        return cls(
            dim_node=dim_node,
            dim_edge=dim_edge,
            sample_param=sample_param,
            memory_param=memory_param,
            gnn_param=gnn_param,
            train_param=train_param,
            num_nodes=num_nodes,
            mailbox=mailbox,
        )


class NodeClassificationModel(nn.Module):
    """Simple 2-layer MLP for node classification.

    Args:
        dim_in: Input dimension
        dim_hid: Hidden dimension
        num_class: Number of classes
    """

    def __init__(self, dim_in: int, dim_hid: int, num_class: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc2 = nn.Linear(dim_hid, num_class)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
