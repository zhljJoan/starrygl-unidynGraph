"""Temporal multi-head attention layer for CTDG (online models).

CTDG-specific component. Uses TimeEncode for temporal deltas and
multi-head attention over sampled temporal neighborhoods.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

try:
    import dgl
    import dgl.function as dglfn
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False

from .time_encode import TimeEncode


class TemporalTransformerConv(nn.Module):
    """Single-layer temporal transformer attention over a DGL bipartite block. CTDG-specific.

    Implements the attention layer from MemShare (TGN variant).
    Combines temporal node features, edge features, and time deltas using
    multi-head scaled dot-product attention.

    Expects block to have:
    - ``srcdata['h']``: [n_dst + n_src, D] node features (dst nodes first)
    - ``edata['dt']``: [E] time deltas
    - ``edata['f']``: [E, dim_edge] edge features

    Args:
        dim_node: Node feature dimension.
        dim_edge: Edge feature dimension.
        dim_time: Time encoding dimension (0 to disable time encoding).
        num_head: Number of attention heads (default 2).
        dim_out: Output dimension.
        dropout: Dropout rate for output projection.
        att_dropout: Dropout rate for attention weights.

    Example::

        conv = TemporalTransformerConv(
            dim_node=128, dim_edge=16, dim_time=100,
            num_head=4, dim_out=128
        )
        output = conv(dgl_block)                     # [n_dst, 128]
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        dim_time: int,
        num_head: int,
        dim_out: int,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert _DGL_AVAILABLE, "dgl is required for TemporalTransformerConv"
        self.num_head = num_head
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_out = dim_out

        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        # Q: dst node feat + zero-time encoding
        self.w_q = nn.Linear(dim_node + dim_time, dim_out)
        # K, V: neighbor feat + edge feat + time delta encoding
        self.w_k = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        self.w_v = nn.Linear(dim_node + dim_edge + dim_time, dim_out)
        # output projection: aggregated + residual dst feat
        self.w_out = nn.Linear(dim_node + dim_out, dim_out)

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, b: "dgl.DGLGraph") -> torch.Tensor:
        """Forward pass over DGL bipartite block.

        Args:
            b: DGL block with srcdata['h'], edata['dt'], edata['f']

        Returns:
            Node embeddings [n_dst, dim_out]
        """
        device = b.device
        n_dst = b.num_dst_nodes()

        if b.num_edges() == 0:
            return torch.zeros(n_dst, self.dim_out, device=device)

        h = b.srcdata['h']          # [n_dst+n_src, D]
        h_dst = h[:n_dst]           # [n_dst, D]
        h_src = h[n_dst:]           # [n_src, D]

        edge_src, edge_dst = b.edges()
        nbr_idx = edge_src - n_dst  # [E] neighbor indices

        dt = b.edata['dt'].float()  # [E]
        ef = b.edata['f'].float()   # [E, dim_edge]

        # --- time encodings ---
        if self.dim_time > 0:
            time_feat = self.time_enc(dt)                                    # [E, dim_time]
            zero_time = self.time_enc(torch.zeros(n_dst, device=device))     # [n_dst, dim_time]
            Q_in = torch.cat([h_dst, zero_time], dim=1)                      # [n_dst, D+dim_time]
            K_in = torch.cat([h_src[nbr_idx], ef, time_feat], dim=1)        # [E, D+E+dim_time]
            V_in = K_in
        else:
            Q_in = h_dst
            K_in = torch.cat([h_src[nbr_idx], ef], dim=1)
            V_in = K_in

        # Q indexed by dst node of each edge
        Q = self.w_q(Q_in)[edge_dst]    # [E, dim_out]
        K = self.w_k(K_in)              # [E, dim_out]
        V = self.w_v(V_in)              # [E, dim_out]

        # multi-head reshape
        Q = Q.reshape(Q.shape[0], self.num_head, -1)  # [E, H, d]
        K = K.reshape(K.shape[0], self.num_head, -1)
        V = V.reshape(V.shape[0], self.num_head, -1)

        # attention scores → per-dst softmax
        att_raw = self.att_act((Q * K).sum(dim=2))    # [E, H]
        att = dgl.ops.edge_softmax(b, att_raw)        # [E, H], softmax per dst
        att = self.att_dropout(att)

        V_weighted = (V * att.unsqueeze(-1)).reshape(V.shape[0], -1)  # [E, dim_out]
        b.edata['_v'] = V_weighted
        b.update_all(dglfn.copy_e('_v', '_m'), dglfn.sum('_m', '_agg'))
        agg = b.dstdata['_agg']   # [n_dst, dim_out]

        # residual + output
        rst = self.w_out(torch.cat([agg, h_dst], dim=1))
        rst = self.layer_norm(F.relu(self.dropout(rst)))
        return rst
