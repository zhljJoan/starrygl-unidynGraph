"""CTDG model components: temporal attention, memory updater, link predictor.

- :class:`TimeEncode` — learnable time encoding (cosine basis).
- :class:`TemporalTransformerConv` — single-layer temporal multi-head
  attention over a DGL bipartite block.
- :func:`build_dgl_block` — convert a BTS ``TemporalGraphBlock`` to DGL.
- :class:`CTDGMemoryUpdater` — GRU-based memory updater over K-slot mailbox.
- :class:`CTDGLinkPredictor` — full link prediction module (conv + scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    import dgl
    import dgl.function as dglfn
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False

from starry_unigraph.runtime.modules import TimeEncode


# ---------------------------------------------------------------------------
# Temporal multi-head attention  (MemShare TransfomerAttentionLayer, DGL-based)
# ---------------------------------------------------------------------------

class TemporalTransformerConv(nn.Module):
    """Single-layer temporal transformer attention over a DGL bipartite block.

    Expects block.srcdata['h'], block.edata['dt'], block.edata['f'].
    The first ``num_dst`` rows of srcdata['h'] correspond to dst nodes
    (same convention as MemShare).
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
        """
        b: DGL block
            srcdata['h']  [n_dst + n_src, D]  – dst nodes first, then neighbors
            edata['dt']   [E]                  – time delta (scalar per edge)
            edata['f']    [E, dim_edge]         – edge features
        Returns: [n_dst, dim_out]
        """
        device = b.device
        n_dst = b.num_dst_nodes()

        if b.num_edges() == 0:
            return torch.zeros(n_dst, self.dim_out, device=device)

        h = b.srcdata['h']          # [n_dst+n_src, D]
        h_dst = h[:n_dst]           # [n_dst, D]
        h_src = h[n_dst:]           # [n_src, D]  (one row per edge in this block)

        edge_src, edge_dst = b.edges()  # edge_src ∈ [n_dst, n_dst+n_src), edge_dst ∈ [0, n_dst)
        # neighbor indices into h_src (subtract n_dst offset)
        nbr_idx = edge_src - n_dst  # [E]

        dt = b.edata['dt'].float()  # [E]
        ef = b.edata['f'].float()   # [E, dim_edge]

        # --- time encodings ---
        if self.dim_time > 0:
            time_feat = self.time_enc(dt)                                            # [E, dim_time]
            zero_time = self.time_enc(torch.zeros(n_dst, device=device))             # [n_dst, dim_time]
            Q_in = torch.cat([h_dst, zero_time], dim=1)                              # [n_dst, D+dim_time]
            K_in = torch.cat([h_src[nbr_idx], ef, time_feat], dim=1)                # [E, D+E+dim_time]
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

        # attention scores → per-dst softmax (MemShare uses dgl.ops.edge_softmax)
        att_raw = self.att_act((Q * K).sum(dim=2))    # [E, H]
        att = dgl.ops.edge_softmax(b, att_raw)        # [E, H], softmax over in-edges per dst
        att = self.att_dropout(att)

        V_weighted = (V * att.unsqueeze(-1)).reshape(V.shape[0], -1)  # [E, dim_out]
        b.edata['_v'] = V_weighted
        b.update_all(dglfn.copy_e('_v', '_m'), dglfn.sum('_m', '_agg'))
        agg = b.dstdata['_agg']   # [n_dst, dim_out]8

        # residual + output
        rst = self.w_out(torch.cat([agg, h_dst], dim=1))
        rst = self.layer_norm(F.relu(self.dropout(rst)))
        return rst


# ---------------------------------------------------------------------------
# Helper: build DGL block from BTS TemporalGraphBlock
# ---------------------------------------------------------------------------

def build_dgl_block(
    bts_block: object,
    roots: torch.Tensor,           # [n_dst] global node ids on device
    edge_feat_all: torch.Tensor,   # [num_edges, E] full edge feature table (any device)
    device: torch.device | str,
) -> "dgl.DGLGraph | None":
    """Convert one BTS TemporalGraphBlock into a DGL bipartite block.

    The returned block has:

    - ``srcdata['__ID']``: global node IDs (dst nodes first, then sampled
      neighbors) for indexing into the memory bank.
    - ``edata['dt']``: time deltas per edge.
    - ``edata['f']``: edge features.

    Args:
        bts_block: A BTS ``TemporalGraphBlock`` from the native sampler.
        roots: Root (dst) node global IDs on *device*, shape ``[n_dst]``.
        edge_feat_all: Full edge feature table, shape ``[num_edges, E]``.
        device: Target device for the output block.

    Returns:
        A DGL block, or ``None`` if no edges were sampled.
    """
    assert _DGL_AVAILABLE, "dgl is required"

    samp_nodes = bts_block.sample_nodes().to(device)   # [E]
    src_index  = bts_block.src_index().to(device)      # [E]
    delta_ts   = bts_block.delta_ts().to(device)       # [E]
    eids       = bts_block.eid().to(device)            # [E]

    n_dst = roots.numel()
    n_src = samp_nodes.numel()

    if n_src == 0:
        return None

    edge_src = torch.arange(n_src, dtype=torch.long, device=device) + n_dst
    edge_dst = src_index.long()

    b = dgl.create_block(
        (edge_src, edge_dst),
        num_src_nodes=n_dst + n_src,
        num_dst_nodes=n_dst,
        device=device,
    )

    # Store global node IDs for caller to index into memory (MemShare pattern)
    b.srcdata['__ID'] = torch.cat([roots, samp_nodes]).long()

    b.edata['dt'] = delta_ts.float()
    b.edata['f']  = edge_feat_all[eids.long()].float().to(device)

    return b


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CTDGModelOutput:
    pos_logits: torch.Tensor
    neg_logits: torch.Tensor


# ---------------------------------------------------------------------------
# Memory updater  (GRU over K-slot mailbox)
# ---------------------------------------------------------------------------

class CTDGMemoryUpdater(nn.Module):
    """GRU-based memory updater over K-slot mailbox history.

    Aggregates K mailbox slots via a GRU (``mail_aggregator``), then fuses
    the aggregated message with the current node memory via a GRU cell.

    Args:
        hidden_dim: Node memory dimension.
        mailbox_slot_dim: Width of each mailbox slot (``2*D + E``).
        mailbox_slots: Number of mailbox slots (K).

    Forward signature::

        new_memory = updater(mailbox_history, current_memory)
        # mailbox_history: [M, K, slot_dim]
        # current_memory:  [M, D]
        # returns:         [M, D]
    """

    def __init__(self, hidden_dim: int, mailbox_slot_dim: int, mailbox_slots: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mailbox_slots = mailbox_slots
        self.mail_aggregator = nn.GRU(
            input_size=mailbox_slot_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.memory_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        mailbox_history: torch.Tensor,  # [M, K, slot_dim]
        current_memory: torch.Tensor,   # [M, D]
    ) -> torch.Tensor:
        M = mailbox_history.size(0)
        if M == 0:
            return current_memory.clone()
        _, agg = self.mail_aggregator(mailbox_history)  # [1, M, D]
        agg = agg.squeeze(0)
        return self.memory_cell(agg, current_memory)


# ---------------------------------------------------------------------------
# Link predictor  (uses TemporalTransformerConv for context)
# ---------------------------------------------------------------------------

class CTDGLinkPredictor(nn.Module):
    """Temporal link prediction module with transformer attention.

    Combines a :class:`TemporalTransformerConv` for computing node
    representations from sampled neighborhoods, with an edge predictor
    that scores ``(src, dst)`` and ``(src, neg_dst)`` pairs.

    Args:
        num_nodes: Total number of nodes in the graph.
        hidden_dim: Node memory / representation dimension.
        edge_feat_dim: Edge feature dimension.
        dim_time: Time encoding dimension (default 100).
        num_head: Number of attention heads (default 2).
        dropout: Dropout rate for output projection.
        att_dropout: Dropout rate for attention weights.

    Forward signature::

        output = predictor(src_conv, dst_conv, neg_conv)
        # Returns CTDGModelOutput with .pos_logits and .neg_logits
    """
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        edge_feat_dim: int,
        dim_time: int = 100,
        num_head: int = 2,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        if _DGL_AVAILABLE:
            self.conv = TemporalTransformerConv(
                dim_node=hidden_dim,
                dim_edge=edge_feat_dim,
                dim_time=dim_time,
                num_head=num_head,
                dim_out=hidden_dim,
                dropout=dropout,
                att_dropout=att_dropout,
            )
        else:
            self.conv = None

        # EdgePredictor aligned with MemShare: src_fc + dst_fc → relu → out_fc
        self.src_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.dst_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc  = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        src_conv: torch.Tensor,   # [B, D]  conv output = node repr
        dst_conv: torch.Tensor,   # [B, D]
        neg_conv: torch.Tensor,   # [B, D]
    ) -> CTDGModelOutput:
        h_src = self.src_fc(src_conv)
        h_dst = self.dst_fc(dst_conv)
        h_neg = self.dst_fc(neg_conv)
        pos_logits = self.out_fc(F.relu(h_src + h_dst)).squeeze(-1)
        neg_logits = self.out_fc(F.relu(h_src + h_neg)).squeeze(-1)
        return CTDGModelOutput(pos_logits=pos_logits, neg_logits=neg_logits)
