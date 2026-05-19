"""CTDG model components: temporal attention, memory updater, link predictor.

- :class:`TimeEncode` — learnable time encoding (cosine basis).
- :class:`TemporalTransformerConv` — single-layer temporal multi-head
  attention over a DGL bipartite block.
- :func:`build_dgl_block` — convert a BTS ``TemporalGraphBlock`` to DGL.
- :class:`CTDGMemoryUpdater` — GRU-based memory updater over K-slot mailbox.
- :class:`CTDGLinkPredictor` — full link prediction module (conv + scoring).
- :func:`build_ctdg_model` — instantiate a CTDG model by family.
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

        edge_src, edge_dst = b.edges()

        dt = b.edata['dt'].float()  # [E]
        ef = b.edata['f'].float()   # [E, dim_edge]

        # --- time encodings ---
        if self.dim_time > 0:
            time_feat = self.time_enc(dt)                                            # [E, dim_time]
            zero_time = self.time_enc(torch.zeros(n_dst, device=device))             # [n_dst, dim_time]
            Q_in = torch.cat([h_dst, zero_time], dim=1)                              # [n_dst, D+dim_time]
            K_in = torch.cat([h[edge_src], ef, time_feat], dim=1)                # [E, D+E+dim_time]
            V_in = K_in
        else:
            Q_in = h_dst
            K_in = torch.cat([h[edge_src], ef], dim=1)
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
    root_ts: torch.Tensor | None,
    edge_feat_all: torch.Tensor,   # [num_edges, E] full edge feature table (any device)
    device: torch.device | str,
) -> "tuple[dgl.DGLGraph, dict[str, torch.Tensor]] | None":
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

    samp_nodes = bts_block.sample_nodes().to(device)      # [E]
    samp_ts    = bts_block.sample_nodes_ts().to(device)   # [E]
    src_index  = bts_block.src_index().to(device)         # [E]
    delta_ts   = bts_block.delta_ts().to(device)          # [E]
    eids       = bts_block.eid().to(device)               # [E]

    root_len = roots.numel()
    edge_len = samp_nodes.numel()

    if edge_len == 0:
        return None

    if root_ts is None:
        root_ts = torch.zeros(root_len, dtype=torch.float32, device=device)
    root_ts = root_ts.to(device).float()
    src_ts = torch.cat([root_ts, samp_ts.float()], dim=0)
    node_ids = torch.cat([roots.long(), samp_nodes.long()], dim=0)

    # Match MemShare's to_block(unique=True): compact duplicate
    # (node_id, timestamp) pairs and remap seed metadata to the compact rows.
    pairs = torch.stack([node_ids.to(torch.float64), src_ts.to(torch.float64)], dim=0)
    compact_pairs, inverse = torch.unique(pairs, dim=1, return_inverse=True)
    first_pos = torch.full((compact_pairs.size(1),), inverse.numel(), dtype=torch.long, device=device)
    first_pos.scatter_reduce_(
        0,
        inverse,
        torch.arange(inverse.numel(), dtype=torch.long, device=device),
        reduce="amin",
        include_self=True,
    )
    first_compact_ids = inverse[torch.sort(first_pos).values]
    compact_id = torch.empty(compact_pairs.size(1), dtype=torch.long, device=device)
    compact_id[first_compact_ids] = torch.arange(first_compact_ids.numel(), dtype=torch.long, device=device)
    original_to_compact = compact_id[inverse]
    compact_pairs = compact_pairs[:, first_compact_ids]

    col = original_to_compact[:root_len]
    row = original_to_compact[root_len : root_len + edge_len]
    edge_dst = col[src_index.long()]
    n_dst = int(col.max().item()) + 1 if col.numel() > 0 else 0
    n_src = max(n_dst, int(row.max().item()) + 1 if row.numel() > 0 else 0)

    b = dgl.create_block(
        (row, edge_dst),
        num_src_nodes=n_src,
        num_dst_nodes=n_dst,
        device=device,
    )

    # Store global node IDs for caller to index into memory (MemShare pattern)
    b.srcdata['__ID'] = compact_pairs[0, b.srcnodes()].long()
    b.srcdata['ts'] = compact_pairs[1, b.srcnodes()].float()

    b.edata['dt'] = delta_ts.float()
    b.edata['f']  = edge_feat_all[eids.long()].float().to(device)

    metadata = {
        "src_pos_index": original_to_compact[:root_len // 3],
        "dst_pos_index": original_to_compact[root_len // 3 : 2 * (root_len // 3)],
        "dst_neg_index": original_to_compact[2 * (root_len // 3) : root_len],
    }
    return b, metadata


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CTDGModelOutput:
    pos_logits: torch.Tensor
    neg_logits: torch.Tensor
    updated_ids: torch.Tensor | None = None
    updated_memory: torch.Tensor | None = None
    updated_ts: torch.Tensor | None = None
    mail_ids: torch.Tensor | None = None
    mail_slots: torch.Tensor | None = None
    mail_ts: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Memory updater  (GRU over K-slot mailbox)
# ---------------------------------------------------------------------------

class CTDGMemoryUpdater(nn.Module):
    """MemShare/TGN-style GRU memory updater.

    The reference path feeds the flattened mailbox plus
    ``TimeEncode(node_ts - memory_ts)`` into a GRUCell whose hidden state is
    the previous node memory.  This keeps the updater directly aligned with
    ``AsyncMemeoryUpdater.rnn_updater`` in MemShare for TGN/JODIE-style
    memory updates.
    """

    def __init__(
        self,
        hidden_dim: int,
        mailbox_slot_dim: int,
        mailbox_slots: int,
        dim_time: int = 100,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mailbox_slots = mailbox_slots
        self.mailbox_slot_dim = mailbox_slot_dim
        self.dim_time = dim_time
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.memory_cell = nn.GRUCell(mailbox_slot_dim * mailbox_slots + dim_time, hidden_dim)

    def forward(
        self,
        mailbox_history: torch.Tensor,  # [M, K, slot_dim]
        current_memory: torch.Tensor,   # [M, D]
        node_ts: torch.Tensor | None = None,
        memory_ts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M = mailbox_history.size(0)
        if M == 0:
            return current_memory.clone()
        mem_input = mailbox_history.reshape(M, -1)
        if self.dim_time > 0:
            if node_ts is None:
                node_ts = torch.zeros(M, dtype=current_memory.dtype, device=current_memory.device)
            if memory_ts is None:
                memory_ts = torch.zeros(M, dtype=current_memory.dtype, device=current_memory.device)
            time_feat = self.time_enc(node_ts.to(current_memory.device) - memory_ts.to(current_memory.device))
            mem_input = torch.cat([mem_input, time_feat], dim=1)
        return self.memory_cell(mem_input, current_memory)


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
        node_feat_dim: int = 0,
        mailbox_slots: int = 1,
        dim_time: int = 100,
        num_head: int = 2,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_feat_dim = int(node_feat_dim)
        self.node_feat_map = nn.Linear(node_feat_dim, hidden_dim) if node_feat_dim > 0 and node_feat_dim != hidden_dim else None
        self.memory_updater = CTDGMemoryUpdater(
            hidden_dim=hidden_dim,
            mailbox_slot_dim=2 * hidden_dim + edge_feat_dim,
            mailbox_slots=mailbox_slots,
            dim_time=dim_time,
        )

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

    def forward_mfg(
        self,
        block: "dgl.DGLGraph",
        metadata: dict[str, torch.Tensor],
        *,
        memory: torch.Tensor,
        memory_ts: torch.Tensor,
        mailbox: torch.Tensor,
        node_feat: torch.Tensor | None = None,
        edge_feat: torch.Tensor | None = None,
    ) -> CTDGModelOutput:
        """MemShare-compatible forward over one prepared MFG block.

        This mirrors ``GeneralModel.forward`` for TGN_large: update memory for
        every block source row, write updated memory into ``srcdata['h']``, run
        transformer attention once, then index embeddings by metadata.
        """
        node_ts = block.srcdata["ts"].to(memory.device).float()
        updated_memory = self.memory_updater(
            mailbox,
            memory,
            node_ts=node_ts,
            memory_ts=memory_ts,
        )
        h = updated_memory
        if node_feat is not None:
            node_feat = node_feat.to(h.device).float()
            if self.node_feat_map is not None:
                h = h + self.node_feat_map(node_feat)
            elif node_feat.shape[1] == h.shape[1]:
                h = h + node_feat
        block.srcdata["h"] = h
        out = self.conv(block) if self.conv is not None else h[: block.num_dst_nodes()]
        src_idx = metadata["src_pos_index"].to(out.device)
        dst_idx = metadata["dst_pos_index"].to(out.device)
        neg_idx = metadata["dst_neg_index"].to(out.device)
        pred = self(
            src_conv=out[src_idx],
            dst_conv=out[dst_idx],
            neg_conv=out[neg_idx],
        )

        update_rows = torch.cat([src_idx, dst_idx], dim=0)
        updated_ids = block.srcdata["__ID"][update_rows].long()
        updated_ts = block.srcdata["ts"][update_rows].float()
        updated_vals = updated_memory[update_rows]

        src_mem = updated_memory[src_idx]
        dst_mem = updated_memory[dst_idx]
        if edge_feat is None:
            edge_feat = torch.zeros(src_mem.size(0), self.conv.dim_edge, device=src_mem.device)
        edge_feat = edge_feat.to(src_mem.device).float()
        edge_dim = self.conv.dim_edge if self.conv is not None else edge_feat.size(-1)
        if edge_feat.size(-1) >= edge_dim:
            edge_feat = edge_feat[:, :edge_dim]
        else:
            edge_feat = torch.cat(
                [edge_feat, torch.zeros(edge_feat.size(0), edge_dim - edge_feat.size(-1), device=edge_feat.device)],
                dim=1,
            )
        src_slot = torch.cat([src_mem, dst_mem, edge_feat], dim=1)
        dst_slot = torch.cat([dst_mem, src_mem, edge_feat], dim=1)
        pred.updated_ids = updated_ids
        pred.updated_memory = updated_vals
        pred.updated_ts = updated_ts
        pred.mail_ids = updated_ids
        pred.mail_slots = torch.cat([src_slot, dst_slot], dim=0)
        pred.mail_ts = updated_ts
        return pred


@dataclass(frozen=True)
class CTDGModelSpec:
    model_cls: type[nn.Module]


CTDG_MODEL_SPECS: dict[str, CTDGModelSpec] = {
    # The current online runtime uses one shared implementation for the
    # supported CTDG families. Family-based dispatch is kept explicit here so
    # specialized implementations can be added without changing the runtime
    # builder contract.
    "tgn": CTDGModelSpec(CTDGLinkPredictor),
    "jodie": CTDGModelSpec(CTDGLinkPredictor),
    "dyrep": CTDGModelSpec(CTDGLinkPredictor),
    "tgat": CTDGModelSpec(CTDGLinkPredictor),
    "apan": CTDGModelSpec(CTDGLinkPredictor),
}


def build_ctdg_model(
    model_family: str,
    *,
    num_nodes: int,
    hidden_dim: int,
    edge_feat_dim: int,
    node_feat_dim: int = 0,
    mailbox_slots: int = 1,
    dim_time: int = 100,
    num_head: int = 2,
    dropout: float = 0.1,
    att_dropout: float = 0.1,
) -> nn.Module:
    """Instantiate a CTDG model by family.

    Args:
        model_family: CTDG model family name from the config.
        num_nodes: Total number of nodes in the graph.
        hidden_dim: Node memory / representation dimension.
        edge_feat_dim: Edge feature dimension.
        dim_time: Time encoding dimension.
        num_head: Number of attention heads.
        dropout: Dropout rate for output projection.
        att_dropout: Dropout rate for attention weights.

    Returns:
        A CTDG model instance.

    Raises:
        KeyError: If the family is not registered.
    """
    family = model_family.lower()
    spec = CTDG_MODEL_SPECS.get(family)
    if spec is None:
        raise KeyError(f"Unsupported CTDG model family: {model_family}")
    return spec.model_cls(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        mailbox_slots=mailbox_slots,
        dim_time=dim_time,
        num_head=num_head,
        dropout=dropout,
        att_dropout=att_dropout,
    )
