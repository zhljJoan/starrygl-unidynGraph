"""Multi-layer GCN message passing primitive. Stable API - v0.1.0+

Reusable GCN layer stack supporting both full graphs and DGL bipartite blocks.
Used as backbone in DTDG Flare models (TGCN, EvolveGCN, MPNN-LSTM).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import dgl
import dgl.function as fn
from dgl import DGLGraph


def _graph_sequence(blob_or_graph) -> list[DGLGraph]:
    """Convert STGraphBlob or single graph to list of graphs."""
    if hasattr(blob_or_graph, '__iter__') and not isinstance(blob_or_graph, DGLGraph):
        return list(blob_or_graph)
    return [blob_or_graph]


def _graph_input(graph: DGLGraph) -> torch.Tensor:
    """Extract node features from graph (handles blocks vs. full graphs)."""
    if graph.is_block:
        if "x" in graph.dstdata:
            return graph.dstdata["x"]
        return graph.srcdata["x"]
    return graph.ndata["x"]


def _gcn_norm(graph: DGLGraph) -> torch.Tensor:
    """Compute GCN normalization weights (symmetric or degree-based)."""
    if "gcn_norm" in graph.edata:
        return graph.edata["gcn_norm"].float()
    weights = (
        graph.edata["w"].float()
        if "w" in graph.edata
        else torch.ones(graph.num_edges(), dtype=torch.float32, device=graph.device)
    )
    if graph.is_block:
        with graph.local_scope():
            graph.edata["w_tmp"] = weights
            graph.update_all(fn.copy_e("w_tmp", "m"), fn.sum("m", "deg"))
            deg = graph.dstdata["deg"].float().clamp_min_(1.0)
            dst, _ = graph.edges(order="eid")
            norm = weights / deg[graph.edges(order="eid")[1]]
    else:
        with graph.local_scope():
            graph.edata["w_tmp"] = weights
            graph.update_all(fn.copy_e("w_tmp", "m"), fn.sum("m", "deg"))
            deg = graph.ndata["deg"].float().clamp_min_(1.0)
            _, dst = graph.edges(order="eid")
            norm = weights / deg[dst]
    return norm.unsqueeze(-1)


def _gcn_message_pass(graph: DGLGraph, x: torch.Tensor) -> torch.Tensor:
    """Single GCN message passing step: normalized neighbor aggregation."""
    with graph.local_scope():
        if graph.is_block:
            if x.size(0) == graph.num_src_nodes():
                src_x = x
            elif x.size(0) == graph.num_dst_nodes():
                # For distributed blocks: use route if available
                route = getattr(graph, "route", None)
                if (
                    route is not None
                    and route.send_index is not None
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    src_x = route.forward(x)
                else:
                    extra_count = graph.num_src_nodes() - graph.num_dst_nodes()
                    extra_x = torch.zeros(
                        extra_count, x.size(-1), dtype=x.dtype, device=x.device
                    )
                    src_x = torch.cat([x, extra_x], dim=0)
            else:
                raise ValueError(
                    f"Unexpected feature rows for block graph: got {x.size(0)}, "
                    f"expected {graph.num_dst_nodes()} or {graph.num_src_nodes()}"
                )
            graph.srcdata["h"] = src_x
        else:
            graph.ndata["h"] = x
        graph.edata["norm"] = _gcn_norm(graph)
        graph.update_all(fn.u_mul_e("h", "norm", "m"), fn.sum("m", "h_new"))
        return graph.dstdata["h_new"] if graph.is_block else graph.ndata["h_new"]


class GCNStack(nn.Module):
    """Multi-layer GCN with message passing on DGL graphs/blocks. Stable API - v0.1.0+

    Each layer performs normalized message passing (GCN_norm * src_feats)
    followed by a learnable linear transformation. ReLU activation is applied
    between layers.

    Can process either:
    - Single DGL graph/block → returns single output tensor
    - STGraphBlob (multi-frame temporal sequence) → returns list of outputs

    Args:
        input_size: Input feature dimension (e.g., node feature dim).
        hidden_size: Hidden and output dimension for all layers.
        num_layers: Number of GCN layers (default 2).

    Example::

        gcn = GCNStack(input_size=2, hidden_size=64, num_layers=2)
        # Single graph
        h = gcn.forward_graph(block)                 # [N, 64]
        # Multi-frame blob
        h_list = gcn.layerwise(blob)                 # list[Tensor]

    Shape:
        - Input graph: node features ``[N, input_size]``
        - Output: ``[N, hidden_size]`` per layer applied sequentially
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_size, bias=False))
            in_dim = hidden_size

    def forward_graph(self, graph: DGLGraph, x: torch.Tensor | None = None) -> torch.Tensor:
        """Apply GCN layers to a single graph.

        Args:
            graph: DGL graph or bipartite block
            x: Optional node features. If None, extracts from graph.ndata['x']
                or graph.dstdata['x'] (for blocks).

        Returns:
            Node embeddings [N, hidden_size]
        """
        h = _graph_input(graph) if x is None else x
        for idx, layer in enumerate(self.layers):
            h = _gcn_message_pass(graph, h)
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def layerwise(self, blob_or_graph) -> list[torch.Tensor]:
        """Apply GCN to a sequence of graphs (STGraphBlob or iterable).

        Args:
            blob_or_graph: STGraphBlob (multi-frame) or single DGLGraph

        Returns:
            List of node embeddings, one per graph in sequence
        """
        outputs = []
        for graph in _graph_sequence(blob_or_graph):
            outputs.append(self.forward_graph(graph))
        return outputs
