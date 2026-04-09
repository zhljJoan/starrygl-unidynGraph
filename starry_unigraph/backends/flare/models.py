from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dgl import DGLGraph

from .stgraph_blob import STGraphBlob


def _graph_sequence(blob_or_graph: STGraphBlob | DGLGraph) -> list[DGLGraph]:
    if isinstance(blob_or_graph, STGraphBlob):
        return list(blob_or_graph)
    return [blob_or_graph]


def _graph_input(graph: DGLGraph) -> torch.Tensor:
    if graph.is_block:
        if "x" in graph.dstdata:
            return graph.dstdata["x"]
        return graph.srcdata["x"]
    return graph.ndata["x"]


def _graph_labels(graph: DGLGraph) -> torch.Tensor | None:
    if graph.is_block:
        if "y" in graph.dstdata:
            return graph.dstdata["y"]
        if "y" in graph.srcdata:
            return graph.srcdata["y"][: graph.num_dst_nodes()]
        return None
    return graph.ndata.get("y")


def _gcn_norm(graph: DGLGraph) -> torch.Tensor:
    if "gcn_norm" in graph.edata:
        return graph.edata["gcn_norm"].float()
    weights = graph.edata["w"].float() if "w" in graph.edata else torch.ones(graph.num_edges(), dtype=torch.float32, device=graph.device)
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
    with graph.local_scope():
        if graph.is_block:
            if x.size(0) == graph.num_src_nodes():
                src_x = x
            elif x.size(0) == graph.num_dst_nodes():
                route = getattr(graph, "route", None)
                if route is not None and route.send_index is not None and dist.is_available() and dist.is_initialized():
                    src_x = route.forward(x)
                else:
                    extra_count = graph.num_src_nodes() - graph.num_dst_nodes()
                    extra_x = torch.zeros(extra_count, x.size(-1), dtype=x.dtype, device=x.device)
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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_size, bias=False))
            in_dim = hidden_size

    def forward_graph(self, graph: DGLGraph, x: torch.Tensor | None = None) -> torch.Tensor:
        h = _graph_input(graph) if x is None else x
        for idx, layer in enumerate(self.layers):
            h = _gcn_message_pass(graph, h)
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def layerwise(self, blob_or_graph: STGraphBlob | DGLGraph) -> list[torch.Tensor]:
        outputs = []
        for graph in _graph_sequence(blob_or_graph):
            outputs.append(self.forward_graph(graph))
        return outputs


class MatGRUCell(nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.update = nn.Linear(in_feats + out_feats, out_feats)
        self.reset = nn.Linear(in_feats + out_feats, out_feats)
        self.htilda = nn.Linear(in_feats + out_feats, out_feats)

    def forward(self, prev_w: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.repeat(prev_w.size(0), 1)
        xc = torch.cat([inputs, prev_w], dim=1)
        z_t = torch.sigmoid(self.update(xc))
        r_t = torch.sigmoid(self.reset(xc))
        g_c = torch.cat([inputs, r_t * prev_w], dim=1)
        h_tilde = torch.tanh(self.htilda(g_c))
        return z_t * prev_w + (1 - z_t) * h_tilde


class FlareEvolveGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.initial_weight = nn.Parameter(torch.empty(input_size, hidden_size))
        nn.init.xavier_uniform_(self.initial_weight)
        self.mat_gru = MatGRUCell(in_feats=input_size, out_feats=hidden_size)
        self.pool_proj = nn.Linear(input_size, input_size)
        self.out = nn.Linear(hidden_size, output_size)

    def _pool_graph_context(self, graph: DGLGraph) -> torch.Tensor:
        x = _graph_input(graph)
        return self.pool_proj(x.mean(dim=0, keepdim=True))

    def forward(self, blob_or_graph: STGraphBlob | DGLGraph, state: torch.Tensor | None = None):
        weights = self.initial_weight if state is None else state
        outputs = []
        for graph in _graph_sequence(blob_or_graph):
            ctx = self._pool_graph_context(graph)
            weights = self.mat_gru(weights, ctx)
            h = _gcn_message_pass(graph, _graph_input(graph)) @ weights
            outputs.append(self.out(h))
        return outputs if isinstance(blob_or_graph, STGraphBlob) else outputs[-1], weights


class FlareTGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gcn = GCNStack(input_size, hidden_size * 3, num_layers=2)
        self.u_t = nn.Linear(hidden_size * 2, hidden_size)
        self.r_t = nn.Linear(hidden_size * 2, hidden_size)
        self.c_t = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, blob_or_graph: STGraphBlob | DGLGraph, state: torch.Tensor | None = None):
        outputs = []
        h = state
        for graph, urc in zip(_graph_sequence(blob_or_graph), self.gcn.layerwise(blob_or_graph)):
            if h is None:
                h = torch.zeros(urc.size(0), self.hidden_size, dtype=urc.dtype, device=urc.device)
            u, r, c = urc.chunk(3, dim=-1)
            u = torch.sigmoid(self.u_t(torch.cat([u, h], dim=-1)))
            r = torch.sigmoid(self.r_t(torch.cat([r, h], dim=-1)))
            c = torch.tanh(self.c_t(torch.cat([c, r * h], dim=-1)))
            h = u * h + (1 - u) * c
            outputs.append(self.out(h))
        return outputs if isinstance(blob_or_graph, STGraphBlob) else outputs[-1], h


class _LSTMCell(nn.Module):
    """Custom LSTM cell matching FlareDTDG reference (returns tuple state, supports tuple addition)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.i_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_t = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = state
        xh = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.i_t(xh))
        f = torch.sigmoid(self.f_t(xh))
        g = torch.tanh(self.g_t(xh))
        o = torch.sigmoid(self.o_t(xh))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class FlareMPNNLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gcn = GCNStack(input_size, hidden_size, num_layers=2)
        self.rnn = nn.ModuleList([
            _LSTMCell(hidden_size, hidden_size),
            _LSTMCell(hidden_size, hidden_size),
        ])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, blob_or_graph: STGraphBlob | DGLGraph, state=None):
        outputs = []
        inps = self.gcn.layerwise(blob_or_graph)
        h = state
        for graph, x in zip(_graph_sequence(blob_or_graph), inps):
            # Restore per-snapshot RNN state from manager (training) or pass through (eval)
            h = graph.flare_fetch_state(h)

            s1 = None if h is None else h[0:2]
            s2 = None if h is None else h[2:4]

            x, _ = s1 = self.rnn[0](x, s1)
            x, _ = s2 = self.rnn[1](x, s2)

            # s1 + s2: tuple concatenation → (h1, c1, h2, c2), stored as combined state
            h = s1 + s2

            graph.flare_store_state(h)
            outputs.append(self.out(x))

        if isinstance(blob_or_graph, STGraphBlob):
            return outputs, h
        else:
            return outputs[-1], h


@dataclass(frozen=True)
class FlareModelSpec:
    model_cls: type[nn.Module]


MODEL_SPECS: dict[str, FlareModelSpec] = {
    "evolvegcn": FlareModelSpec(FlareEvolveGCN),
    "tgcn": FlareModelSpec(FlareTGCN),
    "mpnn_lstm": FlareModelSpec(FlareMPNNLSTM),
    "gcn": FlareModelSpec(FlareTGCN),
}


def build_flare_model(model_name: str, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
    key = model_name.lower()
    if key not in MODEL_SPECS:
        raise KeyError(f"Unsupported flare model: {model_name}")
    return MODEL_SPECS[key].model_cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size)


def extract_graph_labels(blob_or_graph: STGraphBlob | DGLGraph) -> torch.Tensor | None:
    graphs = _graph_sequence(blob_or_graph)
    if not graphs:
        return None
    labels = _graph_labels(graphs[-1])
    if labels is None:
        return None
    return labels.view(-1, 1) if labels.dim() == 1 else labels
