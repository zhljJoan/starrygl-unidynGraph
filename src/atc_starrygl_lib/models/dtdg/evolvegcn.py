from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .gcn import GCNConv
from .async_module import AsyncModule


class MatGRUCell(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.update = nn.Linear(in_feats + out_feats, out_feats)
        self.reset = nn.Linear(in_feats + out_feats, out_feats)
        self.htilda = nn.Linear(in_feats + out_feats, out_feats)

    def forward(self, prev_W: Tensor, inputs: Tensor) -> Tensor:
        inputs = inputs.repeat(prev_W.size(0), 1)
        xc = torch.cat([inputs, prev_W], dim=1)
        z_t = torch.sigmoid(self.update(xc))
        r_t = torch.sigmoid(self.reset(xc))
        g_c = torch.cat([inputs, r_t * prev_W], dim=1)
        h_tilde = torch.tanh(self.htilda(g_c))
        return z_t * prev_W + (1 - z_t) * h_tilde


class _GCNConvAgent(AsyncModule):
    async def async_forward(self, g: Any, w: Tensor) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g)
        if route_first and route is not None:
            x = await g.flare_async_route(x)
        x = GCNConv.msg_pass(g, x)
        return x @ w


class EvolveGCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        num_layers: int = 1,
        pool_mode: str = 'mean',
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pool_mode = pool_mode

        self.initial_weight = nn.Parameter(torch.empty(input_size, hidden_size))
        nn.init.xavier_uniform_(self.initial_weight)

        self.mat_gru = MatGRUCell(in_feats=input_size, out_feats=hidden_size)
        self.pool_proj = nn.Linear(input_size, input_size)
        self.out = nn.Linear(hidden_size, output_size)
        self._conv_agent = _GCNConvAgent()

    def _pool_graph_context(self, g: Any) -> Tensor:
        import dgl
        if g.is_block:
            ntype = "_N_dst" if 'x' in g.dstdata else "_N_src"
            g = dgl.block_to_graph(g)
            h_pool = dgl.max_nodes(g, 'x', ntype=ntype) if self.pool_mode == 'max' else dgl.mean_nodes(g, 'x', ntype=ntype)
        else:
            h_pool = dgl.max_nodes(g, 'x') if self.pool_mode == 'max' else dgl.mean_nodes(g, 'x')
        return self.pool_proj(h_pool)

    def forward(self, blob: Any, state: Optional[Tensor] = None):
        W = state if state is not None else self.initial_weight

        if self.training:
            W = blob[0].flare_fetch_state(W, end=self.input_size)
            Ws: List[Tensor] = []
            for g in blob:
                c_t = self._pool_graph_context(g)
                W = self.mat_gru(W, c_t)
                Ws.append(W)
                g.flare_store_state(W)
            outs = self._conv_agent.layerwise(blob, Ws)
            outs = [self.out(x) for x in outs]
            return outs, W
        else:
            g = blob
            c_t = self._pool_graph_context(g)
            W = self.mat_gru(W, c_t)
            x, _, _ = GCNConv.get_inputs(g)
            x = GCNConv.msg_pass(g, x) @ W
            return self.out(x), W
