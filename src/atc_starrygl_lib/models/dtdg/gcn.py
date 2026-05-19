from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from torch import Tensor

from .async_module import AsyncModule
from .route import Route


class GCNConv(AsyncModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shortcut = bool(shortcut)

        w_in = self.in_features * 2 if shortcut else self.in_features
        self.weight = nn.Parameter(torch.empty(w_in, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    @staticmethod
    def gcn_norm(g: DGLGraph) -> Tensor:
        if 'gcn_norm' in g.edata:
            return g.edata['gcn_norm']
        if g.is_block:
            with g.local_scope():
                g.dstdata['s'] = g.in_degrees().float()
                g.apply_edges(fn.e_div_v('w', 's', 'x') if 'w' in g.edata else lambda e: {'x': 1.0 / e.dst['s'].clamp(min=1)})
                x = g.edata.get('x', torch.ones(g.num_edges(), device=g.device))
        else:
            with g.local_scope():
                g.ndata['s'] = g.in_degrees().float()
                g.ndata['t'] = g.out_degrees().float()
                g.apply_edges(fn.u_mul_v('t', 's', 'x'))
                x = torch.rsqrt(g.edata['x'].clamp(min=1))
        return x.nan_to_num_(0.0)

    @staticmethod
    def msg_pass(g: DGLGraph, x: Tensor) -> Tensor:
        with g.local_scope():
            if g.is_block:
                g.srcdata['x'] = x
            else:
                g.ndata['x'] = x
            if 'gcn_norm' not in g.edata:
                g.edata['gcn_norm'] = GCNConv.gcn_norm(g)
            g.update_all(fn.u_mul_e('x', 'gcn_norm', 'm'), fn.sum('m', 'x'))
            return g.dstdata['x'] if g.is_block else g.ndata['x']

    @staticmethod
    def get_inputs(
        g: DGLGraph,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Route], bool]:
        route: Optional[Route] = getattr(g, "route", None)
        if x is None:
            if g.is_block:
                if 'x' in g.dstdata:
                    return g.dstdata['x'], route, True
                else:
                    return g.srcdata['x'], route, False
            else:
                return g.ndata['x'], route, False
        else:
            if g.is_block:
                if x.size(0) == g.num_src_nodes():
                    route_first = False
                else:
                    assert x.size(0) == g.num_dst_nodes()
                    route_first = True
            else:
                assert x.size(0) == g.num_nodes()
                route_first = False
            return x, route, route_first

    async def async_forward(
        self,
        g: DGLGraph,
        x: Tensor,
        route: Optional[Route] = None,
    ) -> Tensor:
        _x = x
        if route is None:
            await self.yield_forward()
        else:
            x = await route.async_forward(x)
        x = self.msg_pass(g, x)
        if self.shortcut:
            x = torch.cat([x, _x[:x.size(0)]], dim=-1)
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class GCN(AsyncModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        bias: bool = True,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_layers = int(num_layers)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_features if i == 0 else out_features
            self.convs.append(GCNConv(in_ch, out_features, bias=bias, shortcut=shortcut))

    async def async_forward(self, g: DGLGraph, x: Optional[Tensor] = None) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g, x)
        for i, conv in enumerate(self.convs):
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                x = F.relu(x)
                x = await conv.async_forward(g, x, route=route)
        return x
