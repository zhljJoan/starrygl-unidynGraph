"""Async GNN modules with distributed routing support.

Implements FlareDTDG-style async GNN layers that integrate with ChunkPropagationRoute
for distributed training with communication/computation overlap.

Reference: ~/FlareDTDG/flare2/nn/graphconv.py, async_module.py
"""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from torch import Tensor

from starry_unigraph.models.layers.route import ChunkPropagationRoute


__all__ = [
    "AsyncModule",
    "GCNConv",
    "GCN",
    "GraphSAGEConv",
    "GraphSAGE",
    "GATConv",
    "GAT",
]


class AsyncModule(nn.Module):
    """Base class for async neural network modules.

    Supports async forward passes for communication/computation overlap.
    """

    async def async_forward(self, *args, **kwargs):
        """Async forward pass. Subclasses must implement this."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Synchronous wrapper around async_forward."""
        return asyncio.run(self.async_forward(*args, **kwargs))

    async def yield_forward(self):
        """Yield control flow to other async tasks."""
        await asyncio.sleep(0)


class GCNConv(AsyncModule):
    """Graph Convolutional Network layer with async routing support.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to use bias
        shortcut: Whether to use shortcut connection (concat input)
    """

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
        w_out = self.out_features

        self.weight = nn.Parameter(torch.empty(w_in, w_out))
        if bias:
            self.bias = nn.Parameter(torch.empty(w_out))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    @staticmethod
    def gcn_norm(g: DGLGraph) -> Tensor:
        """Compute GCN normalization: 1/sqrt(deg_src * deg_dst)."""
        if 'gcn_norm' in g.edata:
            return g.edata['gcn_norm']

        if g.is_block:
            # Block: use in-degrees of dst nodes
            if 'w' in g.edata:
                with g.local_scope():
                    g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                    g.apply_edges(fn.e_div_v('w', 's', 'x'))
                    x = g.edata['x']
            else:
                with g.local_scope():
                    g.dstdata['s'] = g.in_degrees().float()
                    g.apply_edges(fn.e_div_v('w', 's', 'x'))
                    x = g.edata['x']
        else:
            # Full graph: use sqrt(deg_src * deg_dst)
            if 'w' in g.edata:
                r = dgl.reverse(g, copy_ndata=False, copy_edata=True)
                r.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                r = r.ndata['s']

                with g.local_scope():
                    g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
                    g.ndata['t'] = r
                    g.apply_edges(fn.u_mul_v('t', 's', 'x'))
                    x = g.edata['w'] * torch.rsqrt(g.edata['x'])
            else:
                with g.local_scope():
                    g.ndata['s'] = g.in_degrees().float()
                    g.ndata['t'] = g.out_degrees().float()
                    g.apply_edges(fn.u_mul_v('t', 's', 'x'))
                    x = torch.rsqrt(g.edata['x'])

        x = x.nan_to_num_(0.0)
        return x

    @staticmethod
    def msg_pass(g: DGLGraph, x: Tensor) -> Tensor:
        """Message passing with GCN normalization."""
        with g.local_scope():
            if g.is_block:
                g.srcdata['x'] = x
            else:
                g.ndata['x'] = x

            if 'gcn_norm' not in g.edata:
                g.edata['gcn_norm'] = GCNConv.gcn_norm(g)

            g.update_all(fn.u_mul_e('x', 'gcn_norm', 'm'), fn.sum('m', 'x'))

            if g.is_block:
                x = g.dstdata['x']
            else:
                x = g.ndata['x']
        return x

    @staticmethod
    def get_inputs(
        g: DGLGraph,
        x: Tensor | None = None,
    ) -> Tuple[Tensor, ChunkPropagationRoute | None, bool]:
        """Extract input features and route from graph.

        Returns:
            (x, route, route_first):
                x: Input features
                route: ChunkPropagationRoute if available
                route_first: Whether to apply route before message passing
        """
        route: ChunkPropagationRoute | None = getattr(g, "route", None)

        if x is None:
            if g.is_block:
                if 'x' in g.dstdata:
                    route_first = True
                    x = g.dstdata['x']
                else:
                    route_first = False
                    x = g.srcdata['x']
            else:
                x = g.ndata['x']
                route_first = False
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
        route: ChunkPropagationRoute | None = None,
    ) -> Tensor:
        """Async forward with optional routing.

        Args:
            g: DGL graph or block
            x: Input features
            route: Optional ChunkPropagationRoute for distributed training

        Returns:
            Output features after GCN convolution
        """
        _x = x

        if route is None:
            # Yield control to other tasks
            await self.yield_forward()
        else:
            # Async routing (communication/computation overlap)
            x = await route.async_forward(x)

        # Message passing
        x = self.msg_pass(g, x)

        # Shortcut connection
        if self.shortcut:
            x = torch.cat([x, _x[:x.size(0)]], dim=-1)

        # Linear transformation
        x = x @ self.weight
        if isinstance(self.bias, Tensor):
            x = x + self.bias

        return x


class GCN(AsyncModule):
    """Multi-layer GCN with async routing support.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_layers: Number of GCN layers
        bias: Whether to use bias
        shortcut: Whether to use shortcut connections
    """

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
            in_ch = self.in_features if i == 0 else self.out_features
            out_ch = self.out_features
            self.convs.append(GCNConv(in_ch, out_ch, bias=bias, shortcut=shortcut))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    async def async_forward(
        self,
        g: DGLGraph,
        x: Tensor | None = None,
    ) -> Tensor:
        """Async forward through all GCN layers.

        Args:
            g: DGL graph or block (may have route attached)
            x: Input features (optional, can be in g.srcdata/dstdata)

        Returns:
            Output features after all GCN layers
        """
        x, route, route_first = GCNConv.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                x = F.relu(x)
                r = route
                x = await conv.async_forward(g, x, route=r)

        return x


class GraphSAGEConv(AsyncModule):
    """GraphSAGE layer with async routing support.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        aggregator: "mean", "pool", or "lstm"
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.aggregator = aggregator

        if aggregator == "pool":
            self.fc_pool = nn.Linear(in_features, in_features)
        elif aggregator == "lstm":
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)

        self.fc_self = nn.Linear(in_features, out_features, bias=False)
        self.fc_neigh = nn.Linear(in_features, out_features, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc_self.weight)
        nn.init.xavier_uniform_(self.fc_neigh.weight)
        if hasattr(self, 'fc_pool'):
            nn.init.xavier_uniform_(self.fc_pool.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def aggregate(self, g: DGLGraph, x: Tensor) -> Tensor:
        """Aggregate neighbor features."""
        with g.local_scope():
            if g.is_block:
                g.srcdata['x'] = x
            else:
                g.ndata['x'] = x

            if self.aggregator == "mean":
                g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'neigh'))
            elif self.aggregator == "pool":
                g.srcdata['x'] = F.relu(self.fc_pool(x))
                g.update_all(fn.copy_u('x', 'm'), fn.max('m', 'neigh'))
            elif self.aggregator == "lstm":
                # LSTM aggregation (requires sorting by node ID)
                g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'neigh'))
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator}")

            if g.is_block:
                neigh = g.dstdata['neigh']
            else:
                neigh = g.ndata['neigh']

        return neigh

    async def async_forward(
        self,
        g: DGLGraph,
        x: Tensor,
        route: ChunkPropagationRoute | None = None,
    ) -> Tensor:
        """Async forward with optional routing."""
        if route is None:
            await self.yield_forward()
        else:
            x = await route.async_forward(x)

        # Self features
        if g.is_block:
            x_self = x[:g.num_dst_nodes()]
        else:
            x_self = x

        # Aggregate neighbor features
        neigh = self.aggregate(g, x)

        # Combine self and neighbor
        out = self.fc_self(x_self) + self.fc_neigh(neigh)

        if self.bias is not None:
            out = out + self.bias

        return F.relu(out)


class GraphSAGE(AsyncModule):
    """Multi-layer GraphSAGE with async routing support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.in_features if i == 0 else self.out_features
            out_ch = self.out_features
            self.convs.append(GraphSAGEConv(in_ch, out_ch, aggregator=aggregator, bias=bias))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    async def async_forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                r = route
                x = await conv.async_forward(g, x, route=r)

        return x


class GATConv(AsyncModule):
    """Graph Attention Network layer (simplified version).

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_heads: Number of attention heads
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_heads = int(num_heads)

        self.fc = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_features))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features * num_heads))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    async def async_forward(
        self,
        g: DGLGraph,
        x: Tensor,
        route: ChunkPropagationRoute | None = None,
    ) -> Tensor:
        """Async forward with attention."""
        if route is None:
            await self.yield_forward()
        else:
            x = await route.async_forward(x)

        # Linear transformation
        h = self.fc(x).view(-1, self.num_heads, self.out_features)

        # Attention (simplified - full version would use edge attention)
        with g.local_scope():
            if g.is_block:
                g.srcdata['h'] = h
                g.dstdata['h'] = h[:g.num_dst_nodes()]
            else:
                g.ndata['h'] = h

            # Mean aggregation (simplified GAT)
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))

            if g.is_block:
                out = g.dstdata['h']
            else:
                out = g.ndata['h']

        out = out.view(-1, self.num_heads * self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return F.elu(out)


class GAT(AsyncModule):
    """Multi-layer GAT with async routing support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.in_features if i == 0 else self.out_features * num_heads
            out_ch = self.out_features
            self.convs.append(GATConv(in_ch, out_ch, num_heads=num_heads, bias=bias))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    async def async_forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                r = route
                x = await conv.async_forward(g, x, route=r)

        return x
