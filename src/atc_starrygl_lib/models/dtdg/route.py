from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

import torch
import torch.autograd as autograd
import torch.distributed as dist
import dgl
from dgl.heterograph import DGLBlock
from torch import Tensor


__all__ = ["Route"]


class Route:
    def __init__(
        self,
        send_sizes: List[int],
        recv_sizes: List[int],
        send_index: Optional[Tensor] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.send_sizes = send_sizes
        self.recv_sizes = recv_sizes
        self.send_index = send_index
        self.group = group

    @property
    def send_len(self) -> int:
        return sum(self.send_sizes)

    @property
    def recv_len(self) -> int:
        return sum(self.recv_sizes)

    def forward(self, x: Tensor, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> Tensor:
        group = self.group if group is None else group
        return RouteAgent(self, reverse=reverse, group=group).forward(x)

    async def async_forward(self, x: Tensor, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> Tensor:
        group = self.group if group is None else group
        return await RouteAgent(self, reverse=reverse, group=group).async_forward(x)

    def send(self, x: Tensor, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> Tensor:
        group = self.group if group is None else group
        return RouteAgent(self, reverse=reverse, group=group).send(x)

    def recv(self, ctx: Tensor) -> Tensor:
        return RouteAgent.recv(ctx)

    def pin_memory(self, device=None) -> "Route":
        return type(self)(
            send_index=self.send_index.pin_memory(device=device) if self.send_index is not None else None,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            group=self.group,
        )

    def to(self, device=None, dtype=None, non_blocking=False, copy=False) -> "Route":
        return type(self)(
            send_index=self.send_index.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) if self.send_index is not None else None,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            group=self.group,
        )

    def __getstate__(self):
        return {
            "send_sizes": self.send_sizes,
            "recv_sizes": self.recv_sizes,
            "send_index": self.send_index,
        }

    def __setstate__(self, state):
        self.send_sizes = state["send_sizes"]
        self.recv_sizes = state["recv_sizes"]
        self.send_index = state["send_index"]
        self.group = None

    def __repr__(self):
        p = len(self.send_sizes) if self.send_sizes else -1
        return f"{type(self).__name__}(parts={p})"

    @classmethod
    def from_empty(cls, num_nodes: int, edge_index: Tensor, idtype=torch.int32) -> List[DGLBlock]:
        src = edge_index[0]
        dst = edge_index[1]
        g = dgl.create_block(
            (src, dst),
            num_src_nodes=num_nodes,
            num_dst_nodes=num_nodes,
            idtype=idtype,
        )
        node_ids = torch.arange(num_nodes, dtype=idtype, device=edge_index.device)
        edge_ids = torch.arange(g.num_edges(), dtype=idtype, device=edge_index.device)
        g.srcdata[dgl.NID] = node_ids
        g.dstdata[dgl.NID] = node_ids
        g.edata[dgl.EID] = edge_ids
        g.route = None
        return [g]

    @classmethod
    def from_graph(cls, node_parts, edge_index: Tensor, num_parts: Optional[int] = None, idtype=torch.int32) -> List[DGLBlock]:
        if not isinstance(node_parts, Tensor):
            num_nodes = int(node_parts)
            return cls.from_empty(num_nodes=num_nodes, edge_index=edge_index, idtype=idtype)

        if num_parts is None:
            num_parts = int(node_parts.max().item()) + 1

        dst_ids_list = []
        edge_ids_list = []
        dst_send_ids_list = []
        src_recv_ids_list = []

        xmp = torch.empty_like(node_parts)
        for k in range(num_parts):
            dst_ids = torch.where(node_parts == k)[0]
            edge_ids = torch.where(node_parts[edge_index[1]] == k)[0]
            xmp.zero_()
            xmp[edge_index[0, edge_ids]] = 1
            xmp[dst_ids] = 0
            src_ids = torch.where(xmp != 0)[0]
            src_recv_ids = [src_ids[node_parts[src_ids] == i] for i in range(num_parts)]
            dst_ids_list.append(dst_ids)
            edge_ids_list.append(edge_ids)
            src_recv_ids_list.append(src_recv_ids)

        for k in range(num_parts):
            dst_send_ids_list.append([src_recv_ids_list[i][k] for i in range(num_parts)])

        rets = []
        for k in range(num_parts):
            dst_ids = dst_ids_list[k]
            src_ids = torch.cat([dst_ids, torch.cat(src_recv_ids_list[k], dim=0)], dim=0)
            xmp.fill_(2**63 - 1)
            xmp[src_ids] = torch.arange(src_ids.numel(), dtype=xmp.dtype, device=xmp.device)
            dst_send_ind = xmp[torch.cat(dst_send_ids_list[k], dim=0)]
            dst_send_szs = [t.numel() for t in dst_send_ids_list[k]]
            src_recv_szs = [t.numel() for t in src_recv_ids_list[k]]
            edge_ids = edge_ids_list[k]
            src = xmp[edge_index[0, edge_ids]]
            dst = xmp[edge_index[1, edge_ids]]
            g = dgl.create_block(
                (src, dst),
                num_src_nodes=src_ids.numel(),
                num_dst_nodes=dst_ids.numel(),
                idtype=idtype,
            )
            g.srcdata[dgl.NID] = src_ids.to(dtype=idtype)
            g.dstdata[dgl.NID] = dst_ids.to(dtype=idtype)
            g.edata[dgl.EID] = edge_ids.to(dtype=idtype)
            g.route = cls(send_sizes=dst_send_szs, recv_sizes=src_recv_szs, send_index=dst_send_ind)
            rets.append(g)
        return rets


class RouteAgent:
    def __init__(self, route: Route, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> None:
        self.route = route
        self.group = group
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            return x
        return self.recv(self.send(x))

    async def async_forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            return x
        ctx = self.send(x)
        await asyncio.sleep(0.0)
        return self.recv(ctx)

    def send(self, x: Tensor) -> Tensor:
        if self.route.send_index is None:
            raise RuntimeError("Empty route is unsupported for send")
        return RouteSendFunction.apply(x, self.route, self.reverse, self.group)

    @staticmethod
    def recv(ctx: Tensor) -> Tensor:
        return RouteRecvFunction.apply(ctx)


class RouteSendFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, route: Route, reverse: bool = False, group: Optional[dist.ProcessGroup] = None):
        r_ctx = RouteContext(route, reverse=reverse, group=group)
        r_ctx.forward_send(x)
        ret = torch.empty(0, dtype=torch.float32, device="cpu")
        ret._r_ctx = r_ctx
        ctx.saved_r_ctx = r_ctx
        return ret

    @staticmethod
    def backward(ctx, _):
        r_ctx: RouteContext = ctx.saved_r_ctx
        return r_ctx.backward_recv(), None, None, None


class RouteRecvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, key: Tensor):
        r_ctx: RouteContext = key._r_ctx
        ctx.saved_r_ctx = r_ctx
        return r_ctx.forward_recv()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        r_ctx: RouteContext = ctx.saved_r_ctx
        r_ctx.backward_send(grad_output)
        return torch.empty(0, dtype=torch.float32, device="cpu")


class RouteContext:
    def __init__(self, route: Route, reverse: bool = False, group: Optional[dist.ProcessGroup] = None):
        self.route = route
        self.group = dist.GroupMember.WORLD if group is None else group
        self.reverse = reverse
        self.task = None

    def _send_impl(self, x: Tensor):
        y = x[self.route.send_index]
        outs = torch.empty(self.route.recv_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            outs, y,
            self.route.recv_sizes,
            self.route.send_sizes,
            group=self.group,
            async_op=True,
        )
        return x, outs, work

    def _send_post(self, x: Tensor, outs: Tensor, work: dist.Work):
        work.wait()
        return torch.cat([x, outs], dim=0)

    def _recv_impl(self, x: Tensor):
        n = x.size(0) - self.route.recv_len
        x, y = x[:n], x[n:]
        outs = torch.empty(self.route.send_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            outs, y,
            self.route.send_sizes,
            self.route.recv_sizes,
            group=self.group,
            async_op=True,
        )
        return x, outs, work

    def _recv_post(self, x: Tensor, outs: Tensor, work: dist.Work):
        work.wait()
        s = 0
        for sz in self.route.send_sizes:
            x[self.route.send_index[s:s + sz]] += outs[s:s + sz]
            s += sz
        return x

    @torch.no_grad()
    def forward_send(self, x: Tensor):
        assert self.task is None
        x = x.detach()
        self.task = self._recv_impl(x) if self.reverse else self._send_impl(x)

    @torch.no_grad()
    def forward_recv(self):
        assert not isinstance(self.task, Tensor)
        x = self._recv_post(*self.task) if self.reverse else self._send_post(*self.task)
        self.task = None
        return x

    @torch.no_grad()
    def backward_send(self, g: Tensor):
        assert self.task is None
        g = g.detach()
        self.task = self._send_impl(g) if self.reverse else self._recv_impl(g)

    @torch.no_grad()
    def backward_recv(self):
        assert not isinstance(self.task, Tensor)
        g = self._send_post(*self.task) if self.reverse else self._recv_post(*self.task)
        self.task = None
        return g
