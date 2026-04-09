from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import dgl
import torch
import torch.autograd as autograd
import torch.distributed as dist
from dgl.heterograph import DGLBlock


@dataclass
class Route:
    send_sizes: list[int]
    recv_sizes: list[int]
    send_index: torch.Tensor | None = None
    group: dist.ProcessGroup | None = None

    @property
    def send_len(self) -> int:
        return sum(self.send_sizes)

    @property
    def recv_len(self) -> int:
        return sum(self.recv_sizes)

    def forward(self, x: torch.Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return RouteAgent(self, reverse=reverse, group=self.group if group is None else group).forward(x)

    async def async_forward(self, x: torch.Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return await RouteAgent(self, reverse=reverse, group=self.group if group is None else group).async_forward(x)

    def send(self, x: torch.Tensor, reverse: bool = False, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return RouteAgent(self, reverse=reverse, group=self.group if group is None else group).send(x)

    def recv(self, ctx: torch.Tensor) -> torch.Tensor:
        return RouteAgent.recv(ctx)

    def pin_memory(self, device: str | None = None) -> Route:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.pin_memory(device=device),
            group=self.group,
        )

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Route:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            group=self.group,
        )

    def describe(self) -> dict[str, Any]:
        active_parts = [idx for idx, size in enumerate(self.recv_sizes) if int(size) > 0]
        return {
            "parts": len(self.send_sizes),
            "send_len": self.send_len,
            "recv_len": self.recv_len,
            "send_sizes": [int(size) for size in self.send_sizes],
            "recv_sizes": [int(size) for size in self.recv_sizes],
            "has_send_index": self.send_index is not None,
            "active_recv_parts": active_parts,
        }

    @classmethod
    def from_empty(cls, num_nodes: int, edge_index: torch.Tensor, idtype: torch.dtype = torch.int32) -> list[DGLBlock]:
        src = edge_index[0]
        dst = edge_index[1]
        graph = dgl.create_block((src, dst), num_src_nodes=num_nodes, num_dst_nodes=num_nodes, idtype=idtype)
        node_ids = torch.arange(num_nodes, dtype=idtype, device=edge_index.device)
        edge_ids = torch.arange(graph.num_edges(), dtype=idtype, device=edge_index.device)
        graph.srcdata[dgl.NID] = node_ids
        graph.dstdata[dgl.NID] = node_ids
        graph.edata[dgl.EID] = edge_ids
        graph.route = None
        return [graph]

    @classmethod
    def from_graph(
        cls,
        node_parts: torch.Tensor | int,
        edge_index: torch.Tensor,
        num_parts: int | None = None,
        idtype: torch.dtype = torch.int32,
    ) -> list[DGLBlock]:
        if not isinstance(node_parts, torch.Tensor):
            return cls.from_empty(num_nodes=int(node_parts), edge_index=edge_index, idtype=idtype)

        node_parts = node_parts.long()
        if num_parts is None:
            num_parts = int(node_parts.max().item() + 1) if node_parts.numel() > 0 else 1
        num_parts = max(1, int(num_parts))

        dst_ids_list: list[torch.Tensor] = []
        edge_ids_list: list[torch.Tensor] = []
        dst_send_ids_list: list[list[torch.Tensor]] = []
        src_recv_ids_list: list[list[torch.Tensor]] = []
        xmap = torch.empty_like(node_parts)

        for part_id in range(num_parts):
            dst_ids = torch.where(node_parts == part_id)[0]
            edge_ids = torch.where(node_parts[edge_index[1]] == part_id)[0]
            xmap.zero_()
            xmap[edge_index[0, edge_ids]] = 1
            xmap[dst_ids] = 0
            src_ids = torch.where(xmap != 0)[0]
            src_recv_ids = [src_ids[node_parts[src_ids] == src_part] for src_part in range(num_parts)]
            dst_ids_list.append(dst_ids)
            edge_ids_list.append(edge_ids)
            src_recv_ids_list.append(src_recv_ids)

        for part_id in range(num_parts):
            dst_send_ids = [src_recv_ids_list[src_part][part_id] for src_part in range(num_parts)]
            dst_send_ids_list.append(dst_send_ids)

        blocks: list[DGLBlock] = []
        for part_id in range(num_parts):
            dst_ids = dst_ids_list[part_id]
            remote_src = torch.cat(src_recv_ids_list[part_id], dim=0)
            src_ids = torch.cat([dst_ids, remote_src], dim=0)

            xmap.fill_(2**63 - 1)
            xmap[src_ids] = torch.arange(src_ids.numel(), dtype=xmap.dtype, device=xmap.device)

            dst_send_cat = torch.cat(dst_send_ids_list[part_id], dim=0)
            dst_send_ind = xmap[dst_send_cat] if dst_send_cat.numel() > 0 else torch.zeros((0,), dtype=torch.long, device=xmap.device)
            dst_send_szs = [int(ids.numel()) for ids in dst_send_ids_list[part_id]]
            src_recv_szs = [int(ids.numel()) for ids in src_recv_ids_list[part_id]]

            edge_ids = edge_ids_list[part_id]
            src = xmap[edge_index[0, edge_ids]]
            dst = xmap[edge_index[1, edge_ids]]
            graph = dgl.create_block((src, dst), num_src_nodes=src_ids.numel(), num_dst_nodes=dst_ids.numel(), idtype=idtype)
            graph.srcdata[dgl.NID] = src_ids.to(dtype=idtype)
            graph.dstdata[dgl.NID] = dst_ids.to(dtype=idtype)
            graph.edata[dgl.EID] = edge_ids.to(dtype=idtype)
            graph.route = cls(send_sizes=dst_send_szs, recv_sizes=src_recv_szs, send_index=dst_send_ind.long())
            blocks.append(graph)
        return blocks


class RouteAgent:
    def __init__(self, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None) -> None:
        self.route = route
        self.group = dist.GroupMember.WORLD if group is None else group
        self.reverse = reverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.route.send_index is None:
            return x
        return self.recv(self.send(x))

    async def async_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.route.send_index is None:
            return x
        ctx = self.send(x)
        await asyncio.sleep(0.0)
        return self.recv(ctx)

    def send(self, x: torch.Tensor) -> torch.Tensor:
        if self.route.send_index is None:
            raise RuntimeError("Empty route is unsupported for send")
        return RouteSendFunction.apply(x, self.route, self.reverse, self.group)

    @staticmethod
    def recv(ctx: torch.Tensor) -> torch.Tensor:
        return RouteRecvFunction.apply(ctx)


class RouteSendFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None):
        route_ctx = RouteContext(route, reverse=reverse, group=group)
        route_ctx.forward_send(x)
        ret = torch.empty(0, dtype=torch.float32, device="cpu")
        ret._route_ctx = route_ctx
        ctx.saved_route_ctx = route_ctx
        return ret

    @staticmethod
    def backward(ctx, _):
        route_ctx: RouteContext = ctx.saved_route_ctx
        return route_ctx.backward_recv(), None, None, None


class RouteRecvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, key: torch.Tensor):
        route_ctx: RouteContext = key._route_ctx
        ctx.saved_route_ctx = route_ctx
        return route_ctx.forward_recv()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        route_ctx: RouteContext = ctx.saved_route_ctx
        route_ctx.backward_send(grad_output)
        return torch.empty(0, dtype=torch.float32, device="cpu")


class RouteContext:
    def __init__(self, route: Route, reverse: bool = False, group: dist.ProcessGroup | None = None):
        self.route = route
        self.group = dist.GroupMember.WORLD if group is None else group
        self.reverse = reverse
        self.task: tuple[torch.Tensor, torch.Tensor, dist.Work] | None = None

    def _send_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dist.Work]:
        send_buf = x[self.route.send_index]
        recv_buf = torch.empty(self.route.recv_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(recv_buf, send_buf, self.route.recv_sizes, self.route.send_sizes, group=self.group, async_op=True)
        return x, recv_buf, work

    def _send_post(self, x: torch.Tensor, recv_buf: torch.Tensor, work: dist.Work) -> torch.Tensor:
        work.wait()
        return torch.cat([x, recv_buf], dim=0)

    def _recv_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dist.Work]:
        local_len = x.size(0) - self.route.recv_len
        local_x = x[:local_len]
        remote_x = x[local_len:]
        recv_buf = torch.empty(self.route.send_len, *x.shape[1:], dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(recv_buf, remote_x, self.route.send_sizes, self.route.recv_sizes, group=self.group, async_op=True)
        return local_x, recv_buf, work

    def _recv_post(self, x: torch.Tensor, recv_buf: torch.Tensor, work: dist.Work) -> torch.Tensor:
        work.wait()
        offset = 0
        for size in self.route.send_sizes:
            if size > 0:
                x[self.route.send_index[offset : offset + size]] += recv_buf[offset : offset + size]
            offset += size
        return x

    @torch.no_grad()
    def forward_send(self, x: torch.Tensor) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before using Route.forward")
        self.task = self._recv_impl(x.detach()) if self.reverse else self._send_impl(x.detach())

    @torch.no_grad()
    def forward_recv(self) -> torch.Tensor:
        if self.task is None:
            raise RuntimeError("RouteContext has no pending task")
        x = self._recv_post(*self.task) if self.reverse else self._send_post(*self.task)
        self.task = None
        return x

    @torch.no_grad()
    def backward_send(self, grad: torch.Tensor) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before using Route.backward")
        self.task = self._send_impl(grad.detach()) if self.reverse else self._recv_impl(grad.detach())

    @torch.no_grad()
    def backward_recv(self) -> torch.Tensor:
        if self.task is None:
            raise RuntimeError("RouteContext has no pending task")
        grad = self._send_post(*self.task) if self.reverse else self._recv_post(*self.task)
        self.task = None
        return grad
