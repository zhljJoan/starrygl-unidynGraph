from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.autograd as autograd
import torch.distributed as dist
from torch import Tensor


@dataclass(slots=True)
class StaticRoute:
    send_index: Tensor
    send_sizes: list[int]
    recv_sizes: list[int]
    group: Optional[dist.ProcessGroup] = None

    @property
    def send_len(self) -> int:
        return int(sum(self.send_sizes))

    @property
    def recv_len(self) -> int:
        return int(sum(self.recv_sizes))

    def to(self, device=None, dtype=None, non_blocking: bool = False, copy: bool = False) -> "StaticRoute":
        return StaticRoute(
            send_index=self.send_index.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            send_sizes=list(self.send_sizes),
            recv_sizes=list(self.recv_sizes),
            group=self.group,
        )


class StaticRouteComm:
    def __init__(self, group: Optional[dist.ProcessGroup] = None) -> None:
        self.group = group

    def forward(self, x: Tensor, route: StaticRoute | None, *, reverse: bool = False) -> Tensor:
        if route is None:
            return x
        group = route.group if route.group is not None else self.group
        return _RouteRecvFunction.apply(_RouteSendFunction.apply(x, route, reverse, group))


class _RouteSendFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, route: StaticRoute, reverse: bool, group: Optional[dist.ProcessGroup]):
        r_ctx = _RouteContext(route, reverse=bool(reverse), group=group)
        r_ctx.forward_send(x)
        key = torch.empty(0, dtype=torch.float32, device="cpu")
        key._route_ctx = r_ctx
        ctx.route_ctx = r_ctx
        return key

    @staticmethod
    def backward(ctx: Any, _grad_key: Tensor):
        return ctx.route_ctx.backward_recv(), None, None, None


class _RouteRecvFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, key: Tensor):
        r_ctx = key._route_ctx
        ctx.route_ctx = r_ctx
        return r_ctx.forward_recv()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        ctx.route_ctx.backward_send(grad_output)
        return torch.empty(0, dtype=torch.float32, device="cpu")


class _RouteContext:
    def __init__(self, route: StaticRoute, reverse: bool, group: Optional[dist.ProcessGroup]) -> None:
        self.route = route
        self.reverse = reverse
        self.group = dist.GroupMember.WORLD if group is None else group
        self.task: tuple[Tensor, Tensor, dist.Work] | None = None

    def _send_impl(self, x: Tensor) -> tuple[Tensor, Tensor, dist.Work]:
        y = x[self.route.send_index]
        out = torch.empty((self.route.recv_len, *x.shape[1:]), dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            out,
            y,
            self.route.recv_sizes,
            self.route.send_sizes,
            group=self.group,
            async_op=True,
        )
        return x, out, work

    def _recv_impl(self, x: Tensor) -> tuple[Tensor, Tensor, dist.Work]:
        n = x.size(0) - self.route.recv_len
        x_base = x[:n]
        y = x[n:]
        out = torch.empty((self.route.send_len, *x_base.shape[1:]), dtype=x.dtype, device=x.device)
        work = dist.all_to_all_single(
            out,
            y,
            self.route.send_sizes,
            self.route.recv_sizes,
            group=self.group,
            async_op=True,
        )
        return x_base, out, work

    @staticmethod
    def _send_post(x: Tensor, out: Tensor, work: dist.Work) -> Tensor:
        work.wait()
        return torch.cat([x, out], dim=0)

    def _recv_post(self, x: Tensor, out: Tensor, work: dist.Work) -> Tensor:
        work.wait()
        s = 0
        for size in self.route.send_sizes:
            if size:
                x[self.route.send_index[s : s + size]] += out[s : s + size]
            s += size
        return x

    @torch.no_grad()
    def forward_send(self, x: Tensor) -> None:
        self.task = self._recv_impl(x.detach()) if self.reverse else self._send_impl(x.detach())

    @torch.no_grad()
    def forward_recv(self) -> Tensor:
        assert self.task is not None
        x = self._recv_post(*self.task) if self.reverse else self._send_post(*self.task)
        self.task = None
        return x

    @torch.no_grad()
    def backward_send(self, grad: Tensor) -> None:
        self.task = self._send_impl(grad.detach()) if self.reverse else self._recv_impl(grad.detach())

    @torch.no_grad()
    def backward_recv(self) -> Tensor:
        assert self.task is not None
        grad = self._send_post(*self.task) if self.reverse else self._recv_post(*self.task)
        self.task = None
        return grad
