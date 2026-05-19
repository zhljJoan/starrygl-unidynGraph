"""Autograd propagation route used by DTDG model layers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.autograd as autograd
import torch.distributed as dist
from torch import Tensor


@dataclass(slots=True)
class ChunkPropagationRoute:
    """Differentiable all-to-all route for GNN layer activations."""

    send_sizes: list[int]
    recv_sizes: list[int]
    send_index: Optional[Tensor] = None
    recv_src_rows: Optional[Tensor] = None
    append_recv: bool = True
    group: Optional[dist.ProcessGroup] = None

    @classmethod
    def empty(cls, num_parts: int = 1, append_recv: bool = True) -> "ChunkPropagationRoute":
        return cls(
            send_sizes=[0] * max(1, int(num_parts)),
            recv_sizes=[0] * max(1, int(num_parts)),
            send_index=None,
            recv_src_rows=None,
            append_recv=append_recv,
        )

    @property
    def send_len(self) -> int:
        return int(sum(self.send_sizes))

    @property
    def recv_len(self) -> int:
        return int(sum(self.recv_sizes))

    def forward(self, x: Tensor, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> Tensor:
        route_group = self.group if group is None else group
        return _PropagationAgent(self, reverse=reverse, group=route_group).forward(x)

    async def async_forward(self, x: Tensor, reverse: bool = False, group: Optional[dist.ProcessGroup] = None) -> Tensor:
        route_group = self.group if group is None else group
        return await _PropagationAgent(self, reverse=reverse, group=route_group).async_forward(x)

    def pin_memory(self) -> "ChunkPropagationRoute":
        return ChunkPropagationRoute(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.pin_memory(),
            recv_src_rows=None if self.recv_src_rows is None else self.recv_src_rows.pin_memory(),
            append_recv=self.append_recv,
            group=self.group,
        )

    def to(self, device: str | torch.device) -> "ChunkPropagationRoute":
        return ChunkPropagationRoute(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.to(device),
            recv_src_rows=None if self.recv_src_rows is None else self.recv_src_rows.to(device),
            append_recv=self.append_recv,
            group=self.group,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "parts": len(self.send_sizes),
            "send_len": self.send_len,
            "recv_len": self.recv_len,
            "send_sizes": [int(x) for x in self.send_sizes],
            "recv_sizes": [int(x) for x in self.recv_sizes],
            "has_send_index": self.send_index is not None,
            "has_recv_src_rows": self.recv_src_rows is not None,
            "append_recv": bool(self.append_recv),
        }


PropagationRoute = ChunkPropagationRoute


class _PropagationAgent:
    def __init__(
        self,
        route: ChunkPropagationRoute,
        reverse: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.route = route
        self.reverse = reverse
        self.group = dist.GroupMember.WORLD if group is None else group

    def forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None or not dist.is_available() or not dist.is_initialized():
            if self.route.append_recv:
                return x
            return x.new_empty((0, *x.shape[1:]))
        return self.recv(self.send(x))

    async def async_forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None or not dist.is_available() or not dist.is_initialized():
            if self.route.append_recv:
                return x
            return x.new_empty((0, *x.shape[1:]))
        ctx = self.send(x)
        await asyncio.sleep(0.0)
        return self.recv(ctx)

    def send(self, x: Tensor) -> Tensor:
        return _PropagationSendFunction.apply(x, self.route, self.reverse, self.group)

    @staticmethod
    def recv(ctx: Tensor) -> Tensor:
        return _PropagationRecvFunction.apply(ctx)


class _PropagationSendFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, route: ChunkPropagationRoute, reverse: bool = False, group=None):
        route_ctx = _PropagationContext(route, reverse=reverse, group=group)
        route_ctx.forward_send(x)
        key = torch.empty(0, dtype=torch.float32, device="cpu")
        key._chunk_route_ctx = route_ctx
        ctx.saved_route_ctx = route_ctx
        return key

    @staticmethod
    def backward(ctx, _):
        route_ctx: _PropagationContext = ctx.saved_route_ctx
        return route_ctx.backward_recv(), None, None, None


class _PropagationRecvFunction(autograd.Function):
    @staticmethod
    def forward(ctx, key: Tensor):
        route_ctx: _PropagationContext = key._chunk_route_ctx
        ctx.saved_route_ctx = route_ctx
        return route_ctx.forward_recv()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        route_ctx: _PropagationContext = ctx.saved_route_ctx
        route_ctx.backward_send(grad_output)
        key = torch.empty(0, dtype=torch.float32, device="cpu")
        key._chunk_route_ctx = route_ctx
        return key


class _PropagationContext:
    def __init__(self, route: ChunkPropagationRoute, reverse: bool = False, group=None) -> None:
        self.route = route
        self.reverse = reverse
        self.group = dist.GroupMember.WORLD if group is None else group
        self.send_buf: Optional[Tensor] = None
        self.recv_buf: Optional[Tensor] = None
        self.recv_work = None
        self.x_num_rows = 0
        self.local_x: Optional[Tensor] = None
        self.local_grad: Optional[Tensor] = None

    @property
    def send_sizes(self) -> list[int]:
        return self.route.recv_sizes if self.reverse else self.route.send_sizes

    @property
    def recv_sizes(self) -> list[int]:
        return self.route.send_sizes if self.reverse else self.route.recv_sizes

    def forward_send(self, x: Tensor) -> None:
        if self.route.send_index is None:
            raise RuntimeError("send_index is required for propagation send")
        self.x_num_rows = int(x.size(0))
        if self.route.append_recv:
            self.local_x = x
        self.send_buf = x[self.route.send_index].contiguous()
        self.recv_buf = torch.empty(sum(self.recv_sizes), x.size(1), dtype=x.dtype, device=x.device)
        self.recv_work = dist.all_to_all_single(
            self.recv_buf,
            self.send_buf,
            output_split_sizes=self.recv_sizes,
            input_split_sizes=self.send_sizes,
            group=self.group,
            async_op=True,
        )

    def forward_recv(self) -> Tensor:
        if self.recv_work is not None:
            self.recv_work.wait()
        if self.recv_buf is None:
            raise RuntimeError("forward_recv called before forward_send")
        recv_buf = self.recv_buf
        if self.route.recv_src_rows is not None:
            recv_buf = recv_buf[self.route.recv_src_rows.to(recv_buf.device)]
        if self.route.append_recv:
            if self.local_x is None:
                raise RuntimeError("append_recv requires local input from forward_send")
            return torch.cat([self.local_x, recv_buf], dim=0)
        return recv_buf

    def backward_send(self, grad_output: Tensor) -> None:
        if self.route.append_recv:
            self.local_grad = grad_output[: self.x_num_rows].contiguous()
            remote_grad = grad_output[self.x_num_rows :].contiguous()
        else:
            self.local_grad = None
            remote_grad = grad_output.contiguous()
        if self.route.recv_src_rows is not None and remote_grad.numel() > 0:
            inv = torch.empty_like(self.route.recv_src_rows, device=remote_grad.device)
            inv[self.route.recv_src_rows.to(remote_grad.device)] = torch.arange(
                int(self.route.recv_src_rows.numel()),
                dtype=torch.long,
                device=remote_grad.device,
            )
            remote_grad = remote_grad[inv]
        self.send_buf = remote_grad
        feature_dim = remote_grad.size(1) if remote_grad.dim() > 1 else 1
        self.recv_buf = torch.empty(sum(self.send_sizes), feature_dim, dtype=remote_grad.dtype, device=remote_grad.device)
        self.recv_work = dist.all_to_all_single(
            self.recv_buf,
            self.send_buf,
            output_split_sizes=self.send_sizes,
            input_split_sizes=self.recv_sizes,
            group=self.group,
            async_op=True,
        )

    def backward_recv(self) -> Tensor:
        if self.recv_work is not None:
            self.recv_work.wait()
        if self.recv_buf is None or self.route.send_index is None:
            raise RuntimeError("backward_recv called before backward_send")
        grad_x = torch.zeros(self.x_num_rows, self.recv_buf.size(1), dtype=self.recv_buf.dtype, device=self.recv_buf.device)
        grad_x.index_add_(0, self.route.send_index, self.recv_buf)
        if self.local_grad is not None:
            grad_x = grad_x + self.local_grad
        return grad_x
