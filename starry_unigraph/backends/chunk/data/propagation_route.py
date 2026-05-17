"""Chunk-local copy of the DTDG autograd propagation route primitive.

This keeps chunk's high-level API independent from ``backends.dtdg`` while
preserving the same call shape for model-layer message propagation:

    h_remote_appended = route.forward(h)

The route is for activation/message propagation and is distinct from fetch and
state-sync plans.
"""

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
    group: Optional[dist.ProcessGroup] = None

    @property
    def send_len(self) -> int:
        return int(sum(self.send_sizes))

    @property
    def recv_len(self) -> int:
        return int(sum(self.recv_sizes))

    def forward(
        self,
        x: Tensor,
        reverse: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Tensor:
        return _PropagationAgent(self, reverse=reverse, group=self.group if group is None else group).forward(x)

    async def async_forward(
        self,
        x: Tensor,
        reverse: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Tensor:
        return await _PropagationAgent(self, reverse=reverse, group=self.group if group is None else group).async_forward(x)

    def pin_memory(self) -> "ChunkPropagationRoute":
        return ChunkPropagationRoute(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.pin_memory(),
            group=self.group,
        )

    def to(self, device: str | torch.device) -> "ChunkPropagationRoute":
        return ChunkPropagationRoute(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.to(device),
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
        }


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
            return x
        return self.recv(self.send(x))

    async def async_forward(self, x: Tensor) -> Tensor:
        if self.route.send_index is None or not dist.is_available() or not dist.is_initialized():
            return x
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
        self.send_work = None
        self.recv_work = None
        self.x_num_rows = 0

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
        return self.recv_buf

    def backward_send(self, grad_output: Tensor) -> None:
        self.send_buf = grad_output.contiguous()
        feature_dim = grad_output.size(1) if grad_output.dim() > 1 else 1
        self.recv_buf = torch.empty(sum(self.send_sizes), feature_dim, dtype=grad_output.dtype, device=grad_output.device)
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
        return grad_x
