from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .dist_index import dist_index_loc


@dataclass(slots=True)
class AsyncTensorHandle:
    works: list[Any]
    recv_tensors: tuple[Tensor, ...]
    finalize: Optional[Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]] = None

    def wait(self) -> tuple[Tensor, ...]:
        for work in self.works:
            work.wait()
        if self.finalize is not None:
            return self.finalize(self.recv_tensors)
        return self.recv_tensors


@dataclass(slots=True)
class A2APlan:
    send_counts: Tensor
    recv_counts: Tensor
    send_splits: list[int]
    recv_splits: list[int]
    send_total: int
    recv_total: int


def _ptr_from_counts(counts: Tensor) -> Tensor:
    ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=counts.device)
    ptr[1:] = counts.to(torch.long).cumsum(0)
    return ptr


def _splits_from_counts(counts: Tensor) -> list[int]:
    return [int(x) for x in counts.detach().cpu().tolist()]


def _counts_from_ptr(ptr: Tensor) -> Tensor:
    return (ptr[1:] - ptr[:-1]).long().contiguous()


class DynamicA2AComm:
    """Thin variable-size all-to-all helper.

    The first count exchange is explicit and reusable by all tensors aligned to
    the same rank grouping.
    """

    def __init__(self, device: torch.device | str, group: Optional[dist.ProcessGroup] = None) -> None:
        self.device = torch.device(device)
        self.group = group

    @property
    def world_size(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size(group=self.group)

    @property
    def rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank(group=self.group)

    def exchange_counts(self, send_counts: Tensor) -> A2APlan:
        send_counts = send_counts.to(self.device, dtype=torch.long).contiguous()
        if self.world_size <= 1:
            recv_counts = send_counts.clone()
        else:
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        return A2APlan(
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_splits=_splits_from_counts(send_counts),
            recv_splits=_splits_from_counts(recv_counts),
            send_total=int(send_counts.sum().item()),
            recv_total=int(recv_counts.sum().item()),
        )

    def submit_a2a(self, plan: A2APlan, tensor: Tensor) -> AsyncTensorHandle:
        send = tensor.to(self.device).contiguous()
        recv = send.new_empty((plan.recv_total, *send.shape[1:]))
        if self.world_size <= 1:
            recv.copy_(send)
            return AsyncTensorHandle(works=[], recv_tensors=(recv,))
        work = dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=plan.recv_splits,
            input_split_sizes=plan.send_splits,
            group=self.group,
            async_op=True,
        )
        return AsyncTensorHandle(works=[work], recv_tensors=(recv,))

    def submit_a2a_reverse(self, plan: A2APlan, tensor: Tensor) -> AsyncTensorHandle:
        send = tensor.to(self.device).contiguous()
        recv = send.new_empty((plan.send_total, *send.shape[1:]))
        if self.world_size <= 1:
            recv.copy_(send)
            return AsyncTensorHandle(works=[], recv_tensors=(recv,))
        work = dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=plan.send_splits,
            input_split_sizes=plan.recv_splits,
            group=self.group,
            async_op=True,
        )
        return AsyncTensorHandle(works=[work], recv_tensors=(recv,))


class DynamicFetchComm(DynamicA2AComm):
    """Request-response fetch for row-indexed feature/memory/mailbox state."""

    def submit_row_fetch(
        self,
        read_index: Tensor,
        read_ptr: Tensor,
        gather_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
        *,
        time_slices: Tensor | None = None,
    ) -> AsyncTensorHandle:
        read_index = read_index.to(self.device, dtype=torch.long).contiguous()
        read_ptr = read_ptr.to(self.device, dtype=torch.long).contiguous()
        rows = dist_index_loc(read_index).long().contiguous()
        send_counts = _counts_from_ptr(read_ptr)

        if self.world_size <= 1:
            return AsyncTensorHandle(
                works=[],
                recv_tensors=(gather_fn(rows, None if time_slices is None else time_slices.to(self.device)),),
            )

        rank = self.rank
        local_start = int(read_ptr[rank].item())
        local_end = int(read_ptr[rank + 1].item())
        local_time = None if time_slices is None else time_slices[local_start:local_end].to(self.device)
        local_payload = gather_fn(rows[local_start:local_end], local_time).to(self.device)

        remote_counts = send_counts.clone()
        remote_counts[rank] = 0
        if int(remote_counts.sum().item()) == 0:
            out = local_payload.new_empty((read_index.numel(), *local_payload.shape[1:]))
            out[local_start:local_end] = local_payload
            return AsyncTensorHandle(works=[], recv_tensors=(out,))

        plan = self.exchange_counts(remote_counts)
        remote_rows = torch.cat((rows[:local_start], rows[local_end:]), dim=0).contiguous()
        row_handle = self.submit_a2a(plan, remote_rows)
        (request_rows,) = row_handle.wait()

        request_time = None
        if time_slices is not None:
            remote_time = torch.cat((time_slices[:local_start], time_slices[local_end:]), dim=0).to(self.device)
            time_handle = self.submit_a2a(plan, remote_time)
            (request_time,) = time_handle.wait()

        response = gather_fn(request_rows, request_time).to(self.device).contiguous()
        resp_handle = self.submit_a2a_reverse(plan, response)

        def _finalize(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            (remote_payload,) = tensors
            out = remote_payload.new_empty((read_index.numel(), *remote_payload.shape[1:]))
            if local_end > local_start:
                out[local_start:local_end] = local_payload
            if local_start > 0:
                out[:local_start] = remote_payload[:local_start]
            if local_end < read_index.numel():
                out[local_end:] = remote_payload[local_start:]
            return (out,)

        return AsyncTensorHandle(works=resp_handle.works, recv_tensors=resp_handle.recv_tensors, finalize=_finalize)


class DynamicPushComm(DynamicA2AComm):
    """Single-phase push for owner writeback and replica cache refresh."""

    def submit_push(self, target_index: Tensor, target_ptr: Tensor, *payloads: Tensor) -> AsyncTensorHandle:
        target_index = target_index.to(self.device, dtype=torch.long).contiguous()
        target_ptr = target_ptr.to(self.device, dtype=torch.long).contiguous()
        send_counts = _counts_from_ptr(target_ptr)
        plan = self.exchange_counts(send_counts)
        tensors = (target_index, *tuple(p.to(self.device).contiguous() for p in payloads))
        if self.world_size <= 1:
            return AsyncTensorHandle(works=[], recv_tensors=tensors)
        works: list[Any] = []
        recvs: list[Tensor] = []
        for tensor in tensors:
            handle = self.submit_a2a(plan, tensor)
            works.extend(handle.works)
            recvs.extend(handle.recv_tensors)
        return AsyncTensorHandle(works=works, recv_tensors=tuple(recvs))
