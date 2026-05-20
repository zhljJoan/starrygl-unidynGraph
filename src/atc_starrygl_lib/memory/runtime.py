from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part
from atc_starrygl_lib.comm.dynamic import AsyncTensorHandle, DynamicFetchComm, DynamicPushComm
from atc_starrygl_lib.comm.layouts import MemoryReadLayout, MemoryWriteLayout, ReplicaPushLayout
from atc_starrygl_lib.runtime.index import DistIndexTables
from atc_starrygl_lib.sampling.temporal import SamplingOutput

from .shared_sync import ReplicaPushIndex, build_replica_push_layout
from .store import MemoryStore


def _ptr_from_rank(rank: Tensor, world_size: int) -> Tensor:
    counts = torch.bincount(rank.long(), minlength=int(world_size))
    ptr = torch.zeros(int(world_size) + 1, dtype=torch.long, device=rank.device)
    ptr[1:] = counts.cumsum(0)
    return ptr


@dataclass(slots=True)
class MemoryReadHandle:
    runtime: "MemoryRuntime"
    layout: MemoryReadLayout
    handle: AsyncTensorHandle

    def wait(self) -> tuple[Tensor, Tensor]:
        return self.runtime.wait_read(self.handle, self.layout)


@dataclass(slots=True)
class MemoryWriteHandle:
    runtime: "MemoryRuntime"
    handle: AsyncTensorHandle

    def wait_apply(self) -> None:
        target_index, memory, ts = self.handle.wait()
        self.runtime.apply_write(target_index, memory, ts)


@dataclass(slots=True)
class MemoryReplicaHandle:
    runtime: "MemoryRuntime"
    handle: AsyncTensorHandle

    def wait_apply(self) -> None:
        target_index, memory, ts = self.handle.wait()
        self.runtime.apply_replica_push(target_index, memory, ts)


class MemoryRuntime:
    def __init__(
        self,
        index: DistIndexTables,
        store: MemoryStore,
        fetch_comm: DynamicFetchComm,
        push_comm: DynamicPushComm,
        *,
        world_size: int,
    ) -> None:
        self.index = index
        self.store = store
        self.fetch_comm = fetch_comm
        self.push_comm = push_comm
        self.world_size = int(world_size)

    def build_read_layout_from_sampling(self, output: SamplingOutput) -> MemoryReadLayout:
        read_idx = self.index.read_for(output.node_comm.node_gids)
        rank = dist_index_part(read_idx)
        order = torch.argsort(rank, stable=True)
        read_idx = read_idx[order].contiguous()
        compute_to_grouped = torch.empty_like(order)
        compute_to_grouped[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        compute_to_memory = compute_to_grouped.index_select(0, output.node_comm.compute_to_comm.to(compute_to_grouped.device))
        return MemoryReadLayout(
            read_index=read_idx,
            read_ptr=_ptr_from_rank(rank[order], self.world_size),
            compute_to_memory=compute_to_memory.to(output.node_comm.node_gids.device).long().contiguous(),
        )

    def submit_read(self, layout: MemoryReadLayout) -> AsyncTensorHandle:
        return self.fetch_comm.submit_row_fetch(layout.read_index, layout.read_ptr, self._gather_memory_payload)

    def read(self, layout: MemoryReadLayout) -> MemoryReadHandle:
        return MemoryReadHandle(runtime=self, layout=layout, handle=self.submit_read(layout))

    def wait_read(self, handle: AsyncTensorHandle, layout: MemoryReadLayout) -> tuple[Tensor, Tensor]:
        (payload,) = handle.wait()
        payload = payload.index_select(0, layout.compute_to_memory.to(payload.device))
        return payload[:, :-1].contiguous(), payload[:, -1].contiguous()

    def build_write_layout(self, node_ids: Tensor, source_pos: Optional[Tensor] = None) -> MemoryWriteLayout:
        target = self.index.master_for(node_ids)
        source = torch.arange(target.numel(), dtype=torch.long, device=target.device) if source_pos is None else source_pos.to(target.device).long()
        rank = dist_index_part(target)
        order = torch.argsort(rank, stable=True)
        return MemoryWriteLayout(
            target_index=target[order].contiguous(),
            target_ptr=_ptr_from_rank(rank[order], self.world_size),
            source_pos=source[order].contiguous(),
        )

    def submit_write(self, layout: MemoryWriteLayout, memory: Tensor, ts: Tensor) -> AsyncTensorHandle:
        return self.push_comm.submit_push(
            layout.target_index,
            layout.target_ptr,
            memory.index_select(0, layout.source_pos.to(memory.device)),
            ts.index_select(0, layout.source_pos.to(ts.device)),
        )

    def write(self, layout: MemoryWriteLayout, memory: Tensor, ts: Tensor) -> MemoryWriteHandle:
        return MemoryWriteHandle(runtime=self, handle=self.submit_write(layout, memory, ts))

    def apply_write(self, target_index: Tensor, memory: Tensor, ts: Tensor) -> None:
        self.store.update_rows(dist_index_loc(target_index), memory, ts)

    def build_replica_push_layout(self, node_ids: Tensor, replica_index: ReplicaPushIndex, source_pos: Optional[Tensor] = None) -> ReplicaPushLayout:
        return build_replica_push_layout(node_ids, replica_index, world_size=self.world_size, source_pos=source_pos)

    def submit_replica_push(self, layout: ReplicaPushLayout, memory: Tensor, ts: Tensor) -> AsyncTensorHandle:
        return self.push_comm.submit_push(
            layout.target_index,
            layout.target_ptr,
            memory.index_select(0, layout.source_pos.to(memory.device)),
            ts.index_select(0, layout.source_pos.to(ts.device)),
        )

    def replica_push(self, layout: ReplicaPushLayout, memory: Tensor, ts: Tensor) -> MemoryReplicaHandle:
        return MemoryReplicaHandle(runtime=self, handle=self.submit_replica_push(layout, memory, ts))

    def apply_replica_push(self, target_index: Tensor, memory: Tensor, ts: Tensor) -> None:
        self.store.update_rows(dist_index_loc(target_index), memory, ts)

    def _gather_memory_payload(self, rows: Tensor, _time_slices: Tensor | None = None) -> Tensor:
        memory, ts = self.store.gather_rows(rows)
        return torch.cat([memory, ts.reshape(-1, 1).to(memory.dtype)], dim=1)
