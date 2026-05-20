from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part
from atc_starrygl_lib.comm.dynamic import AsyncTensorHandle, DynamicPushComm
from atc_starrygl_lib.comm.layouts import ReplicaPushLayout


@dataclass(slots=True)
class ReplicaPushIndex:
    replica_ptr: Tensor
    replica_target_index: Tensor


def _ptr_from_rank(rank: Tensor, world_size: int) -> Tensor:
    counts = torch.bincount(rank.long(), minlength=int(world_size))
    ptr = torch.zeros(int(world_size) + 1, dtype=torch.long, device=rank.device)
    ptr[1:] = counts.cumsum(0)
    return ptr


def build_replica_push_layout(
    updated_nodes: Tensor,
    replica_index: ReplicaPushIndex,
    *,
    world_size: int,
    source_pos: Optional[Tensor] = None,
) -> ReplicaPushLayout:
    nodes = updated_nodes.long().to(replica_index.replica_ptr.device)
    ptr = replica_index.replica_ptr
    starts = ptr.index_select(0, nodes)
    ends = ptr.index_select(0, nodes + 1)
    counts = (ends - starts).long()
    total = int(counts.sum().item())
    device = updated_nodes.device
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return ReplicaPushLayout(
            target_index=empty,
            target_ptr=torch.zeros(int(world_size) + 1, dtype=torch.long, device=device),
            source_pos=empty,
        )
    default_source = torch.arange(nodes.numel(), dtype=torch.long, device=ptr.device) if source_pos is None else source_pos.to(ptr.device).long()
    source = torch.repeat_interleave(default_source, counts).to(device).long()
    repeated_starts = torch.repeat_interleave(starts, counts)
    segment_base = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    local_offset = torch.arange(total, dtype=torch.long, device=ptr.device) - segment_base
    target = replica_index.replica_target_index.index_select(0, repeated_starts + local_offset).to(device).long()
    rank = dist_index_part(target)
    order = torch.argsort(rank, stable=True)
    return ReplicaPushLayout(
        target_index=target[order].contiguous(),
        target_ptr=_ptr_from_rank(rank[order], int(world_size)),
        source_pos=source[order].contiguous(),
    )


class SharedStateSync:
    def __init__(self, comm: DynamicPushComm) -> None:
        self.comm = comm

    def submit_push(self, layout: ReplicaPushLayout, *payloads: Tensor) -> AsyncTensorHandle:
        aligned = tuple(payload.index_select(0, layout.source_pos.to(payload.device)) for payload in payloads)
        return self.comm.submit_push(layout.target_index, layout.target_ptr, *aligned)

    @staticmethod
    def target_rows(target_index: Tensor) -> Tensor:
        return dist_index_loc(target_index).long()
