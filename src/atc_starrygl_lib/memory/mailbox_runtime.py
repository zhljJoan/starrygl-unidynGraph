from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part
from atc_starrygl_lib.comm.dynamic import AsyncTensorHandle, DynamicFetchComm, DynamicPushComm
from atc_starrygl_lib.comm.layouts import MailboxReadLayout, MailboxWriteLayout, ReplicaPushLayout
from atc_starrygl_lib.runtime.index import DistIndexTables
from atc_starrygl_lib.sampling.temporal import SamplingOutput

from .mailbox import MailboxStore


def _ptr_from_rank(rank: Tensor, world_size: int) -> Tensor:
    counts = torch.bincount(rank.long(), minlength=int(world_size))
    ptr = torch.zeros(int(world_size) + 1, dtype=torch.long, device=rank.device)
    ptr[1:] = counts.cumsum(0)
    return ptr


@dataclass(slots=True)
class MailboxReadHandle:
    runtime: "MailboxRuntime"
    layout: MailboxReadLayout
    handle: AsyncTensorHandle

    def wait(self) -> tuple[Tensor, Tensor]:
        return self.runtime.wait_read(self.handle, self.layout)


@dataclass(slots=True)
class MailboxWriteHandle:
    runtime: "MailboxRuntime"
    handle: AsyncTensorHandle

    def wait_apply(self) -> None:
        target_index, msg, ts = self.handle.wait()
        self.runtime.apply_write(target_index, msg, ts)


@dataclass(slots=True)
class MailboxReplicaHandle:
    runtime: "MailboxRuntime"
    handle: AsyncTensorHandle

    def wait_apply(self) -> None:
        target_index, mailbox, mailbox_ts = self.handle.wait()
        self.runtime.apply_replica_push(target_index, mailbox, mailbox_ts)


class MailboxRuntime:
    def __init__(
        self,
        index: DistIndexTables,
        store: MailboxStore,
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

    def build_read_layout_from_sampling(self, output: SamplingOutput) -> MailboxReadLayout:
        read_idx = self.index.read_for(output.node_comm.node_gids)
        rank = dist_index_part(read_idx)
        order = torch.argsort(rank, stable=True)
        read_idx = read_idx[order].contiguous()
        compute_to_grouped = torch.empty_like(order)
        compute_to_grouped[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        compute_to_mailbox = compute_to_grouped.index_select(0, output.node_comm.compute_to_comm.to(compute_to_grouped.device))
        return MailboxReadLayout(
            read_index=read_idx,
            read_ptr=_ptr_from_rank(rank[order], self.world_size),
            compute_to_mailbox=compute_to_mailbox.to(output.node_comm.node_gids.device).long().contiguous(),
        )

    def submit_read(self, layout: MailboxReadLayout) -> AsyncTensorHandle:
        return self.fetch_comm.submit_row_fetch(layout.read_index, layout.read_ptr, self._gather_mailbox_payload)

    def read(self, layout: MailboxReadLayout) -> MailboxReadHandle:
        return MailboxReadHandle(runtime=self, layout=layout, handle=self.submit_read(layout))

    def wait_read(self, handle: AsyncTensorHandle, layout: MailboxReadLayout) -> tuple[Tensor, Tensor]:
        (payload,) = handle.wait()
        payload = payload.index_select(0, layout.compute_to_mailbox.to(payload.device))
        k = int(self.store.mailbox.size(1))
        msg_dim = int(self.store.mailbox.size(2))
        mail_width = k * msg_dim
        mailbox = payload[:, :mail_width].reshape(payload.size(0), k, msg_dim).contiguous()
        mailbox_ts = payload[:, mail_width:].reshape(payload.size(0), k).contiguous()
        return mailbox, mailbox_ts

    def build_write_layout(self, node_ids: Tensor, source_pos: Optional[Tensor] = None) -> MailboxWriteLayout:
        target = self.index.master_for(node_ids)
        source = torch.arange(target.numel(), dtype=torch.long, device=target.device) if source_pos is None else source_pos.to(target.device).long()
        rank = dist_index_part(target)
        order = torch.argsort(rank, stable=True)
        return MailboxWriteLayout(
            target_index=target[order].contiguous(),
            target_ptr=_ptr_from_rank(rank[order], self.world_size),
            source_pos=source[order].contiguous(),
        )

    def submit_write(self, layout: MailboxWriteLayout, msg: Tensor, ts: Tensor) -> AsyncTensorHandle:
        return self.push_comm.submit_push(
            layout.target_index,
            layout.target_ptr,
            msg.index_select(0, layout.source_pos.to(msg.device)),
            ts.index_select(0, layout.source_pos.to(ts.device)),
        )

    def write(self, layout: MailboxWriteLayout, msg: Tensor, ts: Tensor) -> MailboxWriteHandle:
        return MailboxWriteHandle(runtime=self, handle=self.submit_write(layout, msg, ts))

    def apply_write(self, target_index: Tensor, msg: Tensor, ts: Tensor) -> None:
        self.store.append_rows(dist_index_loc(target_index), msg, ts)

    def submit_replica_push(self, layout: ReplicaPushLayout, mailbox: Tensor, mailbox_ts: Tensor) -> AsyncTensorHandle:
        return self.push_comm.submit_push(
            layout.target_index,
            layout.target_ptr,
            mailbox.index_select(0, layout.source_pos.to(mailbox.device)),
            mailbox_ts.index_select(0, layout.source_pos.to(mailbox_ts.device)),
        )

    def replica_push(self, layout: ReplicaPushLayout, mailbox: Tensor, mailbox_ts: Tensor) -> MailboxReplicaHandle:
        return MailboxReplicaHandle(runtime=self, handle=self.submit_replica_push(layout, mailbox, mailbox_ts))

    def apply_replica_push(self, target_index: Tensor, mailbox: Tensor, mailbox_ts: Tensor) -> None:
        rows = dist_index_loc(target_index).long().to(self.store.mailbox.device)
        self.store.mailbox[rows] = mailbox.to(device=self.store.mailbox.device, dtype=self.store.mailbox.dtype)
        self.store.mailbox_ts[rows] = mailbox_ts.to(device=self.store.mailbox_ts.device, dtype=self.store.mailbox_ts.dtype)

    def _gather_mailbox_payload(self, rows: Tensor, _time_slices: Tensor | None = None) -> Tensor:
        mailbox, mailbox_ts = self.store.gather_rows(rows)
        return torch.cat([mailbox.flatten(1), mailbox_ts.to(mailbox.dtype)], dim=1)
