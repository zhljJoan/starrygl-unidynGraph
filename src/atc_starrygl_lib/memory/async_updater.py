from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .mailbox_runtime import MailboxRuntime
from .runtime import MemoryRuntime
from .shared_sync import ReplicaPushIndex


class HistoricalBlend(nn.Module):
    def __init__(self, memory_dim: int, learnable_gamma: bool = True) -> None:
        super().__init__()
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor([0.5]))
        else:
            self.register_buffer("gamma", torch.tensor([0.9]))
        self.memory_dim = int(memory_dim)

    def forward(
        self,
        updated: Tensor,
        shared_mask: Tensor,
        historical_memory: Tensor,
        increment: Tensor,
    ) -> Tensor:
        if not shared_mask.any():
            return updated
        gamma = torch.sigmoid(self.gamma)
        out = updated.clone()
        pred = historical_memory + _normalize_increment(increment)
        out[shared_mask] = gamma * updated[shared_mask] + (1 - gamma) * pred[shared_mask]
        return out


@dataclass(slots=True)
class AsyncCommitHandle:
    memory_handle: object | None = None
    mailbox_handle: object | None = None
    memory_replica_handle: object | None = None
    mailbox_replica_handle: object | None = None

    def wait_apply(self) -> None:
        for handle in (
            self.memory_handle,
            self.mailbox_handle,
            self.memory_replica_handle,
            self.mailbox_replica_handle,
        ):
            if handle is not None:
                handle.wait_apply()


@dataclass(slots=True)
class AsyncMemoryUpdateSpec:
    """Runtime commit inputs produced by a CTDG batch.

    ``src``/``dst``/``edge_feat`` build MemShare-style mailbox messages:
    ``[src_memory || dst_memory || edge_feat]`` delivered to ``dst``.
    """

    src: Tensor | None = None
    dst: Tensor | None = None
    ts: Tensor | None = None
    edge_feat: Tensor | None = None
    update_mailbox: bool = True
    memory_nodes: Tensor | None = None
    mailbox_nodes: Tensor | None = None
    memory_replica_index: ReplicaPushIndex | None = None
    mailbox_replica_index: ReplicaPushIndex | None = None
    mailbox_snapshot: Tensor | None = None
    mailbox_snapshot_ts: Tensor | None = None
    wait_apply: bool = False

    @classmethod
    def from_edges(
        cls,
        src: Tensor,
        dst: Tensor,
        ts: Tensor,
        edge_feat: Tensor | None = None,
        *,
        update_mailbox: bool = True,
        wait_apply: bool = False,
    ) -> "AsyncMemoryUpdateSpec":
        return cls(
            src=src,
            dst=dst,
            ts=ts,
            edge_feat=edge_feat,
            update_mailbox=bool(update_mailbox),
            wait_apply=bool(wait_apply),
        )


class AsyncMemoryCommitter:
    """Submit memory/mailbox updates without coupling updater compute to dist."""

    def __init__(self, memory_runtime: MemoryRuntime, mailbox_runtime: MailboxRuntime | None = None) -> None:
        self.memory_runtime = memory_runtime
        self.mailbox_runtime = mailbox_runtime

    def submit(
        self,
        updated_nodes: Tensor,
        updated_memory: Tensor,
        updated_ts: Tensor,
        *,
        mailbox_nodes: Optional[Tensor] = None,
        mailbox_msg: Optional[Tensor] = None,
        mailbox_ts: Optional[Tensor] = None,
        memory_replica_index: Optional[ReplicaPushIndex] = None,
        mailbox_replica_index: Optional[ReplicaPushIndex] = None,
        mailbox_snapshot: Optional[Tensor] = None,
        mailbox_snapshot_ts: Optional[Tensor] = None,
    ) -> AsyncCommitHandle:
        mem_layout = self.memory_runtime.build_write_layout(updated_nodes)
        mem_handle = self.memory_runtime.write(mem_layout, updated_memory, updated_ts)
        mail_handle = None
        mem_replica_handle = None
        mail_replica_handle = None
        if self.mailbox_runtime is not None and mailbox_nodes is not None and mailbox_msg is not None and mailbox_ts is not None:
            mail_layout = self.mailbox_runtime.build_write_layout(mailbox_nodes)
            mail_handle = self.mailbox_runtime.write(mail_layout, mailbox_msg, mailbox_ts)
        if memory_replica_index is not None:
            replica_layout = self.memory_runtime.build_replica_push_layout(updated_nodes, memory_replica_index)
            mem_replica_handle = self.memory_runtime.replica_push(replica_layout, updated_memory, updated_ts)
        if (
            self.mailbox_runtime is not None
            and mailbox_replica_index is not None
            and mailbox_nodes is not None
            and mailbox_snapshot is not None
            and mailbox_snapshot_ts is not None
        ):
            mailbox_replica_layout = self.memory_runtime.build_replica_push_layout(mailbox_nodes, mailbox_replica_index)
            mail_replica_handle = self.mailbox_runtime.replica_push(
                mailbox_replica_layout,
                mailbox_snapshot,
                mailbox_snapshot_ts,
            )
        return AsyncCommitHandle(
            memory_handle=mem_handle,
            mailbox_handle=mail_handle,
            memory_replica_handle=mem_replica_handle,
            mailbox_replica_handle=mail_replica_handle,
        )


class RuntimeAsyncMemoryUpdater(nn.Module):
    """Wrap a CTDG memory updater with row/index-based async commit.

    The wrapped ``base_updater`` remains a pure model component.  This class
    only reads its ``last_updated_*`` tensors and submits memory/mailbox updates
    through ``AsyncMemoryCommitter``.
    """

    def __init__(
        self,
        base_updater: nn.Module,
        committer: AsyncMemoryCommitter,
        historical_blend: HistoricalBlend | None = None,
    ) -> None:
        super().__init__()
        self.base_updater = base_updater
        self.committer = committer
        self.historical_blend = historical_blend
        self.last_updated_memory: Tensor | None = None
        self.last_updated_ts: Tensor | None = None
        self.last_updated_nid: Tensor | None = None
        self.last_commit_handle: AsyncCommitHandle | None = None

    def reset_state(self) -> None:
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.last_commit_handle = None
        for name in ("last_updated_memory", "last_updated_ts", "last_updated_nid"):
            if hasattr(self.base_updater, name):
                setattr(self.base_updater, name, None)

    def forward(self, mfg: Any, spec: AsyncMemoryUpdateSpec | None = None) -> Tensor | None:
        updated = self._run_base_updater(mfg)
        nid = getattr(self.base_updater, "last_updated_nid", None)
        ts = getattr(self.base_updater, "last_updated_ts", None)
        if updated is None:
            updated = getattr(self.base_updater, "last_updated_memory", None)
        if updated is None or nid is None or ts is None:
            return updated

        updated = self._maybe_historical_blend(mfg, updated)
        self.last_updated_memory = updated.detach().clone()
        self.last_updated_ts = ts.detach().clone()
        self.last_updated_nid = nid.detach().clone()

        if spec is not None:
            self.last_commit_handle = self.submit_commit(spec)
            if spec.wait_apply and self.last_commit_handle is not None:
                self.last_commit_handle.wait_apply()
        return updated

    def submit_commit(self, spec: AsyncMemoryUpdateSpec) -> AsyncCommitHandle:
        if self.last_updated_memory is None or self.last_updated_ts is None or self.last_updated_nid is None:
            raise RuntimeError("no updated memory is available; call forward() first")
        nid = self.last_updated_nid
        updated = self.last_updated_memory
        updated_ts = self.last_updated_ts

        if spec.memory_nodes is not None:
            memory_nodes = spec.memory_nodes
        elif spec.src is not None and spec.dst is not None:
            memory_nodes = torch.cat([spec.src, spec.dst], dim=0)
        else:
            memory_nodes = nid
        memory_values = _safe_index(nid, memory_nodes, updated)
        memory_ts = _safe_index(nid, memory_nodes, updated_ts.reshape(-1, 1)).reshape(-1)
        if memory_values is None or memory_ts is None:
            raise RuntimeError("memory_nodes are not covered by updated node ids")

        mailbox_nodes = spec.mailbox_nodes
        mailbox_msg = None
        mailbox_ts = None
        if spec.update_mailbox and spec.src is not None and spec.dst is not None and spec.ts is not None:
            mailbox_nodes = torch.cat([spec.src, spec.dst], dim=0) if mailbox_nodes is None else mailbox_nodes
            mailbox_msg = _build_mailbox_messages(nid, updated, spec.src, spec.dst, spec.edge_feat)
            mailbox_ts = torch.cat([spec.ts, spec.ts], dim=0)

        return self.committer.submit(
            memory_nodes,
            memory_values,
            memory_ts,
            mailbox_nodes=mailbox_nodes,
            mailbox_msg=mailbox_msg,
            mailbox_ts=mailbox_ts,
            memory_replica_index=spec.memory_replica_index,
            mailbox_replica_index=spec.mailbox_replica_index,
            mailbox_snapshot=spec.mailbox_snapshot,
            mailbox_snapshot_ts=spec.mailbox_snapshot_ts,
        )

    def _run_base_updater(self, mfg: Any) -> Tensor | None:
        if hasattr(self.base_updater, "forward_from_mfg"):
            blocks = mfg if isinstance(mfg, (list, tuple)) else [mfg]
            updated = None
            for block in blocks:
                updated = self.base_updater.forward_from_mfg(block)
            return updated
        result = self.base_updater(mfg)
        if isinstance(result, Tensor):
            return result
        return getattr(self.base_updater, "last_updated_memory", None)

    def _maybe_historical_blend(self, mfg: Any, updated: Tensor) -> Tensor:
        if self.historical_blend is None:
            return updated
        block = _first_block(mfg)
        srcdata = getattr(block, "srcdata", None)
        if not isinstance(srcdata, dict):
            return updated
        shared_mask = srcdata.get("shared_mask")
        his_mem = srcdata.get("his_mem")
        increment = srcdata.get("his_increment")
        if shared_mask is None or his_mem is None or increment is None:
            return updated
        return self.historical_blend(
            updated,
            shared_mask.to(device=updated.device).bool(),
            his_mem.to(device=updated.device, dtype=updated.dtype),
            increment.to(device=updated.device, dtype=updated.dtype),
        )


def _normalize_increment(x: Tensor, eps: float = 1e-12) -> Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def _first_block(mfg: Any) -> Any:
    if isinstance(mfg, (list, tuple)):
        if not mfg:
            return None
        first = mfg[0]
        if isinstance(first, (list, tuple)):
            return first[0] if first else None
        return first
    return mfg


def _build_mailbox_messages(
    nid: Tensor,
    updated: Tensor,
    src: Tensor,
    dst: Tensor,
    edge_feat: Tensor | None,
) -> Tensor:
    src_mem = _safe_index(nid, src, updated)
    dst_mem = _safe_index(nid, dst, updated)
    if src_mem is None or dst_mem is None:
        raise RuntimeError("src/dst nodes are not covered by updated node ids")
    src_mem = src_mem.reshape(int(src_mem.size(0)), -1)
    dst_mem = dst_mem.reshape(int(dst_mem.size(0)), -1)
    src_mail = torch.cat([src_mem, dst_mem], dim=-1)
    dst_mail = torch.cat([dst_mem, src_mem], dim=-1)
    if edge_feat is not None:
        edge = edge_feat.to(src_mem.device, dtype=src_mem.dtype).reshape(int(edge_feat.size(0)), -1)
        src_mail = torch.cat([src_mail, edge], dim=-1)
        dst_mail = torch.cat([dst_mail, edge], dim=-1)
    return torch.cat([src_mail, dst_mail], dim=0)


def _safe_index(nid: Tensor, query: Tensor, values: Tensor) -> Tensor | None:
    dev = values.device
    nid_s = nid.to(dev).long().reshape(-1)
    query_s = query.to(dev).long().reshape(-1)
    if nid_s.numel() == 0 or query_s.numel() == 0:
        shape = (0, *values.shape[1:])
        return values.new_empty(shape)
    order = torch.argsort(nid_s)
    sorted_nid = nid_s[order]
    rows = torch.searchsorted(sorted_nid, query_s).clamp_max(nid_s.numel() - 1)
    if not torch.equal(sorted_nid[rows], query_s):
        return None
    return values[order[rows]]
