from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.distributed as dist

from atc_starrygl_lib.comm.layouts import MailboxWriteLayout, MemoryWriteLayout
from .mailbox_runtime import MailboxRuntime
from .runtime import MemoryRuntime
from .shared_sync import ReplicaPushIndex
from atc_starrygl_lib.comm.dist_index import dist_index_is_shared, dist_index_loc, dist_index_part


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
        pred = historical_memory + increment
        out[shared_mask] = gamma * updated[shared_mask] + (1 - gamma) * pred[shared_mask]
        return out


class SharedHistoricalCache(nn.Module):
    """Filter small shared-memory deltas before replica synchronization."""

    def __init__(
        self,
        memory_dim: int,
        *,
        num_nodes: int = 0,
        alpha: float = 0.0,
        times_threshold: int = 10,
        time_threshold: float | None = None,
        preload_candidate_delta: bool = False,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.alpha = float(alpha)
        self.times_threshold = int(times_threshold)
        self.time_threshold = None if time_threshold is None else float(time_threshold)
        self.preload_candidate_delta = bool(preload_candidate_delta)
        self.register_buffer("historical_memory", torch.zeros(int(num_nodes), self.memory_dim))
        self.register_buffer("historical_ts", torch.zeros(int(num_nodes)))
        self.register_buffer("loss_count", torch.zeros(int(num_nodes), dtype=torch.long))

    def reset_state(self) -> None:
        self.historical_memory.zero_()
        self.historical_ts.zero_()
        self.loss_count.zero_()

    def historical_check(self, index: Tensor, new_data: Tensor, ts: Tensor) -> Tensor:
        index = index.long().reshape(-1)
        if index.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=new_data.device)
        self._ensure_capacity(int(index.max().item()) + 1, new_data.device, new_data.dtype)
        hist = self.historical_memory.index_select(0, index).to(device=new_data.device, dtype=new_data.dtype)
        hist_ts = self.historical_ts.index_select(0, index).to(device=ts.device, dtype=ts.dtype)
        loss = self.loss_count.index_select(0, index).to(device=index.device)

        mask = _cosine_distance(new_data, hist) > self.alpha
        if self.time_threshold is not None:
            mask = mask | ((ts - hist_ts) > self.time_threshold)
        mask = mask | (loss > self.times_threshold)

        if mask.any():
            update_index = index[mask].to(self.historical_memory.device)
            self.historical_memory[update_index] = new_data[mask].to(
                device=self.historical_memory.device,
                dtype=self.historical_memory.dtype,
            )
            self.historical_ts[update_index] = ts[mask].to(
                device=self.historical_ts.device,
                dtype=self.historical_ts.dtype,
            )
            self.loss_count[update_index] = 0
        if (~mask).any():
            skipped = index[~mask].to(self.loss_count.device)
            self.loss_count[skipped] += 1
        return mask

    def _ensure_capacity(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        if int(self.historical_memory.size(0)) >= int(size):
            return
        current = int(self.historical_memory.size(0))
        grow = int(size) - current
        self.historical_memory = torch.cat(
            [self.historical_memory, torch.zeros(grow, self.memory_dim, device=device, dtype=dtype)],
            dim=0,
        )
        self.historical_ts = torch.cat(
            [self.historical_ts, torch.zeros(grow, device=device, dtype=torch.float32)],
            dim=0,
        )
        self.loss_count = torch.cat(
            [self.loss_count, torch.zeros(grow, device=device, dtype=torch.long)],
            dim=0,
        )


class HistoricalDeltaFilter(nn.Module):
    """Track historical memory and dense increments for historical compensation."""

    def __init__(
        self,
        memory_dim: int,
        *,
        num_nodes: int = 0,
        preload_candidate_delta: bool = False,
        node_id_to_index: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.preload_candidate_delta = bool(preload_candidate_delta)
        if node_id_to_index is None:
            self.register_buffer("node_id_to_index", torch.empty(0, dtype=torch.long), persistent=False)
        else:
            self.register_buffer("node_id_to_index", node_id_to_index.long().reshape(-1).contiguous(), persistent=False)
        self.register_buffer("historical_memory", torch.zeros(int(num_nodes), self.memory_dim))
        self.register_buffer("historical_ts", torch.zeros(int(num_nodes)))
        self.register_buffer("increment_sum", torch.zeros(int(num_nodes), self.memory_dim))
        self.register_buffer("increment_count", torch.zeros(int(num_nodes), 1))

    def reset_state(self) -> None:
        self.historical_memory.zero_()
        self.historical_ts.zero_()
        self.increment_sum.zero_()
        self.increment_count.zero_()

    def preload_candidate(self, index: Tensor, historical_memory: Tensor, new_data: Tensor, ts: Tensor) -> None:
        if not self.preload_candidate_delta:
            return
        index, keep = self._map_index(index)
        if index.numel() == 0:
            return
        new_data = new_data[keep]
        historical_memory = historical_memory[keep]
        ts = ts.reshape(-1)[keep]
        self._ensure_capacity(int(index.max().item()) + 1, new_data.device, new_data.dtype)
        delta = new_data - historical_memory
        self._update_increment(index, delta)
        update_index = index.to(self.historical_memory.device)
        self.historical_memory[update_index] = new_data.to(
            device=self.historical_memory.device,
            dtype=self.historical_memory.dtype,
        )
        self.historical_ts[update_index] = ts.to(
            device=self.historical_ts.device,
            dtype=self.historical_ts.dtype,
        )

    def record_history(
        self,
        index: Tensor,
        new_data: Tensor,
        ts: Tensor,
        *,
        reset_increment: bool = False,
    ) -> None:
        index, keep = self._map_index(index)
        if index.numel() == 0:
            return
        new_data = new_data[keep]
        ts = ts.reshape(-1)[keep]
        self._ensure_capacity(int(index.max().item()) + 1, new_data.device, new_data.dtype)
        update_index = index.to(self.historical_memory.device)
        self.historical_memory[update_index] = new_data.to(
            device=self.historical_memory.device,
            dtype=self.historical_memory.dtype,
        )
        self.historical_ts[update_index] = ts.to(
            device=self.historical_ts.device,
            dtype=self.historical_ts.dtype,
        )
        if reset_increment:
            self.increment_sum[update_index] = 0
            self.increment_count[update_index] = 0

    def get_increment(self, index: Tensor) -> Tensor:
        raw = index.long().reshape(-1)
        mapped, keep = self._map_index(raw)
        out = self.increment_sum.new_zeros((int(raw.numel()), self.memory_dim))
        if mapped.numel() == 0:
            return out
        self._ensure_capacity(int(mapped.max().item()) + 1, self.increment_sum.device, self.increment_sum.dtype)
        denom = self.increment_count.index_select(0, mapped).clamp_min(1)
        out[keep.to(out.device)] = self.increment_sum.index_select(0, mapped) / denom
        return out

    def get_increment_remote(self, index: Tensor) -> Tensor:
        return self.get_increment(index)

    def get_count(self, index: Tensor) -> Tensor:
        raw = index.long().reshape(-1)
        mapped, keep = self._map_index(raw)
        out = self.increment_count.new_zeros((int(raw.numel()), 1))
        if mapped.numel() == 0:
            return out
        self._ensure_capacity(int(mapped.max().item()) + 1, self.increment_count.device, self.increment_count.dtype)
        out[keep.to(out.device)] = self.increment_count.index_select(0, mapped)
        return out

    def update(self, index: Tensor, change: Tensor) -> None:
        index, keep = self._map_index(index)
        if index.numel() == 0:
            return
        change = change[keep]
        self._ensure_capacity(int(index.max().item()) + 1, change.device, change.dtype)
        value = change.to(device=self.increment_sum.device, dtype=self.increment_sum.dtype).reshape(-1, self.memory_dim)
        self._update_increment(index, value)

    def get_history(self, index: Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        raw = index.long().reshape(-1)
        resolved, keep = self._map_index(raw)
        out_device = self.historical_memory.device if device is None else device
        out_dtype = self.historical_memory.dtype if dtype is None else dtype
        out = torch.zeros((int(raw.numel()), self.memory_dim), device=out_device, dtype=out_dtype)
        if resolved.numel() == 0:
            return out
        self._ensure_capacity(int(resolved.max().item()) + 1, self.historical_memory.device, self.historical_memory.dtype)
        gathered = self.historical_memory.index_select(0, resolved.to(self.historical_memory.device))
        out[keep.to(out.device)] = gathered.to(device=out.device, dtype=out.dtype)
        return out

    def get_history_ts(self, index: Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        raw = index.long().reshape(-1)
        resolved, keep = self._map_index(raw)
        out_device = self.historical_ts.device if device is None else device
        out_dtype = self.historical_ts.dtype if dtype is None else dtype
        out = torch.zeros((int(raw.numel()),), device=out_device, dtype=out_dtype)
        if resolved.numel() == 0:
            return out
        self._ensure_capacity(int(resolved.max().item()) + 1, self.historical_ts.device, self.historical_ts.dtype)
        gathered = self.historical_ts.index_select(0, resolved.to(self.historical_ts.device))
        out[keep.to(out.device)] = gathered.to(device=out.device, dtype=out.dtype)
        return out

    def _update_increment(self, index: Tensor, delta: Tensor) -> None:
        device_index = index.to(self.increment_sum.device)
        self.increment_sum.index_add_(
            0,
            device_index,
            delta.to(device=self.increment_sum.device, dtype=self.increment_sum.dtype),
        )
        ones = torch.ones((device_index.numel(), 1), device=self.increment_count.device, dtype=self.increment_count.dtype)
        self.increment_count.index_add_(0, device_index, ones)

    def _map_index(self, index: Tensor) -> tuple[Tensor, Tensor]:
        raw = index.long().reshape(-1)
        if raw.numel() == 0:
            return raw, torch.zeros((0,), dtype=torch.bool, device=raw.device)
        if int(self.node_id_to_index.numel()) == 0:
            return raw, torch.ones_like(raw, dtype=torch.bool)
        if int(raw.max().item()) >= int(self.node_id_to_index.numel()):
            raise IndexError("node id exceeds historical filter mapping size")
        mapped = self.node_id_to_index.index_select(0, raw.to(self.node_id_to_index.device)).to(raw.device)
        keep = mapped >= 0
        return mapped[keep].long(), keep

    def _ensure_capacity(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        if int(self.increment_sum.size(0)) >= int(size):
            return
        current = int(self.increment_sum.size(0))
        grow = int(size) - current
        self.historical_memory = torch.cat(
            [self.historical_memory, torch.zeros(grow, self.memory_dim, device=device, dtype=dtype)],
            dim=0,
        )
        self.historical_ts = torch.cat(
            [self.historical_ts, torch.zeros(grow, device=device, dtype=torch.float32)],
            dim=0,
        )
        self.loss_count = torch.cat(
            [self.loss_count, torch.zeros(grow, device=device, dtype=torch.long)],
            dim=0,
        )
        self.increment_sum = torch.cat(
            [self.increment_sum, torch.zeros(grow, self.memory_dim, device=device, dtype=dtype)],
            dim=0,
        )
        self.increment_count = torch.cat(
            [self.increment_count, torch.zeros(grow, 1, device=device, dtype=torch.float32)],
            dim=0,
        )


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
    src_rows: Tensor | None = None
    dst_rows: Tensor | None = None
    ts: Tensor | None = None
    edge_feat: Tensor | None = None
    update_mailbox: bool = True
    memory_nodes: Tensor | None = None
    memory_rows: Tensor | None = None
    memory_write_target_index: Tensor | None = None
    memory_write_target_ptr: Tensor | None = None
    memory_write_source_pos: Tensor | None = None
    mailbox_nodes: Tensor | None = None
    mailbox_self_rows: Tensor | None = None
    mailbox_peer_rows: Tensor | None = None
    mailbox_edge_feat: Tensor | None = None
    mailbox_ts: Tensor | None = None
    mailbox_write_target_index: Tensor | None = None
    mailbox_write_target_ptr: Tensor | None = None
    mailbox_write_source_pos: Tensor | None = None
    precomputed_commit: bool = False
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
        src_rows: Tensor | None = None,
        dst_rows: Tensor | None = None,
        update_mailbox: bool = True,
        wait_apply: bool = False,
    ) -> "AsyncMemoryUpdateSpec":
        return cls(
            src=src,
            dst=dst,
            src_rows=src_rows,
            dst_rows=dst_rows,
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

    def submit_p2p_memory(
        self,
        updated_nodes: Tensor,
        updated_memory: Tensor,
        updated_ts: Tensor,
        *,
        write_layout: MemoryWriteLayout | None = None,
    ) -> AsyncCommitHandle:
        mem_layout = (
            write_layout
            if write_layout is not None
            else self.memory_runtime.build_write_layout(updated_nodes)
        )
        mem_handle = self.memory_runtime.write(mem_layout, updated_memory, updated_ts)
        return AsyncCommitHandle(memory_handle=mem_handle)

    def submit_p2p_mailbox(
        self,
        mailbox_nodes: Tensor,
        mailbox_msg: Tensor,
        mailbox_ts: Tensor,
        *,
        write_layout: MailboxWriteLayout | None = None,
    ) -> AsyncCommitHandle:
        if self.mailbox_runtime is None:
            return AsyncCommitHandle()
        mail_layout = (
            write_layout
            if write_layout is not None
            else self.mailbox_runtime.build_write_layout(mailbox_nodes)
        )
        mail_handle = self.mailbox_runtime.write(mail_layout, mailbox_msg, mailbox_ts)
        return AsyncCommitHandle(mailbox_handle=mail_handle)

    def submit_shared(
        self,
        updated_nodes: Tensor,
        updated_memory: Tensor,
        updated_ts: Tensor,
        *,
        memory_replica_index: Optional[ReplicaPushIndex] = None,
        mailbox_nodes: Optional[Tensor] = None,
        mailbox_replica_index: Optional[ReplicaPushIndex] = None,
        mailbox_snapshot: Optional[Tensor] = None,
        mailbox_snapshot_ts: Optional[Tensor] = None,
    ) -> AsyncCommitHandle:
        mem_replica_handle = None
        mail_replica_handle = None
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
            memory_replica_handle=mem_replica_handle,
            mailbox_replica_handle=mail_replica_handle,
        )

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
        memory_write_layout: MemoryWriteLayout | None = None,
        mailbox_write_layout: MailboxWriteLayout | None = None,
    ) -> AsyncCommitHandle:
        return _merge_handles(
            self.submit_shared(
                updated_nodes,
                updated_memory,
                updated_ts,
                memory_replica_index=memory_replica_index,
                mailbox_nodes=mailbox_nodes,
                mailbox_replica_index=mailbox_replica_index,
                mailbox_snapshot=mailbox_snapshot,
                mailbox_snapshot_ts=mailbox_snapshot_ts,
            ),
            self.submit_p2p_memory(
                updated_nodes,
                updated_memory,
                updated_ts,
                write_layout=memory_write_layout,
            ),
            self.submit_p2p_mailbox(
                mailbox_nodes,
                mailbox_msg,
                mailbox_ts,
                write_layout=mailbox_write_layout,
            )
            if mailbox_nodes is not None and mailbox_msg is not None and mailbox_ts is not None
            else AsyncCommitHandle(),
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
        historical_cache: SharedHistoricalCache | None = None,
        historical_filter: HistoricalDeltaFilter | None = None,
        use_staged_commit: bool = False,
        use_shared_filter: bool = True,
        enable_delta_compensation: bool = False,
        delta_compensation_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_updater = base_updater
        self.committer = committer
        self.historical_blend = historical_blend
        self.historical_cache = historical_cache
        self.historical_filter = historical_filter
        self.use_staged_commit = bool(use_staged_commit)
        self.use_shared_filter = bool(use_shared_filter)
        self.enable_delta_compensation = bool(enable_delta_compensation)
        self.delta_compensation_gamma = float(delta_compensation_gamma)
        self.delta_gamma = nn.Parameter(torch.tensor([self.delta_compensation_gamma], dtype=torch.float32))
        self.last_updated_memory: Tensor | None = None
        self.last_updated_ts: Tensor | None = None
        self.last_updated_nid: Tensor | None = None
        self.last_commit_handle: AsyncCommitHandle | None = None
        self.pending_shared_handle: AsyncCommitHandle | None = None
        self.pending_async_handle: AsyncCommitHandle | None = None
        self.pending_shared_nodes: Tensor | None = None
        self._diag_stats: dict[str, float] = {
            "commit_memory_row_path_count": 0.0,
            "commit_memory_row_fallback_count": 0.0,
            "commit_mailbox_row_path_count": 0.0,
            "commit_mailbox_row_fallback_count": 0.0,
        }

    def reset_diag_stats(self) -> None:
        for key in self._diag_stats:
            self._diag_stats[key] = 0.0

    def pop_diag_stats(self) -> dict[str, float]:
        out = dict(self._diag_stats)
        self.reset_diag_stats()
        return out

    def reset_state(self) -> None:
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.last_commit_handle = None
        self.pending_shared_handle = None
        self.pending_async_handle = None
        self.pending_shared_nodes = None
        if self.historical_cache is not None:
            self.historical_cache.reset_state()
        if self.historical_filter is not None:
            self.historical_filter.reset_state()
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
        updated = self._maybe_delta_compensate(updated, nid, ts)
        self.last_updated_memory = updated.detach().clone()
        self.last_updated_ts = ts.detach().clone()
        self.last_updated_nid = nid.detach().clone()

        if spec is not None:
            self.last_commit_handle = self.submit_commit(spec)
            if spec.wait_apply:
                self.synchronize_shared()
                self.handle_last_async()
        return updated

    def submit_commit(self, spec: AsyncMemoryUpdateSpec) -> AsyncCommitHandle:
        prepared = self._prepare_commit_inputs(spec)
        if not self.use_staged_commit:
            handle = self.committer.submit(
                prepared.memory_nodes,
                prepared.memory_values,
                prepared.memory_ts,
                mailbox_nodes=prepared.mailbox_nodes,
                mailbox_msg=prepared.mailbox_msg,
                mailbox_ts=prepared.mailbox_ts,
                memory_replica_index=prepared.spec.memory_replica_index,
                mailbox_replica_index=prepared.spec.mailbox_replica_index,
                mailbox_snapshot=prepared.spec.mailbox_snapshot,
                mailbox_snapshot_ts=prepared.spec.mailbox_snapshot_ts,
                memory_write_layout=prepared.memory_write_layout,
                mailbox_write_layout=prepared.mailbox_write_layout,
            )
            self.pending_shared_handle = None
            self.pending_async_handle = None
            return handle
        self.drain_before_submit()
        return self._submit_staged_prepared(prepared)

    def submit_empty_commit(self, spec: AsyncMemoryUpdateSpec) -> AsyncCommitHandle:
        memory_runtime = self.committer.memory_runtime
        memory = memory_runtime.store.memory
        ts = memory_runtime.store.ts
        device = memory.device
        memory_nodes = torch.empty((0,), dtype=torch.long, device=device)
        memory_values = memory.new_empty((0, int(memory.size(1))))
        memory_ts = ts.new_empty((0,))
        mailbox_nodes = None
        mailbox_msg = None
        mailbox_ts = None
        mailbox_runtime = getattr(self.committer, "mailbox_runtime", None)
        if spec.update_mailbox and mailbox_runtime is not None:
            mailbox_store = mailbox_runtime.store
            mailbox = mailbox_store.mailbox
            mailbox_nodes = torch.empty((0,), dtype=torch.long, device=mailbox.device)
            mailbox_msg = mailbox.new_empty((0, int(mailbox.size(2))))
            mailbox_ts = mailbox_store.mailbox_ts.new_empty((0,))
        if not self.use_staged_commit:
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
        prepared = _PreparedCommitInputs(
            spec=spec,
            memory_nodes=memory_nodes,
            memory_values=memory_values,
            memory_ts=memory_ts,
            mailbox_nodes=mailbox_nodes,
            mailbox_msg=mailbox_msg,
            mailbox_ts=mailbox_ts,
        )
        self.drain_before_submit()
        return self._submit_staged_prepared(prepared)

    def drain_before_submit(self) -> None:
        if not self.use_staged_commit:
            return
        self.synchronize_shared()
        self.handle_last_async()

    def _submit_staged_prepared(self, prepared: "_PreparedCommitInputs") -> AsyncCommitHandle:
        shared_handle = self.submit_shared(prepared)
        async_handle = _merge_handles(
            self.submit_p2p_memory(prepared),
            self.submit_p2p_mailbox(prepared),
        )
        self.pending_shared_handle = shared_handle if _has_any_handle(shared_handle) else None
        self.pending_shared_nodes = prepared.shared_nodes.detach().clone() if prepared.shared_nodes is not None else None
        self.pending_async_handle = async_handle if _has_any_handle(async_handle) else None
        return _merge_handles(shared_handle, async_handle)

    def synchronize_shared(self) -> None:
        handle = self.pending_shared_handle
        shared_nodes = self.pending_shared_nodes
        self.pending_shared_handle = None
        self.pending_shared_nodes = None
        if handle is not None:
            handle.wait_apply()
            self._refresh_shared_history(shared_nodes)

    def handle_last_async(self) -> None:
        handle = self.pending_async_handle
        self.pending_async_handle = None
        if handle is not None:
            handle.wait_apply()

    def wait_pending(self) -> None:
        self.synchronize_shared()
        self.handle_last_async()

    def submit_shared(self, prepared: "_PreparedCommitInputs") -> AsyncCommitHandle:
        shared_nodes = prepared.shared_nodes
        shared_memory = prepared.shared_memory
        shared_ts = prepared.shared_ts
        shared_mailbox_nodes = prepared.shared_mailbox_nodes
        shared_mailbox_snapshot = prepared.shared_mailbox_snapshot
        shared_mailbox_snapshot_ts = prepared.shared_mailbox_snapshot_ts
        need_shared_memory_collective = prepared.spec.memory_replica_index is not None
        need_shared_mailbox_collective = (
            prepared.spec.mailbox_replica_index is not None
            and getattr(self.committer, "mailbox_runtime", None) is not None
        )
        has_shared_memory = (
            shared_nodes is not None
            and shared_memory is not None
            and shared_ts is not None
        )
        has_shared_mailbox = (
            need_shared_mailbox_collective
            and shared_mailbox_nodes is not None
            and shared_mailbox_snapshot is not None
            and shared_mailbox_snapshot_ts is not None
        )
        if not need_shared_memory_collective and not need_shared_mailbox_collective:
            return AsyncCommitHandle()
        if any(value is not None for value in (shared_nodes, shared_memory, shared_ts)) and not has_shared_memory:
            raise RuntimeError("shared memory payload is incomplete")
        if any(
            value is not None
            for value in (shared_mailbox_nodes, shared_mailbox_snapshot, shared_mailbox_snapshot_ts)
        ) and not has_shared_mailbox:
            raise RuntimeError("shared mailbox payload is incomplete")
        if not has_shared_memory:
            shared_nodes = prepared.memory_nodes.new_empty((0,))
            shared_memory = prepared.memory_values.new_empty((0, prepared.memory_values.size(1)))
            shared_ts = prepared.memory_ts.new_empty((0,))
        if not has_shared_mailbox and need_shared_mailbox_collective:
            mailbox_runtime = getattr(self.committer, "mailbox_runtime", None)
            assert mailbox_runtime is not None
            mailbox = mailbox_runtime.store.mailbox
            mailbox_ts = mailbox_runtime.store.mailbox_ts
            shared_mailbox_nodes = prepared.memory_nodes.new_empty((0,))
            shared_mailbox_snapshot = mailbox.new_empty((0, int(mailbox.size(1)), int(mailbox.size(2))))
            shared_mailbox_snapshot_ts = mailbox_ts.new_empty((0, int(mailbox_ts.size(1))))
        return self.committer.submit_shared(
            shared_nodes,
            shared_memory,
            shared_ts,
            memory_replica_index=prepared.spec.memory_replica_index if need_shared_memory_collective else None,
            mailbox_nodes=shared_mailbox_nodes,
            mailbox_replica_index=prepared.spec.mailbox_replica_index if need_shared_mailbox_collective else None,
            mailbox_snapshot=shared_mailbox_snapshot,
            mailbox_snapshot_ts=shared_mailbox_snapshot_ts,
        )

    def submit_p2p_memory(self, prepared: "_PreparedCommitInputs") -> AsyncCommitHandle:
        return self.committer.submit_p2p_memory(
            prepared.memory_nodes,
            prepared.memory_values,
            prepared.memory_ts,
            write_layout=prepared.memory_write_layout,
        )

    def submit_p2p_mailbox(self, prepared: "_PreparedCommitInputs") -> AsyncCommitHandle:
        if prepared.mailbox_nodes is None or prepared.mailbox_msg is None or prepared.mailbox_ts is None:
            return AsyncCommitHandle()
        if prepared.mailbox_write_layout is None:
            return self.committer.submit_p2p_mailbox(
                prepared.mailbox_nodes,
                prepared.mailbox_msg,
                prepared.mailbox_ts,
            )
        return self.committer.submit_p2p_mailbox(
            prepared.mailbox_nodes,
            prepared.mailbox_msg,
            prepared.mailbox_ts,
            write_layout=prepared.mailbox_write_layout,
        )

    def _refresh_shared_history(self, shared_nodes: Tensor | None) -> None:
        if shared_nodes is None or shared_nodes.numel() == 0:
            return
        memory_runtime = getattr(self.committer, "memory_runtime", None)
        if memory_runtime is None or not hasattr(memory_runtime, "store") or not hasattr(memory_runtime, "index"):
            return
        nodes = shared_nodes.reshape(-1).long()
        nodes, order = torch.sort(nodes)
        keep = torch.ones(int(nodes.numel()), dtype=torch.bool, device=nodes.device)
        if int(nodes.numel()) > 1:
            keep[1:] = nodes[1:] != nodes[:-1]
        nodes = nodes[keep]
        if nodes.numel() == 0:
            return
        master_index = memory_runtime.index.master_for(
            nodes.to(memory_runtime.index.master_dist_index.device)
        )
        rows = dist_index_loc(master_index).to(memory_runtime.store.device)
        memory, ts = memory_runtime.store.gather_rows(rows)
        if self.historical_cache is not None:
            self.historical_cache._ensure_capacity(
                int(nodes.max().item()) + 1,
                self.historical_cache.historical_memory.device,
                self.historical_cache.historical_memory.dtype,
            )
            cache_nodes = nodes.to(self.historical_cache.historical_memory.device)
            self.historical_cache.historical_memory[cache_nodes] = memory.to(
                self.historical_cache.historical_memory.device,
                dtype=self.historical_cache.historical_memory.dtype,
            )
            self.historical_cache.historical_ts[cache_nodes] = ts.to(
                self.historical_cache.historical_ts.device,
                dtype=self.historical_cache.historical_ts.dtype,
            )

    def _prepare_commit_inputs(self, spec: AsyncMemoryUpdateSpec) -> "_PreparedCommitInputs":
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
        memory_rows = spec.memory_rows
        if memory_rows is None and spec.src_rows is not None and spec.dst_rows is not None:
            memory_rows = torch.cat([spec.src_rows, spec.dst_rows], dim=0)
        if memory_rows is not None:
            memory_rows = _checked_rows(memory_rows, updated, name="memory_rows")
            if spec.precomputed_commit or _rows_match_nodes(nid, memory_rows, memory_nodes):
                self._diag_stats["commit_memory_row_path_count"] += 1.0
                memory_values = updated.index_select(0, memory_rows)
                memory_ts = updated_ts.reshape(-1, 1).index_select(0, memory_rows).reshape(-1)
            else:
                self._diag_stats["commit_memory_row_fallback_count"] += 1.0
                memory_values = _safe_index(nid, memory_nodes, updated)
                memory_ts = _safe_index(nid, memory_nodes, updated_ts.reshape(-1, 1))
                if memory_values is None or memory_ts is None:
                    raise RuntimeError("memory_nodes are not covered by updated node ids")
                memory_ts = memory_ts.reshape(-1)
        else:
            memory_values = _safe_index(nid, memory_nodes, updated)
            memory_ts = _safe_index(nid, memory_nodes, updated_ts.reshape(-1, 1))
            if memory_values is None or memory_ts is None:
                raise RuntimeError("memory_nodes are not covered by updated node ids")
            memory_ts = memory_ts.reshape(-1)
        if not spec.precomputed_commit:
            memory_nodes, memory_values, memory_ts = _latest_payload_by_key(memory_nodes, memory_values, memory_ts)
        memory_write_layout = _memory_write_layout_from_spec(spec)

        mailbox_nodes = None
        mailbox_msg = None
        mailbox_ts = None
        if (
            spec.update_mailbox
            and spec.mailbox_nodes is not None
            and spec.mailbox_self_rows is not None
            and spec.mailbox_peer_rows is not None
            and spec.mailbox_ts is not None
        ):
            mailbox_nodes = spec.mailbox_nodes
            self_rows = _checked_rows(spec.mailbox_self_rows, updated, name="mailbox_self_rows")
            peer_rows = _checked_rows(spec.mailbox_peer_rows, updated, name="mailbox_peer_rows")
            self_mem = updated.index_select(0, self_rows).reshape(int(self_rows.numel()), -1)
            peer_mem = updated.index_select(0, peer_rows).reshape(int(peer_rows.numel()), -1)
            mailbox_msg = torch.cat([self_mem, peer_mem], dim=-1)
            if spec.mailbox_edge_feat is not None:
                edge = spec.mailbox_edge_feat.to(self_mem.device, dtype=self_mem.dtype).reshape(int(self_rows.numel()), -1)
                mailbox_msg = torch.cat([mailbox_msg, edge], dim=-1)
            mailbox_ts = spec.mailbox_ts.reshape(-1)
        elif spec.update_mailbox and spec.src is not None and spec.dst is not None and spec.ts is not None:
            mailbox_nodes = torch.cat([spec.src, spec.dst], dim=0)
            if spec.src_rows is not None and spec.dst_rows is not None:
                src_rows = _checked_rows(spec.src_rows, updated, name="src_rows")
                dst_rows = _checked_rows(spec.dst_rows, updated, name="dst_rows")
                if spec.precomputed_commit or (
                    _rows_match_nodes(nid, src_rows, spec.src)
                    and _rows_match_nodes(nid, dst_rows, spec.dst)
                ):
                    self._diag_stats["commit_mailbox_row_path_count"] += 1.0
                    mailbox_msg = _build_mailbox_messages_from_rows(updated, src_rows, dst_rows, spec.edge_feat)
                else:
                    self._diag_stats["commit_mailbox_row_fallback_count"] += 1.0
                    mailbox_msg = _build_mailbox_messages(nid, updated, spec.src, spec.dst, spec.edge_feat)
            else:
                mailbox_msg = _build_mailbox_messages(nid, updated, spec.src, spec.dst, spec.edge_feat)
            mailbox_ts = torch.cat([spec.ts, spec.ts], dim=0)
        if mailbox_nodes is not None and mailbox_msg is not None and mailbox_ts is not None and not spec.precomputed_commit:
            mailbox_nodes, mailbox_msg, mailbox_ts = _latest_payload_by_key(mailbox_nodes, mailbox_msg, mailbox_ts)
        mailbox_write_layout = _mailbox_write_layout_from_spec(spec)
        shared_nodes = None
        shared_memory = None
        shared_ts = None
        if spec.memory_replica_index is not None:
            shared_mask = _replicated_node_mask(memory_nodes, spec.memory_replica_index)
            if self.use_shared_filter and self.historical_cache is not None and shared_mask.any():
                shared_index = memory_nodes[shared_mask].to(device=updated.device, dtype=torch.long)
                candidate_shared_memory = memory_values[shared_mask]
                candidate_shared_ts = memory_ts[shared_mask]
                update_mask = self.historical_cache.historical_check(
                    shared_index,
                    candidate_shared_memory,
                    candidate_shared_ts,
                )
                full_mask = torch.zeros_like(shared_mask)
                full_mask[shared_mask] = update_mask.to(device=shared_mask.device)
                shared_mask = full_mask
            if shared_mask.any():
                shared_nodes = memory_nodes[shared_mask]
                shared_memory = memory_values[shared_mask]
                shared_ts = memory_ts[shared_mask]

        shared_mailbox_nodes = None
        shared_mailbox_snapshot = None
        shared_mailbox_snapshot_ts = None
        if spec.mailbox_replica_index is not None and (spec.mailbox_nodes is not None or mailbox_nodes is not None):
            shared_mailbox_nodes = spec.mailbox_nodes if spec.mailbox_nodes is not None else mailbox_nodes
            shared_mailbox_snapshot = spec.mailbox_snapshot
            shared_mailbox_snapshot_ts = spec.mailbox_snapshot_ts

        return _PreparedCommitInputs(
            spec=spec,
            memory_nodes=memory_nodes,
            memory_values=memory_values,
            memory_ts=memory_ts,
            memory_write_layout=memory_write_layout,
            mailbox_nodes=mailbox_nodes,
            mailbox_msg=mailbox_msg,
            mailbox_ts=mailbox_ts,
            mailbox_write_layout=mailbox_write_layout,
            shared_nodes=shared_nodes,
            shared_memory=shared_memory,
            shared_ts=shared_ts,
            shared_mailbox_nodes=shared_mailbox_nodes,
            shared_mailbox_snapshot=shared_mailbox_snapshot,
            shared_mailbox_snapshot_ts=shared_mailbox_snapshot_ts,
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
        if self.historical_blend is None or self.historical_filter is None:
            return updated
        nid = getattr(self.base_updater, "last_updated_nid", None)
        if nid is None:
            return updated
        shared_mask, his_mem, increment, shared_index = self._historical_compensation_inputs(mfg, updated, nid)
        if shared_mask is None or his_mem is None or increment is None or shared_index is None:
            return updated
        out = self.historical_blend(
            updated,
            shared_mask,
            his_mem,
            increment,
        )
        shared_index = shared_index[shared_mask]
        if shared_index.numel() > 0:
            change = out[shared_mask].detach() - his_mem[shared_mask].detach()
            self.historical_filter.update(shared_index, change)
        return out

    def _historical_compensation_inputs(
        self,
        mfg: Any,
        updated: Tensor,
        nid: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.historical_filter is None:
            return None, None, None, None
        index = nid.to(device=updated.device, dtype=torch.long).reshape(-1)
        if index.numel() == 0:
            return None, None, None, None
        block = _first_block(mfg)
        srcdata = getattr(block, "srcdata", None)
        shared_mask = None
        his_mem = None
        if isinstance(srcdata, dict):
            raw_mask = srcdata.get("shared_mask")
            if raw_mask is not None:
                shared_mask = raw_mask.to(device=updated.device).bool().reshape(-1)
            raw_his_mem = srcdata.get("his_mem")
            if raw_his_mem is not None:
                his_mem = raw_his_mem.to(device=updated.device, dtype=updated.dtype)
        if shared_mask is None:
            memory_runtime = getattr(self.committer, "memory_runtime", None)
            if memory_runtime is None or not hasattr(memory_runtime, "index"):
                return None, None, None, None
            master_index = memory_runtime.index.master_for(
                index.to(memory_runtime.index.master_dist_index.device)
            ).to(index.device)
            shared_mask = dist_index_is_shared(master_index).to(device=updated.device)
        if his_mem is None and self.historical_cache is not None:
            his_mem = self.historical_cache.historical_memory.index_select(
                0,
                index.to(self.historical_cache.historical_memory.device),
            ).to(device=updated.device, dtype=updated.dtype)
        if his_mem is None:
            his_mem = self.historical_filter.get_history(index, device=updated.device, dtype=updated.dtype)
        increment = self.historical_filter.get_increment(index).to(device=updated.device, dtype=updated.dtype)
        if shared_mask.numel() != updated.size(0):
            return None, None, None, None
        if not bool(shared_mask.any().item()):
            return shared_mask, his_mem, increment, index
        transition_dense = his_mem[shared_mask] + increment[shared_mask]
        max_val = transition_dense.max()
        if float(max_val.item()) != 0.0:
            transition_dense = transition_dense - transition_dense.min()
            transition_dense = transition_dense / transition_dense.max().clamp_min(1e-12)
            transition_dense = 2 * transition_dense - 1
        full_increment = torch.zeros_like(increment)
        full_increment[shared_mask] = transition_dense - his_mem[shared_mask]
        return shared_mask, his_mem, full_increment, index

    def _maybe_delta_compensate(self, updated: Tensor, nid: Tensor, ts: Tensor) -> Tensor:
        if not self.enable_delta_compensation or self.historical_filter is None:
            return updated
        if self.historical_blend is not None:
            return updated
        if nid.numel() == 0:
            return updated
        index = nid.to(device=updated.device, dtype=torch.long).reshape(-1)
        if index.numel() == 0:
            return updated
        cache = self.historical_filter
        prev_memory = cache.get_history(index, device=updated.device, dtype=updated.dtype)
        with torch.no_grad():
            transition_dense = cache.get_increment(index).to(device=updated.device, dtype=updated.dtype)
            max_val = transition_dense.max()
            if float(max_val.item()) != 0.0:
                transition_dense = transition_dense - transition_dense.min()
                transition_dense = transition_dense / transition_dense.max().clamp_min(1e-12)
                transition_dense = 2 * transition_dense - 1
            pred_memory = prev_memory + transition_dense

        gamma = self.delta_gamma.to(device=updated.device, dtype=updated.dtype)
        inc_count = cache.get_count(index).to(device=updated.device).reshape(-1)
        valid = inc_count > 0
        out = updated.clone()
        if bool(valid.any().item()):
            out[valid] = gamma * pred_memory[valid] + (1.0 - gamma) * updated[valid]

        with torch.no_grad():
            if bool(valid.any().item()):
                change = out.detach() - prev_memory.detach()
                cache.update(index[valid], change[valid])
        return out


def _memory_write_layout_from_spec(spec: AsyncMemoryUpdateSpec) -> MemoryWriteLayout | None:
    if not spec.precomputed_commit:
        return None
    if (
        spec.memory_write_target_index is None
        or spec.memory_write_target_ptr is None
        or spec.memory_write_source_pos is None
    ):
        return None
    return MemoryWriteLayout(
        target_index=spec.memory_write_target_index.long().contiguous(),
        target_ptr=spec.memory_write_target_ptr.long().contiguous(),
        source_pos=spec.memory_write_source_pos.long().contiguous(),
    )


def _mailbox_write_layout_from_spec(spec: AsyncMemoryUpdateSpec) -> MailboxWriteLayout | None:
    if not spec.precomputed_commit:
        return None
    if (
        spec.mailbox_write_target_index is None
        or spec.mailbox_write_target_ptr is None
        or spec.mailbox_write_source_pos is None
    ):
        return None
    return MailboxWriteLayout(
        target_index=spec.mailbox_write_target_index.long().contiguous(),
        target_ptr=spec.mailbox_write_target_ptr.long().contiguous(),
        source_pos=spec.mailbox_write_source_pos.long().contiguous(),
    )


@dataclass(slots=True)
class _PreparedCommitInputs:
    spec: AsyncMemoryUpdateSpec
    memory_nodes: Tensor
    memory_values: Tensor
    memory_ts: Tensor
    memory_write_layout: MemoryWriteLayout | None = None
    mailbox_nodes: Tensor | None = None
    mailbox_msg: Tensor | None = None
    mailbox_ts: Tensor | None = None
    mailbox_write_layout: MailboxWriteLayout | None = None
    shared_nodes: Tensor | None = None
    shared_memory: Tensor | None = None
    shared_ts: Tensor | None = None
    shared_mailbox_nodes: Tensor | None = None
    shared_mailbox_snapshot: Tensor | None = None
    shared_mailbox_snapshot_ts: Tensor | None = None


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
    num_edges = int(src_mem.size(0))
    mem_dim = int(src_mem.size(1))
    edge_dim = 0 if edge_feat is None else int(edge_feat.reshape(num_edges, -1).size(1))
    out = src_mem.new_empty((num_edges * 2, mem_dim * 2 + edge_dim))
    out[:num_edges, :mem_dim] = src_mem
    out[:num_edges, mem_dim : mem_dim * 2] = dst_mem
    out[num_edges:, :mem_dim] = dst_mem
    out[num_edges:, mem_dim : mem_dim * 2] = src_mem
    if edge_feat is not None:
        edge = edge_feat.to(src_mem.device, dtype=src_mem.dtype).reshape(num_edges, -1)
        out[:num_edges, mem_dim * 2 :] = edge
        out[num_edges:, mem_dim * 2 :] = edge
    return out


def _build_mailbox_messages_from_rows(
    updated: Tensor,
    src_rows: Tensor,
    dst_rows: Tensor,
    edge_feat: Tensor | None,
) -> Tensor:
    src_rows = _checked_rows(src_rows, updated, name="src_rows")
    dst_rows = _checked_rows(dst_rows, updated, name="dst_rows")
    src_mem = updated.index_select(0, src_rows).reshape(int(src_rows.numel()), -1)
    dst_mem = updated.index_select(0, dst_rows).reshape(int(dst_rows.numel()), -1)
    num_edges = int(src_mem.size(0))
    mem_dim = int(src_mem.size(1))
    edge_dim = 0 if edge_feat is None else int(edge_feat.reshape(num_edges, -1).size(1))
    out = src_mem.new_empty((num_edges * 2, mem_dim * 2 + edge_dim))
    out[:num_edges, :mem_dim] = src_mem
    out[:num_edges, mem_dim : mem_dim * 2] = dst_mem
    out[num_edges:, :mem_dim] = dst_mem
    out[num_edges:, mem_dim : mem_dim * 2] = src_mem
    if edge_feat is not None:
        edge = edge_feat.to(src_mem.device, dtype=src_mem.dtype).reshape(num_edges, -1)
        out[:num_edges, mem_dim * 2 :] = edge
        out[num_edges:, mem_dim * 2 :] = edge
    return out


def _checked_rows(rows: Tensor, values: Tensor, *, name: str) -> Tensor:
    rows = rows.to(values.device).long().reshape(-1)
    if rows.numel() == 0:
        return rows
    if int(rows.min().item()) < 0 or int(rows.max().item()) >= int(values.size(0)):
        raise RuntimeError(f"{name} contain rows outside updated memory")
    return rows


def _check_rows_match_nodes(nid: Tensor, rows: Tensor, nodes: Tensor) -> None:
    if _rows_match_nodes(nid, rows, nodes):
        return
    raise RuntimeError("commit row ids do not match commit node ids")


def _rows_match_nodes(nid: Tensor, rows: Tensor, nodes: Tensor) -> bool:
    rows = _checked_rows(rows, nid.reshape(-1, 1), name="rows")
    actual = nid.to(rows.device).long().reshape(-1).index_select(0, rows)
    expected = nodes.to(rows.device).long().reshape(-1)
    return torch.equal(actual, expected)


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


def _latest_payload_by_key(key: Tensor, payload: Tensor, ts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    key = key.long().reshape(-1)
    ts = ts.reshape(-1)
    if key.numel() <= 1:
        return key.contiguous(), payload.contiguous(), ts.contiguous()
    unique, inverse = torch.unique(key, return_inverse=True)
    if unique.numel() == key.numel():
        return key.contiguous(), payload.contiguous(), ts.contiguous()
    if torch.is_floating_point(ts):
        init_value: float | int = float("-inf")
    else:
        init_value = torch.iinfo(ts.dtype).min
    latest_ts = torch.full((unique.numel(),), init_value, dtype=ts.dtype, device=ts.device)
    latest_ts.scatter_reduce_(0, inverse.to(ts.device), ts, reduce="amax", include_self=True)
    pos = torch.arange(key.numel(), dtype=torch.long, device=key.device)
    sentinel = torch.full_like(pos, key.numel())
    selected_pos = torch.where(ts.to(key.device) == latest_ts.to(key.device).index_select(0, inverse), pos, sentinel)
    selected = torch.full((unique.numel(),), key.numel(), dtype=torch.long, device=key.device)
    selected.scatter_reduce_(0, inverse, selected_pos, reduce="amin", include_self=True)
    return (
        unique.contiguous(),
        payload.index_select(0, selected.to(payload.device)).contiguous(),
        latest_ts.contiguous(),
    )


def _merge_handles(*handles: AsyncCommitHandle) -> AsyncCommitHandle:
    out = AsyncCommitHandle()
    for handle in handles:
        if handle.memory_handle is not None:
            out.memory_handle = handle.memory_handle
        if handle.mailbox_handle is not None:
            out.mailbox_handle = handle.mailbox_handle
        if handle.memory_replica_handle is not None:
            out.memory_replica_handle = handle.memory_replica_handle
        if handle.mailbox_replica_handle is not None:
            out.mailbox_replica_handle = handle.mailbox_replica_handle
    return out


def _has_any_handle(handle: AsyncCommitHandle) -> bool:
    return any(
        part is not None
        for part in (
            handle.memory_handle,
            handle.mailbox_handle,
            handle.memory_replica_handle,
            handle.mailbox_replica_handle,
        )
    )


def _replicated_node_mask(node_ids: Tensor, replica_index: ReplicaPushIndex) -> Tensor:
    nodes = node_ids.long().to(replica_index.replica_ptr.device)
    starts = replica_index.replica_ptr.index_select(0, nodes)
    ends = replica_index.replica_ptr.index_select(0, nodes + 1)
    return (ends - starts).to(node_ids.device) > 0


def _cosine_distance(x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
    x_norm = x.norm(dim=-1).clamp_min(eps)
    y_norm = y.norm(dim=-1).clamp_min(eps)
    sim = (x * y).sum(dim=-1) / (x_norm * y_norm)
    return 1 - sim.clamp(-1, 1)
