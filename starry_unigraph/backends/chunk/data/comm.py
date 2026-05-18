"""Async communication pipeline for chunk training.

CommPipeline manages three independent async all-to-all channels:

  spatial  — node feature exchange (SpatialRouteData)
  memory   — memory/state cache update (MemoryRouteData)
  replica  — hot-node replica all-gather (MemoryRouteData.replica_*)

Each channel can be submitted and awaited independently, allowing
any overlap pattern between communication and computation.

Typical training-loop pattern (one batch ahead):

    pipeline = CommPipeline(device, group)

    async def train_epoch(slices):
        for t, (route_s, route_m, features, memory, ts) in enumerate(slices):
            # Submit comms for slice t (non-blocking)
            pipeline.submit_spatial(route_s, features)
            pipeline.submit_memory(route_m, memory, ts)

            # Compute on local data for slice t ...

            # Collect results when needed
            feat_result = await pipeline.await_spatial()
            mem_result  = await pipeline.await_memory()
            if mem_result:
                local_store[mem_result.recv_node_ids] = mem_result.recv_memory

C++ layer responsibility: CUDA kernel primitives (gather, scatter,
pinned-memory DMA).  All scheduling and async coordination is here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .dist_index import dist_index_loc
from .route import MemoryRouteData, SpatialRouteData


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SpatialResult:
    """Features received from other partitions after spatial all-to-all.

    Attributes:
        recv_node_ids: [R] global node IDs received.
        recv_features: [R, feat_dim] feature tensors.
    """
    recv_node_ids: Tensor
    recv_features: Tensor


@dataclass
class MemoryResult:
    """Memory vectors received from other partitions after memory all-to-all.

    Attributes:
        recv_node_ids: [R] global node IDs received.
        recv_memory:   [R, mem_dim] memory tensors.
        recv_ts:       [R] timestamps of received updates.
    """
    recv_node_ids: Tensor
    recv_memory:   Tensor
    recv_ts:       Tensor


@dataclass
class FetchResult:
    """Feature/memory rows fetched for one CTDG sampled batch."""

    local_read_index: Optional[Tensor] = None
    remote_read_index: Optional[Tensor] = None
    local_features: Optional[Tensor] = None
    remote_features: Optional[Tensor] = None
    local_memory: Optional[Tensor] = None
    remote_memory: Optional[Tensor] = None


@dataclass(frozen=True)
class CommHandle:
    """Opaque handle returned by non-blocking communication submission."""

    channel: str


# ---------------------------------------------------------------------------
# Internal: await a single dist.Work handle
# ---------------------------------------------------------------------------

async def _wait_work(work: dist.Work) -> None:
    """Await dist.Work without blocking the asyncio event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, work.wait)


def _split_sizes(ptr: Tensor) -> list[int]:
    """Convert CSR ptr [P+1] to a list of per-partition sizes [P]."""
    return (ptr[1:] - ptr[:-1]).tolist()


def validate_training_comm_backend(device: torch.device) -> None:
    """Validate that distributed CUDA training uses NCCL-backed tensors."""
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() <= 1:
        return
    backend = dist.get_backend()
    if device.type != "cuda":
        raise RuntimeError(
            "Chunk distributed training requires CUDA tensors so internal "
            f"communication can use NCCL; got device={device}."
        )
    if backend != "nccl":
        raise RuntimeError(
            "Chunk distributed training requires NCCL for internal communication; "
            f"current process group backend is {backend!r}."
        )


def _ptr_from_counts(counts: Tensor) -> Tensor:
    ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=counts.device)
    ptr[1:] = counts.to(torch.long).cumsum(0)
    return ptr


def _exchange_recv_ptr(send_ptr: Tensor, group: Optional[dist.ProcessGroup]) -> Tensor:
    """Exchange filtered send sizes and return the matching recv ptr."""
    send_counts = (send_ptr[1:] - send_ptr[:-1]).to(dtype=torch.long).contiguous()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return _ptr_from_counts(recv_counts)


def _memory_change_mask(
    memory: Tensor,
    baseline_memory: Tensor,
    threshold: float,
    metric: str,
) -> Tensor:
    """MemShare-style change-rate check for memory communication."""
    if memory.shape != baseline_memory.shape:
        raise ValueError("memory and baseline_memory must have the same shape")
    if threshold <= 0:
        return torch.ones(memory.size(0), dtype=torch.bool, device=memory.device)
    if memory.numel() == 0:
        return torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)

    metric = str(metric)
    if metric == "cos":
        change = 1.0 - F.cosine_similarity(memory, baseline_memory, dim=1, eps=1e-12)
    elif metric in {"l2", "mse"}:
        change = (memory - baseline_memory).pow(2).sum(dim=1)
        if metric == "l2":
            change = change.sqrt()
    else:
        raise ValueError(f"Unknown memory change metric: {metric}")
    return change > float(threshold)


# ---------------------------------------------------------------------------
# Single-channel state
# ---------------------------------------------------------------------------

@dataclass
class _PendingOp:
    """In-flight async operation handles and pre-allocated recv buffers."""
    works:     list           # list of dist.Work handles
    recv_bufs: list[Tensor]   # pre-allocated output buffers (in order)


# ---------------------------------------------------------------------------
# CommPipeline
# ---------------------------------------------------------------------------

class CommPipeline:
    """Three-channel async communication pipeline.

    Channels (independent):
      spatial  — feature all-to-all   (SpatialRouteData)
      memory   — memory all-to-all    (MemoryRouteData)
      replica  — hot-node all-gather  (MemoryRouteData.replica_*)

    Args:
        device: GPU device for communication buffers.
        group:  optional process group (defaults to WORLD).
    """

    def __init__(
        self,
        device: torch.device,
        group:  Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._device = device
        self._group  = group
        self._spatial_pending:  Optional[_PendingOp] = None
        self._memory_pending:   Optional[_PendingOp] = None
        self._replica_pending:  Optional[_PendingOp] = None
        self._fetch_pending:    Optional[_PendingOp] = None
        # Track recv_node_ids so await_ can return them alongside data
        self._spatial_recv_ids: Optional[Tensor] = None
        self._memory_recv_ids:  Optional[Tensor] = None
        self._replica_recv_ids: Optional[Tensor] = None
        self._fetch_result:     Optional[FetchResult] = None

    def _gather_by_dist_index(self, rows: Tensor, index: Optional[Tensor]) -> Optional[Tensor]:
        if index is None or index.numel() == 0:
            return None
        loc = dist_index_loc(index.to(device=rows.device)).long()
        return rows.index_select(0, loc).to(self._device, non_blocking=True)

    @staticmethod
    def _empty_like_rows(rows: Tensor, count: int = 0) -> Tensor:
        return rows.new_empty((int(count), *rows.shape[1:]))

    def _remote_fetch_rows(
        self,
        remote_read_index: Tensor,
        remote_owners: Tensor,
        rows: Tensor,
    ) -> tuple[Tensor, list]:
        """Request remote rows from owner ranks and return recv buffer + works."""
        if remote_read_index.numel() == 0:
            return self._empty_like_rows(rows.to(self._device), 0), []

        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group=self._group) <= 1:
            return self._gather_by_dist_index(rows, remote_read_index), []

        world_size = dist.get_world_size(group=self._group)
        remote_owners = remote_owners.to(device=self._device, dtype=torch.long).contiguous()
        req_locs = dist_index_loc(remote_read_index).to(device=self._device, dtype=torch.long).contiguous()
        send_counts_t = torch.bincount(remote_owners, minlength=world_size).to(device=self._device, dtype=torch.long)
        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=self._group)
        send_counts = send_counts_t.tolist()
        recv_counts = recv_counts_t.tolist()

        recv_locs = torch.empty(int(recv_counts_t.sum().item()), dtype=torch.long, device=self._device)
        req_work = dist.all_to_all_single(
            recv_locs,
            req_locs,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self._group,
            async_op=True,
        )
        req_work.wait()

        rows_dev = rows.to(self._device, non_blocking=True)
        send_values = rows_dev.index_select(0, recv_locs.long()).contiguous()
        recv_values = self._empty_like_rows(rows_dev, int(send_counts_t.sum().item()))
        value_work = dist.all_to_all_single(
            recv_values,
            send_values,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
            group=self._group,
            async_op=True,
        )
        return recv_values, [value_work]

    def submit_fetch(
        self,
        plan: Any,
        *,
        feature_rows: Optional[Tensor] = None,
        memory_rows: Optional[Tensor] = None,
    ) -> Optional[CommHandle]:
        """Fetch remote feature/memory rows described by a ``FetchPlan``."""
        self._drain_fetch_sync()
        if plan is None:
            return None

        local_read_index = getattr(plan, "local_read_index", None)
        remote_read_index = getattr(plan, "remote_read_index", None)
        remote_owners = getattr(plan, "feature_owners", None)
        works: list = []
        result = FetchResult(
            local_read_index=local_read_index,
            remote_read_index=remote_read_index,
        )

        if feature_rows is not None:
            result.local_features = self._gather_by_dist_index(feature_rows, local_read_index)
            if remote_read_index is not None and remote_read_index.numel() > 0:
                if remote_owners is None:
                    raise ValueError("FetchPlan.feature_owners is required for remote feature fetch")
                result.remote_features, feature_works = self._remote_fetch_rows(
                    remote_read_index.long().contiguous(),
                    remote_owners.long().contiguous(),
                    feature_rows,
                )
                works.extend(feature_works)
            else:
                result.remote_features = self._empty_like_rows(feature_rows.to(self._device), 0)

        if memory_rows is not None:
            result.local_memory = self._gather_by_dist_index(memory_rows, local_read_index)
            if remote_read_index is not None and remote_read_index.numel() > 0:
                if remote_owners is None:
                    raise ValueError("FetchPlan.feature_owners is required for remote memory fetch")
                result.remote_memory, memory_works = self._remote_fetch_rows(
                    remote_read_index.long().contiguous(),
                    remote_owners.long().contiguous(),
                    memory_rows,
                )
                works.extend(memory_works)
            else:
                result.remote_memory = self._empty_like_rows(memory_rows.to(self._device), 0)

        self._fetch_result = result
        self._fetch_pending = _PendingOp(works=works, recv_bufs=[])
        return CommHandle("fetch")

    async def await_fetch(self) -> Optional[FetchResult]:
        if self._fetch_pending is None:
            return None
        await asyncio.gather(*[_wait_work(w) for w in self._fetch_pending.works])
        result = self._fetch_result
        self._fetch_pending = None
        self._fetch_result = None
        return result

    # ------------------------------------------------------------------
    # Spatial channel
    # ------------------------------------------------------------------

    def submit_spatial(
        self,
        route:    SpatialRouteData,
        features: Tensor,           # [num_local_nodes, feat_dim]
    ) -> CommHandle:
        """Start async all-to-all for spatial feature exchange.

        Gathers features[route.send_index] and scatters to peers.
        Concurrent call to submit_spatial before await_spatial is safe:
        the in-flight op is synchronously drained first.
        """
        self._drain_spatial_sync()

        send_data  = features[route.send_index]           # [S, feat_dim]
        recv_total = int(route.recv_ptr[-1])
        recv_buf   = torch.empty(
            recv_total, features.size(1),
            dtype=features.dtype, device=self._device,
        )

        in_split  = _split_sizes(route.send_ptr)
        out_split = _split_sizes(route.recv_ptr)

        work = dist.all_to_all_single(
            recv_buf, send_data,
            output_split_sizes=out_split,
            input_split_sizes=in_split,
            group=self._group,
            async_op=True,
        )
        self._spatial_pending  = _PendingOp(works=[work], recv_bufs=[recv_buf])
        self._spatial_recv_ids = route.recv_node_ids
        return CommHandle("spatial")

    async def await_spatial(self) -> Optional[SpatialResult]:
        """Await in-flight spatial op.  Returns None if none was submitted."""
        if self._spatial_pending is None:
            return None
        await asyncio.gather(*[_wait_work(w) for w in self._spatial_pending.works])
        result = SpatialResult(
            recv_node_ids = self._spatial_recv_ids,
            recv_features = self._spatial_pending.recv_bufs[0],
        )
        self._spatial_pending  = None
        self._spatial_recv_ids = None
        return result

    # ------------------------------------------------------------------
    # Memory channel
    # ------------------------------------------------------------------

    def submit_memory(
        self,
        route:  MemoryRouteData,
        memory: Tensor,    # [D, mem_dim]  D = route.num_unique
        ts:     Tensor,    # [D]
        baseline_memory: Optional[Tensor] = None,
        change_threshold: float = 0.0,
        change_metric: str = "cos",
    ) -> CommHandle:
        """Start async all-to-all for memory/state cache updates.

        ``memory`` and ``ts`` must already be indexed by event_pos
        (caller selects latest or sampled candidate per unique node).
        If ``baseline_memory`` is supplied, only nodes whose memory changed
        beyond ``change_threshold`` are communicated.
        """
        self._drain_memory_sync()
        route = route.to(self._device)
        memory = memory.to(self._device, non_blocking=True)
        ts = ts.to(self._device, non_blocking=True)
        if baseline_memory is not None:
            baseline_memory = baseline_memory.to(self._device, non_blocking=True)

        if baseline_memory is not None:
            keep = _memory_change_mask(memory, baseline_memory, change_threshold, change_metric)
            route = route.filter_updates(keep)
            memory = memory[keep]
            ts = ts[keep]

        recv_ptr = _exchange_recv_ptr(route.send_ptr, self._group)

        recv_total = int(recv_ptr[-1])
        recv_mem   = torch.empty(recv_total, memory.size(1), dtype=memory.dtype, device=self._device)
        recv_ts    = torch.empty(recv_total, dtype=ts.dtype, device=self._device)
        recv_ids   = torch.empty(recv_total, dtype=torch.long, device=self._device)

        in_split  = _split_sizes(route.send_ptr)
        out_split = _split_sizes(recv_ptr)

        w_ids = dist.all_to_all_single(
            recv_ids, route.unique_nodes,
            output_split_sizes=out_split, input_split_sizes=in_split,
            group=self._group, async_op=True,
        )
        w_mem = dist.all_to_all_single(
            recv_mem, memory,
            output_split_sizes=out_split, input_split_sizes=in_split,
            group=self._group, async_op=True,
        )
        w_ts = dist.all_to_all_single(
            recv_ts, ts,
            output_split_sizes=out_split, input_split_sizes=in_split,
            group=self._group, async_op=True,
        )

        self._memory_pending   = _PendingOp(works=[w_ids, w_mem, w_ts],
                                            recv_bufs=[recv_ids, recv_mem, recv_ts])
        self._memory_recv_ids  = recv_ids  # same tensor, aliased for clarity
        return CommHandle("memory")

    async def await_memory(self) -> Optional[MemoryResult]:
        """Await in-flight memory update op."""
        if self._memory_pending is None:
            return None
        await asyncio.gather(*[_wait_work(w) for w in self._memory_pending.works])
        ids, mem, ts = self._memory_pending.recv_bufs
        result = MemoryResult(recv_node_ids=ids, recv_memory=mem, recv_ts=ts)
        self._memory_pending  = None
        self._memory_recv_ids = None
        return result

    # ------------------------------------------------------------------
    # Replica (hot-node) channel
    # ------------------------------------------------------------------

    def submit_replica(
        self,
        route:  MemoryRouteData,
        memory: Tensor,    # [D, mem_dim]
    ) -> Optional[CommHandle]:
        """Start async all-to-all for replica (hot) node memory sync."""
        if not route.has_replicas:
            return None
        self._drain_replica_sync()

        rep_idx  = route.replica_idx                          # [R]
        send_mem = memory[rep_idx]                            # [R, mem_dim]
        rep_ids  = route.unique_nodes[rep_idx]                # [R]

        recv_total = int(route.replica_recv_ptr[-1]) if route.replica_recv_ptr is not None else 0
        recv_mem   = torch.empty(recv_total, memory.size(1), dtype=memory.dtype, device=self._device)
        recv_ids   = torch.empty(recv_total, dtype=torch.long, device=self._device)

        in_split  = _split_sizes(route.replica_send_ptr)
        out_split = _split_sizes(route.replica_recv_ptr) if route.replica_recv_ptr is not None \
                    else [0] * (route.replica_send_ptr.numel() - 1)

        w_ids = dist.all_to_all_single(
            recv_ids, rep_ids,
            output_split_sizes=out_split, input_split_sizes=in_split,
            group=self._group, async_op=True,
        )
        w_mem = dist.all_to_all_single(
            recv_mem, send_mem,
            output_split_sizes=out_split, input_split_sizes=in_split,
            group=self._group, async_op=True,
        )
        self._replica_pending  = _PendingOp(works=[w_ids, w_mem], recv_bufs=[recv_ids, recv_mem])
        return CommHandle("replica")

    async def await_replica(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Await in-flight replica sync.

        Returns:
            (recv_node_ids, recv_memory) or None.
        """
        if self._replica_pending is None:
            return None
        await asyncio.gather(*[_wait_work(w) for w in self._replica_pending.works])
        ids, mem = self._replica_pending.recv_bufs
        self._replica_pending = None
        return ids, mem

    async def await_handle(self, handle: Optional[CommHandle]):
        """Await a handle returned by ``submit_*``.

        This keeps the public interface channel-based while preserving the
        existing ``await_spatial`` / ``await_memory`` / ``await_replica`` calls.
        """
        if handle is None:
            return None
        if handle.channel == "spatial":
            return await self.await_spatial()
        if handle.channel == "memory":
            return await self.await_memory()
        if handle.channel == "replica":
            return await self.await_replica()
        if handle.channel == "fetch":
            return await self.await_fetch()
        raise ValueError(f"Unknown communication channel: {handle.channel}")

    # ------------------------------------------------------------------
    # Synchronous drain helpers (safety valves)
    # ------------------------------------------------------------------

    def _drain_spatial_sync(self) -> None:
        if self._spatial_pending is not None:
            for w in self._spatial_pending.works:
                w.wait()
            self._spatial_pending  = None
            self._spatial_recv_ids = None

    def _drain_memory_sync(self) -> None:
        if self._memory_pending is not None:
            for w in self._memory_pending.works:
                w.wait()
            self._memory_pending  = None
            self._memory_recv_ids = None

    def _drain_replica_sync(self) -> None:
        if self._replica_pending is not None:
            for w in self._replica_pending.works:
                w.wait()
            self._replica_pending = None

    def _drain_fetch_sync(self) -> None:
        if self._fetch_pending is not None:
            for w in self._fetch_pending.works:
                w.wait()
            self._fetch_pending = None
            self._fetch_result = None

    def drain_all_sync(self) -> None:
        """Synchronously wait for all in-flight ops (e.g. at epoch end)."""
        self._drain_spatial_sync()
        self._drain_memory_sync()
        self._drain_replica_sync()
        self._drain_fetch_sync()
