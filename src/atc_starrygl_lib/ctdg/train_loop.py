from __future__ import annotations

from collections import defaultdict
import time
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part
from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.memory import AsyncMemoryUpdateSpec
from atc_starrygl_lib.memory.async_updater import (
    _build_mailbox_messages,
    _build_mailbox_messages_from_rows,
    _checked_rows,
    _rows_match_nodes,
)
from atc_starrygl_lib.runtime.async_queue import AsyncWorkQueue


def train_epoch(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    task: Any,
    optimizer: torch.optim.Optimizer,
    *,
    split: str = "train",
    memory_commit: Any = None,
) -> dict[str, float]:
    """Run one explicit CTDG train epoch over session.iter_batches(split)."""
    encoder.train()
    head.train()
    runtime_cfg = dict(getattr(getattr(session, "ctx", None), "config", {}).get("runtime", {}))
    compute_train_metrics = bool(runtime_cfg.get("train_compute_metrics", True))
    sync_timing = bool(runtime_cfg.get("profile_sync_timing", False))
    sync_device = str(getattr(getattr(session, "ctx", None), "device", ""))
    schedule_async_commit = bool(runtime_cfg.get("schedule_async_commit", False))

    epoch_t0 = time.perf_counter()
    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    stage = {
        "batch_wait_seconds": 0.0,
        "encode_seconds": 0.0,
        "head_loss_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_step_seconds": 0.0,
        "memory_commit_seconds": 0.0,
        "metrics_seconds": 0.0,
        "finalize_seconds": 0.0,
        "batches": 0.0,
    }
    sync_stage = {
        "memory_commit_seconds": 0.0,
        "batch_wait_seconds": 0.0,
        "encode_seconds": 0.0,
        "head_loss_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_step_seconds": 0.0,
        "metrics_seconds": 0.0,
        "epoch_tail_seconds": 0.0,
        "finalize_seconds": 0.0,
    }
    backend = getattr(session, "backend", None)
    if backend is not None and hasattr(backend, "reset_profile_stats"):
        backend.reset_profile_stats()
    if memory_commit is not None and hasattr(memory_commit, "reset_profile_stats"):
        memory_commit.reset_profile_stats()
    commit_queue: AsyncWorkQueue[None] | None = None
    if memory_commit is not None and schedule_async_commit:
        commit_queue = AsyncWorkQueue(max_workers=1)
    iterator = iter(session.iter_batches(split))
    try:
        while True:
            t_commit_wait = time.perf_counter()
            if commit_queue is None:
                _wait_pending_memory_commit(memory_commit)
            else:
                if _commit_waits_at_batch_boundary(memory_commit):
                    _drain_commit_queue(commit_queue, wait_all=False)
            sync_stage["memory_commit_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["memory_commit_seconds"] += float(time.perf_counter() - t_commit_wait)
            t_wait = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            sync_stage["batch_wait_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["batch_wait_seconds"] += float(time.perf_counter() - t_wait)
            optimizer.zero_grad(set_to_none=True)
            t_encode = time.perf_counter()
            emb = encode_batch(encoder, batch)
            sync_stage["encode_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["encode_seconds"] += float(time.perf_counter() - t_encode)
            t_head = time.perf_counter()
            output = head(emb, batch)
            loss = task.compute_loss(output, batch)
            sync_stage["head_loss_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["head_loss_seconds"] += float(time.perf_counter() - t_head)

            t_backward = time.perf_counter()
            loss.backward()
            sync_stage["backward_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["backward_seconds"] += float(time.perf_counter() - t_backward)
            t_step = time.perf_counter()
            optimizer.step()
            sync_stage["optimizer_step_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
            stage["optimizer_step_seconds"] += float(time.perf_counter() - t_step)

            if memory_commit is not None:
                if commit_queue is None:
                    memory_commit(encoder, batch)
                    sync_stage["memory_commit_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
                else:
                    # Keep one in-flight commit so i can overlap with i+1 compute.
                    _drain_commit_queue(commit_queue, wait_all=False)
                    commit_queue.submit(_run_memory_commit, memory_commit, encoder, batch)

            losses.append(float(loss.detach().item()))
            if compute_train_metrics:
                t_metrics = time.perf_counter()
                _append_metrics(metrics, task.compute_metrics(output, batch))
                sync_stage["metrics_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
                stage["metrics_seconds"] += float(time.perf_counter() - t_metrics)
            stage["batches"] += 1.0
    finally:
        t_commit_wait = time.perf_counter()
        if commit_queue is None:
            _wait_pending_memory_commit(memory_commit)
        else:
            _drain_commit_queue(commit_queue, wait_all=True)
            commit_queue.close()
        sync_stage["memory_commit_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
        stage["memory_commit_seconds"] += float(time.perf_counter() - t_commit_wait)
    sync_stage["epoch_tail_seconds"] += _timed_sync_cuda(sync_timing, sync_device)

    epoch_wall = float(time.perf_counter() - epoch_t0)
    accounted = float(
        stage["batch_wait_seconds"]
        + stage["encode_seconds"]
        + stage["head_loss_seconds"]
        + stage["backward_seconds"]
        + stage["optimizer_step_seconds"]
        + stage["memory_commit_seconds"]
        + stage["metrics_seconds"]
    )
    stage["wall_seconds"] = epoch_wall
    stage["accounted_seconds"] = accounted
    stage["unaccounted_seconds"] = max(0.0, epoch_wall - accounted)

    t_finalize = time.perf_counter()
    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    out.update({f"stage_{k}": float(v) for k, v in stage.items()})
    if memory_commit is not None and hasattr(memory_commit, "pop_profile_stats"):
        out.update({f"stage_{k}": float(v) for k, v in memory_commit.pop_profile_stats().items()})
    if backend is not None and hasattr(backend, "pop_profile_stats"):
        out.update(backend.pop_profile_stats())
    if optimizer is not None and hasattr(optimizer, "pop_profile_stats"):
        out.update({f"stage_optimizer_{k}": float(v) for k, v in optimizer.pop_profile_stats().items()})
    sync_stage["finalize_seconds"] += _timed_sync_cuda(sync_timing, sync_device)
    out.update({f"stage_sync_{k}": float(v) for k, v in sync_stage.items()})
    out["stage_finalize_seconds"] = float(time.perf_counter() - t_finalize)
    return out


@torch.no_grad()
def evaluate(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    task: Any,
    *,
    split: str = "val",
    memory_commit: Any = None,
) -> dict[str, float]:
    """Evaluate CTDG batches, optionally advancing memory after each prediction."""
    encoder.eval()
    head.eval()

    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    iterator = iter(session.iter_batches(split))
    while True:
        _wait_pending_memory_commit(memory_commit)
        try:
            batch = next(iterator)
        except StopIteration:
            break
        emb = encode_batch(encoder, batch)
        output = head(emb, batch)
        loss = task.compute_loss(output, batch)
        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))
        if memory_commit is not None:
            memory_commit(encoder, batch)
    _wait_pending_memory_commit(memory_commit)

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


@torch.no_grad()
def predict(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    *,
    split: str = "test",
    memory_commit: Any = None,
) -> list[tuple[Any, Batch]]:
    """Return raw outputs, optionally updating memory after each emitted output."""
    encoder.eval()
    head.eval()
    outputs: list[tuple[Any, Batch]] = []
    iterator = iter(session.iter_batches(split))
    while True:
        _wait_pending_memory_commit(memory_commit)
        try:
            batch = next(iterator)
        except StopIteration:
            break
        output = head(encode_batch(encoder, batch), batch)
        outputs.append((output, batch))
        if memory_commit is not None:
            memory_commit(encoder, batch)
    _wait_pending_memory_commit(memory_commit)
    return outputs


def encode_batch(encoder: torch.nn.Module, batch: Batch) -> Tensor:
    if hasattr(encoder, "module"):
        module = encoder.module
        if hasattr(module, "encode"):
            return module.encode(batch.graph)
    if hasattr(encoder, "encode"):
        return encoder.encode(batch.graph)
    return encoder(batch.graph)


def _maybe_sync_cuda(enabled: bool, device: str | None) -> None:
    if not enabled:
        return
    if not torch.cuda.is_available():
        return
    if device is None or not str(device).startswith("cuda"):
        return
    torch.cuda.synchronize(torch.device(str(device)))


def _timed_sync_cuda(enabled: bool, device: str | None) -> float:
    t0 = time.perf_counter()
    _maybe_sync_cuda(enabled, device)
    return float(time.perf_counter() - t0)


class CTDGMemoryCommitHook:
    """Explicit post-step memory/mailbox writeback hook for CTDG training."""

    def __init__(
        self,
        updater: Any = None,
        *,
        memory_replica_index: Any = None,
        mailbox_replica_index: Any = None,
        mailbox_runtime: Any = None,
        wait_apply: bool = False,
        wait_mode: str = "legacy",
    ) -> None:
        self.updater = updater
        self.memory_replica_index = memory_replica_index
        self.mailbox_replica_index = mailbox_replica_index
        self.mailbox_runtime = mailbox_runtime
        self.wait_apply = bool(wait_apply)
        self.wait_mode = str(wait_mode).strip().lower()
        if self.wait_mode not in {"legacy", "memshare"}:
            raise ValueError(f"unsupported wait_mode: {wait_mode!r}")
        self.last_handle = None
        self.last_updater = None
        self._profile_stats = {
            "memory_commit_build_seconds": 0.0,
            "memory_commit_submit_seconds": 0.0,
            "memory_commit_wait_sync_seconds": 0.0,
            "memory_commit_row_path_count": 0.0,
            "memory_commit_row_fallback_count": 0.0,
            "mailbox_commit_row_path_count": 0.0,
            "mailbox_commit_row_fallback_count": 0.0,
        }

    def reset_profile_stats(self) -> None:
        for key in self._profile_stats:
            self._profile_stats[key] = 0.0

    def pop_profile_stats(self) -> dict[str, float]:
        out = dict(self._profile_stats)
        updater = self.updater or self.last_updater
        if updater is not None and hasattr(updater, "pop_diag_stats"):
            diag = updater.pop_diag_stats()
            out["memory_commit_row_path_count"] += float(diag.get("commit_memory_row_path_count", 0.0))
            out["memory_commit_row_fallback_count"] += float(diag.get("commit_memory_row_fallback_count", 0.0))
            out["mailbox_commit_row_path_count"] += float(diag.get("commit_mailbox_row_path_count", 0.0))
            out["mailbox_commit_row_fallback_count"] += float(diag.get("commit_mailbox_row_fallback_count", 0.0))
        self.reset_profile_stats()
        return out

    def __call__(self, encoder: torch.nn.Module, batch: Batch) -> None:
        t_build = time.perf_counter()
        updater = self.updater or _find_memory_updater(encoder)
        if updater is None:
            self._profile_stats["memory_commit_build_seconds"] += float(time.perf_counter() - t_build)
            return
        self.last_updater = updater
        edge_feat = _positive_edge_feature(batch)
        spec = AsyncMemoryUpdateSpec.from_edges(
            batch.src,
            batch.dst,
            batch.ts,
            edge_feat=edge_feat,
            src_rows=batch.commit_src_rows if batch.commit_src_rows is not None else batch.pos_src,
            dst_rows=batch.commit_dst_rows if batch.commit_dst_rows is not None else batch.pos_dst,
            wait_apply=self.wait_apply,
        ) if batch.src is not None and batch.dst is not None and batch.ts is not None else AsyncMemoryUpdateSpec(
            wait_apply=self.wait_apply,
        )
        spec.memory_replica_index = self.memory_replica_index
        self._populate_mailbox_replica_spec(updater, spec)
        self._profile_stats["memory_commit_build_seconds"] += float(time.perf_counter() - t_build)
        t_submit = time.perf_counter()
        if hasattr(updater, "submit_commit"):
            handle = updater.submit_commit(spec)
        elif hasattr(updater, "commit"):
            handle = updater.commit(spec)
        else:
            self._profile_stats["memory_commit_submit_seconds"] += float(time.perf_counter() - t_submit)
            return
        self._profile_stats["memory_commit_submit_seconds"] += float(time.perf_counter() - t_submit)
        self.last_handle = handle
        t_wait = time.perf_counter()
        if self.wait_apply or self.wait_mode == "memshare":
            self._wait_pending_with_updater(updater)
        elif handle is not None and hasattr(handle, "wait_apply") and not self._updater_owns_handle_wait(updater):
            handle.wait_apply()
        self._profile_stats["memory_commit_wait_sync_seconds"] += float(time.perf_counter() - t_wait)

    def wait_pending(self) -> None:
        t_wait = time.perf_counter()
        updater = self.updater or self.last_updater
        if updater is not None:
            self._wait_pending_with_updater(updater)
        handle = self.last_handle
        self.last_handle = None
        if handle is not None and hasattr(handle, "wait_apply") and not self._updater_owns_handle_wait(updater):
            handle.wait_apply()
        self._profile_stats["memory_commit_wait_sync_seconds"] += float(time.perf_counter() - t_wait)

    def should_wait_at_batch_boundary(self) -> bool:
        return self.wait_mode == "legacy"

    def _populate_mailbox_replica_spec(self, updater: Any, spec: AsyncMemoryUpdateSpec) -> None:
        if self.mailbox_replica_index is None or self.mailbox_runtime is None:
            return
        if spec.src is None or spec.dst is None or spec.ts is None:
            return
        updated_nid = getattr(updater, "last_updated_nid", None)
        updated_memory = getattr(updater, "last_updated_memory", None)
        if updated_nid is None or updated_memory is None:
            return
        mailbox_nodes = torch.cat([spec.src, spec.dst], dim=0).long().contiguous()
        if spec.src_rows is not None and spec.dst_rows is not None:
            src_rows = _checked_rows(spec.src_rows, updated_memory, name="src_rows")
            dst_rows = _checked_rows(spec.dst_rows, updated_memory, name="dst_rows")
            if _rows_match_nodes(updated_nid, src_rows, spec.src) and _rows_match_nodes(updated_nid, dst_rows, spec.dst):
                mailbox_msg = _build_mailbox_messages_from_rows(updated_memory, src_rows, dst_rows, spec.edge_feat)
            else:
                mailbox_msg = _build_mailbox_messages(updated_nid, updated_memory, spec.src, spec.dst, spec.edge_feat)
        else:
            mailbox_msg = _build_mailbox_messages(updated_nid, updated_memory, spec.src, spec.dst, spec.edge_feat)
        mailbox_ts = torch.cat([spec.ts, spec.ts], dim=0).contiguous()
        target = self.mailbox_runtime.index.master_for(mailbox_nodes).long()
        rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
        local_owner = dist_index_part(target) == rank
        if not local_owner.any():
            return
        mailbox_nodes = mailbox_nodes[local_owner].contiguous()
        mailbox_msg = mailbox_msg[local_owner].contiguous()
        mailbox_ts = mailbox_ts[local_owner].contiguous()
        target = target[local_owner].contiguous()
        if target.numel() > 1:
            unique_target, inverse = torch.unique(target, return_inverse=True)
            latest_ts = torch.full(
                (unique_target.numel(),),
                float("-inf"),
                dtype=mailbox_ts.dtype,
                device=mailbox_ts.device,
            )
            latest_ts.scatter_reduce_(0, inverse, mailbox_ts, reduce="amax", include_self=True)
            pos = torch.arange(target.numel(), dtype=torch.long, device=target.device)
            sentinel = torch.full_like(pos, target.numel())
            selected_pos = torch.where(mailbox_ts == latest_ts[inverse], pos, sentinel)
            selected = torch.full((unique_target.numel(),), target.numel(), dtype=torch.long, device=target.device)
            selected.scatter_reduce_(0, inverse, selected_pos, reduce="amin", include_self=True)
            target = unique_target
            mailbox_nodes = mailbox_nodes.index_select(0, selected).contiguous()
            mailbox_msg = mailbox_msg.index_select(0, selected).contiguous()
            mailbox_ts = latest_ts.contiguous()
        rows = dist_index_loc(target).long()
        oldest = self.mailbox_runtime.store.mailbox_ts.index_select(0, rows).min(dim=1).values
        keep = mailbox_ts.to(device=oldest.device, dtype=oldest.dtype) > oldest
        if not keep.any():
            return
        mailbox_nodes = mailbox_nodes[keep].contiguous()
        mailbox_msg = mailbox_msg[keep].contiguous()
        mailbox_ts = mailbox_ts[keep].contiguous()
        rows = rows[keep].contiguous()
        snapshot, snapshot_ts = self.mailbox_runtime.store.project_append_rows(rows, mailbox_msg, mailbox_ts)
        spec.mailbox_replica_index = self.mailbox_replica_index
        spec.mailbox_nodes = mailbox_nodes
        spec.mailbox_snapshot = snapshot.to(mailbox_nodes.device)
        spec.mailbox_snapshot_ts = snapshot_ts.to(mailbox_nodes.device)

    @staticmethod
    def _wait_pending_with_updater(updater: Any) -> None:
        if hasattr(updater, "wait_pending"):
            updater.wait_pending()
            return
        CTDGMemoryCommitHook._synchronize_shared(updater)
        CTDGMemoryCommitHook._handle_last_async(updater)

    @staticmethod
    def _updater_owns_handle_wait(updater: Any) -> bool:
        if updater is None:
            return False
        # Legacy updater commit path still relies on hook-level handle.wait_apply().
        if hasattr(updater, "use_staged_commit") and not bool(getattr(updater, "use_staged_commit")):
            return False
        return hasattr(updater, "synchronize_shared") or hasattr(updater, "handle_last_async")

    @staticmethod
    def _synchronize_shared(updater: Any) -> None:
        if hasattr(updater, "synchronize_shared"):
            updater.synchronize_shared()

    @staticmethod
    def _handle_last_async(updater: Any) -> None:
        if hasattr(updater, "handle_last_async"):
            updater.handle_last_async()


def _append_metrics(dst: defaultdict[str, list[float]], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        dst[key].append(float(value))


def _mean_metrics(metrics: defaultdict[str, list[float]]) -> dict[str, float]:
    return {key: _mean(values) for key, values in metrics.items()}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _find_memory_updater(module: torch.nn.Module) -> Any:
    current: Any = module
    if hasattr(current, "module"):
        current = current.module
    for name in ("memory_updater", "updater", "memory"):
        candidate = getattr(current, name, None)
        if candidate is not None and (hasattr(candidate, "submit_commit") or hasattr(candidate, "commit")):
            return candidate
    for child in current.modules() if hasattr(current, "modules") else ():
        if child is current:
            continue
        if hasattr(child, "submit_commit") or hasattr(child, "commit"):
            return child
    return None


def _wait_pending_memory_commit(memory_commit: Any) -> None:
    if memory_commit is None:
        return
    if hasattr(memory_commit, "should_wait_at_batch_boundary") and not bool(memory_commit.should_wait_at_batch_boundary()):
        return
    if hasattr(memory_commit, "wait_pending"):
        memory_commit.wait_pending()


def _commit_waits_at_batch_boundary(memory_commit: Any) -> bool:
    if memory_commit is None:
        return False
    if hasattr(memory_commit, "should_wait_at_batch_boundary"):
        return bool(memory_commit.should_wait_at_batch_boundary())
    return True


def _drain_commit_queue(queue: AsyncWorkQueue[None], *, wait_all: bool) -> None:
    while len(queue) > (0 if wait_all else 1):
        queue.pop_result()


def _run_memory_commit(memory_commit: Any, encoder: torch.nn.Module, batch: Batch) -> None:
    memory_commit(encoder, batch)


def _first_edge_feature(graph: Any) -> Tensor | None:
    block = _first_block(graph)
    edata = getattr(block, "edata", None)
    if edata is None:
        return None
    feature = edata.get("f") if hasattr(edata, "get") else (edata["f"] if "f" in edata else None)
    if feature is None and "feat" in edata:
        feature = edata["feat"]
    return feature


def _positive_edge_feature(batch: Batch) -> Tensor | None:
    if batch.edge_feat is not None:
        return batch.edge_feat
    feature = _first_edge_feature(batch.graph)
    if feature is None or batch.src is None:
        return feature
    if int(feature.size(0)) == int(batch.src.numel()):
        return feature
    if feature.dim() <= 1:
        return None
    return feature.new_zeros((int(batch.src.numel()), int(feature.size(-1))))


def _first_block(graph: Any) -> Any:
    if isinstance(graph, (list, tuple)):
        if not graph:
            return None
        first = graph[0]
        if isinstance(first, (list, tuple)):
            return first[0] if first else None
        return first
    return graph
