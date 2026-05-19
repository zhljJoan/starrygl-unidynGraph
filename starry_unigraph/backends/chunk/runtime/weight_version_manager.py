"""Async gradient synchronization with version ordering.

Implements WeightVersionManager for ordered async gradient sync:
- Step v forward saves flat param buffer to slot v % slots
- Backward hook writes GradBucket(v)
- Bucket full triggers async all_reduce
- Optimizer apply waits for all < v buckets in order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn


@dataclass
class GradBucket:
    """Gradient bucket for one training step version."""
    version: int
    grads: List[Tensor] = field(default_factory=list)
    work: Optional[dist.Work] = None
    ready: bool = False


class WeightVersionManager:
    """Ordered async gradient synchronization manager.

    Maintains a sliding window of parameter backups and gradient buckets.
    Ensures optimizer.step() is applied in version order even when
    all_reduce ops complete out of order.

    Args:
        model: The model whose parameters are managed
        num_backup_slots: Number of parameter backup slots (K+1)
        group: Optional process group for all_reduce
    """

    def __init__(
        self,
        model: nn.Module,
        num_backup_slots: int = 2,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.model = model
        self.num_backup_slots = num_backup_slots
        self.group = group

        self.active_version: int = 0
        self.optimizer_apply_version: int = 0

        # Flat parameter buffer for fast backup/restore
        self._params: List[Tensor] = [p for p in model.parameters() if p.requires_grad]
        self._param_sizes: List[int] = [int(p.numel()) for p in self._params]
        total_params = sum(self._param_sizes)

        # Backup slots: flat buffers
        self._backup_slots: List[Optional[Tensor]] = [None] * num_backup_slots

        # Pending gradient buckets keyed by version
        self._pending_buckets: Dict[int, GradBucket] = {}

        # Next version to apply to optimizer
        self._next_apply: int = 0

    def _flat_params(self) -> Tensor:
        """Return current parameters as a flat tensor."""
        return torch.cat([p.data.view(-1) for p in self._params])

    def _restore_params(self, flat: Tensor) -> None:
        """Restore parameters from a flat tensor."""
        offset = 0
        for p, size in zip(self._params, self._param_sizes):
            p.data.copy_(flat[offset:offset + size].view(p.shape))
            offset += size

    def save_param_backup(self, version: int) -> None:
        """Save current parameters to backup slot for version v.

        Call this before forward pass of step v.

        Args:
            version: Training step version
        """
        slot = version % self.num_backup_slots
        self._backup_slots[slot] = self._flat_params().clone()
        self.active_version = version

    def restore_param_backup(self, version: int) -> None:
        """Restore parameters from backup slot for version v.

        Args:
            version: Training step version to restore
        """
        slot = version % self.num_backup_slots
        backup = self._backup_slots[slot]
        if backup is None:
            raise RuntimeError(f"No backup found for version {version}")
        self._restore_params(backup)

    def submit_grad_bucket(
        self,
        version: int,
        grads: List[Tensor],
    ) -> GradBucket:
        """Submit gradients for async all_reduce.

        Args:
            version: Training step version
            grads: List of gradient tensors

        Returns:
            GradBucket with async work handle
        """
        bucket = GradBucket(version=version, grads=grads)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size(self.group) > 1:
            # Flatten and all_reduce
            flat_grad = torch.cat([g.view(-1) for g in grads])
            work = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=self.group, async_op=True)
            bucket.work = work
            # Store flat grad for later scatter back
            bucket._flat_grad = flat_grad
        else:
            bucket.ready = True

        self._pending_buckets[version] = bucket
        return bucket

    def _finalize_bucket(self, bucket: GradBucket) -> None:
        """Wait for bucket all_reduce and scatter gradients back."""
        if bucket.work is not None:
            bucket.work.wait()
            # Scatter flat grad back to individual grad tensors
            offset = 0
            for g in bucket.grads:
                size = int(g.numel())
                g.copy_(bucket._flat_grad[offset:offset + size].view(g.shape))
                offset += size
        bucket.ready = True

    def maybe_apply_ready(self, optimizer: torch.optim.Optimizer) -> bool:
        """Apply optimizer step if the next version's bucket is ready.

        Applies steps in order: only applies version v if all versions < v
        have already been applied.

        Args:
            optimizer: The optimizer to step

        Returns:
            True if a step was applied
        """
        applied = False

        while self._next_apply in self._pending_buckets:
            bucket = self._pending_buckets[self._next_apply]

            # Finalize (wait for all_reduce if needed)
            if not bucket.ready:
                self._finalize_bucket(bucket)

            # Apply optimizer step
            optimizer.step()
            optimizer.zero_grad()

            del self._pending_buckets[self._next_apply]
            self._next_apply += 1
            self.optimizer_apply_version = self._next_apply
            applied = True

        return applied

    def drain_all(self, optimizer: torch.optim.Optimizer) -> int:
        """Wait for all pending buckets and apply optimizer steps in order.

        Args:
            optimizer: The optimizer to step

        Returns:
            Number of steps applied
        """
        count = 0
        # Sort pending versions to apply in order
        for version in sorted(self._pending_buckets.keys()):
            bucket = self._pending_buckets[version]
            if not bucket.ready:
                self._finalize_bucket(bucket)

        while self._next_apply in self._pending_buckets:
            bucket = self._pending_buckets[self._next_apply]
            optimizer.step()
            optimizer.zero_grad()
            del self._pending_buckets[self._next_apply]
            self._next_apply += 1
            self.optimizer_apply_version = self._next_apply
            count += 1

        return count

    def describe(self) -> dict:
        return {
            "active_version": self.active_version,
            "optimizer_apply_version": self.optimizer_apply_version,
            "pending_versions": sorted(self._pending_buckets.keys()),
            "num_backup_slots": self.num_backup_slots,
        }
