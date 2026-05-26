from __future__ import annotations

import time
from typing import Any, Iterable

import torch
import torch.distributed as dist


class AsyncGradientSyncOptimizer:
    """Optimizer wrapper with StarryUniGraph-style async gradient all-reduce."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.nn.Parameter],
        *,
        group: Any = None,
        average: bool = True,
        enabled: bool | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = [p for p in params if p.requires_grad]
        self.group = group
        self.average = bool(average)
        self.enabled = _dist_enabled(group) if enabled is None else bool(enabled)
        self._profile_stats = {
            "sync_seconds": 0.0,
            "flatten_seconds": 0.0,
            "all_reduce_seconds": 0.0,
            "copyback_seconds": 0.0,
            "steps": 0.0,
        }

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self.synchronize()
        return self.optimizer.step(*args, **kwargs)

    def synchronize(self) -> None:
        if not self.enabled or not self.params:
            return
        t0 = time.perf_counter()
        t_flat = time.perf_counter()
        flat = torch.cat([
            torch.zeros_like(p, memory_format=torch.contiguous_format).view(-1)
            if p.grad is None
            else p.grad.detach().contiguous().view(-1)
            for p in self.params
        ])
        self._profile_stats["flatten_seconds"] += float(time.perf_counter() - t_flat)
        t_reduce = time.perf_counter()
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group, async_op=True)
        work.wait()
        self._profile_stats["all_reduce_seconds"] += float(time.perf_counter() - t_reduce)
        if self.average:
            flat.div_(float(dist.get_world_size(self.group)))
        t_copy = time.perf_counter()
        offset = 0
        for p in self.params:
            size = int(p.numel())
            grad = flat[offset : offset + size].view_as(p)
            if p.grad is None:
                p.grad = grad.clone(memory_format=torch.contiguous_format)
            else:
                p.grad.copy_(grad)
            offset += size
        self._profile_stats["copyback_seconds"] += float(time.perf_counter() - t_copy)
        self._profile_stats["sync_seconds"] += float(time.perf_counter() - t0)
        self._profile_stats["steps"] += 1.0

    def reset_profile_stats(self) -> None:
        for key in self._profile_stats:
            self._profile_stats[key] = 0.0

    def pop_profile_stats(self) -> dict[str, float]:
        out = dict(self._profile_stats)
        self.reset_profile_stats()
        return out

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


def _dist_enabled(group: Any = None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1
