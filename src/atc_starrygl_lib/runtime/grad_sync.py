from __future__ import annotations

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

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self.synchronize()
        return self.optimizer.step(*args, **kwargs)

    def synchronize(self) -> None:
        if not self.enabled or not self.params:
            return
        flat = torch.cat([
            torch.zeros_like(p, memory_format=torch.contiguous_format).view(-1)
            if p.grad is None
            else p.grad.detach().contiguous().view(-1)
            for p in self.params
        ])
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group, async_op=True)
        work.wait()
        if self.average:
            flat.div_(float(dist.get_world_size(self.group)))
        offset = 0
        for p in self.params:
            size = int(p.numel())
            grad = flat[offset : offset + size].view_as(p)
            if p.grad is None:
                p.grad = grad.clone(memory_format=torch.contiguous_format)
            else:
                p.grad.copy_(grad)
            offset += size

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


def _dist_enabled(group: Any = None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1
