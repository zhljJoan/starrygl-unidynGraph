from __future__ import annotations

import os
from copy import deepcopy
from typing import Any
from torch import Tensor
from starry_unigraph.types import DistributedContext


class DistRouteIndex:
    # 编码方案: 
    # [63]: 符号位 (尽量不用)
    # [62]: shared flag
    # [48-61]: part_id (14 bits, 支持 16384 个分区)
    # [0-47]: local_index (48 bits, 支持 281 万亿个 ID)
    
    PART_SHIFT = 48
    SHARED_BIT = 62
    LOC_MASK = 0xFFFFFFFFFFFF # 48 bits
    PART_MASK = 0x3FFF         # 14 bits (1<<14 - 1)

    def __init__(self, index: Tensor, part_ids: Optional[Tensor] = None) -> None:
        if part_ids is None:
            self._data = index.long()
        else:
            # 确保在同一设备
            index = index.long()
            part_ids = part_ids.long().to(index.device)
            self._data = (index & self.LOC_MASK) | ((part_ids & self.PART_MASK) << self.PART_SHIFT)
       
    @property
    def loc(self) -> Tensor:
        return self._data & self.LOC_MASK
    
    @property
    def part(self) -> Tensor:
        return (self._data >> self.PART_SHIFT) & self.PART_MASK
    
    def set_shared(self, indx: Union[slice, Tensor]):
        self._data[indx] |= (1 << self.SHARED_BIT)

    @property
    def is_shared(self) -> Tensor:
        return (self._data >> self.SHARED_BIT).to(torch.bool)
    
    @property
    def dist(self) -> Tensor:
        return self._data
    @property
    def device(self): return self._data.device
    @property
    def shape(self): return self._data.shape
    
    def to(self, device):
        return DistRouteIndex(self._data.to(device))
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def apply_distributed_env(config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    dist_cfg = merged.setdefault("dist", {})
    torchrun_active = "WORLD_SIZE" in os.environ or "RANK" in os.environ or "LOCAL_RANK" in os.environ

    dist_cfg["backend"] = dist_cfg.get("backend", "nccl")
    dist_cfg["world_size"] = _env_int("WORLD_SIZE", int(dist_cfg.get("world_size", 1)))
    dist_cfg["rank"] = _env_int("RANK", int(dist_cfg.get("rank", 0)))
    dist_cfg["local_rank"] = _env_int("LOCAL_RANK", int(dist_cfg.get("local_rank", 0)))
    dist_cfg["local_world_size"] = _env_int("LOCAL_WORLD_SIZE", int(dist_cfg.get("local_world_size", 1)))
    dist_cfg["master_addr"] = os.environ.get("MASTER_ADDR", str(dist_cfg.get("master_addr", "127.0.0.1")))
    dist_cfg["master_port"] = _env_int("MASTER_PORT", int(dist_cfg.get("master_port", 29500)))
    dist_cfg["init_method"] = str(dist_cfg.get("init_method", "env://"))
    dist_cfg["launcher"] = "torchrun" if torchrun_active else str(dist_cfg.get("launcher", "single_process"))

    runtime_cfg = merged.setdefault("runtime", {})
    if runtime_cfg.get("device") == "cuda" and dist_cfg["launcher"] == "torchrun":
        runtime_cfg["device"] = f"cuda:{dist_cfg['local_rank']}"
    return merged


def build_distributed_context(config: dict[str, Any]) -> DistributedContext:
    dist_cfg = config.get("dist", {})
    return DistributedContext(
        backend=str(dist_cfg.get("backend", "nccl")),
        world_size=int(dist_cfg.get("world_size", 1)),
        rank=int(dist_cfg.get("rank", 0)),
        local_rank=int(dist_cfg.get("local_rank", 0)),
        local_world_size=int(dist_cfg.get("local_world_size", 1)),
        master_addr=str(dist_cfg.get("master_addr", "127.0.0.1")),
        master_port=int(dist_cfg.get("master_port", 29500)),
        init_method=str(dist_cfg.get("init_method", "env://")),
        launcher=str(dist_cfg.get("launcher", "single_process")),
    )


def initialize_distributed(ctx: DistributedContext) -> DistributedContext:
    if not ctx.is_distributed:
        return ctx

    import torch
    import torch.distributed as dist

    if ctx.backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=ctx.backend,
            init_method=ctx.init_method,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )
    ctx.initialized = True
    return ctx


def finalize_distributed(ctx: DistributedContext) -> None:
    if not ctx.initialized:
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    ctx.initialized = False
