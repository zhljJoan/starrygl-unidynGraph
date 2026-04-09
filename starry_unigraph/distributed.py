from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from starry_unigraph.types import DistributedContext


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
