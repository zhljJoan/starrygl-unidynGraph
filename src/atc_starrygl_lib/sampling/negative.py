from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class NegativeSamplingRequest:
    pos_src: Tensor
    pos_dst: Tensor
    num_nodes: int
    ratio: int = 1
    split: str = "train"
    dst_pool: Optional[Tensor] = None
    local_dst_pool: Optional[Tensor] = None
    remote_dst_pool: Optional[Tensor] = None
    generator: Optional[torch.Generator] = None


@dataclass(frozen=True)
class NegativeSamplingResult:
    neg_src: Tensor
    neg_dst: Tensor
    ratio: int
    weight: Optional[Tensor] = None


class NegativeSampler(Protocol):
    def sample(self, request: NegativeSamplingRequest) -> NegativeSamplingResult:
        ...


class RandomNegativeSampler:
    """Uniform destination negative sampler for edge prediction."""

    def sample(self, request: NegativeSamplingRequest) -> NegativeSamplingResult:
        if request.num_nodes <= 0 and (request.dst_pool is None or request.dst_pool.numel() == 0):
            raise ValueError("num_nodes must be positive")
        if request.ratio <= 0:
            raise ValueError("ratio must be positive")

        count = int(request.pos_src.numel()) * int(request.ratio)
        neg_src = request.pos_src.repeat_interleave(int(request.ratio))
        neg_dst = _sample_or_uniform(request, count)
        if request.pos_dst.numel() and request.ratio == 1:
            collision = neg_dst == request.pos_dst
            if request.dst_pool is not None and request.dst_pool.numel() > 1:
                neg_dst[collision] = _sample_pool(
                    request.dst_pool,
                    int(collision.sum().item()),
                    request.pos_src.device,
                    request.pos_src.dtype,
                    request.generator,
                )
            else:
                neg_dst[collision] = (neg_dst[collision] + 1) % int(request.num_nodes)
        return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio))


class PoolNegativeSampler:
    """Destination sampler with split-aware local/remote/global pools."""

    def __init__(
        self,
        train_remote_dst_prob: float = 0.0,
        test_policy: str = "global",
        train_local_dst_prob: float | None = None,
        correction: str = "balanced",
    ) -> None:
        if train_local_dst_prob is not None:
            train_remote_dst_prob = 1.0 - float(train_local_dst_prob)
        self.train_remote_dst_prob = min(1.0, max(0.0, float(train_remote_dst_prob)))
        self.test_policy = str(test_policy)
        self.correction = str(correction).strip().lower()

    def sample(self, request: NegativeSamplingRequest) -> NegativeSamplingResult:
        if request.ratio <= 0:
            raise ValueError("ratio must be positive")
        count = int(request.pos_src.numel()) * int(request.ratio)
        neg_src = request.pos_src.repeat_interleave(int(request.ratio))

        if request.split == "train" and request.local_dst_pool is not None and request.local_dst_pool.numel() > 0:
            local_dst = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
            if self.train_remote_dst_prob <= 0:
                return NegativeSamplingResult(neg_src=neg_src, neg_dst=local_dst, ratio=int(request.ratio))
            remote_dst = (
                _sample_pool(request.remote_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
                if request.remote_dst_pool is not None and request.remote_dst_pool.numel() > 0
                else _sample_or_uniform(request, count)
            )
            mask = torch.rand(count, device=request.pos_src.device, generator=request.generator) < self.train_remote_dst_prob
            return NegativeSamplingResult(
                neg_src=neg_src,
                neg_dst=torch.where(mask, remote_dst, local_dst),
                ratio=int(request.ratio),
                weight=_local_remote_weight(
                    mask,
                    correction=self.correction,
                    remote_prob=self.train_remote_dst_prob,
                    local_pool_size=int(request.local_dst_pool.numel()),
                    remote_pool_size=0 if request.remote_dst_pool is None else int(request.remote_dst_pool.numel()),
                ),
            )

        if self.test_policy == "rank_local" and request.local_dst_pool is not None and request.local_dst_pool.numel() > 0:
            neg_dst = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
        else:
            neg_dst = _sample_or_uniform(request, count)
        return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio))


class MemShareLocalNegativeSampler:
    """Replicate MemShare local/global destination sampling and weights."""

    def __init__(self, beta: float = 0.1, test_policy: str = "global") -> None:
        self.beta = min(1.0, max(0.0, float(beta)))
        self.test_policy = str(test_policy)

    def sample(self, request: NegativeSamplingRequest) -> NegativeSamplingResult:
        if request.ratio <= 0:
            raise ValueError("ratio must be positive")
        count = int(request.pos_src.numel()) * int(request.ratio)
        neg_src = request.pos_src.repeat_interleave(int(request.ratio))

        if request.split == "train" and request.local_dst_pool is not None and request.local_dst_pool.numel() > 0:
            global_pool = request.dst_pool if request.dst_pool is not None and request.dst_pool.numel() > 0 else None
            if global_pool is None:
                neg_dst = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
                return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio))
            sampled_global = _sample_pool(global_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
            sampled_local = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
            choose_global = torch.rand(count, device=request.pos_src.device, generator=request.generator) <= self.beta
            neg_dst = torch.where(choose_global, sampled_global, sampled_local)
            local_mask = _membership_mask(neg_dst, request.local_dst_pool)
            weight = _memshare_local_global_weight(
                local_mask=local_mask,
                beta=self.beta,
                local_pool_size=int(request.local_dst_pool.numel()),
                global_pool_size=int(global_pool.numel()),
            )
            return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio), weight=weight)

        if self.test_policy == "rank_local" and request.local_dst_pool is not None and request.local_dst_pool.numel() > 0:
            neg_dst = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
        else:
            neg_dst = _sample_or_uniform(request, count)
        return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio))


def _sample_or_uniform(request: NegativeSamplingRequest, count: int) -> Tensor:
    if request.dst_pool is not None and request.dst_pool.numel() > 0:
        return _sample_pool(request.dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
    return torch.randint(
        0,
        int(request.num_nodes),
        (count,),
        device=request.pos_src.device,
        dtype=request.pos_src.dtype,
        generator=request.generator,
    )


def _sample_pool(pool: Tensor, count: int, device: torch.device, dtype: torch.dtype, generator: Optional[torch.Generator]) -> Tensor:
    pool_dev = pool.to(device=device, dtype=dtype, non_blocking=True)
    idx = torch.randint(0, int(pool_dev.numel()), (count,), device=device, generator=generator)
    return pool_dev[idx]


def _membership_mask(sampled: Tensor, pool: Tensor) -> Tensor:
    if pool.numel() == 0 or sampled.numel() == 0:
        return torch.zeros(sampled.shape, dtype=torch.bool, device=sampled.device)
    pool_cpu = torch.unique(pool.long().cpu(), sorted=True)
    sampled_cpu = sampled.long().cpu()
    pos = torch.searchsorted(pool_cpu, sampled_cpu)
    pos = pos.clamp_max(max(int(pool_cpu.numel()) - 1, 0))
    mask = (torch.searchsorted(pool_cpu, sampled_cpu) < int(pool_cpu.numel())) & (pool_cpu.index_select(0, pos) == sampled_cpu)
    return mask.to(sampled.device)


def _memshare_local_global_weight(*, local_mask: Tensor, beta: float, local_pool_size: int, global_pool_size: int) -> Tensor | None:
    if local_mask.numel() == 0 or local_pool_size <= 0 or global_pool_size <= 0:
        return None
    if beta <= 0.0 or beta >= 1.0:
        return None
    local_ratio = float(local_pool_size) / float(global_pool_size)
    if local_ratio <= 0.0:
        return None
    local_weight = 1.0 / (1.0 - beta + beta * local_ratio)
    remote_weight = 1.0 / (beta * local_ratio)
    weight = torch.empty(local_mask.shape, dtype=torch.float32, device=local_mask.device)
    weight[local_mask] = float(local_weight)
    weight[~local_mask] = float(remote_weight)
    return weight


def _balanced_binary_partition_weight(mask: Tensor) -> Tensor | None:
    count = int(mask.numel())
    if count == 0:
        return None
    remote_count = int(mask.sum().item())
    local_count = count - remote_count
    if remote_count == 0 or local_count == 0:
        return None
    weight = torch.empty((count,), dtype=torch.float32, device=mask.device)
    weight[mask] = float(count) / (2.0 * float(remote_count))
    weight[~mask] = float(count) / (2.0 * float(local_count))
    return weight


def _local_remote_weight(
    mask: Tensor,
    *,
    correction: str,
    remote_prob: float,
    local_pool_size: int,
    remote_pool_size: int,
) -> Tensor | None:
    if correction in {"", "none", "off", "false", "0"}:
        return None
    if correction in {"importance", "probability", "prob", "pool", "density"}:
        return _local_remote_importance_weight(
            mask,
            remote_prob=remote_prob,
            local_pool_size=local_pool_size,
            remote_pool_size=remote_pool_size,
        )
    return _balanced_binary_partition_weight(mask)


def _local_remote_importance_weight(
    mask: Tensor,
    *,
    remote_prob: float,
    local_pool_size: int,
    remote_pool_size: int,
) -> Tensor | None:
    if mask.numel() == 0 or local_pool_size <= 0 or remote_pool_size <= 0:
        return None
    p_remote = min(1.0, max(0.0, float(remote_prob)))
    p_local = 1.0 - p_remote
    if p_remote <= 0.0 or p_local <= 0.0:
        return None
    total = float(local_pool_size + remote_pool_size)
    local_weight = (float(local_pool_size) / total) / p_local
    remote_weight = (float(remote_pool_size) / total) / p_remote
    weight = torch.empty(mask.shape, dtype=torch.float32, device=mask.device)
    weight[mask] = float(remote_weight)
    weight[~mask] = float(local_weight)
    return weight
