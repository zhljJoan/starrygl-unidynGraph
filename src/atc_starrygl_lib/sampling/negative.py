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
        if request.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if request.ratio <= 0:
            raise ValueError("ratio must be positive")

        count = int(request.pos_src.numel()) * int(request.ratio)
        neg_src = request.pos_src.repeat_interleave(int(request.ratio))
        neg_dst = torch.randint(
            low=0,
            high=int(request.num_nodes),
            size=(count,),
            device=request.pos_src.device,
            dtype=request.pos_src.dtype,
            generator=request.generator,
        )
        if request.pos_dst.numel() and request.ratio == 1:
            collision = neg_dst == request.pos_dst
            neg_dst[collision] = (neg_dst[collision] + 1) % int(request.num_nodes)
        return NegativeSamplingResult(neg_src=neg_src, neg_dst=neg_dst, ratio=int(request.ratio))


class PoolNegativeSampler:
    """Destination sampler with split-aware local/global pools."""

    def __init__(self, train_remote_dst_prob: float = 0.0, test_policy: str = "global") -> None:
        self.train_remote_dst_prob = float(train_remote_dst_prob)
        self.test_policy = str(test_policy)

    def sample(self, request: NegativeSamplingRequest) -> NegativeSamplingResult:
        if request.ratio <= 0:
            raise ValueError("ratio must be positive")
        count = int(request.pos_src.numel()) * int(request.ratio)
        neg_src = request.pos_src.repeat_interleave(int(request.ratio))

        if request.split == "train" and request.local_dst_pool is not None and request.local_dst_pool.numel() > 0:
            local_dst = _sample_pool(request.local_dst_pool, count, request.pos_src.device, request.pos_src.dtype, request.generator)
            if self.train_remote_dst_prob <= 0:
                return NegativeSamplingResult(neg_src=neg_src, neg_dst=local_dst, ratio=int(request.ratio))
            global_dst = _sample_or_uniform(request, count)
            mask = torch.rand(count, device=request.pos_src.device, generator=request.generator) < self.train_remote_dst_prob
            return NegativeSamplingResult(neg_src=neg_src, neg_dst=torch.where(mask, global_dst, local_dst), ratio=int(request.ratio))

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
