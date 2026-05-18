"""Sampling hook interfaces for chunk training.

Three hooks:
  NeighborSamplerHook  — temporal k-hop neighbor sampling
  NegativeSamplerHook  — negative edge pair generation
  MFGBuilderHook       — Message Flow Graph construction from sampled neighbors

Usage:
    sampler = NeighborSamplerHook.from_config(cfg)
    neg     = NegativeSamplerHook.from_config(cfg)
    builder = MFGBuilderHook()

    sampled   = sampler.sample(seeds, timestamps, part_data)
    neg_pairs = neg.sample(pos_src, pos_dst, num_nodes)
    mfgs      = builder.build(sampled, features)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Sampled graph container (output of NeighborSamplerHook)
# ---------------------------------------------------------------------------

@dataclass
class SampledGraph:
    """Intermediate result of temporal neighbor sampling.

    Attributes:
        src_nodes:    [S] sampled source node global IDs.
        dst_nodes:    [D] seed destination node global IDs.
        edge_src:     [E] source indices (into src_nodes).
        edge_dst:     [E] destination indices (into dst_nodes).
        edge_ts:      [E] timestamps of sampled edges.
        edge_ids:     [E] global edge IDs (optional).
        layer_ptrs:   [L+1] CSR pointer separating MFG layers, if multi-hop.
    """
    src_nodes:   Tensor
    dst_nodes:   Tensor
    edge_src:    Tensor
    edge_dst:    Tensor
    edge_ts:     Tensor
    edge_ids:    Optional[Tensor] = None
    layer_ptrs:  Optional[Tensor] = None


# ---------------------------------------------------------------------------
# Neighbor sampler
# ---------------------------------------------------------------------------

class NeighborSamplerHook(ABC):
    """Interface for temporal k-hop neighbor sampling.

    Implementations must call the C++ BTS sampler extension.
    """

    @abstractmethod
    def sample(
        self,
        seed_nodes:  Tensor,
        seed_ts:     Tensor,
        part_data:   Any,
    ) -> SampledGraph:
        """Sample k-hop temporal neighbors for seed_nodes at seed_ts.

        Args:
            seed_nodes: [N] global node IDs of seeds.
            seed_ts:    [N] query timestamps (sample events before this ts).
            part_data:  PartitionData providing local Temporal-CSR.

        Returns:
            SampledGraph with sampled neighbors and edges.
        """
        raise NotImplementedError("C++ BTS sampler not yet wired")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> NeighborSamplerHook:
        """Construct sampler from config dict.

        cfg keys: num_neighbors (list), num_layers (int), sample_type (str).
        """
        raise RuntimeError(
            "NeighborSamplerHook.from_config is not a production sampler entry point. "
            "Use MemShareEventEngine.from_config(graph_store, cfg) so CTDG sampling, "
            "deduplication, and native MFG construction stay in the MemShare C++ path."
        )


# ---------------------------------------------------------------------------
# Negative sampler
# ---------------------------------------------------------------------------

class NegativeSamplerHook(ABC):
    """Interface for negative edge pair generation."""

    @abstractmethod
    def sample(
        self,
        pos_src:   Tensor,
        pos_dst:   Tensor,
        num_nodes: int,
        neg_ratio: int = 1,
        split: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        """Generate negative (src, dst) pairs.

        Args:
            pos_src:   [M] positive source nodes.
            pos_dst:   [M] positive destination nodes.
            num_nodes: total graph node count for random sampling.
            neg_ratio: negatives per positive.

        Returns:
            (neg_src [M*neg_ratio], neg_dst [M*neg_ratio])
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> NegativeSamplerHook:
        strategy = cfg.get("neg_strategy", "random")
        if strategy == "random":
            return _RandomNegativeSampler()
        if strategy in {"edge_predict_mixed", "mixed"}:
            return EdgePredictNegativeSampler.from_config(cfg)
        raise ValueError(f"Unknown neg_strategy: {strategy}")


class _RandomNegativeSampler(NegativeSamplerHook):
    """Random uniform negative sampling (CPU, no C++ needed)."""

    def sample(
        self,
        pos_src: Tensor,
        pos_dst: Tensor,
        num_nodes: int,
        neg_ratio: int = 1,
        split: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        M = pos_src.numel()
        neg_dst = torch.randint(0, num_nodes, (M * neg_ratio,), device=pos_src.device)
        neg_src = pos_src.repeat_interleave(neg_ratio)
        return neg_src, neg_dst


@dataclass
class EdgePredictNegativeSampler(NegativeSamplerHook):
    """MemShare-style edge-predict negative sampler.

    Train uses a rank-local destination pool with configurable probability and
    falls back to the global pool. Validation/test use the global-average policy
    unless explicitly overridden. Optional ``neg_weight`` is exposed for callers
    that want importance reweighting when train/eval distributions differ.
    """

    train_remote_dst_prob: float = 0.0
    test_policy: str = "global_average"
    neg_weight: Optional[float] = None
    local_dst_pool: Optional[Tensor] = None
    global_dst_pool: Optional[Tensor] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EdgePredictNegativeSampler":
        local_pool = cfg.get("local_dst_pool")
        global_pool = cfg.get("global_dst_pool", cfg.get("dst_pool"))
        return cls(
            train_remote_dst_prob=float(cfg.get("train_remote_dst_prob", 0.0)),
            test_policy=str(cfg.get("test_policy", cfg.get("negative_policy", "global_average"))),
            neg_weight=None if cfg.get("neg_weight") is None else float(cfg["neg_weight"]),
            local_dst_pool=None if local_pool is None else torch.as_tensor(local_pool, dtype=torch.long),
            global_dst_pool=None if global_pool is None else torch.as_tensor(global_pool, dtype=torch.long),
        )

    def configure_pools(
        self,
        *,
        local_dst_pool: Optional[Tensor] = None,
        global_dst_pool: Optional[Tensor] = None,
    ) -> None:
        if local_dst_pool is not None:
            self.local_dst_pool = local_dst_pool.long().contiguous().cpu()
        if global_dst_pool is not None:
            self.global_dst_pool = global_dst_pool.long().contiguous().cpu()

    def _sample_from_pool(self, pool: Optional[Tensor], count: int, device: torch.device, num_nodes: int) -> Tensor:
        if pool is None or pool.numel() == 0:
            return torch.randint(0, int(num_nodes), (count,), device=device)
        pool_dev = pool.to(device=device, non_blocking=True)
        idx = torch.randint(0, int(pool_dev.numel()), (count,), device=device)
        return pool_dev[idx]

    def sample(
        self,
        pos_src: Tensor,
        pos_dst: Tensor,
        num_nodes: int,
        neg_ratio: int = 1,
        split: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        count = int(pos_src.numel()) * int(neg_ratio)
        neg_src = pos_src.repeat_interleave(int(neg_ratio))
        split = str(split)
        if split == "train":
            local_dst = self._sample_from_pool(self.local_dst_pool, count, pos_src.device, num_nodes)
            if self.train_remote_dst_prob <= 0:
                return neg_src, local_dst
            global_dst = self._sample_from_pool(self.global_dst_pool, count, pos_src.device, num_nodes)
            remote_mask = torch.rand(count, device=pos_src.device) < float(self.train_remote_dst_prob)
            neg_dst = torch.where(remote_mask, global_dst, local_dst)
            return neg_src, neg_dst
        if self.test_policy == "rank_local":
            return neg_src, self._sample_from_pool(self.local_dst_pool, count, pos_src.device, num_nodes)
        return neg_src, self._sample_from_pool(self.global_dst_pool, count, pos_src.device, num_nodes)


# ---------------------------------------------------------------------------
# MFG builder
# ---------------------------------------------------------------------------

class MFGBuilderHook(ABC):
    """Interface for Message Flow Graph (DGL Block) construction."""

    @abstractmethod
    def build(
        self,
        sampled:  SampledGraph,
        features: Optional[Tensor] = None,
    ) -> Any:
        """Build DGL bipartite block(s) from sampled graph.

        Args:
            sampled:  SampledGraph from NeighborSamplerHook.
            features: [num_src, feat_dim] optional pre-fetched features.

        Returns:
            DGL Block or list of Blocks for multi-layer models.
        """
        raise NotImplementedError("MFG construction not yet implemented")

    @classmethod
    def default(cls) -> MFGBuilderHook:
        return DGLMFGBuilder()


class DGLMFGBuilder(MFGBuilderHook):
    """Build a DGL block from a sampled graph using CSC materialization."""

    def build(self, sampled: SampledGraph, features: Optional[Tensor] = None) -> Any:
        import dgl

        if sampled is None:
            raise ValueError("sampled graph is required for DGLMFGBuilder")
        num_src = int(sampled.src_nodes.numel())
        num_dst = int(sampled.dst_nodes.numel())
        if sampled.edge_src.numel() != sampled.edge_dst.numel():
            raise ValueError("sampled.edge_src and sampled.edge_dst must have the same length")

        edge_src = sampled.edge_src.long().contiguous()
        edge_dst = sampled.edge_dst.long().contiguous()
        if edge_src.numel() == 0:
            indptr = torch.zeros(num_dst + 1, dtype=torch.int64, device=edge_src.device)
            indices = edge_src
            eids = torch.empty(0, dtype=torch.int64, device=edge_src.device)
        else:
            order = torch.argsort(edge_dst, stable=True)
            sorted_dst = edge_dst[order]
            indices = edge_src[order].contiguous()
            counts = torch.bincount(sorted_dst, minlength=num_dst)
            indptr = torch.zeros(num_dst + 1, dtype=torch.int64, device=edge_src.device)
            indptr[1:] = counts.cumsum(0)
            if sampled.edge_ids is None:
                eids = order.long().contiguous()
            else:
                eids = sampled.edge_ids.to(device=edge_src.device, dtype=torch.int64)[order].contiguous()

        block = dgl.create_block(
            ("csc", (indptr, indices, eids)),
            num_src_nodes=num_src,
            num_dst_nodes=num_dst,
            idtype=torch.int64,
        )
        block.srcdata[dgl.NID] = sampled.src_nodes.long().contiguous()
        block.dstdata[dgl.NID] = sampled.dst_nodes.long().contiguous()
        block.edata[dgl.EID] = eids.long().contiguous()
        if sampled.edge_ts is not None:
            if edge_src.numel() == 0:
                block.edata["ts"] = sampled.edge_ts.to(device=edge_src.device)[:0].contiguous()
            else:
                block.edata["ts"] = sampled.edge_ts.to(device=edge_src.device)[order].contiguous()
        if features is not None:
            if features.size(0) != num_src:
                raise ValueError(f"features must align with src_nodes: {features.size(0)} != {num_src}")
            block.srcdata["feat"] = features
        return block
