"""Sampling hook interfaces for chunk training.

These are stub interfaces — actual implementations will call C++ extensions.

Three hooks:
  NeighborSamplerHook  — temporal k-hop neighbor sampling
  NegativeSamplerHook  — negative edge pair generation
  MFGBuilderHook       — Message Flow Graph construction from sampled neighbors

Usage (once C++ is ready):
    sampler = NeighborSamplerHook.from_config(cfg)
    neg     = NegativeSamplerHook.from_config(cfg)
    builder = MFGBuilderHook()

    sampled   = sampler.sample(seeds, timestamps, part_data)
    neg_pairs = neg.sample(pos_src, pos_dst, num_nodes)
    mfgs      = builder.build(sampled, features)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
        return _StubNeighborSampler(cfg)


class _StubNeighborSampler(NeighborSamplerHook):
    """Stub: returns seeds as trivial single-hop result (no actual sampling)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.num_neighbors: List[int] = cfg.get("num_neighbors", [20])
        self.num_layers: int = cfg.get("num_layers", 1)

    def sample(self, seed_nodes: Tensor, seed_ts: Tensor, part_data: Any) -> SampledGraph:
        N = seed_nodes.numel()
        return SampledGraph(
            src_nodes  = seed_nodes,
            dst_nodes  = seed_nodes,
            edge_src   = torch.arange(N, dtype=torch.long),
            edge_dst   = torch.arange(N, dtype=torch.long),
            edge_ts    = seed_ts,
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
        raise ValueError(f"Unknown neg_strategy: {strategy}")


class _RandomNegativeSampler(NegativeSamplerHook):
    """Random uniform negative sampling (CPU, no C++ needed)."""

    def sample(
        self,
        pos_src: Tensor,
        pos_dst: Tensor,
        num_nodes: int,
        neg_ratio: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        M = pos_src.numel()
        neg_dst = torch.randint(0, num_nodes, (M * neg_ratio,), device=pos_src.device)
        neg_src = pos_src.repeat_interleave(neg_ratio)
        return neg_src, neg_dst


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
        return _StubMFGBuilder()


class _StubMFGBuilder(MFGBuilderHook):
    """Stub: returns None (no actual MFG construction)."""

    def build(self, sampled: SampledGraph, features: Optional[Tensor] = None) -> None:
        return None
