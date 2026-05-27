from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from torch import Tensor

from .temporal import SamplingOutput, TemporalSamplingRequest


@dataclass(frozen=True)
class NativeSamplerConfig:
    fanouts: tuple[int, ...] = (10,)
    num_layers: int = 1
    policy: str = "recent"
    workers: int = 1
    local_part: int = 0
    world_size: int = 1


@dataclass(frozen=True)
class TemporalGraphData:
    row: Tensor
    col: Tensor
    edge_ids: Tensor
    timestamps: Optional[Tensor]
    num_nodes: int
    node_part: Optional[Tensor] = None
    edge_part: Optional[Tensor] = None
    edge_read_dist_index: Optional[Tensor] = None


class NativeTemporalSampler(Protocol):
    """C++ native temporal sampler boundary."""

    def sample(self, request: TemporalSamplingRequest) -> SamplingOutput:
        ...


class NativeSamplerFactory(Protocol):
    def build(self, graph: TemporalGraphData, config: NativeSamplerConfig) -> NativeTemporalSampler:
        ...


@dataclass
class RawNativeSampleResult:
    mfgs: Any
    node_gids: Tensor
    node_ts: Optional[Tensor]
    layer_ptr: Tensor
    root_gids: Tensor
    root_ts: Optional[Tensor]
    root_lids: Tensor
    edge_gids: Optional[Tensor] = None
    edge_ts: Optional[Tensor] = None
    edge_layer_ptr: Optional[Tensor] = None


class NativeSamplerUnavailable(RuntimeError):
    pass
