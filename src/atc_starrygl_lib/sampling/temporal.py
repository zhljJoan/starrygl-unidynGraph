from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from .negative import NegativeSamplingResult


@dataclass(frozen=True)
class PositiveEdges:
    src: Tensor
    dst: Tensor
    ts: Tensor
    edge_ids: Optional[Tensor] = None
    labels: Optional[Tensor] = None


@dataclass(frozen=True)
class RootSet:
    nodes: Tensor
    ts: Optional[Tensor] = None
    groups: dict[str, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def from_time_slice(
        cls,
        nodes: Tensor,
        time_slice: int | Tensor,
        *,
        groups: Optional[dict[str, tuple[int, int]]] = None,
    ) -> "RootSet":
        nodes = nodes.long().contiguous()
        if isinstance(time_slice, Tensor):
            if time_slice.numel() == 1:
                ts = torch.full(
                    (int(nodes.numel()),),
                    int(time_slice.item()),
                    dtype=torch.long,
                    device=nodes.device,
                )
            else:
                if int(time_slice.numel()) != int(nodes.numel()):
                    raise ValueError("time_slice tensor must be scalar or match nodes length")
                ts = time_slice.to(device=nodes.device, dtype=torch.long).contiguous()
        else:
            ts = torch.full(
                (int(nodes.numel()),),
                int(time_slice),
                dtype=torch.long,
                device=nodes.device,
            )
        return cls(nodes=nodes, ts=ts, groups={} if groups is None else dict(groups))

    def with_time_slice(self, time_slice: int | Tensor) -> "RootSet":
        return RootSet.from_time_slice(self.nodes, time_slice, groups=self.groups)


@dataclass(frozen=True)
class TemporalSamplingRequest:
    roots: RootSet
    fanouts: tuple[int, ...]
    num_layers: int
    policy: str = "recent"
    required_nodes: Optional[Tensor] = None
    positive_edges: Optional[PositiveEdges] = None
    negatives: Optional[NegativeSamplingResult] = None
    meta: dict[str, Any] | None = None


@dataclass
class NodeComputeLayout:
    """Node instances used by MFG computation.

    CTDG compute keys are (node_gid, timestamp). DTDG compute keys are
    (node_gid, time_slice). The local id is the row index in node_gids/node_ts.
    """

    node_gids: Tensor
    node_ts: Optional[Tensor]
    layer_ptr: Tensor
    root_gids: Tensor
    root_ts: Optional[Tensor]
    root_lids: Tensor
    groups: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class EdgeComputeLayout:
    """Deduplicated edge instances used by MFG computation."""

    edge_gids: Tensor
    edge_ts: Optional[Tensor]
    layer_ptr: Tensor


@dataclass
class NodeCommLayout:
    """Feature/memory fetch keys, sorted for communication.

    compute_to_comm maps each compute node local id to a communication entry.
    provider is the rank selected for this fetch; owner is the true owner.
    """

    node_gids: Tensor
    time_slices: Optional[Tensor]
    owner: Tensor
    provider: Tensor
    provider_ptr: Tensor
    compute_to_comm: Tensor
    time_ptr: Optional[Tensor] = None
    replica_mask: Optional[Tensor] = None


@dataclass
class EdgeCommLayout:
    edge_gids: Tensor
    time_slices: Optional[Tensor]
    owner: Optional[Tensor]
    provider: Optional[Tensor]
    provider_ptr: Optional[Tensor]
    compute_to_comm: Tensor
    time_ptr: Optional[Tensor] = None
    replica_mask: Optional[Tensor] = None


@dataclass
class SampledMFG:
    """One sampled MFG in CSC format using compute-layout local ids.

    Message direction is sampled neighbor -> destination/root. csc_indices,
    edge_lids, and delta_t are aligned by edge position.
    """

    layer: int
    dst_lids: Tensor
    src_lids: Tensor
    csc_indptr: Tensor
    csc_indices: Tensor
    edge_lids: Tensor
    src_range: tuple[int, int]
    dst_range: tuple[int, int]
    delta_t: Optional[Tensor] = None
    time_slice: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingOutput:
    mfgs: list[SampledMFG] | list[Any]
    node_compute: NodeComputeLayout
    edge_compute: Optional[EdgeComputeLayout]
    node_comm: NodeCommLayout
    edge_comm: Optional[EdgeCommLayout] = None
    metadata: dict[str, Any] = field(default_factory=dict)


TemporalSamplingResult = SamplingOutput


def build_edge_prediction_request(
    positive_edges: PositiveEdges,
    negatives: NegativeSamplingResult,
    *,
    fanouts: tuple[int, ...],
    num_layers: int,
    policy: str = "recent",
    include_negative_dst_roots: bool = True,
) -> TemporalSamplingRequest:
    num_pos = int(positive_edges.src.numel())
    pos_roots = torch.cat([positive_edges.src, positive_edges.dst], dim=0).long().contiguous()
    pos_ts = torch.cat([positive_edges.ts, positive_edges.ts], dim=0).contiguous()

    roots = pos_roots
    root_ts = pos_ts
    groups = {
        "pos_src": (0, num_pos),
        "pos_dst": (num_pos, 2 * num_pos),
    }
    if include_negative_dst_roots and negatives.neg_dst.numel() > 0:
        neg_ts = positive_edges.ts.repeat_interleave(max(1, int(negatives.ratio)))[: negatives.neg_dst.numel()]
        neg_begin = int(roots.numel())
        roots = torch.cat([roots, negatives.neg_dst.long()], dim=0).contiguous()
        root_ts = torch.cat([root_ts, neg_ts.to(dtype=positive_edges.ts.dtype)], dim=0).contiguous()
        groups["neg_dst"] = (neg_begin, int(roots.numel()))

    return TemporalSamplingRequest(
        roots=RootSet(nodes=roots, ts=root_ts, groups=groups),
        fanouts=tuple(int(v) for v in fanouts),
        num_layers=int(num_layers),
        policy=str(policy),
        required_nodes=None,
        positive_edges=positive_edges,
        negatives=negatives,
    )
