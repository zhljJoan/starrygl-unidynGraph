"""Chunk-local MemShare event execution path.

This module intentionally lives under ``backends/chunk`` instead of importing
legacy CTDG runtime wrappers.  Python only binds the native sampler once per
split and passes contiguous tensors/native blocks through the execution unit;
dedup/MFG construction stay in the native path instead of the dataloader hot
loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import torch
from torch import Tensor

from starry_unigraph.lib import load_bts_sampler_module
from starry_unigraph.backends.chunk.data.graph_store import ChunkGraphStore
from starry_unigraph.backends.chunk.data.plans import CTDGSampleResult, EventView, ExecutionUnit, PlanBundle


def is_memshare_native_available() -> bool:
    try:
        load_bts_sampler_module()
    except Exception:
        return False
    return True


def _build_native_temporal_graph(
    graph_name: str,
    row: Tensor,
    col: Tensor,
    num_nodes: int,
    eid: Tensor,
    timestamp: Optional[Tensor],
):
    mod = load_bts_sampler_module()
    return mod.get_neighbors(
        str(graph_name),
        row.long().contiguous(),
        col.long().contiguous(),
        int(num_nodes),
        0,
        eid.long().contiguous(),
        None,
        None,
        None if timestamp is None else timestamp.to(torch.int64).contiguous(),
    )


@dataclass(slots=True)
class MemShareNativeSampler:
    """Small chunk-local wrapper over the C++ temporal sampler."""

    temporal_graph: Any
    num_nodes: int
    num_edges: int
    fanout: list[int]
    num_layers: int
    workers: int = 1
    policy: str = "recent"
    local_part: int = -1
    edge_part: Optional[Tensor] = None
    node_part: Optional[Tensor] = None
    probability: float = 1.0
    _sampler: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        mod = load_bts_sampler_module()
        if self.edge_part is None:
            self.edge_part = torch.zeros(self.num_edges, dtype=torch.int32)
        if self.node_part is None:
            self.node_part = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._sampler = mod.ParallelSampler(
            self.temporal_graph,
            int(self.num_nodes),
            int(self.num_edges),
            int(self.workers),
            list(self.fanout),
            int(self.num_layers),
            str(self.policy),
            int(self.local_part),
            self.edge_part.to(torch.int32).contiguous(),
            self.node_part.to(torch.int32).contiguous(),
            float(self.probability),
        )

    def sample(self, nodes: Tensor, ts: Optional[Tensor]) -> list[Any]:
        timestamps = None if ts is None or ts.numel() == 0 else ts.to(torch.int64).contiguous()
        self._sampler.neighbor_sample_from_nodes(
            nodes.long().contiguous(),
            timestamps,
            None,
        )
        return list(self._sampler.get_ret())

    def reset(self) -> None:
        self._sampler.reset()


@dataclass(slots=True)
class MemShareEventEngine:
    """Builds chunk execution units and invokes MemShare native sampling."""

    graph_store: ChunkGraphStore
    fanout: list[int]
    num_layers: int
    policy: str = "recent"
    workers: int = 1
    graph_name: str = "chunk_events"
    enabled: bool = True
    _samplers: Dict[str, MemShareNativeSampler] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.fanout = list(self.fanout or [10])
        self.num_layers = int(self.num_layers or len(self.fanout) or 1)
        self.workers = int(self.workers)

    @classmethod
    def from_config(cls, graph_store: ChunkGraphStore, cfg: Dict[str, Any]) -> "MemShareEventEngine":
        fanout = cfg.get("fanout") or cfg.get("num_neighbors") or [10]
        if isinstance(fanout, int):
            fanout = [fanout]
        return cls(
            graph_store=graph_store,
            fanout=list(fanout),
            num_layers=int(cfg.get("num_layers", len(fanout))),
            policy=str(cfg.get("policy", cfg.get("sample_type", "recent"))),
            workers=int(cfg.get("workers", cfg.get("num_workers", 1))),
            enabled=bool(cfg.get("enabled", True)),
        )

    @property
    def available(self) -> bool:
        return self.enabled and is_memshare_native_available()

    def make_unit(self, view: EventView, plans: Optional[PlanBundle] = None) -> ExecutionUnit:
        return ExecutionUnit(
            mode="ctdg",
            block_id=int(view.batch_id),
            placement_version=int(view.placement_version),
            payload=view,
            comm_plan=plans or PlanBundle(),
            profile_hint={
                "num_roots": float(int(view.root_nodes.numel())),
                "event_count": float(max(0, int(view.event_end) - int(view.event_start))),
                "native_ready": float(self.available),
            },
        )

    def iter_units(self, split_slices: list[int], plans_fn=None) -> Iterator[ExecutionUnit]:
        for t in split_slices:
            start, end = self.graph_store.event_range_for_snapshot(int(t))
            view = self.graph_store.ctdg_input_view(
                batch_id=int(t),
                event_start=start,
                event_end=end,
            )
            plans = plans_fn(int(t)) if plans_fn is not None else None
            yield self.make_unit(view, plans)

    def sample(self, unit: ExecutionUnit) -> CTDGSampleResult:
        view = unit.payload
        if not isinstance(view, EventView):
            raise TypeError(f"MemShareEventEngine expects EventView payload, got {type(view).__name__}")
        if not self.available:
            raise RuntimeError("MemShare native sampler is not available")
        if view.temporal_index.timestamps is None:
            raise RuntimeError("MemShare event sampling requires timestamped edges")
        sampler = self._samplers.get("default")
        if sampler is None:
            sampler = self._build_sampler()
            self._samplers["default"] = sampler
        query_ts = view.root_ts.cpu() if view.temporal_index.timestamps is not None and view.root_ts is not None else None
        blocks = sampler.sample(view.root_nodes.cpu(), query_ts)
        return CTDGSampleResult(
            mfgs=blocks,
            input_nodes=view.root_nodes,
            output_nodes=view.root_nodes,
            edge_ids=torch.empty(0, dtype=torch.long),
            node_ts=view.root_ts,
            edge_ts=None,
        )

    def _build_sampler(self) -> MemShareNativeSampler:
        temporal_index = self.graph_store.temporal_index_view()
        if temporal_index.num_edges == 0:
            row = torch.empty(0, dtype=torch.long)
            col = torch.empty(0, dtype=torch.long)
        else:
            counts = temporal_index.indptr[1:] - temporal_index.indptr[:-1]
            col = torch.repeat_interleave(
                torch.arange(temporal_index.num_nodes, dtype=torch.long, device=counts.device),
                counts,
            ).contiguous()
            row = temporal_index.indices.long().contiguous()
        temporal_graph = _build_native_temporal_graph(
            graph_name=self.graph_name,
            row=row.cpu(),
            col=col.cpu(),
            num_nodes=temporal_index.num_nodes,
            eid=temporal_index.edge_ids.cpu(),
            timestamp=None if temporal_index.timestamps is None else temporal_index.timestamps.cpu(),
        )
        return MemShareNativeSampler(
            temporal_graph=temporal_graph,
            num_nodes=temporal_index.num_nodes,
            num_edges=temporal_index.num_edges,
            fanout=self.fanout,
            num_layers=self.num_layers,
            workers=self.workers,
            policy=self.policy,
            node_part=temporal_index.placement.node_owner.cpu().to(torch.int32),
            edge_part=torch.zeros(temporal_index.num_edges, dtype=torch.int32),
        )
