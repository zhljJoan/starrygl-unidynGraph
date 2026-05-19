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
from starry_unigraph.backends.chunk.data.dist_index import dist_index_is_cached, dist_index_part
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


def _maybe_tensor(value: Any) -> Optional[Tensor]:
    if callable(value):
        value = value()
    return value if isinstance(value, Tensor) else None


def _block_sample_nodes(block: Any) -> Optional[Tensor]:
    for name in ("sample_nodes", "input_nodes", "src_nodes", "nodes"):
        value = _maybe_tensor(getattr(block, name, None))
        if value is not None:
            return value
    srcdata = getattr(block, "srcdata", None)
    if isinstance(srcdata, dict):
        for name in ("__ID", "ID", "_ID"):
            value = srcdata.get(name)
            if isinstance(value, Tensor):
                return value
    return None


def _extract_sampled_input_nodes(blocks: list[Any], roots: Tensor, required_nodes: Optional[Tensor]) -> Tensor:
    pieces: list[Tensor] = [roots.long().reshape(-1)]
    if required_nodes is not None and required_nodes.numel() > 0:
        pieces.append(required_nodes.long().reshape(-1))
    for block in blocks:
        nodes = _block_sample_nodes(block)
        if nodes is not None and nodes.numel() > 0:
            pieces.append(nodes.long().reshape(-1))
    if not pieces:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(pieces, dim=0).unique(sorted=True).contiguous()


def _read_index_for_nodes(nodes: Tensor, placement, local_part: int) -> tuple[Tensor, Tensor, Tensor]:
    if placement.read_dist_index is not None:
        read_index = placement.read_dist_index[nodes].long()
        local_mask = dist_index_is_cached(read_index)
        remote_mask = ~local_mask
    elif placement.master_dist_index is not None:
        read_index = placement.master_dist_index[nodes].long()
        owners = dist_index_part(read_index).long()
        remote_mask = owners != int(local_part)
        local_mask = ~remote_mask
    else:
        raise RuntimeError("CTDG sampled fetch requires read_dist_index or master_dist_index in placement")
    return read_index, local_mask, remote_mask


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
    event_batch_size: int = 0
    require_prebuilt_temporal_index: bool = False
    local_part: int = 0
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
            event_batch_size=int(cfg.get("event_batch_size", cfg.get("batch_size", 0))),
            require_prebuilt_temporal_index=bool(cfg.get("require_prebuilt_temporal_index", False)),
            local_part=int(cfg.get("local_part", 0)),
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
        if self.require_prebuilt_temporal_index and not self.graph_store.has_prebuilt_temporal_index:
            raise FileNotFoundError(
                "Chunk CTDG native sampling requires a prepare-time temporal index "
                "artifact at sampling/temporal_index_part_<rank>.pth"
            )
        for t in split_slices:
            event_indices = self.graph_store.event_indices_for_snapshot(int(t), owner_part=int(self.local_part))
            if event_indices.numel() == 0:
                continue
            slice_start, slice_end = self.graph_store.event_range_for_snapshot(int(t))
            view = self.graph_store.ctdg_input_view(
                batch_id=int(t),
                event_start=slice_start,
                event_end=slice_end,
                time_slice_id=int(t),
                batch_offset=0,
                event_indices=event_indices,
            )
            plans = plans_fn(int(t)) if plans_fn is not None else None
            yield self.make_unit(view, plans)

    def sample(
        self,
        unit: ExecutionUnit,
        *,
        extra_root_nodes: Optional[Tensor] = None,
        extra_root_ts: Optional[Tensor] = None,
        required_nodes: Optional[Tensor] = None,
    ) -> CTDGSampleResult:
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
        root_nodes = view.root_nodes.long().contiguous()
        root_ts = view.root_ts
        if extra_root_nodes is not None and extra_root_nodes.numel() > 0:
            root_nodes = torch.cat([root_nodes, extra_root_nodes.long().cpu()], dim=0).contiguous()
            if root_ts is not None and root_ts.numel() > 0:
                if extra_root_ts is None:
                    extra_root_ts = root_ts.new_empty(0)
                root_ts = torch.cat([root_ts.cpu(), extra_root_ts.cpu().to(root_ts.dtype)], dim=0).contiguous()
        query_ts = root_ts.cpu() if view.temporal_index.timestamps is not None and root_ts is not None and root_ts.numel() > 0 else None
        blocks = sampler.sample(root_nodes.cpu(), query_ts)
        unique_nodes = _extract_sampled_input_nodes(blocks, root_nodes, required_nodes)
        placement = view.temporal_index.placement
        read_index, local_mask, remote_mask = _read_index_for_nodes(unique_nodes, placement, self.local_part)
        return CTDGSampleResult(
            mfgs=blocks,
            input_nodes=unique_nodes,
            output_nodes=root_nodes,
            edge_ids=torch.empty(0, dtype=torch.long),
            node_ts=root_ts,
            edge_ts=None,
            memory_node_ids=unique_nodes,
            remote_node_ids=unique_nodes[remote_mask].contiguous(),
            local_node_ids=unique_nodes[local_mask].contiguous(),
            remote_read_index=read_index[remote_mask].contiguous(),
            local_read_index=read_index[local_mask].contiguous(),
            id_map_nodes=unique_nodes,
        )

    def _build_sampler(self) -> MemShareNativeSampler:
        if self.require_prebuilt_temporal_index and not self.graph_store.has_prebuilt_temporal_index:
            raise FileNotFoundError(
                "Chunk CTDG native sampler cannot be built without the prepare-time "
                "temporal index artifact"
            )
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
            local_part=int(self.local_part),
            node_part=dist_index_part(temporal_index.placement.master_dist_index).cpu().to(torch.int32)
            if temporal_index.placement.master_dist_index is not None
            else temporal_index.placement.node_owner.cpu().to(torch.int32),
            edge_part=torch.zeros(temporal_index.num_edges, dtype=torch.int32),
        )
