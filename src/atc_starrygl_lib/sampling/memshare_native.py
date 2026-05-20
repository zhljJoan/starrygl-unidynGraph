from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from atc_starrygl_lib.lib import load_native_utils_module

from atc_starrygl_lib.lib import is_bts_sampler_available, load_bts_sampler_module

from .native import NativeSamplerConfig, NativeSamplerFactory, NativeSamplerUnavailable, NativeTemporalSampler, TemporalGraphData
from .temporal import (
    EdgeCommLayout,
    EdgeComputeLayout,
    NodeCommLayout,
    NodeComputeLayout,
    RootSet,
    SampledMFG,
    SamplingOutput,
    TemporalSamplingRequest,
)


def build_temporal_neighbor_block(graph_name: str, graph: TemporalGraphData) -> Any:
    mod = load_bts_sampler_module()
    return mod.get_neighbors(
        str(graph_name),
        graph.row.long().contiguous(),
        graph.col.long().contiguous(),
        int(graph.num_nodes),
        0,
        graph.edge_ids.long().contiguous(),
        None,
        None,
        None if graph.timestamps is None else graph.timestamps.to(torch.int64).contiguous(),
    )


@dataclass
class MemShareNativeSampler(NativeTemporalSampler):
    temporal_graph: Any
    graph: TemporalGraphData
    config: NativeSamplerConfig
    probability: float = 1.0
    _sampler: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not is_bts_sampler_available():
            raise NativeSamplerUnavailable("libstarrygl_sampler is not available")
        mod = load_bts_sampler_module()
        edge_part = self.graph.edge_part
        node_part = self.graph.node_part
        if edge_part is None:
            edge_part = torch.zeros(self.graph.edge_ids.numel(), dtype=torch.int32)
        if node_part is None:
            node_part = torch.zeros(self.graph.num_nodes, dtype=torch.int32)
        self._sampler = mod.ParallelSampler(
            self.temporal_graph,
            int(self.graph.num_nodes),
            int(self.graph.edge_ids.numel()),
            int(self.config.workers),
            list(self.config.fanouts),
            int(self.config.num_layers),
            str(self.config.policy),
            int(self.config.local_part),
            edge_part.to(torch.int32).contiguous(),
            node_part.to(torch.int32).contiguous(),
            float(self.probability),
        )

    def sample(self, request: TemporalSamplingRequest) -> SamplingOutput:
        root_nodes = request.roots.nodes.long().cpu().contiguous()
        root_ts = request.roots.ts
        timestamps = None
        if root_ts is not None and root_ts.numel() > 0:
            timestamps = root_ts.cpu().to(torch.int64).contiguous()
        self._sampler.neighbor_sample_from_nodes(root_nodes, timestamps, None)
        if hasattr(self._sampler, "get_sampling_output"):
            native = self._sampler.get_sampling_output(root_nodes, timestamps)
            return _convert_native_sampling_output(native, request)
        blocks = list(self._sampler.get_ret())
        node_compute = _build_compat_node_compute(blocks, root_nodes, timestamps, request)
        edge_compute = _build_compat_edge_compute(blocks)
        node_comm = _build_compat_node_comm(node_compute)
        edge_comm = None if edge_compute is None else _build_compat_edge_comm(edge_compute)
        return SamplingOutput(
            mfgs=blocks,
            node_compute=node_compute,
            edge_compute=edge_compute,
            node_comm=node_comm,
            edge_comm=edge_comm,
            metadata={"native": "memshare_bts", **(request.meta or {})},
        )

    def sample_dtdg_uniform(
        self,
        request: TemporalSamplingRequest,
        *,
        t_now: int,
        num_hist: int,
    ) -> list[SamplingOutput]:
        root_nodes = request.roots.nodes.long().cpu().contiguous()
        if int(num_hist) < 1:
            raise ValueError("num_hist must be >= 1")
        if not hasattr(self._sampler, "sample_dtdg_uniform"):
            raise NativeSamplerUnavailable("libstarrygl_sampler does not expose sample_dtdg_uniform")
        native_outputs = list(self._sampler.sample_dtdg_uniform(root_nodes, int(t_now), int(num_hist)))
        start_slice = int(t_now) - int(num_hist) + 1
        outputs: list[SamplingOutput] = []
        for offset, native in enumerate(native_outputs):
            time_slice = start_slice + offset
            meta = {**(request.meta or {}), "time_slice": time_slice, "t_now": int(t_now), "num_hist": int(num_hist)}
            slice_request = TemporalSamplingRequest(
                roots=request.roots.with_time_slice(time_slice),
                fanouts=request.fanouts,
                num_layers=request.num_layers,
                policy="dtdg_uniform",
                required_nodes=request.required_nodes,
                positive_edges=request.positive_edges,
                negatives=request.negatives,
                meta=meta,
            )
            outputs.append(_convert_native_sampling_output(native, slice_request))
        return outputs

    def reset(self) -> None:
        self._sampler.reset()


@dataclass(frozen=True)
class MemShareNativeSamplerFactory(NativeSamplerFactory):
    graph_name: str = "ctdg_events"
    probability: float = 1.0

    def build(self, graph: TemporalGraphData, config: NativeSamplerConfig) -> MemShareNativeSampler:
        temporal_graph = build_temporal_neighbor_block(self.graph_name, graph)
        return MemShareNativeSampler(
            temporal_graph=temporal_graph,
            graph=graph,
            config=config,
            probability=float(self.probability),
        )


def _maybe_tensor(value: Any) -> Optional[Tensor]:
    if callable(value):
        value = value()
    return value if isinstance(value, Tensor) else None


def _convert_native_sampling_output(native: Any, request: TemporalSamplingRequest) -> SamplingOutput:
    mfgs = [
        SampledMFG(
            layer=int(mfg.layer),
            dst_lids=mfg.dst_lids().long().contiguous(),
            src_lids=mfg.src_lids().long().contiguous(),
            csc_indptr=mfg.csc_indptr().long().contiguous(),
            csc_indices=mfg.csc_indices().long().contiguous(),
            edge_lids=mfg.edge_lids().long().contiguous(),
            delta_t=mfg.delta_t().long().contiguous(),
            src_range=(int(mfg.src_begin), int(mfg.src_end)),
            dst_range=(int(mfg.dst_begin), int(mfg.dst_end)),
        )
        for mfg in list(native.mfgs)
    ]
    node_compute = NodeComputeLayout(
        node_gids=native.node_gids().long().contiguous(),
        node_ts=native.node_ts().long().contiguous(),
        layer_ptr=native.node_layer_ptr().long().contiguous(),
        root_gids=native.root_gids().long().contiguous(),
        root_ts=native.root_ts().long().contiguous(),
        root_lids=native.root_lids().long().contiguous(),
        groups=dict(request.roots.groups),
    )
    edge_gids = native.edge_gids().long().contiguous()
    edge_compute = EdgeComputeLayout(
        edge_gids=edge_gids,
        edge_ts=native.edge_ts().long().contiguous(),
        layer_ptr=native.edge_layer_ptr().long().contiguous(),
    )
    node_comm = _build_compat_node_comm(node_compute)
    edge_comm = _build_compat_edge_comm(edge_compute)
    return SamplingOutput(
        mfgs=mfgs,
        node_compute=node_compute,
        edge_compute=edge_compute,
        node_comm=node_comm,
        edge_comm=edge_comm,
        metadata={"native": "memshare_bts_layout", **(request.meta or {})},
    )


def _block_nodes(block: Any) -> Optional[Tensor]:
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


def _block_eids(block: Any) -> Optional[Tensor]:
    for name in ("eid", "edge_ids", "sample_eids"):
        value = _maybe_tensor(getattr(block, name, None))
        if value is not None:
            return value
    return None


def _build_compat_node_compute(
    blocks: list[Any],
    roots: Tensor,
    root_ts: Optional[Tensor],
    request: TemporalSamplingRequest,
) -> NodeComputeLayout:
    unique_roots, root_lids = _stable_unique(roots.long().cpu(), root_ts)
    pieces: list[Tensor] = [unique_roots]
    for block in blocks:
        nodes = _block_nodes(block)
        if nodes is not None and nodes.numel() > 0:
            pieces.append(nodes.long().cpu().reshape(-1))
    node_gids, _ = _stable_unique(torch.cat(pieces, dim=0), None)
    root_end = int(unique_roots.numel())
    if root_ts is not None and root_ts.numel() > 0:
        unique_root_ts = _first_ts_for_lids(root_ts.cpu().to(torch.int64), root_lids, root_end)
    else:
        unique_root_ts = None
    node_ts = None
    if unique_root_ts is not None:
        fill_ts = torch.zeros(max(0, int(node_gids.numel()) - root_end), dtype=unique_root_ts.dtype)
        node_ts = torch.cat([unique_root_ts, fill_ts], dim=0).contiguous()
    return NodeComputeLayout(
        node_gids=node_gids,
        node_ts=node_ts,
        layer_ptr=torch.tensor([0, root_end, int(node_gids.numel())], dtype=torch.long),
        root_gids=roots.long().cpu().contiguous(),
        root_ts=root_ts,
        root_lids=root_lids,
        groups=dict(request.roots.groups),
    )


def _build_compat_edge_compute(blocks: list[Any]) -> Optional[EdgeComputeLayout]:
    pieces: list[Tensor] = []
    for block in blocks:
        eids = _block_eids(block)
        if eids is not None and eids.numel() > 0:
            pieces.append(eids.long().cpu().reshape(-1))
    if not pieces:
        return None
    edge_gids, _ = _stable_unique(torch.cat(pieces, dim=0), None)
    return EdgeComputeLayout(
        edge_gids=edge_gids,
        edge_ts=None,
        layer_ptr=torch.tensor([0, int(edge_gids.numel())], dtype=torch.long),
    )


def _build_compat_node_comm(node_compute: NodeComputeLayout) -> NodeCommLayout:
    node_gids, compute_to_comm = _stable_unique(node_compute.node_gids.long().cpu(), None)
    count = int(node_gids.numel())
    owner = torch.zeros(count, dtype=torch.long)
    provider = torch.zeros(count, dtype=torch.long)
    return NodeCommLayout(
        node_gids=node_gids,
        time_slices=None,
        owner=owner,
        provider=provider,
        provider_ptr=torch.tensor([0, count], dtype=torch.long),
        compute_to_comm=compute_to_comm,
    )


def _build_compat_edge_comm(edge_compute: EdgeComputeLayout) -> EdgeCommLayout:
    edge_gids, compute_to_comm = _stable_unique(edge_compute.edge_gids.long().cpu(), None)
    count = int(edge_gids.numel())
    owner = torch.zeros(count, dtype=torch.long)
    provider = torch.zeros(count, dtype=torch.long)
    return EdgeCommLayout(
        edge_gids=edge_gids,
        time_slices=None,
        owner=owner,
        provider=provider,
        provider_ptr=torch.tensor([0, count], dtype=torch.long),
        compute_to_comm=compute_to_comm,
    )


def _stable_unique(values: Tensor, ts: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    try:
        native = load_native_utils_module()
        if ts is None:
            unique, inverse = native.stable_unique(values.cpu().reshape(-1).contiguous())
            return unique.to(dtype=values.dtype), inverse.long()
        unique, inverse, _ = native.stable_unique_with_ts(
            values.cpu().reshape(-1).contiguous(),
            ts.cpu().reshape(-1).contiguous(),
        )
        return unique.to(dtype=values.dtype), inverse.long()
    except Exception:
        pass
    values = values.cpu().reshape(-1).contiguous()
    if values.numel() == 0:
        return values, torch.empty(0, dtype=torch.long)
    if ts is None:
        _, inverse_sorted = torch.unique(values, sorted=True, return_inverse=True)
    else:
        keys = torch.stack([values.long(), ts.cpu().reshape(-1).long()], dim=1)
        _, inverse_sorted = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    positions = torch.arange(int(values.numel()), dtype=torch.long)
    first_pos = torch.full((int(inverse_sorted.max().item()) + 1,), int(values.numel()), dtype=torch.long)
    first_pos.scatter_reduce_(0, inverse_sorted.long(), positions, reduce="amin", include_self=True)
    stable_order = torch.argsort(first_pos, stable=True)
    sorted_to_stable = torch.empty_like(stable_order)
    sorted_to_stable[stable_order] = torch.arange(int(stable_order.numel()), dtype=torch.long)
    inverse = sorted_to_stable.index_select(0, inverse_sorted.long())
    return values.index_select(0, first_pos.index_select(0, stable_order)).to(dtype=values.dtype), inverse.long()


def _first_ts_for_lids(root_ts: Tensor, root_lids: Tensor, size: int) -> Tensor:
    try:
        native = load_native_utils_module()
        return native.first_ts_for_lids(
            root_ts.cpu().reshape(-1).contiguous(),
            root_lids.cpu().reshape(-1).contiguous(),
            int(size),
        ).to(dtype=root_ts.dtype)
    except Exception:
        pass
    out = torch.zeros(size, dtype=root_ts.dtype)
    if int(size) == 0 or root_lids.numel() == 0:
        return out
    lids = root_lids.cpu().reshape(-1).long()
    positions = torch.arange(int(lids.numel()), dtype=torch.long)
    first_pos = torch.full((int(size),), int(lids.numel()), dtype=torch.long)
    first_pos.scatter_reduce_(0, lids, positions, reduce="amin", include_self=True)
    keep = first_pos < int(lids.numel())
    out[keep] = root_ts.cpu().reshape(-1).index_select(0, first_pos[keep])
    return out
