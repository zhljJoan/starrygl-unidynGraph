from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from atc_starrygl_lib.comm.dynamic import AsyncTensorHandle, DynamicFetchComm
from atc_starrygl_lib.comm.layouts import FeatureReadLayout
from atc_starrygl_lib.comm.static_route import StaticRoute, StaticRouteComm
from atc_starrygl_lib.runtime.index import DistIndexTables, build_feature_read_layout_from_comm
from atc_starrygl_lib.sampling.temporal import SamplingOutput

from .store import FeatureStore


@dataclass(slots=True)
class DTDGSnapshotFeatureView:
    time_slice: int
    dst_rows: Tensor
    edge_rows: Tensor | None = None


class SampledFeatureRuntime:
    """Common feature fetch path for CTDG and sampled-DTDG outputs."""

    def __init__(
        self,
        index: DistIndexTables,
        feature_store: FeatureStore,
        comm: DynamicFetchComm,
        *,
        world_size: int,
    ) -> None:
        self.index = index
        self.feature_store = feature_store
        self.comm = comm
        self.world_size = int(world_size)

    def build_layout_from_sampling(
        self,
        output: SamplingOutput,
        *,
        already_rank_grouped: bool = False,
        deduplicate: bool = False,
    ) -> FeatureReadLayout:
        return build_feature_read_layout_from_comm(
            output.node_comm.node_gids,
            output.node_comm.compute_to_comm,
            self.index.read_dist_index,
            world_size=self.world_size,
            time_slices=output.node_comm.time_slices,
            deduplicate=deduplicate,
            already_rank_grouped=already_rank_grouped,
        )

    def submit_fetch(self, layout: FeatureReadLayout) -> AsyncTensorHandle:
        return self.comm.submit_row_fetch(
            layout.read_index,
            layout.read_ptr,
            self.feature_store.gather_rows,
            time_slices=layout.time_slices,
        )

    def wait_fetch(self, handle: AsyncTensorHandle, layout: FeatureReadLayout) -> Tensor:
        (features,) = handle.wait()
        return features.index_select(0, layout.compute_to_feature.to(features.device))

    def fetch_sampling(self, output: SamplingOutput) -> Tensor:
        layout = self.build_layout_from_sampling(output)
        return self.wait_fetch(self.submit_fetch(layout), layout)


class CTDGFeatureRuntime(SampledFeatureRuntime):
    def __init__(
        self,
        index: DistIndexTables,
        feature_store: FeatureStore,
        comm: DynamicFetchComm,
        *,
        world_size: int,
        edge_dist_index: Tensor | None = None,
    ) -> None:
        super().__init__(index=index, feature_store=feature_store, comm=comm, world_size=world_size)
        self.edge_dist_index = edge_dist_index

    def build_edge_layout_from_sampling(
        self,
        output: SamplingOutput,
        *,
        already_rank_grouped: bool = False,
        deduplicate: bool = False,
    ) -> FeatureReadLayout | None:
        if output.edge_comm is None or self.edge_dist_index is None:
            return None
        return build_feature_read_layout_from_comm(
            output.edge_comm.edge_gids,
            output.edge_comm.compute_to_comm,
            self.edge_dist_index,
            world_size=self.world_size,
            time_slices=output.edge_comm.time_slices,
            deduplicate=deduplicate,
            already_rank_grouped=already_rank_grouped,
        )

    def submit_edge_fetch(self, layout: FeatureReadLayout) -> AsyncTensorHandle:
        return self.comm.submit_row_fetch(
            layout.read_index,
            layout.read_ptr,
            self.feature_store.gather_edge_rows,
            time_slices=layout.time_slices,
        )

    def wait_edge_fetch(self, handle: AsyncTensorHandle, layout: FeatureReadLayout) -> Tensor:
        (features,) = handle.wait()
        return features.index_select(0, layout.compute_to_feature.to(features.device))

    def patch_mfg(self, mfgs: list[Any], features: Tensor, key: str = "h") -> None:
        for block in _flatten(mfgs):
            if hasattr(block, "srcdata"):
                block.srcdata[key] = features.to(next(iter(block.srcdata.values())).device) if block.srcdata else features


class DTDGSampledFeatureRuntime(SampledFeatureRuntime):
    pass


class DTDGSnapshotFeatureRuntime:
    """PartitionData/STGraphLoader feature path for full snapshot DTDG."""

    def __init__(self, feature_store: FeatureStore, route_comm: StaticRouteComm | None = None) -> None:
        self.feature_store = feature_store
        self.route_comm = route_comm or StaticRouteComm()

    def gather_snapshot(self, view: DTDGSnapshotFeatureView) -> tuple[Tensor, Tensor | None]:
        node_ts = view.dst_rows.new_full((view.dst_rows.numel(),), int(view.time_slice))
        node_feat = self.feature_store.gather_node_rows(view.dst_rows, time_slices=node_ts)
        edge_feat = None
        if view.edge_rows is not None:
            edge_ts = view.edge_rows.new_full((view.edge_rows.numel(),), int(view.time_slice))
            edge_feat = self.feature_store.gather_edge_rows(view.edge_rows, time_slices=edge_ts)
        return node_feat, edge_feat

    def apply_static_route(self, dst_x: Tensor, route: StaticRoute | None) -> Tensor:
        return self.route_comm.forward(dst_x, route)


def _flatten(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        out: list[Any] = []
        for item in items:
            out.extend(_flatten(item))
        return out
    return [items]
