"""Canonical chunk graph store and lightweight DTDG/CTDG views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .partition import PartitionData
from .plans import TemporalIndexView, EventView, ChunkPlacement, SnapshotView, PlacementView


def _empty_long() -> Tensor:
    return torch.empty(0, dtype=torch.long)


def _default_placement(part: PartitionData, num_nodes: int) -> ChunkPlacement:
    node_to_chunk = part.node_to_chunk
    if node_to_chunk is None:
        node_to_chunk = torch.zeros(num_nodes, dtype=torch.long)
    node_owner = torch.zeros(num_nodes, dtype=torch.long, device=node_to_chunk.device)
    replica_mask = torch.zeros(num_nodes, dtype=torch.bool, device=node_to_chunk.device)
    return ChunkPlacement(
        placement_version=0,
        node_to_chunk=node_to_chunk.long(),
        node_owner=node_owner,
        node_master=node_owner.clone(),
        replica_mask=replica_mask,
    )


@dataclass(slots=True)
class TemporalEventTable:
    src: Tensor
    dst: Tensor
    ts: Tensor
    edge_ids: Tensor
    snapshot_event_ptr: Tensor


@dataclass(slots=True)
class ChunkGraphStore:
    """Canonical graph data store for chunk-backed DTDG/CTDG execution.

    ``PartitionData`` is the only complete graph artifact.  DTDG and CTDG
    consume lightweight views/indexes derived from it, so switching modes does
    not copy the whole graph.
    """

    part: PartitionData
    placement: ChunkPlacement
    _temporal_index_cache: Optional[TemporalIndexView] = None
    _event_cache: Optional[TemporalEventTable] = None

    @classmethod
    def from_partition_data(
        cls,
        part: PartitionData,
        placement: Optional[ChunkPlacement] = None,
        temporal_index: Optional[TemporalIndexView] = None,
    ) -> "ChunkGraphStore":
        num_nodes = int(part.node_to_chunk.numel()) if part.node_to_chunk is not None else _infer_num_nodes(part)
        return cls(
            part=part,
            placement=placement or _default_placement(part, num_nodes),
            _temporal_index_cache=temporal_index,
        )

    @property
    def placement_version(self) -> int:
        return int(self.placement.placement_version)

    def placement_view(self) -> PlacementView:
        return self.placement.view()

    @property
    def has_prebuilt_temporal_index(self) -> bool:
        return self._temporal_index_cache is not None

    def temporal_index_view(self, sort_by_time: bool = True) -> TemporalIndexView:
        """Return a contiguous temporal adjacency index for sampling.

        The one-time construction sorts edges by ``dst`` and optionally by
        timestamp within each dst node.  CTDG consumes event batches outside
        this index; the index is passed to the sampler/native path only.
        """
        if self._temporal_index_cache is not None and sort_by_time:
            return self._temporal_index_cache

        src, dst, ts, eid = self.part.edge_events(global_ids=True, sort_by_timestamp=False)
        if eid.numel() == 0 and src.numel() > 0:
            eid = torch.arange(src.numel(), dtype=torch.long, device=src.device)
        num_nodes = _infer_num_nodes_from_edges(src, dst, self.placement.node_to_chunk)
        if src.numel() == 0:
            view = TemporalIndexView(
                indptr=torch.zeros(num_nodes + 1, dtype=torch.long),
                indices=_empty_long(),
                edge_ids=_empty_long(),
                timestamps=None if ts.numel() == 0 else ts.contiguous(),
                placement=self.placement_view(),
            )
            if sort_by_time:
                self._temporal_index_cache = view
            return view

        if sort_by_time and ts.numel() == src.numel():
            # Stable two-pass sort keeps timestamp order within each dst group.
            time_order = torch.argsort(ts, stable=True)
            dst_order = torch.argsort(dst[time_order], stable=True)
            order = time_order[dst_order]
        else:
            order = torch.argsort(dst, stable=True)

        sorted_dst = dst[order].long()
        counts = torch.bincount(sorted_dst, minlength=num_nodes)
        indptr = torch.empty(num_nodes + 1, dtype=torch.long, device=src.device)
        indptr[0] = 0
        indptr[1:] = counts.cumsum(0)
        view = TemporalIndexView(
            indptr=indptr.contiguous(),
            indices=src[order].long().contiguous(),
            edge_ids=eid[order].long().contiguous(),
            timestamps=None if ts.numel() != src.numel() else ts[order].contiguous(),
            placement=self.placement_view(),
        )
        if sort_by_time:
            self._temporal_index_cache = view
        return view

    def csc_view(self, sort_by_time: bool = True) -> TemporalIndexView:
        """Compatibility alias; new code should use ``temporal_index_view``."""
        return self.temporal_index_view(sort_by_time=sort_by_time)

    def temporal_events(self) -> TemporalEventTable:
        """Return contiguous event arrays plus snapshot->event offsets.

        This is constructed once from ``PartitionData`` and then reused by
        event-mode loaders.  The hot path slices contiguous tensors only.
        """
        if self._event_cache is not None:
            return self._event_cache

        src_parts = []
        dst_parts = []
        ts_parts = []
        eid_parts = []
        counts = []
        for i in range(len(self.part)):
            src, dst, ts, eid = self.part.edge_events(i, global_ids=True, sort_by_timestamp=False)
            src_parts.append(src.long().contiguous())
            dst_parts.append(dst.long().contiguous())
            if ts.numel() == src.numel():
                ts_parts.append(ts.contiguous())
            else:
                ts_parts.append(torch.zeros(src.numel(), dtype=torch.float32, device=src.device))
            if eid.numel() == src.numel():
                eid_parts.append(eid.long().contiguous())
            else:
                start = sum(counts)
                eid_parts.append(torch.arange(start, start + src.numel(), dtype=torch.long, device=src.device))
            counts.append(int(src.numel()))

        device = src_parts[0].device if src_parts else torch.device("cpu")
        ptr = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
        if counts:
            ptr[1:] = torch.tensor(counts, dtype=torch.long, device=device).cumsum(0)

        self._event_cache = TemporalEventTable(
            src=torch.cat(src_parts, dim=0).contiguous() if src_parts else _empty_long(),
            dst=torch.cat(dst_parts, dim=0).contiguous() if dst_parts else _empty_long(),
            ts=torch.cat(ts_parts, dim=0).contiguous() if ts_parts else torch.empty(0, dtype=torch.float32),
            edge_ids=torch.cat(eid_parts, dim=0).contiguous() if eid_parts else _empty_long(),
            snapshot_event_ptr=ptr.contiguous(),
        )
        return self._event_cache

    def event_range_for_snapshot(self, snapshot_idx: int) -> tuple[int, int]:
        events = self.temporal_events()
        idx = min(max(0, int(snapshot_idx)), max(0, int(events.snapshot_event_ptr.numel()) - 2))
        return int(events.snapshot_event_ptr[idx].item()), int(events.snapshot_event_ptr[idx + 1].item())

    def ctdg_input_view(
        self,
        batch_id: int,
        event_start: int,
        event_end: int,
        *,
        time_slice_id: int | None = None,
        batch_offset: int = 0,
    ) -> EventView:
        events = self.temporal_events()
        event_start = max(0, int(event_start))
        event_end = min(int(event_end), int(events.src.numel()))
        # Keep roots as a contiguous native input.  The MemShare sampler/native
        # MFG path is responsible for internal deduplication.
        root_nodes = torch.cat(
            [events.src[event_start:event_end], events.dst[event_start:event_end]],
            dim=0,
        ).long().contiguous()
        event_ts = events.ts[event_start:event_end].contiguous()
        root_ts = event_ts.repeat(2).contiguous() if event_ts.numel() > 0 else torch.empty(0)
        return EventView(
            batch_id=int(batch_id),
            time_slice_id=int(batch_id if time_slice_id is None else time_slice_id),
            batch_offset=int(batch_offset),
            event_start=event_start,
            event_end=event_end,
            root_nodes=root_nodes.contiguous(),
            root_ts=root_ts,
            temporal_index=self.temporal_index_view(),
            placement_version=self.placement_version,
        )

    def dtdg_input_view(
        self,
        snapshot_id: int,
        window_start: int,
        window_end: int,
        payload=None,
    ) -> SnapshotView:
        src, dst = self.part.to_edge_index(snapshot_index=int(snapshot_id), global_ids=True)
        edge_ids = self.part.edge_ids[int(snapshot_id)].item() if len(self.part.edge_ids) > int(snapshot_id) else None
        node_ids = torch.cat([src, dst]).unique(sorted=True).contiguous() if src.numel() else _empty_long()
        return SnapshotView(
            snapshot_id=int(snapshot_id),
            window_start=int(window_start),
            window_end=int(window_end),
            payload=payload,
            node_ids=node_ids,
            edge_ids=None if edge_ids is None else edge_ids.long().contiguous(),
            placement_version=self.placement_version,
        )


def _infer_num_nodes(part: PartitionData) -> int:
    max_id = -1
    for tensors in (part.src_ids, part.dst_ids):
        data = tensors.data
        if data.numel() > 0:
            max_id = max(max_id, int(data.max().item()))
    return max_id + 1


def _infer_num_nodes_from_edges(src: Tensor, dst: Tensor, node_to_chunk: Tensor) -> int:
    n = int(node_to_chunk.numel()) if node_to_chunk.numel() > 0 else 0
    if src.numel() > 0:
        n = max(n, int(src.max().item()) + 1)
    if dst.numel() > 0:
        n = max(n, int(dst.max().item()) + 1)
    return n
