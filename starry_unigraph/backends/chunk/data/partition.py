"""Chunk-line PartitionData: Enhanced for chunk-based partitioning and load rebalancing.

This module provides chunk-aware data structures for the chunk pipeline:

- :class:`TensorData` — CSR-packed variable-length tensor list (copied from main).
- :class:`RouteData` — Inter-partition routing metadata (copied from main).
- :class:`PartitionData` — Chunk-enhanced partition container with:
  - Chunk assignment tracking (node_to_chunk, dst_chunk indices)
  - Edge index conversion (from_edge_index, to_edge_index, edge_events)
  - CSR construction sorted by dst chunk (ensures chunk-local edges are contiguous)

Key constraint: When building edge CSR, edges are sorted by dst_chunk
to keep edges within the same dst chunk contiguous in storage.

Example::

    # Load from edge_index (CTDG/DTDG events or snapshots)
    part = PartitionData.from_edge_index(
        edge_src=[...], edge_dst=[...], edge_ts=[...],
        node_to_chunk=[...]  # Global chunk assignment
    )

    # Convert back to edge_index with global IDs
    src, dst = part.to_edge_index(global_ids=True)

    # Get temporal events for CTDG sampler
    events = part.edge_events()  # Returns (src, dst, ts, edge_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from dgl.heterograph import DGLBlock

import dgl


def _create_block_from_csc(
    edge_ptr: Tensor,
    edge_src: Tensor,
    edge_ids: Tensor,
    *,
    num_src_nodes: int,
    num_dst_nodes: int,
    idtype: torch.dtype = torch.int32,
) -> DGLBlock:
    """Create a DGL block from local CSC tensors."""
    indptr = edge_ptr.to(dtype=idtype).contiguous()
    indices = edge_src.to(dtype=idtype).contiguous()
    eids = edge_ids.to(dtype=idtype).contiguous()
    return dgl.create_block(
        ("csc", (indptr, indices, eids)),
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        idtype=idtype,
    )


@dataclass
class TensorData:
    """CSR-packed variable-length tensor list (copied from main data layer).

    Stores *N* tensors of possibly different first-dimension sizes as a
    single contiguous ``data`` tensor plus an ``(N+1)``-element ``ptr`` array.

    Attributes:
        ptr: Pointer array of length ``N + 1``.
        data: The packed tensor.
    """

    ptr: list[int]
    data: Tensor

    def __post_init__(self) -> None:
        if not self.ptr:
            raise ValueError("ptr must not be empty")
        if self.ptr[-1] != int(self.data.size(0)):
            raise ValueError(f"ptr[-1] != data.size(0): {self.ptr[-1]} != {self.data.size(0)}")

    def __len__(self) -> int:
        return len(self.ptr) - 1

    def __getitem__(self, index: int | slice) -> TensorData:
        if not isinstance(index, slice):
            index = slice(index, index + 1)
        if index.step not in (None, 1):
            raise ValueError("Only step size of 1 is supported")
        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        offset_start, offset_stop = self.ptr[start], self.ptr[stop]
        ptr = [value - offset_start for value in self.ptr[start : stop + 1]]
        return type(self)(ptr=ptr, data=self.data[offset_start:offset_stop])

    def item(self) -> Tensor:
        items = self.to_tensors()
        if len(items) != 1:
            raise ValueError(f"Expected 1 tensor, got {len(items)}")
        return items[0]

    def to_tensors(self) -> list[Tensor]:
        items: list[Tensor] = []
        for index in range(len(self)):
            start, stop = self.ptr[index], self.ptr[index + 1]
            items.append(self.data[start:stop])
        return items

    def pin_memory(self, device: str | None = None) -> TensorData:
        return type(self)(ptr=self.ptr, data=self.data.pin_memory(device=device))

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> TensorData:
        return type(self)(
            ptr=self.ptr,
            data=self.data.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
        )

    @classmethod
    def from_tensors(cls, tensors: list[Tensor]) -> TensorData:
        """Pack a list of tensors into a single TensorData."""
        if not tensors:
            raise ValueError("tensors must not be empty")
        ptr = [0]
        for tensor in tensors:
            ptr.append(ptr[-1] + int(tensor.size(0)))
        return cls(ptr=ptr, data=torch.cat(tensors, dim=0))


@dataclass
class RouteData:
    """Inter-partition routing metadata (copied from main data layer).

    Stores send/recv sizes and packed reorder indices for each snapshot.

    Attributes:
        send_sizes: [num_snaps][num_parts] — sizes this partition sends
        recv_sizes: [num_snaps][num_parts] — sizes this partition receives
        send_index_ind: Flat local node indices to send
        send_index_ptr: CSR pointers into send_index_ind
    """

    send_sizes: list[list[int]]
    recv_sizes: list[list[int]]
    send_index_ind: Tensor | None
    send_index_ptr: list[int] | None

    def __post_init__(self) -> None:
        if len(self.send_sizes) != len(self.recv_sizes):
            raise ValueError("send_sizes and recv_sizes must have the same length")
        for send_row, recv_row in zip(self.send_sizes, self.recv_sizes):
            if len(send_row) != len(recv_row):
                raise ValueError("send_sizes[i] and recv_sizes[i] must have the same length")
        if (
            self.send_index_ind is not None
            and self.send_index_ptr is not None
            and self.send_index_ptr[-1] != int(self.send_index_ind.numel())
        ):
            raise ValueError("send_index_ptr[-1] must equal send_index_ind.numel()")

    def __len__(self) -> int:
        return len(self.send_sizes)

    def __getitem__(self, index: int | slice) -> RouteData:
        if not isinstance(index, slice):
            index = slice(index, index + 1)
        if index.step not in (None, 1):
            raise ValueError("Only step size of 1 is supported")
        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        if self.send_index_ind is None or self.send_index_ptr is None:
            send_index_ind = self.send_index_ind
            send_index_ptr = self.send_index_ptr
        else:
            offset_start, offset_stop = self.send_index_ptr[start], self.send_index_ptr[stop]
            send_index_ind = self.send_index_ind[offset_start:offset_stop]
            send_index_ptr = [value - offset_start for value in self.send_index_ptr[start : stop + 1]]
        return type(self)(
            send_sizes=self.send_sizes[start:stop],
            recv_sizes=self.recv_sizes[start:stop],
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
        )

    def pin_memory(self, device: str | None = None) -> RouteData:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.pin_memory(device=device),
            send_index_ptr=self.send_index_ptr,
        )

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> RouteData:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=(
                None
                if self.send_index_ind is None
                else self.send_index_ind.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            ),
            send_index_ptr=self.send_index_ptr,
        )


@dataclass
class PartitionData:
    """Chunk-enhanced per-partition dataset for chunk pipeline.

    Extends the base PartitionData with:
    - Chunk assignment tracking (node_to_chunk)
    - CSR construction sorted by dst_chunk
    - Edge index conversion methods (from_edge_index, to_edge_index, edge_events)

    The key constraint: When edges are stored, they are sorted by dst_chunk
    to ensure edges within the same chunk are contiguous.

    Attributes:
        src_ids: Remote source node IDs per snapshot
        dst_ids: Local destination node IDs per snapshot
        edge_ids: Global edge IDs per snapshot
        edge_src: Edge source indices (into combined src+dst space)
        edge_dst: Destination row indices aligned with edge_ptr; the final
            entry is a sentinel, so length is num_dst_nodes + 1 per snapshot.
        edge_ptr: CSR-style pointer into edge_src, length num_dst_nodes + 1 per snapshot
        dst_chunk: [num_edges] Chunk ID of each destination node
        node_data: Dict of per-snapshot node features
        edge_data: Dict of per-snapshot edge features
        routes: Optional routing metadata
        node_to_chunk: [num_nodes] Global chunk assignment (for reference)
    """

    src_ids: TensorData
    dst_ids: TensorData
    edge_ids: TensorData
    edge_src: TensorData
    edge_dst: TensorData
    edge_ptr: TensorData
    dst_chunk: TensorData  # [num_edges per snapshot] dst chunk IDs
    node_data: dict[str, TensorData] = field(default_factory=dict)
    edge_data: dict[str, TensorData] = field(default_factory=dict)
    routes: RouteData | None = None
    node_to_chunk: Tensor | None = None  # [num_nodes] global chunk assignment

    def __post_init__(self) -> None:
        num_snaps = len(self.dst_ids)
        for name, value in self.__dict__.items():
            if isinstance(value, TensorData) and name != "node_to_chunk":
                if len(value) != num_snaps:
                    raise ValueError(f"Expected {num_snaps} entries for {name}, got {len(value)}")
        for name, value in self.node_data.items():
            if len(value) != num_snaps:
                raise ValueError(f"Expected {num_snaps} entries for node_data[{name}], got {len(value)}")
        for name, value in self.edge_data.items():
            if len(value) != num_snaps:
                raise ValueError(f"Expected {num_snaps} entries for edge_data[{name}], got {len(value)}")
        if self.routes is not None and len(self.routes) != num_snaps:
            raise ValueError(f"Expected {num_snaps} entries for routes, got {len(self.routes)}")
        for index in range(num_snaps):
            edge_ptr = self.edge_ptr[index].item()
            edge_dst = self.edge_dst[index].item()
            if edge_ptr.numel() != self.dst_ids[index].item().numel() + 1:
                raise ValueError(f"Expected edge_ptr[{index}] length dst_ids + 1")
            if edge_dst.numel() != edge_ptr.numel():
                raise ValueError(f"Expected edge_dst[{index}] length to equal edge_ptr length")
            if edge_ptr.numel() > 0 and int(edge_ptr[-1]) != int(self.edge_src[index].item().numel()):
                raise ValueError(f"Expected edge_ptr[{index}][-1] to equal number of edges")

    def __len__(self) -> int:
        return len(self.dst_ids)

    def __getitem__(self, index: int | slice) -> PartitionData:
        return type(self)(
            src_ids=self.src_ids[index],
            dst_ids=self.dst_ids[index],
            edge_ids=self.edge_ids[index],
            edge_src=self.edge_src[index],
            edge_dst=self.edge_dst[index],
            edge_ptr=self.edge_ptr[index],
            dst_chunk=self.dst_chunk[index],
            node_data={key: value[index] for key, value in self.node_data.items()},
            edge_data={key: value[index] for key, value in self.edge_data.items()},
            routes=None if self.routes is None else self.routes[index],
            node_to_chunk=self.node_to_chunk,
        )

    @property
    def num_snaps(self) -> int:
        return len(self)

    @property
    def num_dst_nodes(self) -> int:
        return int(self.dst_ids[0].item().numel())

    def to_block(
        self,
        snapshot_index: int = 0,
        *,
        keep_ids: bool = True,
        idtype: torch.dtype = torch.int32,
    ) -> DGLBlock:
        """Materialize one snapshot as a DGL block directly from CSC storage.

        ``edge_ptr`` is already the CSC pointer by compact dst row and
        ``edge_src`` stores compact source rows.  Building the block from CSC
        avoids constructing a repeated flat dst tensor in the dataloader path.
        """
        src_ids = self.src_ids[snapshot_index].item()
        dst_ids = self.dst_ids[snapshot_index].item()
        edge_src = self.edge_src[snapshot_index].item()
        edge_ptr = self.edge_ptr[snapshot_index].item()
        edge_ids = self.edge_ids[snapshot_index].item()
        num_dst_nodes = int(dst_ids.numel())
        num_src_nodes = int(src_ids.numel() + num_dst_nodes)

        g = _create_block_from_csc(
            edge_ptr=edge_ptr,
            edge_src=edge_src,
            edge_ids=edge_ids,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            idtype=idtype,
        )

        if keep_ids:
            compact_src_ids = torch.cat([dst_ids, src_ids], dim=0)
            g.srcdata[dgl.NID] = compact_src_ids.long()
            g.dstdata[dgl.NID] = dst_ids.long()
            g.edata[dgl.EID] = edge_ids.long()

        for key, val in self.node_data.items():
            data = val[snapshot_index].item()
            if data.size(0) == num_src_nodes:
                g.srcdata[key] = data
            elif data.size(0) == num_dst_nodes:
                g.dstdata[key] = data
            else:
                raise ValueError(f"Node data {key} has invalid size {data.size(0)}")

        for key, val in self.edge_data.items():
            g.edata[key] = val[snapshot_index].item()

        g.route = None if self.routes is None else self.routes[snapshot_index]
        return g

    def to_blocks(
        self,
        *,
        keep_ids: bool = True,
        idtype: torch.dtype = torch.int32,
    ) -> list[DGLBlock]:
        """Materialize all snapshots as CSC-backed DGL blocks."""
        return [self.to_block(i, keep_ids=keep_ids, idtype=idtype) for i in range(len(self))]

    def add_ndata(self, key: str, data: TensorData) -> None:
        if len(data) != len(self):
            raise ValueError(f"Expected {len(self)} entries for node_data[{key}], got {len(data)}")
        self.node_data[key] = data

    def pop_ndata(self, key: str) -> TensorData | None:
        return self.node_data.pop(key, None)

    def add_edata(self, key: str, data: TensorData) -> None:
        if len(data) != len(self):
            raise ValueError(f"Expected {len(self)} entries for edge_data[{key}], got {len(data)}")
        self.edge_data[key] = data

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> PartitionData:
        return type(self)(
            src_ids=self.src_ids.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            dst_ids=self.dst_ids.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            edge_ids=self.edge_ids.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            edge_src=self.edge_src.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            edge_dst=self.edge_dst.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            edge_ptr=self.edge_ptr.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            dst_chunk=self.dst_chunk.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            node_data={k: v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.node_data.items()},
            edge_data={k: v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.edge_data.items()},
            routes=None if self.routes is None else self.routes.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            node_to_chunk=None if self.node_to_chunk is None else self.node_to_chunk.to(device=device, dtype=dtype),
        )

    def pin_memory(self, device: str | None = None) -> PartitionData:
        return type(self)(
            src_ids=self.src_ids.pin_memory(device=device),
            dst_ids=self.dst_ids.pin_memory(device=device),
            edge_ids=self.edge_ids.pin_memory(device=device),
            edge_src=self.edge_src.pin_memory(device=device),
            edge_dst=self.edge_dst.pin_memory(device=device),
            edge_ptr=self.edge_ptr.pin_memory(device=device),
            dst_chunk=self.dst_chunk.pin_memory(device=device),
            node_data={k: v.pin_memory(device=device) for k, v in self.node_data.items()},
            edge_data={k: v.pin_memory(device=device) for k, v in self.edge_data.items()},
            routes=None if self.routes is None else self.routes.pin_memory(device=device),
            node_to_chunk=None if self.node_to_chunk is None else self.node_to_chunk.pin_memory(device),
        )

    @classmethod
    def from_edge_index(
        cls,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_timestamps: Optional[Tensor] = None,
        edge_ids: Optional[Tensor] = None,
        node_to_chunk: Optional[Tensor] = None,
        num_snapshots: int = 1,
    ) -> PartitionData:
        """Construct PartitionData from edge_index (CTDG events or DTDG snapshots).

        Edges are sorted by dst_chunk to ensure chunk-local edges are contiguous.
        For single snapshot, wraps edges in TensorData. For multi-snapshot,
        distributes edges across snapshots.

        Args:
            edge_src: [num_edges] Global source node IDs
            edge_dst: [num_edges] Global destination node IDs
            edge_timestamps: [num_edges] Optional timestamps
            edge_ids: [num_edges] Optional global edge IDs
            node_to_chunk: [num_nodes] Chunk assignment for sorting
            num_snapshots: Number of snapshots (default 1)

        Returns:
            A new PartitionData with edges sorted by dst_chunk.
        """
        from .edge_index_utils import snapshot_to_partitiondata_tensors

        # Convert single snapshot to PartitionData tensors
        tensors_dict = snapshot_to_partitiondata_tensors(
            edge_src, edge_dst, edge_timestamps, edge_ids, node_to_chunk
        )

        # Wrap in TensorData (treating as single snapshot for now)
        src_ids = TensorData.from_tensors([tensors_dict["src_ids"]])
        dst_ids = TensorData.from_tensors([tensors_dict["dst_ids"]])
        edge_ids_td = TensorData.from_tensors([tensors_dict["edge_ids"]]) if tensors_dict["edge_ids"] is not None else TensorData.from_tensors([torch.arange(int(tensors_dict["edge_src"].numel()))])
        edge_src_td = TensorData.from_tensors([tensors_dict["edge_src"]])
        edge_dst_td = TensorData.from_tensors([tensors_dict["edge_dst"]])
        edge_ptr_td = TensorData.from_tensors([tensors_dict["edge_ptr"]])
        dst_chunk_td = TensorData.from_tensors([tensors_dict["dst_chunk"]])
        edge_data = {}
        if tensors_dict["edge_ts"] is not None:
            edge_data["timestamps"] = TensorData.from_tensors([tensors_dict["edge_ts"]])

        return cls(
            src_ids=src_ids,
            dst_ids=dst_ids,
            edge_ids=edge_ids_td,
            edge_src=edge_src_td,
            edge_dst=edge_dst_td,
            edge_ptr=edge_ptr_td,
            dst_chunk=dst_chunk_td,
            node_data={},
            edge_data=edge_data,
            routes=None,
            node_to_chunk=node_to_chunk,
        )

    @classmethod
    def from_edge_events(
        cls,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_timestamps: Tensor,
        edge_ids: Optional[Tensor] = None,
        node_to_chunk: Optional[Tensor] = None,
        event_slice: Optional[slice] = None,
        num_snapshots: int = 1,
    ) -> PartitionData:
        """Construct PartitionData from temporal edge events.

        Args:
            edge_src: [num_events] Global source node IDs
            edge_dst: [num_events] Global destination node IDs
            edge_timestamps: [num_events] Event timestamps
            edge_ids: [num_events] Optional global edge IDs
            node_to_chunk: [num_nodes] Chunk assignment for sorting
            event_slice: Optional slice applied to all event tensors before conversion
            num_snapshots: Number of snapshots (currently only single-snapshot wrapping)

        Returns:
            A new PartitionData containing the selected edge events.
        """
        if event_slice is not None:
            edge_src = edge_src[event_slice]
            edge_dst = edge_dst[event_slice]
            edge_timestamps = edge_timestamps[event_slice]
            edge_ids = None if edge_ids is None else edge_ids[event_slice]

        return cls.from_edge_index(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_timestamps=edge_timestamps,
            edge_ids=edge_ids,
            node_to_chunk=node_to_chunk,
            num_snapshots=num_snapshots,
        )

    def to_edge_index(
        self, snapshot_index: Optional[int] = None, global_ids: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Convert PartitionData back to edge_index format.

        Args:
            snapshot_index: If None, concatenate all snapshots
            global_ids: If True, return global IDs; else return local indices

        Returns:
            (edge_src, edge_dst) tensors
        """
        from .edge_index_utils import reconstruct_edge_index_from_snapshot

        if snapshot_index is None:
            # Concatenate all snapshots
            all_src = []
            all_dst = []
            for idx in range(len(self)):
                src_ids = self.src_ids[idx].item()
                dst_ids = self.dst_ids[idx].item()
                edge_src = self.edge_src[idx].item()
                edge_dst = self.edge_dst[idx].item()
                edge_ptr = self.edge_ptr[idx].item()

                if global_ids:
                    src_global, dst_global = reconstruct_edge_index_from_snapshot(
                        src_ids, dst_ids, edge_src, edge_dst, edge_ptr, global_ids=True
                    )
                    all_src.append(src_global)
                    all_dst.append(dst_global)
                else:
                    src_local, dst_local = reconstruct_edge_index_from_snapshot(
                        src_ids, dst_ids, edge_src, edge_dst, edge_ptr, global_ids=False
                    )
                    all_src.append(src_local)
                    all_dst.append(dst_local)

            return torch.cat(all_src, dim=0), torch.cat(all_dst, dim=0)
        else:
            # Single snapshot
            src_ids = self.src_ids[snapshot_index].item()
            dst_ids = self.dst_ids[snapshot_index].item()
            edge_src = self.edge_src[snapshot_index].item()
            edge_dst = self.edge_dst[snapshot_index].item()
            edge_ptr = self.edge_ptr[snapshot_index].item()

            if global_ids:
                return reconstruct_edge_index_from_snapshot(
                    src_ids, dst_ids, edge_src, edge_dst, edge_ptr, global_ids=True
                )
            else:
                return reconstruct_edge_index_from_snapshot(
                    src_ids, dst_ids, edge_src, edge_dst, edge_ptr, global_ids=False
                )

    def edge_events(
        self,
        snapshot_indices: Optional[int | slice] = None,
        global_ids: bool = True,
        sort_by_timestamp: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get temporal events for CTDG sampler.

        Args:
            snapshot_indices: If None, return all snapshots. If int, return one
                snapshot. If slice, return that contiguous snapshot range.
            global_ids: If True, return global node IDs; else return local indices.
            sort_by_timestamp: If True, sort returned events by timestamp.

        Returns:
            (src, dst, timestamps, edge_ids) for all edges in snapshot(s)
        """
        edge_src_list = []
        edge_dst_list = []
        edge_ts_list = []
        edge_id_list = []

        if snapshot_indices is None:
            start_idx, end_idx = 0, len(self)
        elif isinstance(snapshot_indices, slice):
            if snapshot_indices.step not in (None, 1):
                raise ValueError("Only step size of 1 is supported")
            start_idx, end_idx, _ = snapshot_indices.indices(len(self))
        else:
            start_idx, end_idx = snapshot_indices, snapshot_indices + 1

        for idx in range(start_idx, end_idx):
            src, dst = self.to_edge_index(snapshot_index=idx, global_ids=global_ids)
            edge_src_list.append(src)
            edge_dst_list.append(dst)

            if "timestamps" in self.edge_data:
                edge_ts_list.append(self.edge_data["timestamps"][idx].item())

            if len(self.edge_ids) > 0:
                edge_id_list.append(self.edge_ids[idx].item())

        result_src = torch.cat(edge_src_list, dim=0) if edge_src_list else torch.tensor([], dtype=torch.long)
        result_dst = torch.cat(edge_dst_list, dim=0) if edge_dst_list else torch.tensor([], dtype=torch.long)
        result_ts = torch.cat(edge_ts_list, dim=0) if edge_ts_list else torch.tensor([], dtype=torch.float32)
        result_ids = torch.cat(edge_id_list, dim=0) if edge_id_list else torch.tensor([], dtype=torch.long)

        if sort_by_timestamp:
            if result_ts.numel() != result_src.numel():
                raise ValueError("Cannot sort edge events by timestamp: timestamps are missing or incomplete")
            order = torch.argsort(result_ts, stable=True)
            result_src = result_src[order]
            result_dst = result_dst[order]
            result_ts = result_ts[order]
            result_ids = result_ids[order] if result_ids.numel() == order.numel() else result_ids

        return result_src, result_dst, result_ts, result_ids

    def save(self, path: str | Path) -> None:
        """Serialize this PartitionData to disk via torch.save."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> PartitionData:
        """Load a PartitionData from a .pth file."""
        loaded = torch.load(Path(path).expanduser().resolve(), mmap=mmap, weights_only=False)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded
