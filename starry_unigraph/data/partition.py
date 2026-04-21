"""Partitioned graph data containers for distributed DTDG training.

This module defines the core data structures used to store per-partition
snapshot graph data on disk and in memory:

- :class:`TensorData` — variable-length tensor list packed into a single
  contiguous tensor with a pointer array (CSR-style).
- :class:`RouteData` — inter-partition communication metadata (send/recv
  sizes and reorder indices) for all snapshots in a partition.
- :class:`PartitionData` — complete per-partition dataset: node IDs, edge
  topology, features, labels, and optional routing info for every snapshot.

These containers support slicing, ``pin_memory()``, ``.to(device)``,
serialization via ``torch.save``/``torch.load``, and conversion to/from
DGL blocks.

Example::

    # Load a partition file produced by the preprocessing pipeline
    part = PartitionData.load("artifacts/flare/part_000.pth")
    print(len(part))          # number of snapshots
    print(part.num_dst_nodes) # local nodes in this partition

    # Slice a subset of snapshots
    train_part = part[0:140]

    # Convert to DGL blocks for model input
    blocks = train_part.to_blocks(keep_ids=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import dgl
import torch
from dgl.heterograph import DGLBlock
from torch import Tensor


@dataclass
class RouteData:
    """Per-snapshot inter-partition routing metadata for all snapshots.

    Stores the send/recv sizes and a packed reorder index that tells each
    partition which local node features to send to which peer during the
    distributed ``all_to_all`` communication step.

    Attributes:
        send_sizes: ``[num_snaps][num_parts]`` — number of features this
            partition sends to each peer for every snapshot.
        recv_sizes: ``[num_snaps][num_parts]`` — number of features this
            partition receives from each peer.
        send_index_ind: Flat packed local-node indices to gather before
            sending.  ``None`` when routing is disabled (single-rank).
        send_index_ptr: CSR-style pointers into *send_index_ind*, one per
            snapshot (length ``num_snaps + 1``).

    Example::

        route_data = RouteData.from_routes(route_list)
        single_route = route_data[5].item()       # Route for snapshot 5
        route_data_gpu = route_data.to("cuda:0")
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
        if self.send_index_ind is not None and self.send_index_ptr is not None and self.send_index_ptr[-1] != int(self.send_index_ind.numel()):
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

    def item(self, group=None):
        """Return the single :class:`~starry_unigraph.runtime.flare.route.Route` in this container.

        Args:
            group: Optional ``torch.distributed`` process group attached to
                the returned Route.

        Returns:
            A :class:`Route` instance.

        Raises:
            ValueError: If this RouteData contains != 1 snapshot.
        """
        from starry_unigraph.backends.dtdg.runtime.route import Route

        routes = self.to_routes(group=group)
        if len(routes) != 1:
            raise ValueError(f"Expected 1 route, got {len(routes)}")
        return routes[0]

    def to_routes(self, group=None) -> list:
        """Convert all snapshots into a list of :class:`Route` objects.

        Args:
            group: Optional distributed process group for each Route.

        Returns:
            List of :class:`Route`, one per snapshot.
        """
        from starry_unigraph.backends.dtdg.runtime.route import Route

        routes: list[Route] = []
        for index in range(len(self)):
            if self.send_index_ind is None or self.send_index_ptr is None:
                send_index = None
            else:
                start, stop = self.send_index_ptr[index], self.send_index_ptr[index + 1]
                send_index = self.send_index_ind[start:stop]
            routes.append(Route(send_index=send_index, send_sizes=self.send_sizes[index], recv_sizes=self.recv_sizes[index], group=group))
        return routes

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
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            send_index_ptr=self.send_index_ptr,
        )

    @classmethod
    def from_routes(cls, routes: list) -> RouteData:
        """Construct a RouteData from a list of :class:`Route` objects.

        Args:
            routes: List of Route instances (one per snapshot).

        Returns:
            A new :class:`RouteData` packing all routes into flat tensors.
        """
        send_index_parts: list[Tensor] = []
        send_index_ptr = [0]
        send_sizes: list[list[int]] = []
        recv_sizes: list[list[int]] = []
        for route in routes:
            if route.send_index is not None:
                send_index_parts.append(route.send_index.long())
                send_index_ptr.append(send_index_ptr[-1] + int(route.send_index.numel()))
            send_sizes.append([int(size) for size in route.send_sizes])
            recv_sizes.append([int(size) for size in route.recv_sizes])
        send_index_ind = torch.cat(send_index_parts, dim=0) if send_index_parts else None
        return cls(
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr if send_index_ind is not None else None,
        )


@dataclass
class TensorData:
    """CSR-packed variable-length tensor list.

    Stores *N* tensors of possibly different first-dimension sizes as a
    single contiguous ``data`` tensor plus an ``(N+1)``-element ``ptr``
    array.  Element *i* is ``data[ptr[i]:ptr[i+1]]``.

    Supports integer/slice indexing, ``.to(device)``, ``pin_memory()``.

    Attributes:
        ptr: Pointer array of length ``N + 1``.  ``ptr[0] == 0`` and
            ``ptr[-1] == data.size(0)``.
        data: The packed tensor.

    Example::

        td = TensorData.from_tensors([torch.ones(3, 4), torch.zeros(5, 4)])
        print(len(td))       # 2
        print(td[0].item())  # tensor of shape (3, 4)
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
        """Pack a list of tensors into a single TensorData.

        Args:
            tensors: Non-empty list of tensors.  All tensors must have the
                same number of dimensions and compatible trailing shapes.

        Returns:
            A new :class:`TensorData` with ``len(tensors)`` elements.

        Raises:
            ValueError: If *tensors* is empty.
        """
        if not tensors:
            raise ValueError("tensors must not be empty")
        ptr = [0]
        for tensor in tensors:
            ptr.append(ptr[-1] + int(tensor.size(0)))
        return cls(ptr=ptr, data=torch.cat(tensors, dim=0))


@dataclass
class PartitionData:
    """Complete per-partition snapshot dataset for Flare DTDG training.

    Each instance holds *N* snapshots belonging to one graph partition.
    For every snapshot the container stores: local dst node IDs, remote src
    node IDs (boundary nodes), edge topology (src/dst indices, edge IDs),
    arbitrary node/edge feature dicts, and optional routing metadata.

    The class supports:

    * **Indexing / slicing** — ``part[0:100]`` returns a new PartitionData
      with only the first 100 snapshots.
    * **Device transfer** — ``part.to("cuda:0")`` / ``part.pin_memory()``.
    * **Serialization** — ``part.save(path)`` / ``PartitionData.load(path)``.
    * **DGL conversion** — ``part.to_blocks()`` returns a list of DGLBlocks.
    * **Round-trip** — ``PartitionData.from_blocks(blocks)`` reconstructs
      a PartitionData from DGL blocks.

    Attributes:
        src_ids: Remote (boundary) source node IDs per snapshot.
        dst_ids: Local destination node IDs per snapshot.
        edge_ids: Global edge IDs per snapshot.
        edge_src: Edge source indices (into the combined src+dst space).
        edge_dst: Edge destination indices (into the dst-only space).
        node_data: Dict of named per-snapshot node feature TensorData
            (e.g. ``"x"`` for features, ``"y"`` for labels, ``"c"`` for
            chunk assignment).
        edge_data: Dict of named per-snapshot edge feature TensorData
            (e.g. ``"w"`` for weights, ``"gcn_norm"``).
        routes: Optional :class:`RouteData` for distributed communication.

    Example::

        # Preprocessing creates partition files
        part = PartitionData.from_blocks(block_list)
        part.save("artifacts/flare/part_000.pth")

        # Training loads and slices
        part = PartitionData.load("artifacts/flare/part_000.pth")
        train_part = part[0:140]
        blocks = train_part.to_blocks(keep_ids=True)
    """

    src_ids: TensorData
    dst_ids: TensorData
    edge_ids: TensorData
    edge_src: TensorData
    edge_dst: TensorData
    node_data: dict[str, TensorData] = field(default_factory=dict)
    edge_data: dict[str, TensorData] = field(default_factory=dict)
    routes: RouteData | None = None

    def __post_init__(self) -> None:
        num_snaps = len(self.dst_ids)
        for name, value in self.__dict__.items():
            if isinstance(value, TensorData) and len(value) != num_snaps:
                raise ValueError(f"Expected {num_snaps} entries for {name}, got {len(value)}")
        for name, value in self.node_data.items():
            if len(value) != num_snaps:
                raise ValueError(f"Expected {num_snaps} entries for node_data[{name}], got {len(value)}")
        for name, value in self.edge_data.items():
            if len(value) != num_snaps:
                raise ValueError(f"Expected {num_snaps} entries for edge_data[{name}], got {len(value)}")
        if self.routes is not None and len(self.routes) != num_snaps:
            raise ValueError(f"Expected {num_snaps} entries for routes, got {len(self.routes)}")

    def __len__(self) -> int:
        return len(self.dst_ids)

    def __getitem__(self, index: int | slice) -> PartitionData:
        return type(self)(
            src_ids=self.src_ids[index],
            dst_ids=self.dst_ids[index],
            edge_ids=self.edge_ids[index],
            edge_src=self.edge_src[index],
            edge_dst=self.edge_dst[index],
            node_data={key: value[index] for key, value in self.node_data.items()},
            edge_data={key: value[index] for key, value in self.edge_data.items()},
            routes=None if self.routes is None else self.routes[index],
        )

    @property
    def num_snaps(self) -> int:
        return len(self)

    @property
    def num_dst_nodes(self) -> int:
        return int(self.dst_ids[0].item().numel())

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
            node_data={k: v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.node_data.items()},
            edge_data={k: v.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for k, v in self.edge_data.items()},
            routes=None if self.routes is None else self.routes.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
        )

    def pin_memory(self, device: str | None = None) -> PartitionData:
        return type(self)(
            src_ids=self.src_ids.pin_memory(device=device),
            dst_ids=self.dst_ids.pin_memory(device=device),
            edge_ids=self.edge_ids.pin_memory(device=device),
            edge_src=self.edge_src.pin_memory(device=device),
            edge_dst=self.edge_dst.pin_memory(device=device),
            node_data={k: v.pin_memory(device=device) for k, v in self.node_data.items()},
            edge_data={k: v.pin_memory(device=device) for k, v in self.edge_data.items()},
            routes=None if self.routes is None else self.routes.pin_memory(device=device),
        )

    def to_blocks(self, node_perm: torch.Tensor | None = None, keep_ids: bool = False) -> list[DGLBlock]:
        """Convert all snapshots to DGL bipartite blocks.

        Args:
            node_perm: Optional permutation tensor of shape ``[num_dst_nodes]``
                that reorders local nodes (used for chunk-decay training).
            keep_ids: If True, store global node/edge IDs in
                ``srcdata[NID]``, ``dstdata[NID]``, ``edata[EID]``.

        Returns:
            List of :class:`DGLBlock`, one per snapshot.  Each block has
            node/edge features from ``node_data`` / ``edge_data`` attached,
            and ``block.route`` set when routing info is available.
        """
        from starry_unigraph.backends.dtdg.runtime.route import Route

        blocks: list[DGLBlock] = []
        for index in range(len(self)):
            orig_dst_ids = self.dst_ids[index].item()
            dst_ids = orig_dst_ids
            src_suffix = self.src_ids[index].item()
            src_ids = torch.cat([dst_ids, src_suffix], dim=0)
            src = self.edge_src[index].item()
            dst = self.edge_dst[index].item()
            if node_perm is not None:
                if int(node_perm.numel()) != int(dst_ids.numel()):
                    raise ValueError(f"Expected node_perm size {dst_ids.numel()}, got {node_perm.numel()}")
                node_imap = torch.empty_like(node_perm)
                node_imap[node_perm] = torch.arange(node_perm.numel(), dtype=node_perm.dtype, device=node_perm.device)
                dst = node_imap[dst]
                src = src.clone()
                inner_src = src < dst_ids.numel()
                src[inner_src] = node_imap[src[inner_src]].to(src.dtype)
                dst_ids = dst_ids[node_perm]
            block = dgl.create_block((src, dst), num_src_nodes=int(src_ids.numel()), num_dst_nodes=int(dst_ids.numel()), idtype=torch.int32)
            if keep_ids:
                if node_perm is not None:
                    src_ids = torch.cat([dst_ids, src_suffix], dim=0)
                block.srcdata[dgl.NID] = src_ids
                block.dstdata[dgl.NID] = dst_ids
                block.edata[dgl.EID] = self.edge_ids[index].item()
            for key, value in self.node_data.items():
                item = value[index].item()
                if item.size(0) == src_ids.size(0):
                    if node_perm is not None:
                        item = torch.cat([item[node_perm], item[dst_ids.numel() :]], dim=0)
                    block.srcdata[key] = item
                elif item.size(0) == dst_ids.size(0):
                    block.dstdata[key] = item[node_perm] if node_perm is not None else item
                else:
                    raise ValueError(f"Unexpected node_data[{key}] size: {tuple(item.size())}")
            for key, value in self.edge_data.items():
                block.edata[key] = value[index].item()
            if self.routes is not None:
                route = self.routes[index].item()
                if node_perm is not None:
                    permuted_prefix = torch.arange(orig_dst_ids.numel(), dtype=torch.long, device=src_ids.device)[node_perm]
                    if route.send_index is not None:
                        remap_index = torch.full((orig_dst_ids.numel(),), -1, dtype=torch.long, device=src_ids.device)
                        remap_index[permuted_prefix] = torch.arange(orig_dst_ids.numel(), dtype=torch.long, device=src_ids.device)
                        keep_local = route.send_index < orig_dst_ids.numel()
                        send_index = route.send_index.clone()
                        send_index[keep_local] = remap_index[route.send_index[keep_local]]
                        route = Route(send_sizes=route.send_sizes, recv_sizes=route.recv_sizes, send_index=send_index, group=route.group)
                    dst_recv_sizes = route.recv_sizes
                    send_sizes = route.send_sizes
                    route = Route(send_sizes=send_sizes, recv_sizes=dst_recv_sizes, send_index=route.send_index, group=route.group)
                block.route = route
            else:
                block.route = None
            blocks.append(block)
        return blocks

    def item(self, node_perm: torch.Tensor | None = None, keep_ids: bool = False) -> DGLBlock:
        blocks = self.to_blocks(node_perm=node_perm, keep_ids=keep_ids)
        if len(blocks) != 1:
            raise ValueError(f"Expected 1 block, got {len(blocks)}")
        return blocks[0]

    def save(self, path: str | Path) -> None:
        """Serialize this PartitionData to disk via ``torch.save``.

        Args:
            path: File path (parent directories are created automatically).
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> PartitionData:
        """Load a PartitionData from a ``.pth`` file.

        Args:
            path: Path to the serialized file.
            mmap: If True, memory-map the file (useful for large datasets
                that don't fit in RAM).

        Returns:
            The deserialized :class:`PartitionData`.

        Raises:
            TypeError: If the file does not contain a PartitionData.
        """
        loaded = torch.load(Path(path).expanduser().resolve(), mmap=mmap, weights_only=False)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded

    @classmethod
    def from_blocks(cls, blocks: list[DGLBlock]) -> PartitionData:
        """Construct a PartitionData from a list of DGL blocks.

        This is the inverse of :meth:`to_blocks`.  Each block must store
        global node IDs in ``srcdata[NID]`` / ``dstdata[NID]`` and global
        edge IDs in ``edata[EID]``.

        Args:
            blocks: List of DGL bipartite blocks (one per snapshot).

        Returns:
            A new :class:`PartitionData` containing all snapshots.
        """
        src_ids: list[torch.Tensor] = []
        dst_ids: list[torch.Tensor] = []
        edge_ids: list[torch.Tensor] = []
        edge_src: list[torch.Tensor] = []
        edge_dst: list[torch.Tensor] = []
        src_node_data: dict[str, list[torch.Tensor]] = {}
        dst_node_data: dict[str, list[torch.Tensor]] = {}
        edge_data: dict[str, list[torch.Tensor]] = {}
        routes: list = []
        for block in blocks:
            src, dst = block.edges()
            edge_src.append(src)
            edge_dst.append(dst)
            src_id = block.srcdata[dgl.NID] if dgl.NID in block.srcdata else torch.arange(block.num_src_nodes())
            dst_id = block.dstdata[dgl.NID] if dgl.NID in block.dstdata else torch.arange(block.num_dst_nodes())
            edge_id = block.edata[dgl.EID] if dgl.EID in block.edata else torch.arange(block.num_edges())
            if torch.any(src_id[: dst_id.numel()] != dst_id):
                raise ValueError("prefix of src_ids and dst_ids must be the same")
            src_ids.append(src_id[dst_id.numel() :])
            dst_ids.append(dst_id)
            edge_ids.append(edge_id)
            for key in block.srcdata.keys():
                if key != dgl.NID:
                    src_node_data.setdefault(key, []).append(block.srcdata[key])
            for key in block.dstdata.keys():
                if key != dgl.NID:
                    dst_node_data.setdefault(key, []).append(block.dstdata[key])
            for key in block.edata.keys():
                if key != dgl.EID:
                    edge_data.setdefault(key, []).append(block.edata[key])
            route = getattr(block, "route", None)
            if route is not None:
                routes.append(route)
        merged_node_data = {k: TensorData.from_tensors(v) for k, v in dst_node_data.items() if len(v) == len(blocks)}
        for key, values in src_node_data.items():
            if len(values) == len(blocks):
                merged_node_data[key] = TensorData.from_tensors(values)
        return cls(
            src_ids=TensorData.from_tensors(src_ids),
            dst_ids=TensorData.from_tensors(dst_ids),
            edge_ids=TensorData.from_tensors(edge_ids),
            edge_src=TensorData.from_tensors(edge_src),
            edge_dst=TensorData.from_tensors(edge_dst),
            node_data=merged_node_data,
            edge_data={k: TensorData.from_tensors(v) for k, v in edge_data.items() if len(v) == len(blocks)},
            routes=RouteData.from_routes(routes) if len(routes) == len(blocks) else None,
        )
