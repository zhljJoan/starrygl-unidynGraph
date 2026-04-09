from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import dgl
import torch
from dgl.heterograph import DGLBlock

from .collection import RouteData, TensorData
from .route import Route


@dataclass
class PartitionData:
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
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> PartitionData:
        loaded = torch.load(Path(path).expanduser().resolve(), mmap=mmap, weights_only=False)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded

    @classmethod
    def from_blocks(cls, blocks: list[DGLBlock]) -> PartitionData:
        src_ids: list[torch.Tensor] = []
        dst_ids: list[torch.Tensor] = []
        edge_ids: list[torch.Tensor] = []
        edge_src: list[torch.Tensor] = []
        edge_dst: list[torch.Tensor] = []
        src_node_data: dict[str, list[torch.Tensor]] = {}
        dst_node_data: dict[str, list[torch.Tensor]] = {}
        edge_data: dict[str, list[torch.Tensor]] = {}
        routes: list[Route] = []
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
