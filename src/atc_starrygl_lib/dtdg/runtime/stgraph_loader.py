from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from atc_starrygl_lib.models.dtdg.route import Route


@dataclass(slots=True)
class STGraphSnapshot:
    graph: Any
    src_ids: Tensor
    dst_ids: Tensor
    edge_ids: Tensor
    edge_src: Tensor
    edge_dst: Tensor
    snapshot_id: int


class SlidingWindowStateManager:
    def __init__(
        self,
        *,
        num_state_slots: int,
        mode: str = "pad",
        disable_routes: bool = False,
    ) -> None:
        if num_state_slots <= 0:
            raise ValueError("num_state_slots must be positive")
        self._state_list: list[Any] = [None] * int(num_state_slots)
        self._graph_list: list[Any] = []
        self._mode = str(mode)
        self._disable_routes = bool(disable_routes)

    def __len__(self) -> int:
        return len(self._graph_list)

    def __getitem__(self, index: int) -> Any:
        return self._graph_list[index]

    def add(self, graph: Any, *, ends_list: list[int | None]) -> None:
        while len(self._graph_list) >= len(self._state_list):
            self._graph_list.pop(0)
        for index, old_graph in enumerate(self._graph_list):
            old_ends = getattr(old_graph, "flare_window_ends_list", ends_list)
            self._graph_list[index] = self._patch_state_methods(old_graph, ends_list=old_ends)
        self._graph_list.append(self._patch_state_methods(graph, ends_list=ends_list))

    def _patch_state_methods(self, graph: Any, *, ends_list: list[int | None]) -> Any:
        snapshot_id = getattr(graph, "flare_snapshot_id", None)
        if snapshot_id is None:
            snapshot_id = -1
        previous_idx = getattr(graph, "flare_rnn_state_idx", None)
        if previous_idx is None or int(previous_idx) < 0:
            state_idx = len(ends_list) - 1
        else:
            state_idx = int(previous_idx) - 1
        end = ends_list[state_idx]
        if self._disable_routes or end is not None:
            graph = self.truncate_graph(graph, end=end)
            graph.route = None
        graph.flare_window_ends_list = ends_list
        graph.flare_rnn_state_idx = state_idx
        graph.flare_snapshot_id = snapshot_id
        graph.flare_is_full_snapshot = end is None
        graph.flare_fetch_state = types.MethodType(self._flare_fetch_state, graph)
        graph.flare_store_state = types.MethodType(self._flare_store_state, graph)
        return STGraphWindow.patch_route_methods(graph)

    def _flare_fetch_state(self, graph: Any, state: Any, end: int | None = None) -> Any:
        state_idx = int(graph.flare_rnn_state_idx)
        old_state = self._state_list[state_idx]
        if end is None:
            end = int(graph.num_dst_nodes()) if getattr(graph, "is_block", False) else int(graph.num_nodes())
        if old_state is None or self._mode == "pad":
            return self.state_padding(state, end=end)
        if self._mode == "mix":
            return self.state_mixing(state, old_state=old_state, end=end)
        raise ValueError(f"unknown sliding window state mode: {self._mode!r}")

    def _flare_store_state(self, graph: Any, state: Any) -> None:
        state_idx = int(graph.flare_rnn_state_idx)
        if self._mode == "pad":
            self._state_list[state_idx] = None
        elif self._mode == "mix":
            self._state_list[state_idx] = self.state_detach(state)
        else:
            raise ValueError(f"unknown sliding window state mode: {self._mode!r}")

    @classmethod
    def state_detach(cls, state: Any) -> Any:
        return cls._apply_state(lambda value: value.detach() if isinstance(value, Tensor) else value, state)

    @classmethod
    def state_padding(cls, state: Any, *, end: int) -> Any:
        def pad(value: Any) -> Any:
            if not isinstance(value, Tensor):
                return value
            if value.size(0) > end:
                return value[:end]
            if value.size(0) < end:
                padding = tuple([0] * (value.dim() * 2 - 1)) + (end - value.size(0),)
                return F.pad(value, padding)
            return value

        return cls._apply_state(pad, state)

    @classmethod
    def state_mixing(cls, state: Any, *, old_state: Any, end: int) -> Any:
        def mix(cur: Any, old: Any) -> Any:
            if not isinstance(cur, Tensor):
                return cur
            if cur.size(0) > end:
                return cur[:end]
            if cur.size(0) == end:
                return cur
            if old.size(0) >= end:
                return torch.cat([cur, old[cur.size(0):end]], dim=0)
            padding = tuple([0] * (old.dim() * 2 - 1)) + (end - old.size(0),)
            return torch.cat([cur, F.pad(old[cur.size(0):], padding)], dim=0)

        return cls._apply_state(mix, state, old_state=old_state)

    @classmethod
    def _apply_state(cls, fn: Any, state: Any, *, old_state: Any = None) -> Any:
        if isinstance(state, tuple):
            if old_state is None:
                return tuple(cls._apply_state(fn, item) for item in state)
            return tuple(cls._apply_state(fn, item, old_state=old) for item, old in zip(state, old_state))
        if isinstance(state, list):
            if old_state is None:
                return [cls._apply_state(fn, item) for item in state]
            return [cls._apply_state(fn, item, old_state=old) for item, old in zip(state, old_state)]
        if isinstance(state, dict):
            if old_state is None:
                return {key: cls._apply_state(fn, value) for key, value in state.items()}
            return {key: cls._apply_state(fn, value, old_state=old_state[key]) for key, value in state.items()}
        if old_state is None:
            return fn(state)
        return fn(state, old_state)

    @classmethod
    def truncate_graph(cls, graph: Any, end: int | None = None) -> Any:
        import dgl

        if getattr(graph, "is_block", False):
            num_dst_nodes = int(graph.num_dst_nodes())
            end = num_dst_nodes if end is None else int(end)
            if end > num_dst_nodes:
                raise ValueError("end must be less than or equal to num_dst_nodes")
            src, dst = graph.edges()
            keep = (src < end) & (dst < end)
            out = dgl.graph(
                (src[keep], dst[keep]),
                num_nodes=end,
                idtype=graph.idtype,
                device=graph.device,
            )
            for key, value in graph.srcdata.items():
                out.ndata[key] = value[:end]
            for key, value in graph.edata.items():
                out.edata[key] = value[keep]
            return out

        num_nodes = int(graph.num_nodes())
        end = num_nodes if end is None else int(end)
        if end > num_nodes:
            raise ValueError("end must be less than or equal to num_nodes")
        if end == num_nodes:
            return graph
        nodes = torch.arange(end, dtype=graph.idtype, device=graph.device)
        return dgl.node_subgraph(graph, nodes, store_ids=False)


class STGraphWindow:
    def __init__(self, state: SlidingWindowStateManager) -> None:
        self.state = state

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, index: int) -> Any:
        return self.patch_route_methods(self.state[index])

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    @property
    def latest_graph(self) -> Any:
        return self[-1]

    @classmethod
    def patch_route_methods(cls, graph: Any) -> Any:
        graph.flare_apply_route = types.MethodType(_flare_apply_route, graph)
        graph.flare_async_route = types.MethodType(_flare_async_route, graph)
        return graph


class STGraphLoader:
    """Native DTDG loader for sliding-window sampling over partition_data artifacts."""

    def __init__(
        self,
        *,
        partition_data: dict[str, Any],
        device: torch.device | str,
        rank: int,
        world_size: int,
    ) -> None:
        self.data = partition_data
        self.device = torch.device(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.chunk_count = _infer_chunk_count(partition_data)

    def __len__(self) -> int:
        return _td_len(self.data["dst_ids"])

    def iter_snapshots(self, indices: Iterable[int]) -> Iterator[STGraphSnapshot]:
        for sid in indices:
            yield self.fetch_snapshot(int(sid))

    def iter_sliding_windows(
        self,
        *,
        snapshot_ids: Iterable[int] | None = None,
        chunk_order: Tensor | None = None,
        chunk_decay: list[int] | None = None,
        num_full_snapshots: int = 1,
        disable_states: bool = True,
        disable_routes: bool = False,
    ) -> Iterator[STGraphWindow | STGraphSnapshot]:
        ids = range(len(self)) if snapshot_ids is None else snapshot_ids
        if chunk_order is None:
            for sid in ids:
                snapshot = self.fetch_snapshot(int(sid))
                _patch_dummy_state_methods(snapshot.graph)
                STGraphWindow.patch_route_methods(snapshot.graph)
                yield snapshot
            return

        if num_full_snapshots <= 0:
            raise ValueError("num_full_snapshots must be positive")
        state_slots = len(chunk_decay or []) + int(num_full_snapshots)
        manager = SlidingWindowStateManager(
            num_state_slots=state_slots,
            mode="pad" if disable_states else "mix",
            disable_routes=disable_routes,
        )
        for sid in ids:
            sid = int(sid)
            dst_perm = self._dst_permutation(sid, chunk_order=chunk_order)
            ends_list = self._window_ends_for_snapshot(
                sid,
                chunk_order=chunk_order,
                chunk_decay=chunk_decay,
                num_full_snapshots=num_full_snapshots,
                dst_perm=dst_perm,
            )
            snapshot = self.fetch_snapshot(sid, dst_perm=dst_perm)
            manager.add(snapshot.graph, ends_list=ends_list)
            yield STGraphWindow(manager)

    def fetch_graph(self, sid: int) -> Any:
        return self.fetch_snapshot(sid).graph

    def fetch_snapshot(self, sid: int, *, dst_perm: Tensor | None = None) -> STGraphSnapshot:
        import dgl

        src_tail = _td_item(self.data["src_ids"], sid).long()
        dst_ids = _td_item(self.data["dst_ids"], sid).long()
        edge_ids = _td_item(self.data["edge_ids"], sid).long()
        edge_src = _td_item(self.data["edge_src"], sid).long()
        edge_dst = _td_item(self.data["edge_dst"], sid).long()
        if dst_perm is not None and dst_perm.numel() > 0:
            dst_perm = dst_perm.cpu().long()
            dst_ids, edge_src, edge_dst = _reorder_dst_prefix(
                dst_ids=dst_ids,
                src_tail=src_tail,
                edge_src=edge_src,
                edge_dst=edge_dst,
                dst_perm=dst_perm,
            )
        src_ids = torch.cat([dst_ids, src_tail], dim=0).long().contiguous()
        block = dgl.create_block(
            (edge_src, edge_dst),
            num_src_nodes=int(src_ids.numel()),
            num_dst_nodes=int(dst_ids.numel()),
        )
        block.srcdata["ID"] = src_ids
        block.dstdata["ID"] = dst_ids
        block.srcdata["__ID"] = torch.arange(int(src_ids.numel()), dtype=torch.long)
        block.dstdata["__ID"] = torch.arange(int(dst_ids.numel()), dtype=torch.long)
        block.edata["ID"] = edge_ids
        block.edata["__ID"] = torch.arange(int(edge_ids.numel()), dtype=torch.long)
        block = block.to(self.device)
        _patch_node_data(block, self.data.get("node_data", {}), sid, device=self.device, dst_perm=dst_perm)
        _patch_edge_data(block, self.data.get("edge_data", {}), sid, device=self.device)
        route = _route_for_slice(self.data.get("route"), sid, group=None)
        if self.world_size <= 1:
            route = None
        if route is not None and route.send_index is not None:
            route = route.to(device=self.device)
        block.route = route
        block.flare_snapshot_id = int(sid)
        _patch_dummy_state_methods(block)
        STGraphWindow.patch_route_methods(block)
        return STGraphSnapshot(
            graph=block,
            src_ids=src_ids.to(self.device),
            dst_ids=dst_ids.to(self.device),
            edge_ids=edge_ids.to(self.device),
            edge_src=edge_src.to(self.device),
            edge_dst=edge_dst.to(self.device),
            snapshot_id=int(sid),
        )

    def _dst_permutation(self, sid: int, *, chunk_order: Tensor) -> Tensor | None:
        dst_chunk = _td_item(self.data["dst_chunk"], sid).long()
        if dst_chunk.numel() == 0:
            return None
        order = chunk_order.cpu().long().index_select(0, dst_chunk)
        return torch.argsort(order, stable=True)

    def _window_ends_for_snapshot(
        self,
        sid: int,
        *,
        chunk_order: Tensor,
        chunk_decay: list[int] | None,
        num_full_snapshots: int,
        dst_perm: Tensor | None,
    ) -> list[int | None]:
        if not chunk_decay:
            return [None] * int(num_full_snapshots)
        dst_chunk = _td_item(self.data["dst_chunk"], sid).long()
        if dst_perm is not None and dst_perm.numel() > 0:
            dst_chunk = dst_chunk.index_select(0, dst_perm.cpu().long())
        order = chunk_order.cpu().long().index_select(0, dst_chunk) if dst_chunk.numel() else dst_chunk
        ends = [int((order < int(decay)).sum().item()) for decay in reversed(chunk_decay)]
        ends.extend([None] * int(num_full_snapshots))
        return ends


def _td_len(td: dict[str, Tensor]) -> int:
    return int(td["ptr"].numel()) - 1


def _td_item(td: dict[str, Tensor], index: int) -> Tensor:
    begin, end = int(td["ptr"][index]), int(td["ptr"][index + 1])
    return td["data"][begin:end]


def _patch_node_data(
    block: Any,
    node_data: dict[str, dict[str, Tensor]],
    sid: int,
    *,
    device: torch.device,
    dst_perm: Tensor | None = None,
) -> None:
    for key, td in node_data.items():
        value = _td_item(td, sid)
        if key in {"y", "c"}:
            if dst_perm is not None and dst_perm.numel() > 0:
                value = value.index_select(0, dst_perm.cpu().long())
            block.dstdata[key] = value.to(device)
        else:
            if dst_perm is not None and dst_perm.numel() > 0:
                num_dst = int(block.num_dst_nodes())
                value = torch.cat([value[:num_dst].index_select(0, dst_perm.cpu().long()), value[num_dst:]], dim=0)
            block.srcdata[key] = value.to(device)


def _patch_edge_data(block: Any, edge_data: dict[str, dict[str, Tensor]], sid: int, *, device: torch.device) -> None:
    for key, td in edge_data.items():
        block.edata[key] = _td_item(td, sid).to(device)


def _route_for_slice(route_data: dict[str, Any] | None, sid: int, group: Any = None) -> Route | None:
    if not route_data:
        return None
    send_ptr = route_data.get("send_index_ptr")
    send_index = route_data.get("send_index")
    if send_ptr is None or send_index is None:
        send = None
    else:
        begin, end = int(send_ptr[sid]), int(send_ptr[sid + 1])
        send = send_index[begin:end].long().contiguous()
    send_sizes = [int(v) for v in route_data["send_sizes"][sid]]
    recv_sizes = [int(v) for v in route_data["recv_sizes"][sid]]
    return Route(send_sizes=send_sizes, recv_sizes=recv_sizes, send_index=send, group=group)


def _patch_dummy_state_methods(graph: Any) -> None:
    if not hasattr(graph, "flare_snapshot_id"):
        graph.flare_snapshot_id = -1
    graph.flare_rnn_state_idx = -1
    graph.flare_is_full_snapshot = True
    graph.flare_fetch_state = types.MethodType(lambda self, state, end=None: state, graph)
    graph.flare_store_state = types.MethodType(lambda self, state: None, graph)


def _flare_apply_route(graph: Any, x: Tensor, reverse: bool = False) -> Tensor:
    route = getattr(graph, "route", None)
    return x if route is None else route.forward(x, reverse=reverse)


async def _flare_async_route(graph: Any, x: Tensor, reverse: bool = False) -> Tensor:
    route = getattr(graph, "route", None)
    return x if route is None else await route.async_forward(x, reverse=reverse)


def _reorder_dst_prefix(
    *,
    dst_ids: Tensor,
    src_tail: Tensor,
    edge_src: Tensor,
    edge_dst: Tensor,
    dst_perm: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    num_dst = int(dst_ids.numel())
    if int(dst_perm.numel()) != num_dst:
        raise ValueError("dst_perm must cover every dst node")
    inv_perm = torch.empty_like(dst_perm)
    inv_perm[dst_perm] = torch.arange(num_dst, dtype=dst_perm.dtype)
    new_edge_dst = inv_perm.index_select(0, edge_dst.long())
    new_edge_src = edge_src.clone()
    dst_mask = new_edge_src < num_dst
    if bool(dst_mask.any()):
        new_edge_src[dst_mask] = inv_perm.index_select(0, new_edge_src[dst_mask].long())
    new_dst_ids = dst_ids.index_select(0, dst_perm)
    return new_dst_ids, new_edge_src.long().contiguous(), new_edge_dst.long().contiguous()


def _infer_chunk_count(partition_data: dict[str, Any]) -> int:
    dst_chunk = partition_data.get("dst_chunk")
    if not dst_chunk:
        return 0
    data = dst_chunk.get("data")
    if data is None or int(data.numel()) == 0:
        return 0
    return int(data.max().item()) + 1
