from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import encode_dist_index

RANK_FORMAT = "atc_rank_v1"


def build_all_rank_artifacts(
    *,
    dist_plan: dict[str, Any],
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    time_ptr_2: Tensor,
    split_time_ptr: dict[str, Tensor] | None = None,
    split: Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build rank-local shared layouts from a global dist plan."""

    src = src.long().cpu().contiguous()
    dst = dst.long().cpu().contiguous()
    time_ptr_2 = time_ptr_2.long().cpu().contiguous()
    split_time_ptr_cpu = None if split_time_ptr is None else {k: v.long().cpu().contiguous() for k, v in split_time_ptr.items()}
    ts_cpu = torch.arange(src.numel(), dtype=torch.float32) if ts is None else ts.cpu().contiguous()
    world_size = int(dist_plan["world_size"])
    layouts = [
        build_rank_layout(
            dist_plan=dist_plan,
            src=src,
            dst=dst,
            rank=rank,
        )
        for rank in range(world_size)
    ]
    dist_plan, layouts = finalize_dist_index(dist_plan=dist_plan, layouts=layouts)
    layouts = [
        finalize_update_nodes(
            layout=layout,
            src=src,
            dst=dst,
            ts=ts_cpu,
            time_ptr_2=time_ptr_2,
        )
        for layout in layouts
    ]
    layouts = finalize_memory_routes(layouts=layouts, dist_plan=dist_plan)
    split_cpu = torch.full((int(time_ptr_2.size(0)),), 0, dtype=torch.uint8) if split is None else split.to(torch.uint8).cpu()
    rank_artifacts = [
        {
            "format": RANK_FORMAT,
            "rank": int(layout["rank"]),
            "world_size": world_size,
            "local_node_ids": layout["local_node_ids"],
            "replica_count": int(layout["replica_count"]),
            "owned_count": int(layout["owned_count"]),
            "shadow_count": int(layout["shadow_count"]),
            "local_edge_ids": layout["local_edge_ids"],
            "read_dist_index": layout["read_dist_index"],
            "owned_chunks": layout["owned_chunks"],
            "owned_node_ids": layout["owned_node_ids"],
            "local_chunk_ids": layout["local_chunk_ids"],
            "local_node_to_chunk": layout["local_node_to_chunk"],
            "local_chunk_ptr": layout["local_chunk_ptr"],
            "local_chunk_nodes": layout["local_chunk_nodes"],
            "time_ptr_2": time_ptr_2,
            "split_event_pos": _build_local_split_event_pos(
                local_edge_ids=layout["local_edge_ids"],
                split_time_ptr=split_time_ptr_cpu,
                time_ptr_2=time_ptr_2,
            ),
            "split_time_ptr": _build_local_split_time_ptr(
                local_edge_ids=layout["local_edge_ids"],
                split_time_ptr=split_time_ptr_cpu,
                time_ptr_2=time_ptr_2,
            ),
            "split": split_cpu,
            "update_node_ptr": layout["update_node_ptr"],
            "update_node_ids": layout["update_node_ids"],
            "update_node_ts": layout["update_node_ts"],
            "update_local_row": layout["update_local_row"],
            "memory_route": layout["memory_route"],
        }
        for layout in layouts
    ]
    return dist_plan, rank_artifacts


def _build_local_split_time_ptr(
    *,
    local_edge_ids: Tensor,
    split_time_ptr: dict[str, Tensor] | None,
    time_ptr_2: Tensor,
) -> dict[str, Tensor]:
    local_edge_ids = local_edge_ids.long().cpu().contiguous()
    if split_time_ptr is None:
        names = ("train", "val", "test")
        windows = {
            "train": time_ptr_2,
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        }
    else:
        names = tuple(split_time_ptr.keys())
        windows = split_time_ptr
    out: dict[str, Tensor] = {}
    for name in names:
        win = windows[name].long().cpu().contiguous()
        if win.numel() == 0:
            out[name] = torch.zeros((0, 2), dtype=torch.long)
            continue
        sorted_edges = torch.sort(local_edge_ids).values
        left = torch.searchsorted(sorted_edges, win[:, 0].contiguous())
        right = torch.searchsorted(sorted_edges, win[:, 1].contiguous())
        counts = (right - left).long()
        starts = torch.zeros_like(counts)
        if counts.numel() > 1:
            starts[1:] = counts.cumsum(0)[:-1]
        out[name] = torch.stack([starts, starts + counts], dim=1).long().contiguous()
    return out


def _build_local_split_event_pos(
    *,
    local_edge_ids: Tensor,
    split_time_ptr: dict[str, Tensor] | None,
    time_ptr_2: Tensor,
) -> dict[str, dict[str, Tensor]]:
    local_edge_ids = torch.sort(local_edge_ids.long().cpu().contiguous()).values
    if split_time_ptr is None:
        names = ("train", "val", "test")
        windows = {
            "train": time_ptr_2,
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        }
    else:
        names = tuple(split_time_ptr.keys())
        windows = split_time_ptr
    out: dict[str, dict[str, Tensor]] = {}
    for name in names:
        win = windows[name].long().cpu().contiguous()
        if win.numel() == 0 or local_edge_ids.numel() == 0:
            out[name] = {
                "data": torch.empty(0, dtype=torch.long),
                "ptr": torch.zeros(int(win.size(0)) + 1, dtype=torch.long),
            }
            continue
        left = torch.searchsorted(local_edge_ids, win[:, 0].contiguous())
        right = torch.searchsorted(local_edge_ids, win[:, 1].contiguous())
        parts = [local_edge_ids[int(begin):int(end)] for begin, end in zip(left.tolist(), right.tolist())]
        counts = (right - left).long()
        ptr = torch.zeros(int(counts.numel()) + 1, dtype=torch.long)
        if counts.numel() > 0:
            ptr[1:] = counts.cumsum(0)
        out[name] = {
            "data": torch.cat(parts, dim=0).long().contiguous() if parts else torch.empty(0, dtype=torch.long),
            "ptr": ptr.long().contiguous(),
        }
    return out


def build_rank_layout(
    *,
    dist_plan: dict[str, Any],
    src: Tensor,
    dst: Tensor,
    rank: int,
) -> dict[str, Any]:
    rank = int(rank)
    edge_owner = dist_plan["edge_owner"].long().cpu()
    node_master = dist_plan["node_master"].long().cpu()
    replica_mask = dist_plan["replica_mask"].bool().cpu()
    node_to_chunk = dist_plan["node_to_chunk"].long().cpu()
    chunk_owner = dist_plan["chunk_owner"].long().cpu()
    local_edge_ids = _edge_ids_for_rank(dist_plan, edge_owner=edge_owner, rank=rank)
    if "local_node_ids_by_part" in dist_plan:
        replica = dist_plan["replica_node_ids_by_part"][rank].long().cpu().contiguous()
        owned = dist_plan["owned_node_ids_by_part"][rank].long().cpu().contiguous()
        shadow = dist_plan["shadow_node_ids_by_part"][rank].long().cpu().contiguous()
        owned = _append_missing_owned_masters(
            owned=owned,
            replica=replica,
            shadow=shadow,
            node_master=node_master,
            replica_mask=replica_mask,
            rank=rank,
        )
    else:
        replica = replica_mask.nonzero(as_tuple=True)[0].long().cpu()
        owned = ((node_master == rank) & ~replica_mask).nonzero(as_tuple=True)[0].long().cpu()
        touched = _unique_touched_nodes(src=src, dst=dst, event_ids=local_edge_ids)
        local_base = torch.zeros(int(node_master.numel()), dtype=torch.bool)
        local_base[replica] = True
        local_base[owned] = True
        shadow = touched[~local_base[touched]].long().cpu()
    local_node_ids = torch.cat([replica, owned, shadow], dim=0).long().contiguous()
    local_row = _build_local_row(local_node_ids, num_nodes=int(node_master.numel()))
    owned_chunks = (chunk_owner == rank).nonzero(as_tuple=True)[0].long().cpu().contiguous()
    local_chunk_ids, local_node_to_chunk, local_chunk_ptr, local_chunk_nodes = build_local_chunk_view(
        local_node_ids=local_node_ids,
        node_to_chunk=node_to_chunk,
        replica_count=int(replica.numel()),
        owned_count=int(owned.numel()),
    )
    return {
        "rank": rank,
        "local_node_ids": local_node_ids,
        "replica_count": int(replica.numel()),
        "owned_count": int(owned.numel()),
        "shadow_count": int(shadow.numel()),
        "local_row": local_row,
        "local_edge_ids": local_edge_ids,
        "owned_chunks": owned_chunks,
        "owned_node_ids": owned,
        "local_chunk_ids": local_chunk_ids,
        "local_node_to_chunk": local_node_to_chunk,
        "local_chunk_ptr": local_chunk_ptr,
        "local_chunk_nodes": local_chunk_nodes,
    }


def finalize_dist_index(
    *,
    dist_plan: dict[str, Any],
    layouts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_master = dist_plan["node_master"].long().cpu()
    replica_mask = dist_plan["replica_mask"].bool().cpu()
    edge_owner = dist_plan["edge_owner"].long().cpu()
    num_nodes = int(node_master.numel())
    master_dist_index = torch.empty(num_nodes, dtype=torch.long)
    for rank, layout in enumerate(layouts):
        nodes = (node_master == int(rank)).nonzero(as_tuple=True)[0].long()
        if nodes.numel() == 0:
            continue
        local = layout["local_row"].index_select(0, nodes).long()
        master_dist_index[nodes] = encode_dist_index(
            local,
            torch.full((int(nodes.numel()),), int(rank), dtype=torch.long),
            shared=replica_mask.index_select(0, nodes),
        )
    for layout in layouts:
        local_row = layout["local_row"]
        rank = int(layout["rank"])
        read_dist_index = master_dist_index.clone()
        local_nodes = layout["local_node_ids"]
        if local_nodes.numel() > 0:
            rows = local_row.index_select(0, local_nodes).long()
            shared = rows < int(layout["replica_count"])
            cached = rows >= int(layout["replica_count"]) + int(layout["owned_count"])
            read_dist_index[local_nodes] = encode_dist_index(
                rows,
                torch.full((int(rows.numel()),), rank, dtype=torch.long),
                shared=shared,
                cached=cached,
            )
        layout["read_dist_index"] = read_dist_index.long().contiguous()
    dist_plan = dict(dist_plan)
    dist_plan["master_dist_index"] = master_dist_index.long().contiguous()
    dist_plan["edge_dist_index"] = encode_dist_index(
        _rank_local_edge_rows(edge_owner),
        edge_owner,
    ).long().contiguous()
    return dist_plan, layouts


def finalize_update_nodes(
    *,
    layout: dict[str, Any],
    src: Tensor,
    dst: Tensor,
    ts: Tensor,
    time_ptr_2: Tensor,
) -> dict[str, Any]:
    local_row = layout["local_row"]
    local_edge_ids = layout["local_edge_ids"]
    edge_keep = torch.zeros(int(src.numel()), dtype=torch.bool)
    edge_keep[local_edge_ids] = True
    ptr = [0]
    node_parts: list[Tensor] = []
    ts_parts: list[Tensor] = []
    row_parts: list[Tensor] = []
    for begin, end in time_ptr_2.tolist():
        begin, end = int(begin), int(end)
        if end <= begin:
            ptr.append(ptr[-1])
            continue
        eids = torch.arange(begin, end, dtype=torch.long)
        eids = eids[edge_keep[eids]]
        if eids.numel() == 0:
            ptr.append(ptr[-1])
            continue
        cand_nodes = torch.cat([src.index_select(0, eids), dst.index_select(0, eids)], dim=0).long()
        cand_ts = torch.cat([ts.index_select(0, eids), ts.index_select(0, eids)], dim=0)
        rows = local_row.index_select(0, cand_nodes)
        keep = rows >= 0
        if not bool(keep.any()):
            ptr.append(ptr[-1])
            continue
        nodes, max_ts = _unique_nodes_with_max_ts(cand_nodes[keep], cand_ts[keep])
        node_parts.append(nodes)
        ts_parts.append(max_ts)
        row_parts.append(local_row.index_select(0, nodes).long())
        ptr.append(ptr[-1] + int(nodes.numel()))
    layout["update_node_ptr"] = torch.tensor(ptr, dtype=torch.long)
    layout["update_node_ids"] = torch.cat(node_parts, dim=0).long().contiguous() if node_parts else torch.empty(0, dtype=torch.long)
    layout["update_node_ts"] = torch.cat(ts_parts, dim=0).contiguous() if ts_parts else torch.empty(0, dtype=ts.dtype)
    layout["update_local_row"] = torch.cat(row_parts, dim=0).long().contiguous() if row_parts else torch.empty(0, dtype=torch.long)
    return layout


def finalize_memory_routes(
    *,
    layouts: list[dict[str, Any]],
    dist_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    sends = [_empty_send_accum(int(layout["update_node_ptr"].numel()) - 1) for layout in layouts]
    recvs = [_empty_recv_accum(int(layout["update_node_ptr"].numel()) - 1) for layout in layouts]
    for src_rank, layout in enumerate(layouts):
        update_ptr = layout["update_node_ptr"]
        update_nodes = layout["update_node_ids"]
        update_rows = layout["update_local_row"]
        for t in range(int(update_ptr.numel()) - 1):
            begin, end = int(update_ptr[t]), int(update_ptr[t + 1])
            if end <= begin:
                continue
            nodes = update_nodes[begin:end].long()
            positions = torch.arange(begin, end, dtype=torch.long)
            local_rows = update_rows[begin:end].long()
            for dst_rank, dst_layout in enumerate(layouts):
                if dst_rank == src_rank:
                    continue
                target_rows = dst_layout["local_row"].index_select(0, nodes)
                keep = target_rows >= 0
                if not bool(keep.any()):
                    continue
                target_rows = target_rows[keep].long().contiguous()
                target_index = encode_dist_index(
                    target_rows,
                    torch.full((int(target_rows.numel()),), dst_rank, dtype=torch.long),
                )
                sends[src_rank]["rank"][t].append(torch.full((int(target_rows.numel()),), dst_rank, dtype=torch.long))
                sends[src_rank]["update_pos"][t].append(positions[keep].long().contiguous())
                sends[src_rank]["local_row"][t].append(local_rows[keep].long().contiguous())
                sends[src_rank]["dist_index"][t].append(target_index.long().contiguous())
                recvs[dst_rank]["rank"][t].append(torch.full((int(target_rows.numel()),), src_rank, dtype=torch.long))
                recvs[dst_rank]["local_row"][t].append(target_rows)
                recvs[dst_rank]["dist_index"][t].append(target_index.long().contiguous())
    for rank, layout in enumerate(layouts):
        layout["memory_route"] = _pack_memory_route(sends[rank], recvs[rank])
    return layouts


def build_local_chunk_view(
    *,
    local_node_ids: Tensor,
    node_to_chunk: Tensor,
    replica_count: int,
    owned_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    local_node_ids = local_node_ids.long().cpu().contiguous()
    node_to_chunk = node_to_chunk.long().cpu()
    local_node_to_chunk = torch.full((int(local_node_ids.numel()),), -1, dtype=torch.long)
    active_end = int(replica_count) + int(owned_count)
    if active_end == 0:
        return torch.empty(0, dtype=torch.long), local_node_to_chunk, torch.zeros(1, dtype=torch.long), torch.empty(0, dtype=torch.long)
    global_chunks = node_to_chunk.index_select(0, local_node_ids[:active_end])
    local_chunk_ids = torch.unique(global_chunks, sorted=True)
    compact = torch.searchsorted(local_chunk_ids, global_chunks).long()
    local_node_to_chunk[:active_end] = compact
    order = torch.argsort(compact, stable=True)
    counts = torch.bincount(compact, minlength=int(local_chunk_ids.numel()))
    ptr = torch.zeros(int(local_chunk_ids.numel()) + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return local_chunk_ids.long().contiguous(), local_node_to_chunk, ptr, order.long().contiguous()


def _edge_ids_for_rank(dist_plan: dict[str, Any], *, edge_owner: Tensor, rank: int) -> Tensor:
    if "edge_ids_by_part" in dist_plan:
        return dist_plan["edge_ids_by_part"][rank].long().cpu().contiguous()
    return (edge_owner == int(rank)).nonzero(as_tuple=True)[0].long().cpu().contiguous()


def _unique_touched_nodes(*, src: Tensor, dst: Tensor, event_ids: Tensor) -> Tensor:
    if event_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat([src.index_select(0, event_ids), dst.index_select(0, event_ids)], dim=0), sorted=True).long()


def _build_local_row(local_node_ids: Tensor, *, num_nodes: int) -> Tensor:
    local_row = torch.full((int(num_nodes),), -1, dtype=torch.long)
    if local_node_ids.numel() > 0:
        local_row[local_node_ids] = torch.arange(int(local_node_ids.numel()), dtype=torch.long)
    return local_row


def _append_missing_owned_masters(
    *,
    owned: Tensor,
    replica: Tensor,
    shadow: Tensor,
    node_master: Tensor,
    replica_mask: Tensor,
    rank: int,
) -> Tensor:
    """Keep native speed-partition layouts total over all master-owned nodes."""

    masters = ((node_master == int(rank)) & ~replica_mask).nonzero(as_tuple=True)[0].long().cpu()
    if masters.numel() == 0:
        return owned
    present = torch.zeros(int(node_master.numel()), dtype=torch.bool)
    if replica.numel() > 0:
        present[replica.long()] = True
    if owned.numel() > 0:
        present[owned.long()] = True
    if shadow.numel() > 0:
        present[shadow.long()] = True
    missing = masters[~present.index_select(0, masters)]
    if missing.numel() == 0:
        return owned
    return torch.cat([owned.long(), missing.long()], dim=0).contiguous()


def _rank_local_edge_rows(edge_owner: Tensor) -> Tensor:
    rows = torch.empty_like(edge_owner.long())
    for rank in torch.unique(edge_owner.long(), sorted=True).tolist():
        ids = (edge_owner == int(rank)).nonzero(as_tuple=True)[0]
        rows[ids] = torch.arange(int(ids.numel()), dtype=torch.long)
    return rows


def _unique_nodes_with_max_ts(nodes: Tensor, ts: Tensor) -> tuple[Tensor, Tensor]:
    unique, inverse = torch.unique(nodes.long(), sorted=True, return_inverse=True)
    max_ts = torch.full((int(unique.numel()),), -float("inf"), dtype=ts.dtype)
    max_ts.scatter_reduce_(0, inverse.long(), ts, reduce="amax", include_self=True)
    return unique.long().contiguous(), max_ts.contiguous()


def _empty_send_accum(num_slices: int) -> dict[str, list[list[Tensor]]]:
    return {
        "rank": [[] for _ in range(num_slices)],
        "update_pos": [[] for _ in range(num_slices)],
        "local_row": [[] for _ in range(num_slices)],
        "dist_index": [[] for _ in range(num_slices)],
    }


def _empty_recv_accum(num_slices: int) -> dict[str, list[list[Tensor]]]:
    return {
        "rank": [[] for _ in range(num_slices)],
        "local_row": [[] for _ in range(num_slices)],
        "dist_index": [[] for _ in range(num_slices)],
    }


def _pack_memory_route(send: dict[str, list[list[Tensor]]], recv: dict[str, list[list[Tensor]]]) -> dict[str, Tensor]:
    send_rank, send_pos, send_row, send_index, send_ptr = _pack_slices(
        send["rank"],
        send["update_pos"],
        send["local_row"],
        send["dist_index"],
    )
    recv_rank, recv_row, recv_index, recv_ptr = _pack_slices(
        recv["rank"],
        recv["local_row"],
        recv["dist_index"],
    )
    return {
        "send_ptr": send_ptr,
        "send_rank": send_rank,
        "send_update_pos": send_pos,
        "send_local_row": send_row,
        "send_dist_index": send_index,
        "recv_ptr": recv_ptr,
        "recv_rank": recv_rank,
        "recv_local_row": recv_row,
        "recv_dist_index": recv_index,
    }


def _pack_slices(*items: list[list[Tensor]]) -> tuple[Tensor, ...]:
    ptr = [0]
    flat_items = [[] for _ in items]
    num_slices = len(items[0]) if items else 0
    for t in range(num_slices):
        count = sum(int(part.numel()) for part in items[0][t])
        ptr.append(ptr[-1] + count)
        for out, item in zip(flat_items, items):
            out.extend(item[t])
    tensors = [torch.cat(values, dim=0).long().contiguous() if values else torch.empty(0, dtype=torch.long) for values in flat_items]
    tensors.append(torch.tensor(ptr, dtype=torch.long))
    return tuple(tensors)
