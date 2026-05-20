from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from atc_starrygl_lib.lib import load_native_utils_module

PARTITION_DATA_FORMAT = "atc_partition_data_v1"


def build_all_partition_data_artifacts(
    *,
    rank_artifacts: list[dict[str, Any]],
    dist_plan: dict[str, Any],
    src: Tensor,
    dst: Tensor,
    time_ptr_2: Tensor,
    edge_ids: Tensor | None = None,
    node_feat: Tensor | None = None,
    node_feat_time_varying: bool = False,
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
    node_label_time_varying: bool = False,
    edge_label: Tensor | None = None,
    edge_weight: Tensor | None = None,
    build_gcn_norm: bool = True,
    dst_node_scope: str = "active",
) -> list[dict[str, Any]]:
    dst_node_scope = _normalize_dst_node_scope(dst_node_scope)
    artifacts = [
        build_partition_data_artifact(
            rank_artifact=rank_artifact,
            dist_plan=dist_plan,
            src=src,
            dst=dst,
            time_ptr_2=time_ptr_2,
            edge_ids=edge_ids,
            node_feat=node_feat,
            node_feat_time_varying=node_feat_time_varying,
            edge_feat=edge_feat,
            node_label=node_label,
            node_label_time_varying=node_label_time_varying,
            edge_label=edge_label,
            edge_weight=edge_weight,
            build_gcn_norm=build_gcn_norm,
            dst_node_scope=dst_node_scope,
        )
        for rank_artifact in rank_artifacts
    ]
    _attach_master_routes(artifacts=artifacts, rank_artifacts=rank_artifacts, dist_plan=dist_plan)
    return artifacts


def build_partition_data_artifact(
    *,
    rank_artifact: dict[str, Any],
    dist_plan: dict[str, Any],
    src: Tensor,
    dst: Tensor,
    time_ptr_2: Tensor,
    edge_ids: Tensor | None = None,
    node_feat: Tensor | None = None,
    node_feat_time_varying: bool = False,
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
    node_label_time_varying: bool = False,
    edge_label: Tensor | None = None,
    edge_weight: Tensor | None = None,
    build_gcn_norm: bool = True,
    dst_node_scope: str = "active",
) -> dict[str, Any]:
    dst_node_scope = _normalize_dst_node_scope(dst_node_scope)
    src = src.long().cpu().contiguous()
    dst = dst.long().cpu().contiguous()
    time_ptr_2 = time_ptr_2.long().cpu().contiguous()
    edge_ids = torch.arange(src.numel(), dtype=torch.long) if edge_ids is None else edge_ids.long().cpu().contiguous()
    local_edge_ids = rank_artifact["local_edge_ids"].long().cpu().contiguous()
    full_dst_ids = _full_dst_ids_for_rank(rank_artifact) if dst_node_scope == "full" else None
    native = _build_partition_topology_native(
        local_edge_ids=local_edge_ids,
        time_ptr_2=time_ptr_2,
        src=src,
        dst=dst,
        edge_ids=edge_ids,
        node_to_chunk=dist_plan["node_to_chunk"],
        edge_weight=edge_weight,
        build_gcn_norm=build_gcn_norm,
        full_dst_ids=full_dst_ids,
    )
    if native is not None:
        node_data: dict[str, dict[str, Tensor]] = {"c": _td_from(native["dst_chunk_data"], native["dst_chunk_ptr"])}
        if node_feat is not None:
            node_data["x"] = _td_from(
                _select_node_tensor_for_slices(
                    node_feat,
                    native["combined_data"].long(),
                    native["combined_ptr"].long(),
                    time_varying=bool(node_feat_time_varying),
                ),
                native["combined_ptr"],
            )
        if node_label is not None:
            node_data["y"] = _td_from(
                _select_node_tensor_for_slices(
                    node_label,
                    native["dst_data"].long(),
                    native["dst_ptr"].long(),
                    time_varying=bool(node_label_time_varying),
                ),
                native["dst_ptr"],
            )
        edge_data: dict[str, dict[str, Tensor]] = {}
        if edge_feat is not None:
            edge_data["feat"] = _td_from(
                edge_feat.cpu().contiguous().index_select(0, native["edge_id_data"].long()),
                native["edge_id_ptr"],
            )
        if edge_label is not None:
            edge_data["label"] = _td_from(
                edge_label.cpu().contiguous().index_select(0, native["edge_id_data"].long()),
                native["edge_id_ptr"],
            )
        if edge_weight is not None:
            edge_data["w"] = _td_from(
                edge_weight.cpu().contiguous().index_select(0, native["edge_id_data"].long()),
                native["edge_id_ptr"],
            )
        if build_gcn_norm and native.get("gcn_norm_data") is not None:
            edge_data["gcn_norm"] = _td_from(native["gcn_norm_data"], native["gcn_norm_ptr"])
        return {
            "format": PARTITION_DATA_FORMAT,
            "rank": int(rank_artifact["rank"]),
            "dst_node_scope": dst_node_scope,
            "src_ids": _td_from(native["src_data"], native["src_ptr"]),
            "dst_ids": _td_from(native["dst_data"], native["dst_ptr"]),
            "edge_ids": _td_from(native["edge_id_data"], native["edge_id_ptr"]),
            "edge_src": _td_from(native["edge_src_data"], native["edge_src_ptr"]),
            "edge_dst": _td_from(native["edge_dst_data"], native["edge_dst_ptr"]),
            "edge_ptr": _td_from(native["edge_ptr_data"], native["edge_ptr_ptr"]),
            "dst_chunk": _td_from(native["dst_chunk_data"], native["dst_chunk_ptr"]),
            "node_data": node_data,
            "edge_data": edge_data,
            "route": None,
        }

    tensors = _empty_partition_tensors()
    local_edge_ids_sorted = torch.sort(local_edge_ids).values
    for sid, (begin, end) in enumerate(time_ptr_2.tolist()):
        left = int(torch.searchsorted(local_edge_ids_sorted, torch.tensor(int(begin), dtype=torch.long)))
        right = int(torch.searchsorted(local_edge_ids_sorted, torch.tensor(int(end), dtype=torch.long)))
        eids = local_edge_ids_sorted[left:right]
        block = _build_slice_block(
            eids=eids,
            src=src,
            dst=dst,
            edge_ids=edge_ids,
            sid=sid,
            dist_plan=dist_plan,
            node_feat=node_feat,
            node_feat_time_varying=node_feat_time_varying,
            edge_feat=edge_feat,
            node_label=node_label,
            node_label_time_varying=node_label_time_varying,
            edge_label=edge_label,
            edge_weight=edge_weight,
            build_gcn_norm=build_gcn_norm,
            full_dst_ids=full_dst_ids,
        )
        _append_block(tensors, block)
    return {
        "format": PARTITION_DATA_FORMAT,
        "rank": int(rank_artifact["rank"]),
        "dst_node_scope": dst_node_scope,
        "src_ids": _td(tensors["src_ids"]),
        "dst_ids": _td(tensors["dst_ids"]),
        "edge_ids": _td(tensors["edge_ids"]),
        "edge_src": _td(tensors["edge_src"]),
        "edge_dst": _td(tensors["edge_dst"]),
        "edge_ptr": _td(tensors["edge_ptr"]),
        "dst_chunk": _td(tensors["dst_chunk"]),
        "node_data": {key: _td(value) for key, value in tensors["node_data"].items()},
        "edge_data": {key: _td(value) for key, value in tensors["edge_data"].items()},
        "route": None,
    }


def _build_partition_topology_native(
    *,
    local_edge_ids: Tensor,
    time_ptr_2: Tensor,
    src: Tensor,
    dst: Tensor,
    edge_ids: Tensor,
    node_to_chunk: Tensor,
    edge_weight: Tensor | None,
    build_gcn_norm: bool,
    full_dst_ids: Tensor | None,
) -> dict[str, Tensor | None] | None:
    try:
        native = load_native_utils_module()
        weight = (
            torch.empty(0, dtype=torch.float32)
            if edge_weight is None
            else edge_weight.float().cpu().contiguous()
        )
        if full_dst_ids is None:
            out = native.build_partition_topology_with_norm(
                local_edge_ids,
                time_ptr_2,
                src,
                dst,
                edge_ids,
                node_to_chunk.long().cpu().contiguous(),
                weight,
                bool(build_gcn_norm),
            )
        else:
            out = native.build_partition_topology_full_with_norm(
                local_edge_ids,
                time_ptr_2,
                src,
                dst,
                edge_ids,
                node_to_chunk.long().cpu().contiguous(),
                full_dst_ids.long().cpu().contiguous(),
                weight,
                bool(build_gcn_norm),
            )
    except Exception:
        return None
    keys = [
        "src_data",
        "src_ptr",
        "dst_data",
        "dst_ptr",
        "edge_id_data",
        "edge_id_ptr",
        "event_pos_data",
        "event_pos_ptr",
        "edge_src_data",
        "edge_src_ptr",
        "edge_dst_data",
        "edge_dst_ptr",
        "edge_ptr_data",
        "edge_ptr_ptr",
        "dst_chunk_data",
        "dst_chunk_ptr",
        "combined_data",
        "combined_ptr",
    ]
    result: dict[str, Tensor | None] = {key: value for key, value in zip(keys, out)}
    if len(out) >= 20:
        result["gcn_norm_data"] = out[18]
        result["gcn_norm_ptr"] = out[19]
    else:
        result["gcn_norm_data"] = None
        result["gcn_norm_ptr"] = None
    return result


def _build_slice_block(
    *,
    eids: Tensor,
    src: Tensor,
    dst: Tensor,
    edge_ids: Tensor,
    sid: int,
    dist_plan: dict[str, Any],
    node_feat: Tensor | None,
    node_feat_time_varying: bool,
    edge_feat: Tensor | None,
    node_label: Tensor | None,
    node_label_time_varying: bool,
    edge_label: Tensor | None,
    edge_weight: Tensor | None,
    build_gcn_norm: bool,
    full_dst_ids: Tensor | None = None,
) -> dict[str, Any]:
    if eids.numel() == 0:
        dst_ids = torch.empty(0, dtype=torch.long) if full_dst_ids is None else full_dst_ids.long().cpu().contiguous()
        empty_long = torch.empty(0, dtype=torch.long)
        node_data: dict[str, Tensor] = _empty_node_data(node_feat, node_label)
        if full_dst_ids is not None:
            node_data = {}
            if node_feat is not None:
                node_data["x"] = _select_node_tensor(node_feat, dst_ids.long(), sid=sid, time_varying=bool(node_feat_time_varying))
            if node_label is not None:
                node_data["y"] = _select_node_tensor(node_label, dst_ids.long(), sid=sid, time_varying=bool(node_label_time_varying))
            node_data["c"] = _local_chunk_for_nodes(dst_ids, dist_plan)
        return {
            "src_ids": empty_long,
            "dst_ids": dst_ids,
            "edge_ids": empty_long,
            "edge_src": empty_long,
            "edge_dst": empty_long,
            "edge_ptr": torch.zeros(int(dst_ids.numel()) + 1, dtype=torch.long),
            "dst_chunk": _local_chunk_for_nodes(dst_ids, dist_plan) if full_dst_ids is not None else empty_long,
            "node_data": node_data,
            "edge_data": _empty_edge_data(edge_feat, edge_label, edge_weight, build_gcn_norm),
        }
    native_block = _build_slice_block_native(
        eids=eids,
        src=src,
        dst=dst,
        edge_ids=edge_ids,
        node_to_chunk=dist_plan["node_to_chunk"],
        full_dst_ids=full_dst_ids,
    )
    if native_block is not None:
        src_ids = native_block["src_ids"]
        dst_ids = native_block["dst_ids"]
        gids = native_block["edge_ids"]
        combined = torch.cat([dst_ids, src_ids], dim=0)
        node_data: dict[str, Tensor] = {}
        if node_feat is not None:
            node_data["x"] = _select_node_tensor(node_feat, combined.long(), sid=sid, time_varying=bool(node_feat_time_varying))
        if node_label is not None:
            node_data["y"] = _select_node_tensor(node_label, dst_ids.long(), sid=sid, time_varying=bool(node_label_time_varying))
        node_data["c"] = _local_chunk_for_nodes(dst_ids, dist_plan)
        edge_data: dict[str, Tensor] = {}
        if edge_feat is not None:
            edge_data["feat"] = edge_feat.cpu().contiguous().index_select(0, gids)
        if edge_label is not None:
            edge_data["label"] = edge_label.cpu().contiguous().index_select(0, gids)
        if edge_weight is not None:
            edge_data["w"] = edge_weight.cpu().contiguous().index_select(0, gids)
        if build_gcn_norm:
            order = _edge_positions_for_gids(edge_ids=edge_ids, gids=gids)
            edge_data["gcn_norm"] = _gcn_norm(
                s=src.index_select(0, order),
                d=dst.index_select(0, order),
                edge_weight=edge_data.get("w"),
            )
        native_block["node_data"] = node_data
        native_block["edge_data"] = edge_data
        return native_block
    s = src.index_select(0, eids)
    d = dst.index_select(0, eids)
    gids = edge_ids.index_select(0, eids)
    edge_dst_chunk = dist_plan["node_to_chunk"].long().cpu().index_select(0, d)
    eid_scale = int(edge_ids.numel()) + 1
    node_scale = int(dst.max().item()) + 1
    order_key = edge_dst_chunk * node_scale * eid_scale + d * eid_scale + gids
    order = torch.argsort(order_key, stable=True)
    eids = eids.index_select(0, order)
    s = s.index_select(0, order)
    d = d.index_select(0, order)
    gids = gids.index_select(0, order)
    dst_ids = torch.unique(d, sorted=True) if full_dst_ids is None else full_dst_ids.long().cpu().contiguous()
    dst_chunk = dist_plan["node_to_chunk"].long().cpu().index_select(0, dst_ids.long())
    dst_rows = torch.searchsorted(dst_ids, d.long()).long()
    src_unique = torch.unique(s, sorted=True)
    dst_lookup = torch.searchsorted(dst_ids, src_unique)
    src_in_dst = (dst_lookup < int(dst_ids.numel())) & (dst_ids.index_select(0, dst_lookup.clamp_max(max(int(dst_ids.numel()) - 1, 0))) == src_unique)
    src_ids = src_unique[~src_in_dst].long().contiguous()
    combined = torch.cat([dst_ids, src_ids], dim=0)
    src_dst_lookup = torch.searchsorted(dst_ids, s.long())
    src_is_dst = (src_dst_lookup < int(dst_ids.numel())) & (dst_ids.index_select(0, src_dst_lookup.clamp_max(max(int(dst_ids.numel()) - 1, 0))) == s)
    src_tail_lookup = torch.searchsorted(src_ids, s.long())
    src_rows = torch.empty_like(s, dtype=torch.long)
    src_rows[src_is_dst] = src_dst_lookup[src_is_dst]
    src_rows[~src_is_dst] = int(dst_ids.numel()) + src_tail_lookup[~src_is_dst]
    edge_ptr = torch.zeros(int(dst_ids.numel()) + 1, dtype=torch.long)
    counts = torch.bincount(dst_rows, minlength=int(dst_ids.numel()))
    edge_ptr[1:] = counts.cumsum(0)
    node_data: dict[str, Tensor] = {}
    if node_feat is not None:
        node_data["x"] = _select_node_tensor(node_feat, combined.long(), sid=sid, time_varying=bool(node_feat_time_varying))
    if node_label is not None:
        node_data["y"] = _select_node_tensor(node_label, dst_ids.long(), sid=sid, time_varying=bool(node_label_time_varying))
    node_data["c"] = _local_chunk_for_nodes(dst_ids, dist_plan)
    edge_data: dict[str, Tensor] = {}
    if edge_feat is not None:
        edge_data["feat"] = edge_feat.cpu().contiguous().index_select(0, gids)
    if edge_label is not None:
        edge_data["label"] = edge_label.cpu().contiguous().index_select(0, gids)
    if edge_weight is not None:
        edge_data["w"] = edge_weight.cpu().contiguous().index_select(0, gids)
    if build_gcn_norm:
        edge_data["gcn_norm"] = _gcn_norm(s=s, d=d, edge_weight=edge_data.get("w"))
    return {
        "src_ids": src_ids.long().contiguous(),
        "dst_ids": dst_ids.long().contiguous(),
        "edge_ids": gids.long().contiguous(),
        "edge_src": src_rows.long().contiguous(),
        "edge_dst": dst_rows.long().contiguous(),
        "edge_ptr": edge_ptr.long().contiguous(),
        "dst_chunk": dst_chunk.long().contiguous(),
        "node_data": node_data,
        "edge_data": edge_data,
    }


def _build_slice_block_native(
    *,
    eids: Tensor,
    src: Tensor,
    dst: Tensor,
    edge_ids: Tensor,
    node_to_chunk: Tensor,
    full_dst_ids: Tensor | None,
) -> dict[str, Any] | None:
    try:
        native = load_native_utils_module()
        if full_dst_ids is None:
            out = native.build_slice_topology(eids, src, dst, edge_ids, node_to_chunk)
        else:
            out = native.build_slice_topology_full(eids, src, dst, edge_ids, node_to_chunk, full_dst_ids)
    except Exception:
        return None
    src_ids, dst_ids, gids, edge_src, edge_dst, edge_ptr, dst_chunk = out
    return {
        "src_ids": src_ids.long().contiguous(),
        "dst_ids": dst_ids.long().contiguous(),
        "edge_ids": gids.long().contiguous(),
        "edge_src": edge_src.long().contiguous(),
        "edge_dst": edge_dst.long().contiguous(),
        "edge_ptr": edge_ptr.long().contiguous(),
        "dst_chunk": dst_chunk.long().contiguous(),
    }


def _edge_positions_for_gids(*, edge_ids: Tensor, gids: Tensor) -> Tensor:
    if int(edge_ids.numel()) == 0 or int(gids.numel()) == 0:
        return torch.empty(0, dtype=torch.long)
    if int(edge_ids.numel()) > int(edge_ids.max().item()) and bool(torch.equal(edge_ids, torch.arange(int(edge_ids.numel()), dtype=edge_ids.dtype))):
        return gids.long()
    order = torch.argsort(edge_ids.long(), stable=True)
    sorted_ids = edge_ids.long().index_select(0, order)
    pos = torch.searchsorted(sorted_ids, gids.long())
    valid_pos = pos.clamp_max(max(int(sorted_ids.numel()) - 1, 0))
    if not bool(((pos < int(sorted_ids.numel())) & (sorted_ids.index_select(0, valid_pos) == gids.long())).all()):
        raise RuntimeError("edge gids are not all present in edge_ids")
    return order.index_select(0, pos).long()


def _attach_master_routes(
    *,
    artifacts: list[dict[str, Any]],
    rank_artifacts: list[dict[str, Any]],
    dist_plan: dict[str, Any],
) -> None:
    world_size = len(artifacts)
    node_master = dist_plan["node_master"].long().cpu()
    num_slices = _td_len(artifacts[0]["dst_ids"]) if artifacts else 0
    send_sizes = [[[0 for _ in range(world_size)] for _ in range(num_slices)] for _ in range(world_size)]
    recv_sizes = [[[0 for _ in range(world_size)] for _ in range(num_slices)] for _ in range(world_size)]
    send_rows = [[[[] for _ in range(world_size)] for _ in range(num_slices)] for _ in range(world_size)]
    recv_src_rows = [[[] for _ in range(num_slices)] for _ in range(world_size)]
    for sid in range(num_slices):
        provider_dst_ids = [_td_item(artifact["dst_ids"], sid).long() for artifact in artifacts]
        for requester, artifact in enumerate(artifacts):
            src_ids = _td_item(artifact["src_ids"], sid).long()
            if src_ids.numel() == 0:
                continue
            requester_dst_count = int(_td_item(artifact["dst_ids"], sid).numel())
            providers = node_master.index_select(0, src_ids)
            src_rows = torch.arange(
                requester_dst_count,
                requester_dst_count + int(src_ids.numel()),
                dtype=torch.long,
            )
            for provider in range(world_size):
                if provider == requester:
                    continue
                mask = providers == int(provider)
                if not bool(mask.any()):
                    continue
                remote_nodes = src_ids[mask]
                dst_ids = provider_dst_ids[provider]
                if dst_ids.numel() == 0:
                    continue
                lookup = torch.searchsorted(dst_ids, remote_nodes)
                valid = (lookup < int(dst_ids.numel())) & (dst_ids.index_select(0, lookup.clamp_max(int(dst_ids.numel()) - 1)) == remote_nodes)
                if not bool(valid.any()):
                    continue
                provider_rows = lookup[valid].long().contiguous()
                requester_rows = src_rows[mask][valid].long().contiguous()
                count = int(provider_rows.numel())
                recv_sizes[requester][sid][provider] += count
                send_sizes[provider][sid][requester] += count
                send_rows[provider][sid][requester].append(provider_rows)
                recv_src_rows[requester][sid].append(requester_rows)
    for rank, artifact in enumerate(artifacts):
        send_index_parts: list[Tensor] = []
        recv_src_row_parts: list[Tensor] = []
        send_ptr = [0]
        recv_src_row_ptr = [0]
        for sid in range(num_slices):
            flat_parts = [part for rows in send_rows[rank][sid] for part in rows]
            flat = torch.cat(flat_parts, dim=0) if flat_parts else torch.empty(0, dtype=torch.long)
            if flat.numel() > 0:
                send_index_parts.append(flat.long().contiguous())
            send_ptr.append(send_ptr[-1] + int(flat.numel()))
            recv_parts = recv_src_rows[rank][sid]
            recv_rows = torch.cat(recv_parts, dim=0) if recv_parts else torch.empty(0, dtype=torch.long)
            if recv_rows.numel() > 0:
                recv_src_row_parts.append(recv_rows.long().contiguous())
            recv_src_row_ptr.append(recv_src_row_ptr[-1] + int(recv_rows.numel()))
        artifact["route"] = {
            "send_sizes": send_sizes[rank],
            "recv_sizes": recv_sizes[rank],
            "send_index_ptr": torch.tensor(send_ptr, dtype=torch.long),
            "send_index": torch.cat(send_index_parts, dim=0) if send_index_parts else torch.empty(0, dtype=torch.long),
            "recv_src_row_ptr": torch.tensor(recv_src_row_ptr, dtype=torch.long),
            "recv_src_row": torch.cat(recv_src_row_parts, dim=0) if recv_src_row_parts else torch.empty(0, dtype=torch.long),
        }


def _empty_partition_tensors() -> dict[str, Any]:
    return {
        "src_ids": [],
        "dst_ids": [],
        "edge_ids": [],
        "edge_src": [],
        "edge_dst": [],
        "edge_ptr": [],
        "dst_chunk": [],
        "node_data": {},
        "edge_data": {},
    }


def _append_block(tensors: dict[str, Any], block: dict[str, Any]) -> None:
    for key in ("src_ids", "dst_ids", "edge_ids", "edge_src", "edge_dst", "edge_ptr", "dst_chunk"):
        tensors[key].append(block[key])
    for key, value in block["node_data"].items():
        tensors["node_data"].setdefault(key, []).append(value)
    for key, value in block["edge_data"].items():
        tensors["edge_data"].setdefault(key, []).append(value)


def _td(items: list[Tensor]) -> dict[str, Tensor]:
    ptr = [0]
    for item in items:
        ptr.append(ptr[-1] + int(item.size(0)))
    data = torch.cat(items, dim=0) if items else torch.empty(0, dtype=torch.long)
    return {"ptr": torch.tensor(ptr, dtype=torch.long), "data": data.contiguous()}


def _td_from(data: Tensor, ptr: Tensor) -> dict[str, Tensor]:
    return {"ptr": ptr.long().cpu().contiguous(), "data": data.cpu().contiguous()}


def _select_node_tensor_for_slices(
    value: Tensor,
    node_ids: Tensor,
    ptr: Tensor,
    *,
    time_varying: bool,
) -> Tensor:
    value = value.cpu().contiguous()
    node_ids = node_ids.long().cpu().contiguous()
    ptr = ptr.long().cpu().contiguous()
    if not time_varying:
        return value.index_select(0, node_ids)
    parts = [
        _select_node_tensor(value, _td_item({"data": node_ids, "ptr": ptr}, sid), sid=sid, time_varying=True)
        for sid in range(int(ptr.numel()) - 1)
    ]
    if parts:
        return torch.cat(parts, dim=0).contiguous()
    return torch.empty((0, *_node_value_shape(value, time_varying=True)), dtype=value.dtype)


def _select_node_tensor(value: Tensor, node_ids: Tensor, *, sid: int, time_varying: bool) -> Tensor:
    value = value.cpu().contiguous()
    node_ids = node_ids.long().cpu().contiguous()
    source = value[min(int(sid), int(value.size(0)) - 1)] if time_varying else value
    return source.index_select(0, node_ids)


def _node_value_shape(value: Tensor, *, time_varying: bool) -> tuple[int, ...]:
    if not time_varying:
        return tuple(value.shape[1:])
    return tuple(value.shape[2:])


def _td_len(td: dict[str, Tensor]) -> int:
    return int(td["ptr"].numel()) - 1


def _td_item(td: dict[str, Tensor], index: int) -> Tensor:
    begin, end = int(td["ptr"][index]), int(td["ptr"][index + 1])
    return td["data"][begin:end]


def _local_chunk_for_nodes(node_ids: Tensor, dist_plan: dict[str, Any]) -> Tensor:
    return dist_plan["node_to_chunk"].long().cpu().index_select(0, node_ids.long()).long().cpu()


def _normalize_dst_node_scope(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "active": "active",
        "active_dst": "active",
        "active-dst": "active",
        "snapshot": "active",
        "full": "full",
        "full_snapshot": "full",
        "full-snapshot": "full",
        "owned": "full",
        "partition": "full",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"unsupported dst_node_scope: {value!r}") from exc


def _full_dst_ids_for_rank(rank_artifact: dict[str, Any]) -> Tensor:
    if "owned_node_ids" in rank_artifact:
        replica = rank_artifact["local_node_ids"][: int(rank_artifact.get("replica_count", 0))].long().cpu()
        owned = rank_artifact["owned_node_ids"].long().cpu()
        nodes = torch.cat([replica, owned], dim=0) if replica.numel() else owned
    else:
        owned_count = int(rank_artifact.get("owned_count", 0))
        replica_count = int(rank_artifact.get("replica_count", 0))
        nodes = rank_artifact["local_node_ids"][: replica_count + owned_count].long().cpu()
    return torch.unique(nodes, sorted=True).long().contiguous()


def _gcn_norm(*, s: Tensor, d: Tensor, edge_weight: Tensor | None) -> Tensor:
    weight = torch.ones(int(s.numel()), dtype=torch.float32) if edge_weight is None else edge_weight.float().cpu()
    num_nodes = int(torch.cat([s, d]).max().item()) + 1 if s.numel() > 0 else 0
    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    in_deg.scatter_add_(0, d.long(), weight)
    out_deg.scatter_add_(0, s.long(), weight)
    denom = torch.sqrt(in_deg.clamp_min(1e-12).index_select(0, d.long())) * torch.sqrt(out_deg.clamp_min(1e-12).index_select(0, s.long()))
    return (weight / denom).nan_to_num(0.0)


def _empty_node_data(node_feat: Tensor | None, node_label: Tensor | None) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {"c": torch.empty(0, dtype=torch.long)}
    if node_feat is not None:
        feat_shape = tuple(node_feat.shape[2:]) if node_feat.dim() >= 3 else tuple(node_feat.shape[1:])
        out["x"] = torch.empty((0, *feat_shape), dtype=node_feat.dtype)
    if node_label is not None:
        label_shape = tuple(node_label.shape[2:]) if node_label.dim() >= 3 else tuple(node_label.shape[1:])
        out["y"] = torch.empty((0, *label_shape), dtype=node_label.dtype)
    return out


def _empty_edge_data(
    edge_feat: Tensor | None,
    edge_label: Tensor | None,
    edge_weight: Tensor | None,
    build_gcn_norm: bool,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    if edge_feat is not None:
        out["feat"] = torch.empty((0, *edge_feat.shape[1:]), dtype=edge_feat.dtype)
    if edge_label is not None:
        out["label"] = torch.empty((0, *edge_label.shape[1:]), dtype=edge_label.dtype)
    if edge_weight is not None:
        out["w"] = torch.empty((0, *edge_weight.shape[1:]), dtype=edge_weight.dtype)
    if build_gcn_norm:
        out["gcn_norm"] = torch.empty(0, dtype=torch.float32)
    return out
