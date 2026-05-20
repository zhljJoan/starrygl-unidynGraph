from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

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
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
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
            edge_feat=edge_feat,
            node_label=node_label,
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
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
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
    edge_keep = torch.zeros(int(src.numel()), dtype=torch.bool)
    edge_keep[local_edge_ids] = True
    tensors = _empty_partition_tensors()
    full_dst_ids = _full_dst_ids_for_rank(rank_artifact) if dst_node_scope == "full" else None
    for begin, end in time_ptr_2.tolist():
        eids = torch.arange(int(begin), int(end), dtype=torch.long)
        eids = eids[edge_keep[eids]]
        block = _build_slice_block(
            eids=eids,
            src=src,
            dst=dst,
            edge_ids=edge_ids,
            dist_plan=dist_plan,
            node_feat=node_feat,
            edge_feat=edge_feat,
            node_label=node_label,
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


def _build_slice_block(
    *,
    eids: Tensor,
    src: Tensor,
    dst: Tensor,
    edge_ids: Tensor,
    dist_plan: dict[str, Any],
    node_feat: Tensor | None,
    edge_feat: Tensor | None,
    node_label: Tensor | None,
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
                node_data["x"] = node_feat.cpu().contiguous().index_select(0, dst_ids.long())
            if node_label is not None:
                node_data["y"] = node_label.cpu().contiguous().index_select(0, dst_ids.long())
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
        node_data["x"] = node_feat.cpu().contiguous().index_select(0, combined.long())
    if node_label is not None:
        node_data["y"] = node_label.cpu().contiguous().index_select(0, dst_ids.long())
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


def _td_len(td: dict[str, Tensor]) -> int:
    return int(td["ptr"].numel()) - 1


def _td_item(td: dict[str, Tensor], index: int) -> Tensor:
    begin, end = int(td["ptr"][index]), int(td["ptr"][index + 1])
    return td["data"][begin:end]


def _local_chunk_for_nodes(node_ids: Tensor, dist_plan: dict[str, Any]) -> Tensor:
    chunks = dist_plan["node_to_chunk"].long().cpu().index_select(0, node_ids.long())
    _, compact = torch.unique(chunks, sorted=True, return_inverse=True)
    return compact.long().cpu()


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
        out["x"] = torch.empty((0, *node_feat.shape[1:]), dtype=node_feat.dtype)
    if node_label is not None:
        out["y"] = torch.empty((0, *node_label.shape[1:]), dtype=node_label.dtype)
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
