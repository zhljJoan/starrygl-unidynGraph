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
) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
    src = src.long().cpu().contiguous()
    dst = dst.long().cpu().contiguous()
    time_ptr_2 = time_ptr_2.long().cpu().contiguous()
    edge_ids = torch.arange(src.numel(), dtype=torch.long) if edge_ids is None else edge_ids.long().cpu().contiguous()
    local_edge_ids = rank_artifact["local_edge_ids"].long().cpu().contiguous()
    edge_keep = torch.zeros(int(src.numel()), dtype=torch.bool)
    edge_keep[local_edge_ids] = True
    tensors = _empty_partition_tensors()
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
        )
        _append_block(tensors, block)
    return {
        "format": PARTITION_DATA_FORMAT,
        "rank": int(rank_artifact["rank"]),
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
) -> dict[str, Any]:
    if eids.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long)
        return {
            "src_ids": empty_long,
            "dst_ids": empty_long,
            "edge_ids": empty_long,
            "edge_src": empty_long,
            "edge_dst": empty_long,
            "edge_ptr": torch.zeros(1, dtype=torch.long),
            "dst_chunk": empty_long,
            "node_data": _empty_node_data(node_feat, node_label),
            "edge_data": _empty_edge_data(edge_feat, edge_label, edge_weight, build_gcn_norm),
        }
    s = src.index_select(0, eids)
    d = dst.index_select(0, eids)
    gids = edge_ids.index_select(0, eids)
    dst_chunk = dist_plan["node_to_chunk"].long().cpu().index_select(0, d)
    eid_scale = int(edge_ids.numel()) + 1
    node_scale = int(dst.max().item()) + 1
    order_key = dst_chunk * node_scale * eid_scale + d * eid_scale + gids
    order = torch.argsort(order_key, stable=True)
    eids = eids.index_select(0, order)
    s = s.index_select(0, order)
    d = d.index_select(0, order)
    gids = gids.index_select(0, order)
    dst_chunk = dst_chunk.index_select(0, order)
    dst_ids = torch.unique(d, sorted=True)
    dst_pos = _index_map(dst_ids)
    dst_rows = torch.tensor([dst_pos[int(n)] for n in d.tolist()], dtype=torch.long)
    src_unique = torch.unique(s, sorted=True)
    dst_set = set(dst_ids.tolist())
    src_ids = torch.tensor([int(n) for n in src_unique.tolist() if int(n) not in dst_set], dtype=torch.long)
    combined = torch.cat([dst_ids, src_ids], dim=0)
    src_pos = _index_map(combined)
    src_rows = torch.tensor([src_pos[int(n)] for n in s.tolist()], dtype=torch.long)
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
    local_rows = [
        {int(nid): row for row, nid in enumerate(rank_artifact["local_node_ids"].tolist())}
        for rank_artifact in rank_artifacts
    ]
    for rank, artifact in enumerate(artifacts):
        num_slices = _td_len(artifact["dst_ids"])
        send_sizes = [[0 for _ in range(world_size)] for _ in range(num_slices)]
        recv_sizes = [[0 for _ in range(world_size)] for _ in range(num_slices)]
        send_index_parts: list[Tensor] = []
        send_ptr = [0]
        for sid in range(num_slices):
            send_rows_by_peer: list[list[int]] = [[] for _ in range(world_size)]
            for nid in _td_item(artifact["src_ids"], sid).tolist():
                master = int(node_master[int(nid)])
                if master == rank:
                    continue
                recv_sizes[sid][master] += 1
                if int(nid) in local_rows[master]:
                    send_rows_by_peer[master].append(local_rows[master][int(nid)])
                    send_sizes[sid][master] += 1
            flat = [row for rows in send_rows_by_peer for row in rows]
            if flat:
                send_index_parts.append(torch.tensor(flat, dtype=torch.long))
            send_ptr.append(send_ptr[-1] + len(flat))
        artifact["route"] = {
            "send_sizes": send_sizes,
            "recv_sizes": recv_sizes,
            "send_index_ptr": torch.tensor(send_ptr, dtype=torch.long),
            "send_index": torch.cat(send_index_parts, dim=0) if send_index_parts else torch.empty(0, dtype=torch.long),
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


def _index_map(values: Tensor) -> dict[int, int]:
    return {int(value): idx for idx, value in enumerate(values.tolist())}


def _local_chunk_for_nodes(node_ids: Tensor, dist_plan: dict[str, Any]) -> Tensor:
    chunks = dist_plan["node_to_chunk"].long().cpu().index_select(0, node_ids.long())
    unique = torch.unique(chunks, sorted=True)
    cmap = {int(cid): idx for idx, cid in enumerate(unique.tolist())}
    return torch.tensor([cmap[int(cid)] for cid in chunks.tolist()], dtype=torch.long)


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
