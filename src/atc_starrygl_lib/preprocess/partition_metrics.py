from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def count_cross_partition_edges(*, src: Tensor, dst: Tensor, node_owner: Tensor) -> int:
    src = src.long().cpu()
    dst = dst.long().cpu()
    node_owner = node_owner.long().cpu()
    src_owner = node_owner.index_select(0, src)
    dst_owner = node_owner.index_select(0, dst)
    return int((src_owner != dst_owner).sum().item())


def aggregate_rank_loads(*, chunk_load: Tensor, chunk_owner: Tensor, world_size: int) -> Tensor:
    chunk_load = chunk_load.float().cpu()
    chunk_owner = chunk_owner.long().cpu()
    rank_load = torch.zeros((int(chunk_load.size(0)), int(world_size)), dtype=torch.float32)
    for rank in range(int(world_size)):
        mask = chunk_owner == rank
        if bool(mask.any()):
            rank_load[:, rank] = chunk_load[:, mask].sum(dim=1)
    return rank_load


def summarize_rank_loads(rank_load: Tensor) -> dict[str, Any]:
    rank_load = rank_load.float().cpu()
    if rank_load.numel() == 0:
        return {
            "num_batches": 0,
            "avg_batch_load_ratio": float("nan"),
            "zero_min_load_batches": 0,
            "all_zero_batches": 0,
        }

    ratios: list[float] = []
    zero_min_load_batches = 0
    all_zero_batches = 0
    for row in rank_load:
        positive = row[row > 0]
        if positive.numel() == 0:
            all_zero_batches += 1
            continue
        if positive.numel() < row.numel():
            zero_min_load_batches += 1
        ratios.append(float((positive.max() / positive.min()).item()))

    avg_ratio = float(sum(ratios) / len(ratios)) if ratios else float("nan")
    return {
        "num_batches": int(rank_load.size(0)),
        "avg_batch_load_ratio": avg_ratio,
        "zero_min_load_batches": int(zero_min_load_batches),
        "all_zero_batches": int(all_zero_batches),
    }


def compute_rank_load_from_edge_owner(
    *,
    src: Tensor,
    dst: Tensor,
    edge_owner: Tensor,
    time_ptr_2: Tensor,
    world_size: int,
    node_count_weight: float = 1.0,
) -> Tensor:
    src = src.long().cpu()
    dst = dst.long().cpu()
    edge_owner = edge_owner.long().cpu()
    time_ptr_2 = time_ptr_2.long().cpu()
    rank_load = torch.zeros((int(time_ptr_2.size(0)), int(world_size)), dtype=torch.float32)
    num_nodes = int(max(src.max().item(), dst.max().item()) + 1) if src.numel() > 0 else 0
    stride = num_nodes + 1
    for t in range(int(time_ptr_2.size(0))):
        begin = int(time_ptr_2[t, 0])
        end = int(time_ptr_2[t, 1])
        if end <= begin:
            continue
        owner = edge_owner[begin:end]
        s = src[begin:end]
        d = dst[begin:end]
        event_count = torch.bincount(owner, minlength=int(world_size)).float()
        pair_src = owner * stride + s
        pair_dst = owner * stride + d
        unique_pairs = torch.unique(torch.cat([pair_src, pair_dst], dim=0))
        unique_owner = (unique_pairs // stride).long()
        node_count = torch.bincount(unique_owner, minlength=int(world_size)).float()
        rank_load[t] = event_count + float(node_count_weight) * node_count
    return rank_load
