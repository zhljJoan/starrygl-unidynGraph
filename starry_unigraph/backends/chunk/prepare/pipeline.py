"""Chunk prepare pipeline.

The entrypoint builds a node master partition when one is not supplied, assigns
nodes to chunks, computes load, rebalances chunk ownership, and optionally builds
route artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .chunk_assignment import ChunkAssignment, build_chunk_assignment
from .load_stats import ChunkLoadStats, compute_chunk_load_stats, compute_chunk_load_stats_from_windows
from .rebalancer import ChunkReassignmentManifest, rebalance_chunks
from .route_builder import (
    build_memory_route_phase1,
    assign_memory_route_ptrs,
    build_spatial_routes,
)


@dataclass
class PrepareArtifacts:
    """Artifacts returned by `prepare()`."""

    assignment: ChunkAssignment
    node_owner: Tensor
    load_stats: Dict[int, ChunkLoadStats]
    rebalance_manifest: ChunkReassignmentManifest
    node_partition: Tensor
    node_to_partition: Tensor
    hot_node_mask: Tensor
    hot_node_ids: Tensor
    replica_mask: Tensor
    partition_strategy: str = "metis"
    mem_routes: Optional[list[list[Any]]] = None
    spatial_routes: Optional[list[list[Any]]] = None
    time_ptr: Optional[Tensor] = None


def _num_nodes(edge_src: Tensor, edge_dst: Tensor, num_nodes: Optional[int]) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if edge_src.numel() == 0 and edge_dst.numel() == 0:
        return 0
    return int(torch.cat([edge_src, edge_dst]).max().item()) + 1


def _degree(edge_src: Tensor, edge_dst: Tensor, num_nodes: int) -> Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_src.device)
    if edge_src.numel() == 0:
        return deg
    one = torch.ones(edge_src.numel(), dtype=torch.long, device=edge_src.device)
    deg.scatter_add_(0, edge_src.long(), one)
    deg.scatter_add_(0, edge_dst.long(), one)
    return deg


def _hot_mask_from_degree(deg: Tensor, hot_topk: int, hot_ratio: float) -> Tensor:
    n = int(deg.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=deg.device)
    k = int(hot_topk)
    if k <= 0 and hot_ratio > 0.0:
        k = int(round(n * float(hot_ratio)))
    k = max(0, min(k, n))
    mask = torch.zeros(n, dtype=torch.bool, device=deg.device)
    if k == 0:
        return mask
    hot = torch.topk(deg, k=k, largest=True, sorted=False).indices
    mask[hot] = deg[hot] > 0
    return mask


def _balanced_by_degree(deg: Tensor, nodes: Tensor, num_parts: int) -> Tensor:
    """Degree-aware fallback partitioner with vectorized round-robin buckets."""
    part = torch.zeros(int(deg.numel()), dtype=torch.long, device=deg.device)
    if nodes.numel() == 0:
        return part
    order = torch.argsort(deg[nodes], descending=True, stable=True)
    ranked_nodes = nodes[order]
    ranked_parts = torch.arange(ranked_nodes.numel(), device=deg.device) % num_parts
    part[ranked_nodes] = ranked_parts
    return part


def _try_metis_partition(edge_src: Tensor, edge_dst: Tensor, nodes: Tensor, num_nodes: int, num_parts: int) -> Optional[Tensor]:
    """Use pymetis when available; return None for deterministic fallback."""
    if nodes.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long, device=edge_src.device)
    try:
        import pymetis  # type: ignore
    except Exception:
        return None

    keep = torch.zeros(num_nodes, dtype=torch.bool, device=edge_src.device)
    keep[nodes] = True
    edge_keep = keep[edge_src] & keep[edge_dst]
    sub_src = edge_src[edge_keep]
    sub_dst = edge_dst[edge_keep]
    if sub_src.numel() == 0:
        return None

    local = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_src.device)
    local[nodes] = torch.arange(nodes.numel(), device=edge_src.device)
    ls = local[sub_src].cpu().tolist()
    ld = local[sub_dst].cpu().tolist()
    adj: list[set[int]] = [set() for _ in range(int(nodes.numel()))]
    for u, v in zip(ls, ld):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    try:
        _, sub_part = pymetis.part_graph(int(num_parts), adjacency=[list(x) for x in adj])
    except Exception:
        return None

    part = torch.zeros(num_nodes, dtype=torch.long, device=edge_src.device)
    part[nodes] = torch.tensor(sub_part, dtype=torch.long, device=edge_src.device)
    return part


def _assign_hot_masters(
    edge_src: Tensor,
    edge_dst: Tensor,
    hot_mask: Tensor,
    cold_part: Tensor,
    deg: Tensor,
    num_parts: int,
) -> Tensor:
    """Assign each replicated hot node a master with neighbor-affinity and balance."""
    hot_ids = hot_mask.nonzero(as_tuple=True)[0]
    if hot_ids.numel() == 0:
        return cold_part

    flat_score = torch.zeros(hot_mask.numel() * num_parts, dtype=torch.long, device=edge_src.device)
    hot_src = hot_mask[edge_src]
    if hot_src.any():
        idx = edge_src[hot_src] * num_parts + cold_part[edge_dst[hot_src]]
        flat_score.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
    hot_dst = hot_mask[edge_dst]
    if hot_dst.any():
        idx = edge_dst[hot_dst] * num_parts + cold_part[edge_src[hot_dst]]
        flat_score.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))

    score = flat_score.view(hot_mask.numel(), num_parts)[hot_ids].float()
    base_load = torch.bincount(cold_part[~hot_mask], weights=deg[~hot_mask].float(), minlength=num_parts)
    hot_weight = deg[hot_ids].float().clamp_min(1.0)
    target = (base_load.sum() + hot_weight.sum()) / max(num_parts, 1)
    penalty = (base_load / target.clamp_min(1.0)).view(1, -1)
    owner = torch.argmax(score - penalty, dim=1)

    part = cold_part.clone()
    part[hot_ids] = owner.long()
    return part


def build_node_partition(
    *,
    edge_src: Tensor,
    edge_dst: Tensor,
    num_partitions: int,
    num_nodes: Optional[int] = None,
    strategy: str = "metis",
    hot_topk: int = 0,
    hot_ratio: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build master node partitions plus hot-node replica metadata.

    Hot nodes are selected by degree and replicated on every worker. Their master
    is the partition with the strongest cold-neighbor affinity under a light load
    penalty. Cold nodes use either METIS (when available) or a deterministic
    degree-balanced fallback; `mem_share` uses hot-neighbor affinity first and the
    same balanced fallback for nodes without hot affinity.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    n = _num_nodes(edge_src, edge_dst, num_nodes)
    deg = _degree(edge_src, edge_dst, n)
    hot_mask = _hot_mask_from_degree(deg, hot_topk=hot_topk, hot_ratio=hot_ratio)
    cold_nodes = (~hot_mask).nonzero(as_tuple=True)[0]
    strategy = str(strategy or "metis").lower()

    if strategy == "metis":
        cold_part = _try_metis_partition(edge_src, edge_dst, cold_nodes, n, num_partitions)
        if cold_part is None:
            cold_part = _balanced_by_degree(deg, cold_nodes, num_partitions)
    elif strategy in {"mem_share", "mem-share", "memory_share"}:
        cold_part = _balanced_by_degree(deg, cold_nodes, num_partitions)
        if cold_nodes.numel() > 0 and hot_mask.any():
            flat_score = torch.zeros(n * num_partitions, dtype=torch.long, device=edge_src.device)
            hot_src = hot_mask[edge_src] & ~hot_mask[edge_dst]
            if hot_src.any():
                hot_master = _assign_hot_masters(edge_src, edge_dst, hot_mask, cold_part, deg, num_partitions)
                idx = edge_dst[hot_src] * num_partitions + hot_master[edge_src[hot_src]]
                flat_score.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
            hot_dst = hot_mask[edge_dst] & ~hot_mask[edge_src]
            if hot_dst.any():
                hot_master = _assign_hot_masters(edge_src, edge_dst, hot_mask, cold_part, deg, num_partitions)
                idx = edge_src[hot_dst] * num_partitions + hot_master[edge_dst[hot_dst]]
                flat_score.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
            score = flat_score.view(n, num_partitions)[cold_nodes]
            has_affinity = score.sum(dim=1) > 0
            if has_affinity.any():
                cold_part[cold_nodes[has_affinity]] = torch.argmax(score[has_affinity], dim=1).long()
    else:
        raise ValueError("partition strategy must be 'metis' or 'mem_share'")

    part = _assign_hot_masters(edge_src, edge_dst, hot_mask, cold_part, deg, num_partitions)
    replica_mask = hot_mask.clone()
    return part.long().cpu(), hot_mask.cpu(), replica_mask.cpu()


def prepare(
    *,
    edge_src: Tensor,
    edge_dst: Tensor,
    node_partition: Optional[Tensor] = None,
    node_to_partition: Optional[Tensor] = None,
    num_partitions: int,
    num_nodes: Optional[int] = None,
    partition_strategy: str = "metis",
    hot_topk: int = 0,
    hot_ratio: float = 0.0,
    num_chunks_per_partition: int = 32,
    edge_timestamps: Optional[Tensor] = None,
    time_ptr: Optional[Tensor] = None,
    max_imbalance_ratio: float = 1.2,
    max_migrations: Optional[int] = None,
    build_mem_routes: bool = False,
    build_spatial_routes_flag: bool = False,
    dst_ids_per_part: Optional[list[list[Any]]] = None,
    src_ids_per_part: Optional[list[list[Any]]] = None,
    num_candidates: int = 3,
    replica_mask: Optional[Tensor] = None,
) -> PrepareArtifacts:
    """Full chunk preprocessing pipeline."""
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")

    if node_partition is None:
        node_partition, hot_node_mask, generated_replica_mask = build_node_partition(
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_partitions=num_partitions,
            num_nodes=num_nodes,
            strategy=partition_strategy,
            hot_topk=hot_topk,
            hot_ratio=hot_ratio,
        )
    else:
        node_partition = node_partition.long().cpu()
        hot_node_mask = torch.zeros(int(node_partition.numel()), dtype=torch.bool)
        generated_replica_mask = hot_node_mask.clone()

    if node_to_partition is None:
        node_to_partition = node_partition
    else:
        node_to_partition = node_to_partition.long().cpu()
    if replica_mask is None:
        replica_mask = generated_replica_mask
    else:
        replica_mask = replica_mask.bool().cpu()

    edge_src_cpu = edge_src.long().cpu()
    edge_dst_cpu = edge_dst.long().cpu()
    edge_ts_cpu = None if edge_timestamps is None else edge_timestamps.cpu()
    time_ptr_cpu = None if time_ptr is None else time_ptr.long().cpu()

    assignment = build_chunk_assignment(
        node_partition=node_partition,
        num_chunks_per_partition=num_chunks_per_partition,
    )

    if time_ptr_cpu is not None and int(time_ptr_cpu.numel()) > 1:
        w_src: list[Tensor] = []
        w_dst: list[Tensor] = []
        for i in range(int(time_ptr_cpu.numel()) - 1):
            s, e = int(time_ptr_cpu[i]), int(time_ptr_cpu[i + 1])
            w_src.append(edge_src_cpu[s:e])
            w_dst.append(edge_dst_cpu[s:e])
        load_stats = compute_chunk_load_stats_from_windows(
            window_edge_srcs=w_src,
            window_edge_dsts=w_dst,
            node_to_chunk=assignment.node_to_chunk,
            chunk_to_owner_partition=assignment.chunk_to_owner_partition,
            node_to_partition=node_to_partition,
        )
    else:
        load_stats = compute_chunk_load_stats(
            edge_src=edge_src_cpu,
            edge_dst=edge_dst_cpu,
            node_to_chunk=assignment.node_to_chunk,
            chunk_to_owner_partition=assignment.chunk_to_owner_partition,
            node_to_partition=node_to_partition,
            edge_timestamps=edge_ts_cpu,
        )

    assignment, node_owner, manifest = rebalance_chunks(
        assignment=assignment,
        load_stats=load_stats,
        num_partitions=num_partitions,
        max_imbalance_ratio=max_imbalance_ratio,
        max_migrations=max_migrations,
    )

    # Replicated hot nodes keep their affinity-selected master after chunk rebalance.
    if hot_node_mask.any():
        node_owner = node_owner.clone()
        node_owner[hot_node_mask] = node_partition[hot_node_mask]

    mem_routes: Optional[list[list[Any]]] = None
    if build_mem_routes:
        if edge_ts_cpu is None or time_ptr_cpu is None:
            raise ValueError("build_mem_routes requires edge_timestamps and time_ptr")
        routes_p1 = build_memory_route_phase1(
            edge_src=edge_src_cpu,
            edge_dst=edge_dst_cpu,
            edge_ts=edge_ts_cpu,
            time_ptr=time_ptr_cpu,
            num_candidates=num_candidates,
            replica_mask=replica_mask,
        )
        mem_routes = assign_memory_route_ptrs(
            routes=routes_p1,
            node_owner=node_owner,
            num_parts=num_partitions,
        )

    spatial: Optional[list[list[Any]]] = None
    if build_spatial_routes_flag:
        if dst_ids_per_part is None or src_ids_per_part is None:
            raise ValueError("build_spatial_routes_flag requires dst_ids_per_part and src_ids_per_part")
        spatial = build_spatial_routes(
            dst_ids_per_part=dst_ids_per_part,
            src_ids_per_part=src_ids_per_part,
            node_owner=node_owner,
            num_parts=num_partitions,
        )

    return PrepareArtifacts(
        assignment=assignment,
        node_owner=node_owner,
        load_stats=load_stats,
        rebalance_manifest=manifest,
        node_partition=node_partition,
        node_to_partition=node_to_partition,
        hot_node_mask=hot_node_mask,
        hot_node_ids=hot_node_mask.nonzero(as_tuple=True)[0],
        replica_mask=replica_mask,
        partition_strategy=partition_strategy,
        mem_routes=mem_routes,
        spatial_routes=spatial,
        time_ptr=time_ptr_cpu,
    )
