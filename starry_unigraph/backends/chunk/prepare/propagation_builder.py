"""Build DTDG model-layer propagation routes from snapshot boundaries.

Provides two construction modes:
1. build_propagation_routes() - Legacy mode, uses global node ownership
2. build_propagation_routes_from_snapshots() - DTDG snapshot mode, uses compact row indexing
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from starry_unigraph.models.layers.route import ChunkPropagationRoute


def build_propagation_routes_from_snapshots(
    *,
    part_data,  # PartitionData
    node_owner: Tensor,
    num_parts: int,
    num_layers: int = 1,
) -> List[List[List[ChunkPropagationRoute]]]:
    """Build propagation routes using DTDG snapshot compact row indexing.

    This version aligns with PartitionData snapshot structure:
    - Uses dst_ids and src_ids from each snapshot
    - Generates recv_src_rows for compact row indexing
    - Supports DTDG block's [dst, remote_src] row layout

    Args:
        part_data: PartitionData with snapshot structure
        node_owner: [num_nodes] Node ownership mapping
        num_parts: Number of partitions
        num_layers: Number of GNN layers

    Returns:
        routes[rank][snapshot][layer] -> ChunkPropagationRoute
    """
    num_parts = int(num_parts)
    num_layers = max(1, int(num_layers))
    node_owner = node_owner.long().cpu()

    # Get owned nodes for each rank
    owned_nodes = [(node_owner == p).nonzero(as_tuple=True)[0].long().contiguous() for p in range(num_parts)]
    routes: List[List[List[ChunkPropagationRoute]]] = [[] for _ in range(num_parts)]

    # Check if part_data has snapshot structure
    if not hasattr(part_data, 'dst_ids') or not hasattr(part_data, 'src_ids'):
        # Fallback to legacy mode
        import warnings
        warnings.warn("PartitionData missing dst_ids/src_ids, falling back to legacy propagation route")
        return build_propagation_routes(
            edge_src=part_data.edge_src if hasattr(part_data, 'edge_src') else torch.empty(0, dtype=torch.long),
            edge_dst=part_data.edge_dst if hasattr(part_data, 'edge_dst') else torch.empty(0, dtype=torch.long),
            time_ptr=part_data.time_ptr if hasattr(part_data, 'time_ptr') else torch.tensor([0], dtype=torch.long),
            node_owner=node_owner,
            num_parts=num_parts,
            num_layers=num_layers,
        )

    num_snapshots = len(part_data.dst_ids)

    for sid in range(num_snapshots):
        # Get snapshot structure
        dst_ids = part_data.dst_ids[sid].item() if hasattr(part_data.dst_ids[sid], 'item') else part_data.dst_ids[sid]
        src_ids = part_data.src_ids[sid].item() if hasattr(part_data.src_ids[sid], 'item') else part_data.src_ids[sid]

        dst_ids = dst_ids.long().cpu()
        src_ids = src_ids.long().cpu()

        num_dst = int(dst_ids.numel())
        num_remote_src = int(src_ids.numel())

        # Build needs matrix for remote src
        needs = _remote_need_matrix_from_snapshot(dst_ids, src_ids, node_owner, num_parts)

        # Build routes for each rank
        snapshot_routes = []
        for rank in range(num_parts):
            route = _route_for_rank_with_snapshot(
                rank=rank,
                needs=needs,
                owned_nodes=owned_nodes[rank],
                dst_ids=dst_ids,
                src_ids=src_ids,
                node_owner=node_owner,
                num_parts=num_parts,
            )
            snapshot_routes.append(route)

        for rank in range(num_parts):
            routes[rank].append([snapshot_routes[rank] for _ in range(num_layers)])

    return routes


def _remote_need_matrix_from_snapshot(
    dst_ids: Tensor,
    src_ids: Tensor,
    node_owner: Tensor,
    num_parts: int,
) -> list[list[Tensor]]:
    """Build remote need matrix from snapshot dst_ids and src_ids.

    Returns needs[recv_rank][send_rank] = remote src nodes needed by recv_rank from send_rank
    """
    needs: list[list[Tensor]] = [
        [torch.empty(0, dtype=torch.long) for _ in range(num_parts)]
        for _ in range(num_parts)
    ]

    if src_ids.numel() == 0:
        return needs

    src_owner = node_owner[src_ids].long()

    # For each dst rank, find which remote src it needs
    for recv_rank in range(num_parts):
        recv_dst_mask = node_owner[dst_ids] == recv_rank
        if not bool(recv_dst_mask.any()):
            continue

        # All remote src are potentially needed by this rank
        # (In practice, we'd need edge connectivity to be precise)
        for send_rank in range(num_parts):
            send_src_mask = src_owner == send_rank
            if bool(send_src_mask.any()):
                needs[recv_rank][send_rank] = src_ids[send_src_mask].unique(sorted=True).long().contiguous()

    return needs


def _route_for_rank_with_snapshot(
    rank: int,
    needs: list[list[Tensor]],
    owned_nodes: Tensor,
    dst_ids: Tensor,
    src_ids: Tensor,
    node_owner: Tensor,
    num_parts: int,
) -> ChunkPropagationRoute:
    """Build route with recv_src_rows for snapshot compact indexing."""
    send_parts: list[Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    recv_src_parts: list[Tensor] = []

    num_dst = int(dst_ids.numel())

    for peer in range(num_parts):
        send_nodes = needs[peer][rank]
        send_sizes.append(int(send_nodes.numel()))
        if send_nodes.numel() > 0:
            rows = torch.searchsorted(owned_nodes, send_nodes)
            valid = rows < owned_nodes.numel()
            matched = torch.zeros_like(valid, dtype=torch.bool)
            if bool(valid.any()):
                matched[valid] = owned_nodes[rows[valid]] == send_nodes[valid]
            if not bool(matched.all()):
                raise ValueError("propagation route send node is missing from sender owned rows")
            send_parts.append(rows.long())

        # Recv: find positions of received nodes in [dst, remote_src] compact layout
        recv_nodes = needs[rank][peer]
        recv_sizes.append(int(recv_nodes.numel()))
        if recv_nodes.numel() > 0:
            # Map recv_nodes to compact row indices
            # Compact layout: [dst_ids, src_ids]
            recv_rows = torch.searchsorted(src_ids, recv_nodes)
            valid = recv_rows < src_ids.numel()
            matched = torch.zeros_like(valid, dtype=torch.bool)
            if bool(valid.any()):
                matched[valid] = src_ids[recv_rows[valid]] == recv_nodes[valid]
            # Offset by num_dst (remote src starts after dst)
            recv_rows = recv_rows + num_dst
            recv_src_parts.append(recv_rows.long())

    send_index = torch.cat(send_parts, dim=0).long().contiguous() if send_parts else None
    recv_src_rows = torch.cat(recv_src_parts, dim=0).long().contiguous() if recv_src_parts else None

    return ChunkPropagationRoute(
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_index=send_index,
        recv_src_rows=recv_src_rows,  # ✅ New field for compact row indexing
        append_recv=True,
    )


def build_propagation_routes(
    *,
    edge_src: Tensor,
    edge_dst: Tensor,
    time_ptr: Tensor,
    node_owner: Tensor,
    num_parts: int,
    num_layers: int = 1,
) -> List[List[List[ChunkPropagationRoute]]]:
    """Build propagation_routes[rank][snapshot][layer].

    Row convention: route ``send_index`` indexes the sender rank's owned-node
    activation rows, where owned rows are sorted global node ids satisfying
    ``node_owner == rank``.  A receiver appends incoming remote activations to
    its local activation tensor.
    """
    num_parts = int(num_parts)
    num_layers = max(1, int(num_layers))
    num_slices = max(0, int(time_ptr.numel()) - 1)
    edge_src = edge_src.long().cpu()
    edge_dst = edge_dst.long().cpu()
    time_ptr = time_ptr.long().cpu()
    node_owner = node_owner.long().cpu()

    owned_nodes = [(node_owner == p).nonzero(as_tuple=True)[0].long().contiguous() for p in range(num_parts)]
    routes: List[List[List[ChunkPropagationRoute]]] = [[] for _ in range(num_parts)]

    for sid in range(num_slices):
        start, end = int(time_ptr[sid]), int(time_ptr[sid + 1])
        src = edge_src[start:end]
        dst = edge_dst[start:end]
        needs = _remote_need_matrix(src, dst, node_owner, num_parts)
        base_routes = [
            _route_for_rank(rank, needs, owned_nodes[rank], num_parts)
            for rank in range(num_parts)
        ]
        for rank in range(num_parts):
            routes[rank].append([base_routes[rank] for _ in range(num_layers)])
    return routes


def _remote_need_matrix(src: Tensor, dst: Tensor, node_owner: Tensor, num_parts: int) -> list[list[Tensor]]:
    needs: list[list[Tensor]] = [
        [torch.empty(0, dtype=torch.long) for _ in range(num_parts)]
        for _ in range(num_parts)
    ]
    if src.numel() == 0:
        return needs
    src_owner = node_owner[src].long()
    dst_owner = node_owner[dst].long()
    remote = src_owner != dst_owner
    for recv_rank in range(num_parts):
        recv_mask = remote & (dst_owner == recv_rank)
        if not bool(recv_mask.any()):
            continue
        for send_rank in range(num_parts):
            mask = recv_mask & (src_owner == send_rank)
            if bool(mask.any()):
                needs[recv_rank][send_rank] = src[mask].unique(sorted=True).long().contiguous()
    return needs


def _route_for_rank(
    rank: int,
    needs: list[list[Tensor]],
    owned_nodes: Tensor,
    num_parts: int,
) -> ChunkPropagationRoute:
    send_parts: list[Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    for peer in range(num_parts):
        send_nodes = needs[peer][rank]
        send_sizes.append(int(send_nodes.numel()))
        if send_nodes.numel() > 0:
            rows = torch.searchsorted(owned_nodes, send_nodes)
            valid = rows < owned_nodes.numel()
            matched = torch.zeros_like(valid, dtype=torch.bool)
            if bool(valid.any()):
                matched[valid] = owned_nodes[rows[valid]] == send_nodes[valid]
            if not bool(matched.all()):
                raise ValueError("propagation route send node is missing from sender owned rows")
            send_parts.append(rows.long())
        recv_sizes.append(int(needs[rank][peer].numel()))
    send_index = torch.cat(send_parts, dim=0).long().contiguous() if send_parts else None
    return ChunkPropagationRoute(
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_index=send_index,
        append_recv=True,
    )
