"""Route builder: two-phase construction of MemoryRouteData and SpatialRouteData.

Phase 1 (pre-partition) — graph-structure independent:
    build_memory_route_phase1(edge_src, edge_dst, edge_ts, time_ptr, K)
    → List[MemoryRouteData] with unique_nodes + cand_pos filled,
      send/recv ptrs empty.  Cheap to keep across repartitions.

Phase 2 (post-partition) — lightweight O(D·P):
    assign_memory_route_ptrs(routes_p1, node_owner, num_parts)
    → fills send_ptr / recv_ptr / recv_node_ids in-place.
    Call this again whenever chunk ownership changes.

Spatial route:
    build_spatial_routes(partition_node_sets, edge_sets, node_owner, num_parts)
    → one SpatialRouteData per partition per time slice.

All construction is vectorised — no Python loops over individual edges.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from starry_unigraph.backends.chunk.data.route import MemoryRouteData, SpatialRouteData, CPUMemoryLayout
from starry_unigraph.backends.chunk.prepare.chunk_assignment import ChunkAssignment


# ---------------------------------------------------------------------------
# Shared primitive: dedup events, keep K latest positions per node
# ---------------------------------------------------------------------------

def _dedup_latest_k(
    edge_src:  Tensor,
    edge_dst:  Tensor,
    edge_ts:   Tensor,
    num_candidates: int,
) -> Tuple[Tensor, Tensor]:
    """Return (unique_nodes [D], cand_pos [D, K]) for src ∪ dst of one slice.

    Fully vectorised (two stable argsort passes + repeat_interleave + scatter).
    unique_nodes is in node-ID order at this point (not yet sorted by owner).
    """
    E = int(edge_src.numel())
    if E == 0:
        empty_cand = torch.zeros(0, num_candidates, dtype=torch.long, device=edge_src.device)
        return torch.zeros(0, dtype=torch.long, device=edge_src.device), empty_cand

    all_nodes = torch.cat([edge_src, edge_dst])          # [2E]
    all_ts    = torch.cat([edge_ts,  edge_ts])           # [2E]
    all_pos   = torch.arange(2 * E, dtype=torch.long, device=edge_src.device)

    # Two-pass stable sort: primary ts desc, secondary node asc
    ts_ord   = torch.argsort(all_ts, descending=True, stable=True)
    node_ord = torch.argsort(all_nodes[ts_ord], stable=True)
    final    = ts_ord[node_ord]

    sorted_nodes = all_nodes[final]
    sorted_pos   = all_pos[final]

    unique_nodes, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
    D = int(unique_nodes.numel())

    # node_ptr[d] = start of group d in sorted arrays
    node_ptr = torch.zeros(D + 1, dtype=torch.long, device=edge_src.device)
    node_ptr[1:] = counts.cumsum(0)

    # Rank of each event within its node group (0 = latest)
    node_idx = torch.repeat_interleave(
        torch.arange(D, dtype=torch.long, device=edge_src.device), counts
    )
    rank = torch.arange(2 * E, dtype=torch.long, device=edge_src.device) - node_ptr[node_idx]

    keep = rank < num_candidates
    cand_pos = torch.full((D, num_candidates), -1, dtype=torch.long, device=edge_src.device)
    cand_pos[node_idx[keep], rank[keep]] = sorted_pos[keep]

    return unique_nodes, cand_pos


# ---------------------------------------------------------------------------
# Phase 1: graph-partition independent
# ---------------------------------------------------------------------------

def build_memory_route_phase1(
    edge_src:       Tensor,
    edge_dst:       Tensor,
    edge_ts:        Tensor,
    time_ptr:       Tensor,
    num_candidates: int = 3,
    replica_mask:   Optional[Tensor] = None,
) -> List[MemoryRouteData]:
    """Build MemoryRouteData Phase 1 for all time slices (pre-partition).

    unique_nodes and cand_pos are computed here.
    send_ptr / recv_ptr / recv_node_ids are left zeroed (filled in Phase 2).
    replica_idx is computed if replica_mask is provided.

    Args:
        edge_src, edge_dst, edge_ts: [E_total] full event stream.
        time_ptr: [T+1] CSR, events for slice t = [time_ptr[t]:time_ptr[t+1]].
        num_candidates: K candidate positions per unique node.
        replica_mask: [num_nodes] bool, True = this node has cross-partition replicas.

    Returns:
        List[MemoryRouteData] of length T, Phase 1 fields filled.
    """
    T = int(time_ptr.numel()) - 1
    routes: List[MemoryRouteData] = []

    for t in range(T):
        s, e = int(time_ptr[t]), int(time_ptr[t + 1])
        unique_nodes, cand_pos = _dedup_latest_k(
            edge_src[s:e], edge_dst[s:e], edge_ts[s:e], num_candidates
        )
        D = int(unique_nodes.numel())
        dev = edge_src.device

        # Replica index (optional)
        replica_idx = None
        if replica_mask is not None and D > 0:
            rep_mask = replica_mask[unique_nodes]
            if rep_mask.any():
                replica_idx = rep_mask.nonzero(as_tuple=True)[0]

        routes.append(MemoryRouteData(
            unique_nodes      = unique_nodes,
            cand_pos          = cand_pos,
            send_ptr          = torch.zeros(1, dtype=torch.long, device=dev),  # placeholder
            recv_ptr          = torch.zeros(1, dtype=torch.long, device=dev),
            recv_node_ids     = torch.zeros(0, dtype=torch.long, device=dev),
            replica_idx       = replica_idx,
            replica_send_ptr  = None,
            replica_recv_ptr  = None,
        ))

    return routes


# ---------------------------------------------------------------------------
# Phase 2: assign routing pointers (can be repeated on repartition)
# ---------------------------------------------------------------------------

def assign_memory_route_ptrs(
    routes:     List[MemoryRouteData],
    node_owner: Tensor,
    num_parts:  int,
    master_dist_index: Optional[Tensor] = None,
) -> List[List[MemoryRouteData]]:
    """Assign send_ptr / recv_ptr / recv_node_ids to Phase 1 routes.

    For each slice, builds one MemoryRouteData per partition (since each
    partition only sends nodes it owns to others, and receives nodes owned
    by it from others).

    Args:
        routes:     Phase 1 output, length T.
        node_owner: [num_nodes] owner partition per global node.
        num_parts:  total partition count.

    Returns:
        per_part_routes[p][t] = MemoryRouteData for partition p, slice t.
    """
    T = len(routes)
    # per_part_routes[p] is the list of T routes for partition p
    per_part: List[List[MemoryRouteData]] = [[] for _ in range(num_parts)]

    for t, r in enumerate(routes):
        if r.unique_nodes.numel() == 0:
            empty = _empty_memory_route(num_parts, r.num_candidates, r.cand_pos.device)
            for p in range(num_parts):
                per_part[p].append(empty)
            continue

        # Sort unique_nodes by owner
        owners   = node_owner[r.unique_nodes]            # [D]
        sort_o   = torch.argsort(owners, stable=True)    # [D]
        s_nodes  = r.unique_nodes[sort_o]
        s_owners = owners[sort_o]
        s_cand   = r.cand_pos[sort_o]
        s_index = None
        if master_dist_index is not None:
            s_index = master_dist_index[r.unique_nodes].to(device=s_nodes.device)[sort_o]

        send_counts = torch.bincount(s_owners, minlength=num_parts)  # [P]
        send_ptr    = torch.zeros(num_parts + 1, dtype=torch.long)
        send_ptr[1:] = send_counts.cumsum(0)

        # Replica mapping (same sort permutation)
        rep_idx_sorted = rep_send_ptr = None
        if r.replica_idx is not None:
            inv = torch.empty_like(sort_o)
            inv[sort_o] = torch.arange(len(sort_o), device=sort_o.device)
            rep_idx_sorted = inv[r.replica_idx]
            rep_owners     = s_owners[rep_idx_sorted]
            rep_cnt        = torch.bincount(rep_owners, minlength=num_parts)
            rep_send_ptr   = torch.zeros(num_parts + 1, dtype=torch.long)
            rep_send_ptr[1:] = rep_cnt.cumsum(0)

        # Build one route per partition (partition p = owner of nodes it receives)
        for p in range(num_parts):
            ps, pe = int(send_ptr[p]), int(send_ptr[p + 1])
            p_route = MemoryRouteData(
                unique_nodes     = s_nodes,
                cand_pos         = s_cand,
                send_ptr         = send_ptr,
                recv_ptr         = torch.zeros(num_parts + 1, dtype=torch.long),  # Phase 3
                recv_node_ids    = torch.zeros(0, dtype=torch.long),
                unique_index     = s_index,
                replica_idx      = rep_idx_sorted,
                replica_send_ptr = rep_send_ptr,
                replica_recv_ptr = None,
            )
            per_part[p].append(p_route)

    # Fill recv_ptr / recv_node_ids by transposing send information
    _fill_memory_recv_ptrs(per_part, num_parts, master_dist_index=master_dist_index)

    return per_part


def _fill_memory_recv_ptrs(
    per_part: List[List[MemoryRouteData]],
    num_parts: int,
    master_dist_index: Optional[Tensor] = None,
) -> None:
    """Transpose send_ptr to fill recv_ptr / recv_node_ids.  O(P² · T)."""
    if not per_part or not per_part[0]:
        return
    T = len(per_part[0])
    for t in range(T):
        for p in range(num_parts):
            recv_counts = torch.zeros(num_parts, dtype=torch.long)
            recv_parts: List[Tensor] = []
            for q in range(num_parts):
                rq = per_part[q][t]
                qs, qe = int(rq.send_ptr[p]), int(rq.send_ptr[p + 1])
                recv_counts[q] = qe - qs
                recv_parts.append(rq.unique_nodes[qs:qe])

            recv_ptr = torch.zeros(num_parts + 1, dtype=torch.long)
            recv_ptr[1:] = recv_counts.cumsum(0)
            recv_nodes = torch.cat(recv_parts) if recv_parts else \
                         torch.zeros(0, dtype=torch.long)

            per_part[p][t].recv_ptr      = recv_ptr
            per_part[p][t].recv_node_ids = recv_nodes
            if master_dist_index is not None and recv_nodes.numel() > 0:
                per_part[p][t].recv_index = master_dist_index[recv_nodes]

            # Replica recv_ptr
            rp = per_part[p][t]
            if rp.replica_send_ptr is not None:
                rep_recv = []
                for q in range(num_parts):
                    rq = per_part[q][t]
                    if rq.replica_send_ptr is None:
                        rep_recv.append(0)
                    else:
                        rep_recv.append(int(rq.replica_send_ptr[p + 1]) - int(rq.replica_send_ptr[p]))
                rep_recv_t = torch.tensor(rep_recv, dtype=torch.long)
                rp.replica_recv_ptr = torch.zeros(num_parts + 1, dtype=torch.long)
                rp.replica_recv_ptr[1:] = rep_recv_t.cumsum(0)


def _empty_memory_route(num_parts: int, K: int, device) -> MemoryRouteData:
    ptr = torch.zeros(num_parts + 1, dtype=torch.long, device=device)
    return MemoryRouteData(
        unique_nodes      = torch.zeros(0, dtype=torch.long, device=device),
        cand_pos          = torch.zeros(0, K, dtype=torch.long, device=device),
        send_ptr          = ptr,
        recv_ptr          = ptr.clone(),
        recv_node_ids     = torch.zeros(0, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Spatial route construction
# ---------------------------------------------------------------------------

def build_spatial_routes(
    dst_ids_per_part:  List[List[Tensor]],
    src_ids_per_part:  List[List[Tensor]],
    node_owner:        Tensor,
    num_parts:         int,
) -> List[List[SpatialRouteData]]:
    """Build SpatialRouteData for all partitions and time slices.

    Args:
        dst_ids_per_part: [P][T] list — local dst node IDs per partition per slice.
        src_ids_per_part: [P][T] list — combined src+dst space node IDs per slice.
        node_owner: [num_nodes] owner partition.
        num_parts: total partitions.

    Returns:
        routes[p][t] = SpatialRouteData for partition p, slice t.
    """
    P = num_parts
    T = len(dst_ids_per_part[0])
    per_part: List[List[SpatialRouteData]] = [[] for _ in range(P)]

    for p in range(P):
        for t in range(T):
            dst_ids = dst_ids_per_part[p][t]   # [num_local_dst]
            src_ids = src_ids_per_part[p][t]   # [num_src_total = dst + remote]

            if dst_ids.numel() == 0:
                empty_ptr = torch.zeros(P + 1, dtype=torch.long)
                per_part[p].append(SpatialRouteData(
                    send_index    = torch.zeros(0, dtype=torch.long),
                    send_ptr      = empty_ptr,
                    recv_ptr      = empty_ptr.clone(),
                    recv_node_ids = torch.zeros(0, dtype=torch.long),
                ))
                continue

            # Which nodes does partition p send?
            # For feature exchange: send nodes owned by p that other partitions need.
            # Here we send local dst_ids, sorted by the partition that needs them.
            # (In practice, this is the owner of the edges' src ends.)
            owners    = node_owner[dst_ids]                      # [num_local_dst]
            sort_o    = torch.argsort(owners, stable=True)
            s_owners  = owners[sort_o]

            send_counts = torch.bincount(s_owners, minlength=P)
            send_ptr    = torch.zeros(P + 1, dtype=torch.long)
            send_ptr[1:] = send_counts.cumsum(0)

            per_part[p].append(SpatialRouteData(
                send_index    = sort_o,              # local indices into dst_ids
                send_ptr      = send_ptr,
                recv_ptr      = torch.zeros(P + 1, dtype=torch.long),  # filled below
                recv_node_ids = torch.zeros(0, dtype=torch.long),
            ))

    # Fill recv_ptr / recv_node_ids by transposing
    for t in range(T):
        for p in range(P):
            recv_counts = torch.zeros(P, dtype=torch.long)
            recv_parts: List[Tensor] = []
            for q in range(P):
                rq = per_part[q][t]
                qs, qe = int(rq.send_ptr[p]), int(rq.send_ptr[p + 1])
                recv_counts[q] = qe - qs
                # The global IDs of nodes q sends to p
                q_dst = dst_ids_per_part[q][t]
                recv_parts.append(q_dst[rq.send_index[qs:qe]])

            recv_ptr = torch.zeros(P + 1, dtype=torch.long)
            recv_ptr[1:] = recv_counts.cumsum(0)
            recv_nodes = torch.cat(recv_parts) if recv_parts else torch.zeros(0, dtype=torch.long)

            per_part[p][t].recv_ptr      = recv_ptr
            per_part[p][t].recv_node_ids = recv_nodes

    return per_part


# ---------------------------------------------------------------------------
# CPU memory layout
# ---------------------------------------------------------------------------

def build_cpu_memory_layout(
    hot_nodes:  Tensor,
    cold_nodes: Tensor,
    num_nodes:  int,
) -> CPUMemoryLayout:
    """Wrapper around CPUMemoryLayout.build (documented there)."""
    return CPUMemoryLayout.build(hot_nodes, cold_nodes, num_nodes)
