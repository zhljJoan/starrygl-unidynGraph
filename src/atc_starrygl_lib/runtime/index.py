from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from atc_starrygl_lib.comm.dist_index import LOCAL_BITS, dist_index_loc, dist_index_part
from atc_starrygl_lib.comm.layouts import FeatureReadLayout


@dataclass(slots=True)
class CTDGNodeTable:
    """Rank-local CTDG row order.

    ``DistIndexTables`` stays global because CTDG negative sampling and dynamic
    temporal sampling may touch any node.  This table only describes rows that
    are readable on the current rank.

    Rows are partitioned as:
      [0, shared_size)                             shared/hot cache rows
      [shared_size, shared_size + remote_1hop)     optional remote 1-hop cache rows
      [shared_size + remote_1hop, N)               owned authoritative rows

    The remote 1-hop segment is disabled by default.  Shared/hot rows are a
    fixed prefix to match the MemShare hot-cache convention.
    """

    rank: int
    world_size: int
    node_ids: Tensor
    owner: Tensor
    shared_size: int = 0
    remote_1hop_size: int = 0

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        self.world_size = int(self.world_size)
        self.shared_size = int(self.shared_size)
        self.remote_1hop_size = int(self.remote_1hop_size)
        if self.shared_size < 0:
            raise ValueError("shared_size must be non-negative")
        if self.remote_1hop_size < 0:
            raise ValueError("remote_1hop_size must be non-negative")
        if self.shared_size + self.remote_1hop_size > int(self.node_ids.numel()):
            raise ValueError("shared/remote 1-hop segments exceed local node table length")

    @property
    def num_rows(self) -> int:
        return int(self.node_ids.numel())

    @property
    def shared_begin(self) -> int:
        return 0

    @property
    def shared_end(self) -> int:
        return self.shared_size

    @property
    def remote_1hop_begin(self) -> int:
        return self.shared_size

    @property
    def remote_1hop_end(self) -> int:
        return self.shared_size + self.remote_1hop_size

    @property
    def owned_begin(self) -> int:
        return self.shared_size + self.remote_1hop_size

    @property
    def has_shared(self) -> bool:
        return self.shared_size > 0

    @property
    def has_remote_1hop(self) -> bool:
        return self.remote_1hop_size > 0

    @classmethod
    def owned_only(
        cls,
        *,
        rank: int,
        world_size: int,
        owned_node_ids: Tensor,
    ) -> "CTDGNodeTable":
        owner = torch.full_like(owned_node_ids.long(), int(rank))
        return cls(
            rank=int(rank),
            world_size=int(world_size),
            node_ids=owned_node_ids.long().contiguous(),
            owner=owner.contiguous(),
            shared_size=0,
            remote_1hop_size=0,
        )


@dataclass(slots=True)
class DTDGNodeTable:
    """Rank-local DTDG block order: [dst/local prefix][remote 1-hop tail]."""

    rank: int
    world_size: int
    dst_node_ids: Tensor
    remote_src_node_ids: Tensor | None = None

    @property
    def num_dst_rows(self) -> int:
        return int(self.dst_node_ids.numel())

    @property
    def num_remote_src_rows(self) -> int:
        return 0 if self.remote_src_node_ids is None else int(self.remote_src_node_ids.numel())

    @property
    def num_src_rows(self) -> int:
        return self.num_dst_rows + self.num_remote_src_rows


@dataclass(slots=True)
class DistIndexTables:
    """Dense node -> distributed position tables.

    master_dist_index points at the authoritative owner/master row.
    read_dist_index points at the current rank's preferred read position.
    """

    master_dist_index: Tensor
    read_dist_index: Tensor

    def master_for(self, node_ids: Tensor) -> Tensor:
        return self.master_dist_index.index_select(0, node_ids.long().to(self.master_dist_index.device)).to(node_ids.device)

    def read_for(self, node_ids: Tensor) -> Tensor:
        return self.read_dist_index.index_select(0, node_ids.long().to(self.read_dist_index.device)).to(node_ids.device)


def _ptr_from_rank(rank: Tensor, world_size: int) -> Tensor:
    counts = torch.bincount(rank.long(), minlength=int(world_size))
    ptr = torch.zeros(int(world_size) + 1, dtype=torch.long, device=rank.device)
    ptr[1:] = counts.cumsum(0)
    return ptr


def _unique_sorted_by_rank(index: Tensor, world_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return unique dist indices grouped by dist_index_part.

    Returns (grouped_index, ptr, inverse_to_original_unique_order).
    """

    unique, inverse = torch.unique(index.long(), sorted=False, return_inverse=True)
    if unique.numel() == 0:
        ptr = torch.zeros(int(world_size) + 1, dtype=torch.long, device=index.device)
        return unique, ptr, inverse
    rank = dist_index_part(unique)
    order = _rank_local_order(unique)
    grouped = unique[order].contiguous()
    ptr = _ptr_from_rank(rank[order], int(world_size))
    unique_to_grouped = torch.empty_like(order)
    unique_to_grouped[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
    return grouped, ptr, unique_to_grouped[inverse]


def build_feature_read_layout(
    compute_nodes: Tensor,
    read_dist_index: Tensor,
    *,
    world_size: int,
    local_rank: Optional[int] = None,
) -> FeatureReadLayout:
    """Build a batch-local feature read layout from sampled compute nodes."""

    read_idx = read_dist_index.index_select(0, compute_nodes.long().to(read_dist_index.device)).to(compute_nodes.device)
    grouped, ptr, compute_to_feature = _unique_sorted_by_rank(read_idx, int(world_size))
    return FeatureReadLayout(
        read_index=grouped,
        read_ptr=ptr,
        compute_to_feature=compute_to_feature.long().contiguous(),
    )


def build_feature_read_layout_from_index(
    read_idx: Tensor,
    *,
    world_size: int,
    local_rank: Optional[int] = None,
    deduplicate: bool = False,
    already_rank_grouped: bool = True,
) -> FeatureReadLayout:
    """Build a feature read layout when the sampler already produced read indices.

    ``deduplicate=False`` avoids the ``torch.unique`` hot path and keeps a
    direct identity compute mapping.  Use it when native sampling already emits
    the exact compute/read layout expected by the model.
    """

    read_idx = read_idx.long().contiguous()
    if not deduplicate and already_rank_grouped:
        grouped = read_idx
        ptr = _ptr_from_rank(dist_index_part(grouped), int(world_size))
        compute_to_feature = torch.arange(grouped.numel(), dtype=torch.long, device=grouped.device)
    elif deduplicate:
        grouped, ptr, compute_to_feature = _unique_sorted_by_rank(read_idx, int(world_size))
    else:
        rank = dist_index_part(read_idx)
        order = _rank_local_order(read_idx)
        grouped = read_idx[order].contiguous()
        ptr = _ptr_from_rank(rank[order], int(world_size))
        compute_to_grouped = torch.empty_like(order)
        compute_to_grouped[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        compute_to_feature = compute_to_grouped

    return FeatureReadLayout(
        read_index=grouped,
        read_ptr=ptr,
        compute_to_feature=compute_to_feature.long().contiguous(),
    )


def build_feature_read_layout_from_comm(
    comm_node_ids: Tensor,
    compute_to_comm: Tensor,
    read_dist_index: Tensor,
    *,
    world_size: int,
    time_slices: Tensor | None = None,
    deduplicate: bool = False,
    already_rank_grouped: bool = False,
) -> FeatureReadLayout:
    """Build a feature read layout from a sampler-produced comm layout.

    ``comm_node_ids`` are the fetch keys, usually already deduplicated by the
    sampler.  ``compute_to_comm`` maps model compute rows to those fetch keys.
    The returned ``compute_to_feature`` maps model compute rows directly to the
    fetched feature tensor.
    """

    read_idx = read_dist_index.index_select(
        0,
        comm_node_ids.long().to(read_dist_index.device),
    ).to(comm_node_ids.device)
    time = None if time_slices is None else time_slices.to(comm_node_ids.device).contiguous()
    if time is not None and deduplicate:
        raise ValueError("temporal feature fetch does not support deduplicate=True")

    if not deduplicate and already_rank_grouped:
        grouped = read_idx.long().contiguous()
        ptr = _ptr_from_rank(dist_index_part(grouped), int(world_size))
        comm_to_feature = torch.arange(grouped.numel(), dtype=torch.long, device=grouped.device)
        grouped_time = time
    elif deduplicate:
        grouped, ptr, comm_to_feature = _unique_sorted_by_rank(read_idx, int(world_size))
        grouped_time = None
    else:
        rank = dist_index_part(read_idx)
        order = _rank_local_order(read_idx)
        grouped = read_idx[order].contiguous()
        ptr = _ptr_from_rank(rank[order], int(world_size))
        comm_to_feature = torch.empty_like(order)
        comm_to_feature[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        grouped_time = None if time is None else time[order].contiguous()

    compute_to_feature = comm_to_feature.index_select(
        0,
        compute_to_comm.long().to(comm_to_feature.device),
    )
    return FeatureReadLayout(
        read_index=grouped,
        read_ptr=ptr,
        compute_to_feature=compute_to_feature.to(comm_node_ids.device).long().contiguous(),
        time_slices=grouped_time,
    )


def _rank_local_order(index: Tensor) -> Tensor:
    key = (dist_index_part(index) << LOCAL_BITS) | dist_index_loc(index)
    return torch.argsort(key, stable=True)
