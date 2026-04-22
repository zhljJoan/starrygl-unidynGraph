"""Snapshot graph loader with async CPU-to-GPU prefetch.

:class:`STGraphLoader` wraps a :class:`PartitionData` and converts each
snapshot into a DGL block on the target device.  When ``device`` is a CUDA
device, data is pinned to page-locked memory and transferred via a dedicated
CUDA stream for overlap with computation (mirrors the original FlareDTDG).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock

from starry_unigraph.data.partition import PartitionData
from .state import RNNStateManager, STGraphBlob, STWindowState


class STGraphLoader:
    """Loads partition snapshots as DGL blocks with optional GPU prefetch.

    The loader owns a :class:`PartitionData`, a CUDA stream for async
    transfer, and chunk-order metadata.  Iterating (``__call__`` or
    ``__iter__``) yields :class:`STGraphBlob` objects that carry an
    :class:`RNNStateManager` for stateful training.

    Args:
        rank: This worker's distributed rank.
        size: Total number of distributed workers.
        data: The :class:`PartitionData` to iterate over.
        device: Target ``torch.device`` for output blocks.
        chunk_count: Number of intra-partition chunks (for chunk-decay).
        chunk_index: Per-node chunk assignment tensor on *device*.
        stream: Optional dedicated CUDA stream for async H2D copies.

    Example::

        loader = STGraphLoader.from_partition_data(
            data=partition_data,
            device="cuda:0",
            chunk_index=chunk_idx,
            rank=0, size=1,
        )
        for blob in loader:
            pred, state = model(blob)
    """
    def __init__(
        self,
        rank: int,
        size: int,
        data: PartitionData,
        device: torch.device,
        chunk_count: int,
        chunk_index: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.rank = rank
        self.size = size
        self.data = data
        self.device = device
        self.chunk_count = chunk_count
        self.chunk_index = chunk_index
        self.stream = stream

    @classmethod
    def from_partition_data(
        cls,
        data: PartitionData,
        device: str | torch.device,
        chunk_index: torch.Tensor,
        rank: int = 0,
        size: int = 1,
    ) -> STGraphLoader:
        """Create a loader from a PartitionData with automatic pin_memory.

        Args:
            data: Source partition data (kept on CPU, optionally pinned).
            device: Target device (``"cuda:0"``, ``"cpu"``, etc.).
            chunk_index: Per-node chunk assignment, shape ``[num_dst_nodes]``.
            rank: Distributed rank (default 0 for single-rank).
            size: Distributed world size (default 1).

        Returns:
            A configured :class:`STGraphLoader`.
        """
        dev = torch.device(device)
        # pin_memory for faster async CPU→GPU transfer (mirrors FlareDTDG)
        if dev.type == "cuda":
            data = data.pin_memory(device=None)
        stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        chunk_index = chunk_index.long().to(dev)
        chunk_count = int(chunk_index.max().item() + 1) if chunk_index.numel() > 0 else 0
        return cls(
            rank=rank,
            size=size,
            data=data,
            device=dev,
            chunk_count=chunk_count,
            chunk_index=chunk_index,
            stream=stream,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: slice) -> STGraphLoader:
        if not isinstance(index, slice):
            raise TypeError("Only slice access is supported")
        return STGraphLoader(
            rank=self.rank,
            size=self.size,
            data=self.data[index],
            device=self.device,
            chunk_count=self.chunk_count,
            chunk_index=self.chunk_index,
            stream=self.stream,
        )

    @staticmethod
    def reorder_chunks(order: torch.Tensor, score: torch.Tensor | None = None) -> torch.Tensor:
        order = order.long()
        if score is None:
            return order
        ranked = score[order].argsort(dim=0, descending=True)
        remapped = torch.empty_like(ranked)
        remapped[ranked] = torch.arange(ranked.numel(), dtype=ranked.dtype, device=ranked.device)
        return remapped

    def _get_perm_ends(
        self,
        chunk_order: torch.Tensor | None = None,
        chunk_decay: list[int] | None = None,
        num_full_snaps: int = 1,
    ) -> tuple[torch.Tensor | None, list[int | None] | None]:
        if chunk_order is None:
            return None, None
        if int(chunk_order.numel()) != int(self.chunk_count):
            raise ValueError(f"Expected chunk_order size {self.chunk_count}, got {chunk_order.numel()}")
        if not chunk_decay:
            return None, [None] * max(1, num_full_snaps)

        # Move chunk_order to device via stream (async, mirrors FlareDTDG)
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                chunk_order = chunk_order.to(self.device, non_blocking=True)
                ordered_chunks = chunk_order[self.chunk_index]
                _, perm = ordered_chunks.sort(dim=0)
                counts = torch.bincount(ordered_chunks, minlength=self.chunk_count)
                ends = torch.cumsum(counts, dim=0)
        else:
            chunk_order = chunk_order.to(self.device)
            ordered_chunks = chunk_order[self.chunk_index]
            _, perm = ordered_chunks.sort(dim=0)
            counts = torch.bincount(ordered_chunks, minlength=self.chunk_count)
            ends = torch.cumsum(counts, dim=0)

        ends_list_raw = ends.tolist()
        ends_list = [ends_list_raw[max(0, min(self.chunk_count - 1, int(step) - 1))] for step in reversed(chunk_decay)]
        ends_list.extend([None] * max(1, num_full_snaps))
        return perm, ends_list

    def _synchronize(self) -> None:
        """Sync data stream and dist barrier (mirrors FlareDTDG.synchronize)."""
        if self.stream is None:
            return
        work = None
        if dist.is_available() and dist.is_initialized():
            work = dist.barrier(async_op=True)
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if work is not None:
            work.wait()

    def __call__(
        self,
        chunk_order: torch.Tensor | None = None,
        chunk_decay: list[int] | None = None,
        num_full_snaps: int = 1,
        disable_states: bool = True,
        disable_routes: bool = False,
    ):
        """Iterate over all snapshots, yielding :class:`STGraphBlob` objects.

        Args:
            chunk_order: Permutation of chunk IDs controlling which chunks
                are loaded in which order (for curriculum-style training).
            chunk_decay: List of chunk-decay steps — each step truncates the
                node set to only the top-N chunks.
            num_full_snaps: Number of trailing snapshots that use the full
                node set (no truncation).
            disable_states: If True, RNN state is padded (not mixed).
            disable_routes: If True, suppress distributed route exchange.

        Yields:
            :class:`STGraphBlob` — one per snapshot, carrying an
            :class:`RNNStateManager` for stateful forward passes.
        """
        perm, ends_list = self._get_perm_ends(
            chunk_order=chunk_order,
            chunk_decay=chunk_decay,
            num_full_snaps=num_full_snaps,
        )

        if ends_list is None:
            for i in range(len(self)):
                g = self.fetch_graph(index=i, perm=perm)
                g = RNNStateManager.patch_dummy_methods(g)
                states = RNNStateManager(ends_list=[None], mode="pad", disable_routes=disable_routes)
                states.add(g)
                yield STGraphBlob(states)
        else:
            states = RNNStateManager(
                ends_list=ends_list,
                mode="pad" if disable_states else "mix",
                disable_routes=disable_routes,
            )
            for i in range(len(self)):
                g = self.fetch_graph(index=i, perm=perm)
                states.add(g)
                yield STGraphBlob(states)

    def __iter__(self):
        yield from self.__call__()

    def fetch_graph(
        self,
        index: int,
        perm: torch.Tensor | None = None,
        ends: int | None = None,
        disable_routes: bool = False,
        disable_states: bool = True,
    ) -> DGLBlock:
        """Fetch a single snapshot as a DGLBlock, transferring to GPU.

        Args:
            index: Snapshot index within this loader's data.
            perm: Optional node permutation for chunk-decay reordering.
            ends: If set, truncate the block to this many dst nodes.
            disable_routes: If True, force ``block.route = None``.
            disable_states: If True, do not attach RNN state methods.

        Returns:
            A :class:`DGLBlock` on ``self.device`` with global IDs in
            ``srcdata[NID]`` / ``dstdata[NID]`` / ``edata[EID]``, plus
            node/edge features and optional ``block.route``.
        """
        # Async CPU→GPU transfer via dedicated stream (mirrors FlareDTDG.fetch_graph)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                snap_data = self.data[index].to(device=self.device, non_blocking=True)
            self._synchronize()
        else:
            snap_data = self.data[index].to(device=self.device, non_blocking=False)

        block = snap_data.item(node_perm=perm, keep_ids=True)

        # single-rank: disable route communication
        if self.size == 1:
            block.route = None
        elif block.route is not None:
            # attach distributed group to route
            if dist.is_available() and dist.is_initialized():
                block.route.group = dist.GroupMember.WORLD

        block.flare_snapshot_id = index
        block.flare_rnn_state_idx = 1  # will be re-patched by RNNStateManager.add()
        block.flare_is_full_snapshot = ends is None
        return block

    def build_snapshot_index(self) -> dict[str, Any]:
        return {
            "snaps": len(self.data),
            "partition_rank": self.rank,
            "partition_size": self.size,
            "chunk_count": self.chunk_count,
            "index_ready": True,
        }
