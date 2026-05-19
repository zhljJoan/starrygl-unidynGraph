"""STGraphLoader: Spatio-Temporal Graph Loader with Chunk Decay Strategy.

Implements FlareDTDG-style historical snapshot loading with chunk-based decay weights.
The key difference from FlareDTDG: decay endpoints are computed from the chunk CSR
(chunk_ptr) rather than a flat node permutation, so each historical layer exposes
exactly the nodes belonging to the first k chunks in decay order.

Reference: ~/FlareDTDG/flare2/data/stc_loader.py
"""

from __future__ import annotations

import types
from typing import Any, Iterator, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from torch import Tensor

from starry_unigraph.backends.chunk.data.partition import PartitionData
from starry_unigraph.backends.chunk.prepare.chunk_assignment import ChunkAssignment
from starry_unigraph.models.layers.route import ChunkPropagationRoute


__all__ = [
    "RNNStateManager",
    "STGraphBlob",
    "STGraphLoader",
]


# ─────────────────────────────────────────────────────────────────────────────
# RNNStateManager
# ─────────────────────────────────────────────────────────────────────────────

class RNNStateManager:
    """Manages RNN hidden states across temporal snapshots.

    Each historical layer has its own state slot.  When a new snapshot is
    added the oldest graph is evicted and all remaining graphs are
    re-patched so their ``flare_rnn_state_idx`` points to the correct slot.

    Args:
        ends_list: Truncation endpoint for each slot.  ``None`` = full graph.
        mode: ``"pad"`` zeros out the state at every step;
              ``"mix"`` blends the previous state with the current one.
        disable_routes: When True the graph is truncated to ``end`` nodes
                        even if ``end is None`` (used for single-GPU runs).
    """

    def __init__(
        self,
        ends_list: List[Optional[int]],
        mode: Literal["pad", "mix"] = "pad",
        disable_routes: bool = False,
    ):
        ends_list = [e for e in ends_list if e is None or e > 0]
        self._ends_list: List[Optional[int]] = ends_list
        self._state_list: List[Any] = [None] * len(ends_list)
        self._graph_list: List[DGLBlock | DGLGraph] = []
        self._mode = mode
        self._disable_routes = disable_routes
        self._snapshot_count = 0

    def __len__(self) -> int:
        return len(self._graph_list)

    def __getitem__(self, i: int) -> DGLBlock | DGLGraph:
        return self._graph_list[i]

    # ── public helpers ────────────────────────────────────────────────────────

    @classmethod
    def patch_dummy_methods(cls, g: DGLBlock | DGLGraph) -> DGLBlock | DGLGraph:
        """Attach no-op state methods to a graph (single-GPU / no-history path)."""
        g.flare_snapshot_id = -1
        g.flare_rnn_state_idx = -1
        g.flare_is_full_snapshot = True
        g.flare_fetch_state = types.MethodType(lambda self, x, end=None: x, g)
        g.flare_store_state = types.MethodType(lambda self, x: None, g)
        return g

    def add(self, g: DGLBlock | DGLGraph) -> None:
        """Enqueue a new graph, evicting the oldest if the buffer is full."""
        while len(self._graph_list) >= len(self._state_list):
            self._graph_list.pop(0)
        # Re-patch all existing graphs (their slot index shifts by one)
        for i in range(len(self._graph_list)):
            self._graph_list[i] = self._patch_methods(self._graph_list[i])
        self._graph_list.append(self._patch_methods(g))

    # ── internal ──────────────────────────────────────────────────────────────

    def _patch_methods(self, g: DGLBlock | DGLGraph) -> DGLBlock | DGLGraph:
        snap_id = getattr(g, "flare_snapshot_id", None)
        if snap_id is None:
            snap_id = self._snapshot_count
            self._snapshot_count += 1

        # Slot index: most-recent graph gets slot len-1, oldest gets slot 0
        idx = getattr(g, "flare_rnn_state_idx", len(self._ends_list)) - 1
        end = self._ends_list[idx]

        if self._disable_routes or end is not None:
            g = self.truncate_graph(g, end=end)

        g.flare_rnn_state_idx = idx
        g.flare_fetch_state = types.MethodType(self._flare_fetch_state, g)
        g.flare_store_state = types.MethodType(self._flare_store_state, g)
        g.flare_snapshot_id = snap_id
        g.flare_is_full_snapshot = (end is None)
        return g

    def _flare_fetch_state(self, this: DGLBlock | DGLGraph, state: Any, end: Optional[int] = None) -> Any:
        idx = this.flare_rnn_state_idx
        assert 0 <= idx < len(self._state_list)
        old = self._state_list[idx]
        if end is None:
            end = this.num_dst_nodes() if this.is_block else this.num_nodes()
        if old is None or self._mode == "pad":
            return self.state_padding(state, end=end)
        return self.state_mixing(state, old_state=old, end=end)

    def _flare_store_state(self, this: DGLBlock | DGLGraph, state: Any) -> None:
        idx = this.flare_rnn_state_idx
        assert 0 <= idx < len(self._state_list)
        if self._mode == "pad":
            self._state_list[idx] = None
        else:
            self._state_list[idx] = self.state_detach(state)

    # ── state helpers (recursive over nested structures) ──────────────────────

    @classmethod
    def apply_state(cls, fn, state: Any, old_state: Any = None) -> Any:
        if isinstance(state, (tuple, list)):
            T = type(state)
            if old_state is None:
                return T(fn(s) for s in state)
            return T(fn(s, t) for s, t in zip(state, old_state))
        if isinstance(state, dict):
            if old_state is None:
                return {k: fn(v) for k, v in state.items()}
            return {k: fn(v, old_state[k]) for k, v in state.items()}
        return fn(state) if old_state is None else fn(state, old_state)

    @classmethod
    def state_detach(cls, state: Any) -> Any:
        return cls.apply_state(lambda s: s.detach() if isinstance(s, Tensor) else s, state)

    @classmethod
    def state_padding(cls, state: Any, end: int) -> Any:
        def _pad(s):
            if not isinstance(s, Tensor):
                return s
            if s.size(0) > end:
                return s[:end]
            if s.size(0) < end:
                pad = [0] * (2 * s.ndim - 1) + [end - s.size(0)]
                return F.pad(s, pad)
            return s
        return cls.apply_state(_pad, state)

    @classmethod
    def state_mixing(cls, state: Any, old_state: Any, end: int) -> Any:
        def _mix(cur, old):
            if not isinstance(cur, Tensor):
                return cur
            cur = cls.state_padding(cur, end=end)
            old = cls.state_padding(old, end=end)
            return (cur + old) * 0.5
        return cls.apply_state(_mix, state, old_state)

    @staticmethod
    def truncate_graph(g: DGLBlock | DGLGraph, end: Optional[int]) -> DGLBlock | DGLGraph:
        """Truncate graph to first ``end`` dst nodes (no-op if end is None)."""
        if end is None:
            return g
        if g.is_block:
            n = g.num_dst_nodes()
            if n <= end:
                return g
            dst_nodes = torch.arange(end, dtype=torch.long, device=g.device)
            try:
                return g.sample_neighbors(dst_nodes, -1)
            except Exception:
                return g
        else:
            n = g.num_nodes()
            if n <= end:
                return g
            nodes = torch.arange(end, dtype=torch.long, device=g.device)
            return g.subgraph(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# STGraphBlob
# ─────────────────────────────────────────────────────────────────────────────

class STGraphBlob:
    """A window of historical graphs with route helpers attached.

    Wraps an ``RNNStateManager`` and exposes each graph with
    ``flare_apply_route`` / ``flare_async_route`` methods so model layers
    can call routing without knowing the underlying route type.
    """

    def __init__(self, state: RNNStateManager):
        self.state = state

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, idx: int) -> DGLBlock | DGLGraph:
        g = self.state[idx]
        return self._patch_methods(g)

    def __iter__(self) -> Iterator[DGLBlock | DGLGraph]:
        for i in range(len(self)):
            yield self[i]

    @property
    def current_graph(self) -> DGLBlock | DGLGraph:
        return self[-1]

    @property
    def flare_is_full_snapshot(self) -> bool:
        return bool(getattr(self.current_graph, "flare_is_full_snapshot", True))

    @property
    def snapshot_index(self) -> int:
        return int(getattr(self.current_graph, "flare_snapshot_id", 0))

    @classmethod
    def _patch_methods(cls, g: DGLBlock | DGLGraph) -> DGLBlock | DGLGraph:
        g.flare_apply_route = types.MethodType(cls._apply_route, g)
        g.flare_async_route = types.MethodType(cls._async_route, g)
        return g

    @staticmethod
    def _apply_route(this: DGLBlock | DGLGraph, x: Tensor, reverse: bool = False) -> Tensor:
        route: Optional[ChunkPropagationRoute] = getattr(this, "route", None)
        return x if route is None else route.forward(x, reverse=reverse)

    @staticmethod
    async def _async_route(this: DGLBlock | DGLGraph, x: Tensor, reverse: bool = False) -> Tensor:
        route: Optional[ChunkPropagationRoute] = getattr(this, "route", None)
        return x if route is None else await route.async_forward(x, reverse=reverse)

    def __repr__(self) -> str:
        parts = []
        for g in self:
            k = g.flare_snapshot_id
            u = g.flare_rnn_state_idx
            shape = f"{g.num_src_nodes()}->{g.num_dst_nodes()}" if g.is_block else str(g.num_nodes())
            tag = f"G[{k}|{u}]{{{shape}}}"
            if getattr(g, "route", None) is not None:
                tag += "(R)"
            if g.flare_is_full_snapshot:
                tag += "*"
            parts.append(tag)
        return f"STGraphBlob[{len(self)}]( {' => '.join(parts)} )"


# ─────────────────────────────────────────────────────────────────────────────
# STGraphLoader
# ─────────────────────────────────────────────────────────────────────────────

class STGraphLoader:
    """Spatio-Temporal Graph Loader with chunk-based decay strategy.

    Iterates over snapshots and yields ``STGraphBlob`` objects.  Each blob
    contains a sliding window of historical graphs where older graphs are
    truncated to fewer nodes according to the chunk decay schedule.

    Chunk decay strategy (differs from FlareDTDG):
    - FlareDTDG uses a flat ``chunk_index`` permutation and ``ind2ptr``.
    - Here we use ``ChunkAssignment.chunk_ptr`` (CSR) to compute the
      truncation endpoint for each decay level directly, so the endpoint
      is exactly ``chunk_ptr[decay_chunks[k] + 1]`` — the number of nodes
      belonging to the first ``decay_chunks[k]+1`` chunks in decay order.

    Args:
        partition_data: PartitionData for this rank
        chunk_assignment: ChunkAssignment with CSR chunk_ptr / chunk_nodes
        device: Target CUDA device
        num_full_snaps: Number of full (non-truncated) snapshots in the blob
        chunk_decay: Ordered list of chunk IDs from most-recent to oldest.
                     ``None`` disables decay (all snapshots are full).
        rnn_state_mode: ``"pad"`` or ``"mix"``
        disable_routes: Disable distributed routing (single-GPU)
    """

    def __init__(
        self,
        partition_data: PartitionData,
        chunk_assignment: Optional[ChunkAssignment] = None,
        device: str | torch.device = "cuda",
        num_full_snaps: int = 1,
        chunk_decay: Optional[List[int]] = None,
        rnn_state_mode: Literal["pad", "mix"] = "pad",
        disable_routes: bool = False,
    ):
        self.partition_data = partition_data
        self.chunk_assignment = chunk_assignment
        self.device = torch.device(device)
        self.num_full_snaps = num_full_snaps
        self.chunk_decay = chunk_decay
        self.rnn_state_mode = rnn_state_mode
        self.disable_routes = disable_routes

        # Distributed context
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.group = dist.GroupMember.WORLD
        else:
            self.rank = 0
            self.world_size = 1
            self.group = None

        # CUDA stream for async data loading
        if self.device.type == "cuda":
            self.stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(self.device)
        else:
            self.stream = None

        # Pre-compute ends_list from chunk CSR
        self._ends_list: Optional[List[Optional[int]]] = self._build_ends_list()

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.partition_data)

    def __iter__(self) -> Iterator[STGraphBlob]:
        yield from self()

    def __call__(
        self,
        chunk_order: Optional[Tensor] = None,
        chunk_decay: Optional[List[int]] = None,
        num_full_snaps: Optional[int] = None,
        disable_states: bool = True,
        disable_routes: Optional[bool] = None,
    ) -> Iterator[STGraphBlob]:
        """Iterate over snapshots, yielding STGraphBlob objects.

        Args:
            chunk_order: Optional [num_chunks] tensor overriding the decay
                         order at runtime (e.g. sorted by load score).
            disable_states: If True use ``"pad"`` mode regardless of config.
        """
        old_decay = self.chunk_decay
        old_full = self.num_full_snaps
        if chunk_decay is not None:
            self.chunk_decay = list(chunk_decay)
        if num_full_snaps is not None:
            self.num_full_snaps = max(1, int(num_full_snaps))
        ends_list = self._build_ends_list()

        # Runtime override: recompute ends_list from chunk_order
        if chunk_order is not None and self.chunk_assignment is not None:
            ends_list = self._build_ends_list_from_order(chunk_order)
        routes_disabled = self.disable_routes if disable_routes is None else bool(disable_routes)

        try:
            if ends_list is None:
                # No decay: keep Flare-compatible behavior and yield a graph.
                for i in range(len(self)):
                    g = self._fetch_graph(i)
                    g = RNNStateManager.patch_dummy_methods(g)
                    g = STGraphBlob._patch_methods(g)
                    yield g
            else:
                mode = "pad" if disable_states else self.rnn_state_mode
                states = RNNStateManager(
                    ends_list=ends_list,
                    mode=mode,
                    disable_routes=routes_disabled or self.world_size == 1,
                )
                for i in range(len(self)):
                    g = self._fetch_graph(i)
                    states.add(g)
                    yield STGraphBlob(states)
        finally:
            self.chunk_decay = old_decay
            self.num_full_snaps = old_full

    # ── chunk decay helpers ───────────────────────────────────────────────────

    def _build_ends_list(self) -> Optional[List[Optional[int]]]:
        """Pre-compute truncation endpoints from chunk_assignment.chunk_ptr.

        Returns None when no decay is configured.
        """
        if self.chunk_decay is None or self.chunk_assignment is None:
            return None

        chunk_ptr = self.chunk_assignment.chunk_ptr  # [num_chunks + 1]
        ends: List[Optional[int]] = []

        # Historical layers: each level exposes one more chunk
        for k in self.chunk_decay:
            # endpoint = number of nodes in chunks 0..k (inclusive)
            end = int(chunk_ptr[k + 1].item())
            ends.append(end)

        # Full snapshots at the end
        ends.extend([None] * self.num_full_snaps)
        return ends

    def _build_ends_list_from_order(self, chunk_order: Tensor) -> List[Optional[int]]:
        """Recompute ends_list when chunk_order is provided at runtime.

        ``chunk_order[c]`` is the priority rank of chunk c (lower = higher
        priority / more recent).  We sort chunks by priority and compute
        cumulative node counts from chunk_ptr.
        """
        assert self.chunk_assignment is not None
        chunk_ptr = self.chunk_assignment.chunk_ptr.to(chunk_order.device)
        chunk_sizes = chunk_ptr[1:] - chunk_ptr[:-1]  # [num_chunks]

        # Sort chunks by priority (ascending order = most important first)
        sorted_chunks = chunk_order.argsort()  # [num_chunks]
        sorted_sizes = chunk_sizes[sorted_chunks]
        cumulative = sorted_sizes.cumsum(0)  # [num_chunks]

        ends: List[Optional[int]] = []
        num_decay = len(self.chunk_decay) if self.chunk_decay else 0
        for i in range(num_decay):
            # Expose first (i+1) chunks in priority order
            end = int(cumulative[i].item())
            ends.append(end)

        ends.extend([None] * self.num_full_snaps)
        return ends

    # ── graph loading ─────────────────────────────────────────────────────────

    def _fetch_graph(self, k: int) -> DGLBlock | DGLGraph:
        """Load snapshot k to device, optionally via a CUDA stream."""
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                data = self.partition_data[k].to(device=self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        else:
            data = self.partition_data[k].to(device=self.device)

        g = data.to_block(0)  # PartitionData[k] has a single snapshot
        if "x" not in g.srcdata:
            g.srcdata["x"] = torch.ones(g.num_src_nodes(), 1, dtype=torch.float32, device=g.device)
        if "x" not in g.dstdata:
            g.dstdata["x"] = g.srcdata["x"][: g.num_dst_nodes()]
        g.flare_snapshot_id = int(k)
        g.flare_rnn_state_idx = 1
        g.flare_is_full_snapshot = True

        # Disable route on single-GPU
        if self.disable_routes or self.world_size == 1:
            setattr(g, "route", None)
        elif hasattr(g, "route") and g.route is not None:
            setattr(g.route, "group", self.group)

        return g

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_partition_data(
        cls,
        partition_data: PartitionData,
        device: str | torch.device,
        chunk_assignment: Optional[ChunkAssignment] = None,
        chunk_index: Optional[Tensor] = None,
        num_full_snaps: int = 1,
        chunk_decay: Optional[List[int]] = None,
        rnn_state_mode: Literal["pad", "mix"] = "pad",
        disable_routes: bool = False,
    ) -> "STGraphLoader":
        if chunk_assignment is None and chunk_index is not None:
            chunk_index = chunk_index.long().cpu()
            num_chunks = int(chunk_index.max().item() + 1) if chunk_index.numel() > 0 else 1
            sort_order = torch.argsort(chunk_index, stable=True)
            counts = torch.bincount(chunk_index[sort_order], minlength=num_chunks)
            chunk_ptr = torch.zeros(num_chunks + 1, dtype=torch.long)
            chunk_ptr[1:] = counts.cumsum(0)
            chunk_assignment = ChunkAssignment(
                num_chunks_per_partition=max(1, num_chunks),
                node_to_chunk=chunk_index,
                chunk_ptr=chunk_ptr,
                chunk_nodes=sort_order.long(),
                chunk_to_initial_partition=torch.zeros(num_chunks, dtype=torch.long),
                chunk_to_owner_partition=torch.zeros(num_chunks, dtype=torch.long),
            )
        return cls(
            partition_data=partition_data,
            chunk_assignment=chunk_assignment,
            device=device,
            num_full_snaps=num_full_snaps,
            chunk_decay=chunk_decay,
            rnn_state_mode=rnn_state_mode,
            disable_routes=disable_routes or not dist.is_initialized() or dist.get_world_size() == 1,
        )
