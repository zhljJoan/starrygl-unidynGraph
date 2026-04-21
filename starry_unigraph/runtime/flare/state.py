"""RNN state management for multi-snapshot Flare training.

Provides :class:`RNNStateManager` (per-snapshot hidden state storage with
padding / mixing across a sliding window) and :class:`STGraphBlob` (a
sequence of DGL graphs with attached state management methods).
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RNNStateManager
# ---------------------------------------------------------------------------

class RNNStateManager:
    """Per-snapshot RNN state storage with padding / mixing support.

    Manages a sliding window of DGL graphs.  Each graph is monkey-patched
    with ``flare_fetch_state(h)`` and ``flare_store_state(h)`` methods so
    that models can transparently read/write per-snapshot hidden state
    without knowing the underlying state management strategy.

    Two modes are supported:

    - **pad** — always zero-pad state to the required number of nodes
      (used during chunk-decay warmup or eval).
    - **mix** — merge current state with a detached copy of the previous
      state (used during multi-frame training with overlapping windows).

    Args:
        ends_list: List of truncation endpoints (one per window slot).
            ``None`` means "full snapshot, no truncation".
        mode: ``"pad"`` or ``"mix"`` — state initialization strategy.
        disable_routes: If True, remove routing info from added graphs.

    Example::

        states = RNNStateManager(ends_list=[None], mode="pad")
        states.add(graph)
        blob = STGraphBlob(states)
        # model.forward(blob) will call graph.flare_fetch_state / flare_store_state
    """

    def __init__(
        self,
        ends_list: list[int | None],
        mode: str = "pad",
        disable_routes: bool = False,
    ) -> None:
        ends_list = [e for e in ends_list if e is None or e > 0]
        self._ends_list: list[int | None] = ends_list
        self._state_list: list[Any] = [None] * len(ends_list)
        self._graph_list: list[Any] = []
        self._mode = mode
        self._disable_routes = disable_routes
        self._snapshot_count = 0

    def __len__(self) -> int:
        return len(self._graph_list)

    def __getitem__(self, i: int) -> Any:
        return self._graph_list[i]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @classmethod
    def patch_dummy_methods(cls, g: Any) -> Any:
        """Attach no-op ``flare_fetch_state`` / ``flare_store_state`` to a graph.

        Used for single-graph eval/predict where no state manager exists.

        Args:
            g: A DGL graph or block.

        Returns:
            The same graph *g* with dummy methods attached in-place.
        """
        g.flare_snapshot_id = -1
        g.flare_rnn_state_idx = -1
        g.flare_is_full_snapshot = True
        g.flare_fetch_state = types.MethodType(lambda self, x, end=None: x, g)
        g.flare_store_state = types.MethodType(lambda self, x: None, g)
        return g

    def add(self, g: Any) -> None:
        """Add a new graph to the sliding window.

        Evicts the oldest graph if the window is full, then patches all
        graphs with state fetch/store methods bound to this manager.

        Args:
            g: A DGL graph (block or plain) for the next snapshot.
        """
        # keep window size bounded
        while len(self._graph_list) >= len(self._state_list):
            self._graph_list.pop(0)
        self._graph_list = [self._patch_methods(h) for h in self._graph_list]
        self._graph_list.append(self._patch_methods(g))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _patch_methods(self, g: Any) -> Any:
        snap_id = getattr(g, "flare_snapshot_id", None)
        if snap_id is None:
            snap_id = self._snapshot_count
            self._snapshot_count += 1

        idx = getattr(g, "flare_rnn_state_idx", len(self._ends_list)) - 1
        end = self._ends_list[idx]

        if self._disable_routes or end is not None:
            g = self._truncate_graph(g, end=end)

        g.flare_rnn_state_idx = idx
        g.flare_fetch_state = types.MethodType(self._flare_fetch_state, g)
        g.flare_store_state = types.MethodType(self._flare_store_state, g)
        g.flare_snapshot_id = snap_id
        g.flare_is_full_snapshot = end is None
        return g

    def _flare_fetch_state(self, this: Any, state: Any, end: int | None = None) -> Any:
        idx = this.flare_rnn_state_idx
        old = self._state_list[idx]
        if end is None:
            end = this.num_dst_nodes() if this.is_block else this.num_nodes()
        if old is None:
            return self.state_padding(state, end=end)
        if self._mode == "pad":
            return self.state_padding(state, end=end)
        if self._mode == "mix":
            return self.state_mixing(state, old_state=old, end=end)
        raise ValueError(f"Unknown mode: {self._mode}")

    def _flare_store_state(self, this: Any, state: Any) -> None:
        idx = this.flare_rnn_state_idx
        if self._mode == "pad":
            self._state_list[idx] = None
        elif self._mode == "mix":
            self._state_list[idx] = self.state_detach(state)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    # ------------------------------------------------------------------
    # State helpers (class methods, reusable)
    # ------------------------------------------------------------------

    @classmethod
    def apply_state(cls, fn, state: Any, old_state: Any = None) -> Any:
        if isinstance(state, (tuple, list)):
            paired = zip(state, old_state) if old_state is not None else ((s, None) for s in state)
            result = [fn(s, t) if old_state is not None else fn(s) for s, t in paired]
            return type(state)(result) if isinstance(state, tuple) else result
        if isinstance(state, dict):
            if old_state is None:
                return {k: fn(v) for k, v in state.items()}
            return {k: fn(v, old_state[k]) for k, v in state.items()}
        if old_state is None:
            return fn(state)
        return fn(state, old_state)

    @classmethod
    def state_detach(cls, state: Any) -> Any:
        def _detach(s: Any) -> Any:
            return s.detach() if isinstance(s, Tensor) else s
        return cls.apply_state(_detach, state)

    @classmethod
    def state_padding(cls, state: Any, end: int) -> Any:
        """Pad or truncate every tensor in *state* to *end* rows.

        Args:
            state: Nested structure (tensor, tuple, list, dict) of hidden
                state tensors.
            end: Target number of rows for dim-0.

        Returns:
            Same nested structure with all tensors resized to *end* rows.
        """
        def _pad(s: Any) -> Any:
            if not isinstance(s, Tensor):
                return s
            n = s.size(0)
            if n > end:
                return s[:end]
            if n < end:
                pad = [0] * (s.dim() * 2 - 1) + [end - n]
                return F.pad(s, pad)
            return s
        return cls.apply_state(_pad, state)

    @classmethod
    def state_mixing(cls, state: Any, old_state: Any, end: int) -> Any:
        """Mix current state with a previous state to fill *end* rows.

        Rows ``[0, cur_size)`` come from *state*; remaining rows
        ``[cur_size, end)`` come from *old_state* (detached).

        Args:
            state: Current hidden state tensors.
            old_state: Previous hidden state tensors (same structure).
            end: Target number of rows.

        Returns:
            Mixed state with exactly *end* rows per tensor.
        """
        def _mix(cur: Any, old: Any) -> Any:
            if not isinstance(cur, Tensor):
                return cur
            if cur.size(0) >= end:
                return cur[:end]
            # fill remainder from old_state
            need = end - cur.size(0)
            if old.size(0) >= end:
                tail = old[cur.size(0):end]
            else:
                tail = F.pad(old[cur.size(0):], [0] * (old.dim() * 2 - 1) + [end - old.size(0)])
            return torch.cat([cur, tail[:need]], dim=0)
        return cls.apply_state(_mix, state, old_state=old_state)

    @classmethod
    def _truncate_graph(cls, g: Any, end: int | None) -> Any:
        import dgl
        if not g.is_block:
            num_nodes = g.num_nodes()
            end = num_nodes if end is None else end
            if end >= num_nodes:
                return g
            inds = torch.arange(end, dtype=g.idtype, device=g.device)
            g = dgl.node_subgraph(g, inds, store_ids=False)
            g.create_formats_()
            return g

        num_dst = g.num_dst_nodes()
        end = num_dst if end is None else end
        if end > num_dst:
            end = num_dst
        g = dgl.block_to_graph(g)
        if end < g.num_src_nodes():
            nodes = torch.arange(end, dtype=g.idtype, device=g.device)
            g = dgl.node_subgraph(g, {"_N_src": nodes, "_N_dst": nodes}, store_ids=False)
        # re-pack as plain graph (same as FlareDTDG)
        for fmt in sorted(g.formats()["created"], reverse=True):
            adjs = g.adj_tensors(fmt)
            s = dgl.graph((fmt, adjs), num_nodes=end, idtype=g.idtype, device=g.device)
            for key, data in g.ndata.items():
                if "_N_src" in data:
                    s.ndata[key] = data["_N_src"]
                elif "_N_dst" in data:
                    s.ndata[key] = data["_N_dst"]
            for key, val in g.edata.items():
                s.edata[key] = val
            break
        g = s
        g.create_formats_()
        return g


# ---------------------------------------------------------------------------
# STWindowState  (kept for describe / metadata only; not used in training)
# ---------------------------------------------------------------------------

class STWindowState:
    """Lightweight metadata tracker (not used in training path)."""

    def __init__(self, full_snapshot_limit: int = 1) -> None:
        self.full_snapshot_limit = full_snapshot_limit

    def describe(self) -> dict[str, Any]:
        return {"full_snapshot_limit": self.full_snapshot_limit}


# ---------------------------------------------------------------------------
# STGraphBlob
# ---------------------------------------------------------------------------

class STGraphBlob:
    """Sequence of DGL graphs with attached RNN state management.

    In **training**, backed by :class:`RNNStateManager` (multi-frame
    window, state padding/mixing).  In **eval/predict**, typically wraps
    a single graph with dummy state methods.

    The blob is iterable and supports ``len()`` and ``__getitem__``.
    The most recent graph is accessible via :attr:`current_graph`.

    Args:
        state: The :class:`RNNStateManager` that owns the graph window.

    Example::

        blob = STGraphBlob(states)
        for graph in blob:          # iterate over window frames
            x = graph.dstdata["x"]
        last = blob.current_graph   # most recent snapshot
    """

    def __init__(self, state: RNNStateManager) -> None:
        self.state = state

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, idx: int) -> Any:
        return self.state[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self.state[i]

    # ------------------------------------------------------------------
    # Convenience properties (from current_graph = last entry)
    # ------------------------------------------------------------------

    @property
    def current_graph(self) -> Any:
        return self.state[-1]

    @property
    def flare_is_full_snapshot(self) -> bool:
        return bool(getattr(self.current_graph, "flare_is_full_snapshot", True))

    @property
    def snapshot_index(self) -> int:
        return int(getattr(self.current_graph, "flare_snapshot_id", 0))

    def describe(self) -> dict[str, Any]:
        return {
            "len": len(self),
            "snapshot_index": self.snapshot_index,
            "is_full_snapshot": self.flare_is_full_snapshot,
        }
