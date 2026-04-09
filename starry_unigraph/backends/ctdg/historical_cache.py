from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class AdaParameter:
    """Adaptive threshold for HistoricalCache — adjusts based on sync/compute latency ratio."""

    alpha: float = 0.5
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    _step: int = field(default=0, init=False, repr=False)

    def update_from_latency(self, sync_ms: float, compute_ms: float) -> None:
        """Tighten threshold when sync is cheap relative to compute; loosen otherwise."""
        if compute_ms <= 0:
            return
        ratio = sync_ms / (compute_ms + 1e-9)
        # ratio > 1 → sync dominates → raise alpha (filter more aggressively)
        # ratio < 1 → compute dominates → lower alpha (allow more syncs)
        delta = 0.01 * (ratio - 1.0)
        self.alpha = float(
            max(self.min_alpha, min(self.max_alpha, self.alpha + delta))
        )
        self._step += 1


class CTDGHistoricalCache:
    """Per-rank cache for shared-node memory snapshots.

    Tracks the last-synced memory for shared nodes and provides a boolean mask
    indicating which nodes have changed enough to warrant a new sync.
    """

    def __init__(
        self,
        num_shared: int,
        hidden_dim: int,
        device: str,
        num_nodes: int | None = None,
        shared_node_ids: torch.Tensor | None = None,
        ada_param: AdaParameter | None = None,
    ) -> None:
        self.num_shared = num_shared
        self.hidden_dim = hidden_dim
        self.device = device
        self.ada_param = ada_param or AdaParameter()

        # Snapshot of memory at last sync time
        self._cached_memory: torch.Tensor = torch.zeros(
            num_shared, hidden_dim, dtype=torch.float32, device=device
        )
        self._cached_memory_ts: torch.Tensor = torch.full(
            (num_shared,), -1.0, dtype=torch.float32, device=device
        )
        self._slot_by_global: torch.Tensor | None = None
        if num_nodes is not None and shared_node_ids is not None:
            self.bind_shared_nodes(num_nodes=num_nodes, shared_node_ids=shared_node_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def historical_check(
        self,
        shared_node_local_indices: torch.Tensor,  # [S] local node_ids of shared nodes
        current_memory: torch.Tensor,              # [S, D] current memory for those nodes
    ) -> torch.Tensor:                             # bool mask [S], True = changed enough
        """Return True for nodes whose memory has drifted beyond alpha threshold."""
        if shared_node_local_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        # Map local node ids → cache slot indices
        cache_slots = self._local_to_slot(shared_node_local_indices)
        valid = cache_slots >= 0
        if not valid.any():
            # No cache entry yet → all need sync
            return torch.ones(shared_node_local_indices.numel(), dtype=torch.bool, device=self.device)

        cached = self._cached_memory[cache_slots.clamp(min=0)]
        cur = current_memory.to(self.device).float()

        # Cosine distance: 1 - cos_sim; range [0, 2]
        cos_sim = F.cosine_similarity(cur, cached, dim=1)  # [S]
        dist = 1.0 - cos_sim  # higher = more change

        changed = dist > (1.0 - self.ada_param.alpha)  # [S] bool
        # Nodes without valid cache always need sync
        changed = changed | (~valid)
        return changed

    def update_cache(
        self,
        shared_indices: torch.Tensor,  # [S] cache slot indices (0-based)
        new_memory: torch.Tensor,       # [S, D]
        new_ts: torch.Tensor,           # [S]
    ) -> None:
        """Update cache after a successful sync."""
        if shared_indices.numel() == 0:
            return
        idx = shared_indices.long().to(self.device)
        self._cached_memory[idx] = new_memory.float().to(self.device)
        self._cached_memory_ts[idx] = new_ts.float().to(self.device)

    def synchronize_shared_update(
        self,
        received_node_ids: torch.Tensor,   # [R] global/local node ids received from other ranks
        received_values: torch.Tensor,     # [R, D]
        received_ts: torch.Tensor,         # [R]
    ) -> None:
        """Apply received updates to cached snapshot, keeping only newer timestamps."""
        if received_node_ids.numel() == 0:
            return
        cache_slots = self._local_to_slot(received_node_ids)
        valid = cache_slots >= 0
        if not valid.any():
            return
        slots = cache_slots[valid].long()
        vals = received_values[valid].float().to(self.device)
        ts = received_ts[valid].float().to(self.device)

        # Only apply if received timestamp is newer than cached
        current_ts = self._cached_memory_ts[slots]
        newer = ts > current_ts
        if not newer.any():
            return
        slots_n = slots[newer]
        vals_n = vals[newer]
        ts_n = ts[newer]
        self._cached_memory[slots_n] = vals_n
        self._cached_memory_ts[slots_n] = ts_n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_to_slot(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Map global node IDs to cache slot indices."""
        if self._slot_by_global is None:
            return torch.full_like(node_ids, -1, dtype=torch.long, device=self.device)
        ids = node_ids.long().to(self.device)
        valid = (ids >= 0) & (ids < self._slot_by_global.numel())
        out = torch.full(ids.shape, -1, dtype=torch.long, device=self.device)
        if valid.any():
            out[valid] = self._slot_by_global[ids[valid]]
        return out

    def slot_indices(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self._local_to_slot(node_ids)

    def bind_shared_nodes(self, num_nodes: int, shared_node_ids: torch.Tensor) -> None:
        """Bind fixed global shared-node IDs to cache slot indices."""
        slot_by_global = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)
        ids = shared_node_ids.long().to(self.device)
        if ids.numel() > 0:
            slot_by_global[ids] = torch.arange(ids.numel(), dtype=torch.long, device=self.device)
        self._slot_by_global = slot_by_global
