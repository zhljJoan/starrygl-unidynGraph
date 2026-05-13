"""Chunk-line MemoryStore: Interface for state/memory management.

This module defines the protocol for chunk training state management.
Actual implementations (RNN state, temporal memory, etc.) are provided
by specific trainers and StateManagers.

The MemoryStore is intentionally minimal — it serves as a contract
between the data layer and the runtime layer, allowing different
state management strategies to be plugged in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch import Tensor


class MemoryStore(ABC):
    """Abstract interface for chunk training state/memory management.

    Subclasses implement specific state management strategies:
    - RNNStateManager: RNN hidden states per snapshot
    - TemporalMemory: Temporal embedding caches
    - EventMemory: Online event aggregation (CTDG-style)

    This is a protocol/interface layer — the actual state tensor management
    happens elsewhere (in SchedulerSession, runtime loaders, etc.).
    """

    @abstractmethod
    def prepare(self, node_ids: Tensor, timestamps: Optional[Tensor] = None) -> None:
        """Prepare state for a new chunk/snapshot.

        Args:
            node_ids: [N] Node IDs involved in this computation
            timestamps: [E] Optional timestamps for temporal context
        """
        pass

    @abstractmethod
    def update(self, output: Any, chunk_id: Optional[Any] = None) -> None:
        """Update internal state after forward pass.

        Args:
            output: Model output or intermediate state
            chunk_id: Optional chunk identifier for bookkeeping
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all state (e.g., at epoch boundary)."""
        pass

    @abstractmethod
    def get_state(self, node_id: Optional[int] = None) -> Any:
        """Retrieve state for a node or all nodes.

        Args:
            node_id: If provided, return state for this node only

        Returns:
            State tensor(s) or dict
        """
        pass

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Return a description of the state structure and metadata."""
        pass


class DummyMemoryStore(MemoryStore):
    """Placeholder MemoryStore for testing."""

    def prepare(self, node_ids: Tensor, timestamps: Optional[Tensor] = None) -> None:
        pass

    def update(self, output: Any, chunk_id: Optional[Any] = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_state(self, node_id: Optional[int] = None) -> Any:
        return None

    def describe(self) -> Dict[str, Any]:
        return {"type": "dummy", "size": 0}
