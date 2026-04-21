"""GraphBackend Protocol: Abstracts how data flows through training loop.

Defines the interface for iterating over graph data, independent of whether
it's online (CTDG), snapshot-based (DTDG), or chunked (Chunk) processing.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Protocol

from torch import Tensor

from starry_unigraph.data.chunk_atomic import ChunkAtomic


class GraphBackend(Protocol):
    """Abstract interface for graph data provisioning.

    Implementations: CTDGOnlineBackend, FlareSnapshotBackend, ChunkBackend
    """

    def iter_batches(
        self,
        split: str,        # "train", "val", "test"
        batch_size: int,   # Samples per batch (or snapshots per batch)
    ) -> Iterator[ChunkAtomic]:
        """Iterate over data chunks/batches for an epoch.

        Each chunk contains graph structure, timestamps, and metadata.
        The caller (PipelineEngine) will apply task-specific sampling on each chunk.

        Args:
            split: Data split ("train", "val", "test")
            batch_size: Batch size (meaning depends on backend)

        Yields:
            ChunkAtomic: Data chunk with graph structure
        """
        ...

    def reset(self) -> None:
        """Reset state between epochs."""
        ...

    def describe(self) -> Dict[str, Any]:
        """Return metadata about this backend for logging/debugging.

        Example: {'backend': 'flare', 'num_snapshots': 100, 'world_size': 4}
        """
        ...


class StateManager(Protocol):
    """Abstract interface for managing RNN/embedding state across iterations.

    Implementations: CTDGStateManager, RNNStateManager, ChunkStateManager
    """

    def prepare(
        self,
        node_ids: Tensor,        # [N] Nodes in current batch
        timestamps: Optional[Tensor] = None,  # [E] Edge timestamps
    ) -> Dict[str, Any]:
        """Prepare state for forward pass on current batch.

        Args:
            node_ids: [N] Node IDs in this batch
            timestamps: Optional edge timestamps (for temporal models)

        Returns:
            State dict to pass to model.forward() (e.g., RNN hidden states)
        """
        ...

    def update(
        self,
        model_output: Dict[str, Tensor],
        chunk: ChunkAtomic,
    ) -> None:
        """Update state after backward pass on current chunk.

        Args:
            model_output: Dictionary of tensors from model.forward()
            chunk: The chunk that was just processed
        """
        ...

    def reset(self) -> None:
        """Reset state (called at epoch boundaries, before eval, etc.)."""
        ...

    def describe(self) -> Dict[str, Any]:
        """Return metadata about this state manager for logging.

        Example: {'manager': 'rnn', 'hidden_dim': 128, 'num_layers': 2}
        """
        ...
