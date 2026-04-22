"""GraphBackend adapters for existing runtimes.

These adapters wrap CTDGSession, FlareRuntimeLoader, and ChunkRuntimeLoader
to implement the unified GraphBackend protocol.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import torch
from torch import Tensor

from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
from starry_unigraph.backends.dtdg import FlareRuntimeLoader
from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.runtime.backend import GraphBackend, StateManager
from starry_unigraph.runtime.chunk import ChunkRuntimeLoader


class CTDGGraphBackend(GraphBackend):
    """Adapter wrapping CTDGSession to implement GraphBackend protocol."""

    def __init__(self, ctdg_session: CTDGSession):
        self.ctdg_session = ctdg_session
        self._current_iterator = None

    def iter_batches(
        self,
        split: str,
        batch_size: int,
    ) -> Iterator[ChunkAtomic]:
        """Iterate over events/batches in CTDG online stream.

        For CTDG, 'batch' is typically a sequence of events grouped by time.
        batch_size represents events per batch (or time windows).
        """
        # CTDG iter_train / iter_eval returns iterator of batches
        if split == "train":
            iterator = self.ctdg_session.iter_train(self.ctdg_session.ctx)
        else:
            iterator = self.ctdg_session.iter_eval(self.ctdg_session.ctx, split=split)

        # Each item from iterator is a batch dict
        # Wrap as ChunkAtomic (placeholder for now)
        for batch in iterator:
            # Create minimal ChunkAtomic from batch
            chunk = self._batch_to_chunk(batch)
            yield chunk

    def reset(self) -> None:
        """Reset state between epochs."""
        pass

    def describe(self) -> Dict[str, Any]:
        """Return metadata about this backend."""
        return {
            "backend": "ctdg_online",
            "graph_mode": "ctdg",
        }

    @staticmethod
    def _batch_to_chunk(batch: Dict[str, Any]) -> ChunkAtomic:
        """Convert CTDG batch dict to ChunkAtomic (placeholder)."""
        # In real implementation, would extract graph structure from batch
        # For now, create empty ChunkAtomic with metadata
        chunk = ChunkAtomic(
            chunk_id=(0, 0),  # Dummy IDs
            time_range=(0.0, 1.0),
            node_set=torch.tensor([]),
            tcsr_rowptr=torch.tensor([]),
            tcsr_col=torch.tensor([]),
            tcsr_ts=torch.tensor([]),
            tcsr_edge_id=torch.tensor([]),
            cross_node_ids=torch.tensor([]),
            cross_node_home=torch.tensor([]),
            cross_edge_count=torch.tensor([]),
        )
        return chunk


class FlareGraphBackend(GraphBackend):
    """Adapter wrapping FlareRuntimeLoader to implement GraphBackend protocol."""

    def __init__(self, flare_loader: FlareRuntimeLoader):
        self.flare_loader = flare_loader

    def iter_batches(
        self,
        split: str,
        batch_size: int,
    ) -> Iterator[ChunkAtomic]:
        """Iterate over snapshots in DTDG Flare mode."""
        if split == "train":
            iterator = self.flare_loader.iter_train(split=split)
        else:
            iterator = self.flare_loader.iter_eval(split=split)

        for stgraph_blob in iterator:
            chunk = self._blob_to_chunk(stgraph_blob)
            yield chunk

    def reset(self) -> None:
        """Reset state between epochs."""
        pass

    def describe(self) -> Dict[str, Any]:
        """Return metadata about this backend."""
        return {
            "backend": "flare_snapshot",
            "graph_mode": "dtdg",
        }

    @staticmethod
    def _blob_to_chunk(blob: Any) -> ChunkAtomic:
        """Convert FlareRuntimeLoader's STGraphBlob to ChunkAtomic (placeholder)."""
        chunk = ChunkAtomic(
            chunk_id=(0, 0),
            time_range=(0.0, 1.0),
            node_set=torch.tensor([]),
            tcsr_rowptr=torch.tensor([]),
            tcsr_col=torch.tensor([]),
            tcsr_ts=torch.tensor([]),
            tcsr_edge_id=torch.tensor([]),
            cross_node_ids=torch.tensor([]),
            cross_node_home=torch.tensor([]),
            cross_edge_count=torch.tensor([]),
        )
        return chunk


class ChunkGraphBackend(GraphBackend):
    """Adapter wrapping ChunkRuntimeLoader to implement GraphBackend protocol."""

    def __init__(self, chunk_loader: ChunkRuntimeLoader):
        self.chunk_loader = chunk_loader

    def iter_batches(
        self,
        split: str,
        batch_size: int,
    ) -> Iterator[ChunkAtomic]:
        """Iterate over time-window chunks in Chunk mode."""
        if split == "train":
            iterator = self.chunk_loader.iter_train(split=split)
        else:
            iterator = self.chunk_loader.iter_eval(split=split)

        for chunk in iterator:
            yield chunk

    def reset(self) -> None:
        """Reset state between epochs."""
        pass

    def describe(self) -> Dict[str, Any]:
        """Return metadata about this backend."""
        return {
            "backend": "chunk_windowed",
            "graph_mode": "chunk",
        }


class DummyStateManager(StateManager):
    """Minimal StateManager for testing/prototyping.

    In real use, this would be replaced with CTDG/Flare-specific state managers.
    """

    def __init__(self):
        self.state = {}

    def prepare(
        self,
        node_ids: Tensor,
        timestamps: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Return current state (or empty if not available)."""
        return self.state.copy()

    def update(
        self,
        model_output: Dict[str, Tensor],
        chunk: ChunkAtomic,
    ) -> None:
        """Update state after batch (dummy: do nothing)."""
        pass

    def reset(self) -> None:
        """Reset state."""
        self.state.clear()

    def describe(self) -> Dict[str, Any]:
        """Return metadata."""
        return {"manager": "dummy", "state_size": len(self.state)}
