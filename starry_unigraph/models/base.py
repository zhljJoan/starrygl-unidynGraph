"""Model protocols and base classes for task-agnostic temporal GNNs."""
from __future__ import annotations

from typing import Dict, Optional, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.data.batch_data import BatchData


class TemporalModel(Protocol):
    """Protocol for core temporal GNN without task awareness.

    Implementations compute node embeddings given a graph and state,
    but do not know about the downstream task (EdgePredict, NodeRegress, etc.)
    """

    def forward(
        self,
        mfg: any,          # Message flow graph (DGL/PyTorch)
        state: Dict[str, Tensor],  # RNN hidden states / memory
    ) -> Tensor:
        """Forward pass to compute node embeddings.

        Args:
            mfg: Message flow graph containing the local neighborhood
            state: Dictionary of RNN states (h, c) or memory state

        Returns:
            Node embeddings [N, hidden_dim]
        """
        ...

    def compute_state_update(
        self,
        embeddings: Tensor,
        batch: BatchData,
    ) -> Dict[str, Tensor]:
        """Compute state updates to be applied after this batch.

        Args:
            embeddings: [N, hidden_dim] output from forward()
            batch: BatchData with timestamps and node_ids

        Returns:
            Dictionary of state updates (for StateManager to apply)
        """
        ...
