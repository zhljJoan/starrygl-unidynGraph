"""TaskAdapter Protocol: Encapsulates task-specific logic independent of graph mode.

Defines the interface for sampling configuration, loss computation, metrics,
and output formatting. Separates Task Type (EdgePredict, NodeRegress, NodeClassify)
from Graph Mode (CTDG, DTDG, Chunk).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import torch
from torch import Tensor


@dataclass
class SampleConfig:
    """Configuration passed to chunk.materialize() to control sampling.

    Fields are task-specific. EdgePredict uses pos_src/pos_dst.
    NodeRegress/Classify use target_nodes/target_labels.
    """
    # EdgePredict fields
    pos_src: Optional[Tensor] = None     # Source nodes of positive edges
    pos_dst: Optional[Tensor] = None     # Destination nodes of positive edges
    neg_strategy: str = "none"           # "cache_aware", "random", "none"
    neg_ratio: int = 1                   # Negative samples per positive

    # NodeRegress/Classify fields
    target_nodes: Optional[Tensor] = None  # Node IDs with labels
    target_labels: Optional[Tensor] = None # Target labels/values

    # Common sampling hyperparams
    num_neighbors: List[int] = field(default_factory=lambda: [20, 10])
    num_layers: int = 2
    sample_type: str = "temporal"        # "temporal" or "random"

    # Additional fields for future extension
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchData:
    """Unified batch container, populated by chunk.materialize().

    Contains graph structure (mfg), node info, and task-specific fields.
    All runtimes and tasks work with this unified interface.
    """
    mfg: Any                            # Message Flow Graph (DGL/PyTorch)
    node_ids: Tensor                    # [N] All involved node IDs (global)

    # EdgePredict fields
    pos_src: Optional[Tensor] = None    # [M_pos] Source of positive edges
    pos_dst: Optional[Tensor] = None    # [M_pos] Destination of positive edges
    neg_src: Optional[Tensor] = None    # [M_neg] Source of negative edges
    neg_dst: Optional[Tensor] = None    # [M_neg] Destination of negative edges

    # NodeRegress/Classify fields
    target_nodes: Optional[Tensor] = None  # [M] Node IDs with labels
    labels: Optional[Tensor] = None        # [M] Labels or regression values

    # Timing info
    timestamps: Optional[Tensor] = None    # [E] Edge timestamps in this batch

    # Metadata
    chunk_id: Optional[tuple] = None       # (time_slice_id, node_cluster_id)
    local_node_mask: Optional[Tensor] = None  # [N] Which nodes are local vs remote

    # For distributed sampling (optional)
    remote_manifest: Optional[Dict[str, Any]] = None


class TaskAdapter(Protocol):
    """Protocol for task-specific logic independent of graph mode.

    Implementations: EdgePredictAdapter, NodeRegressAdapter, NodeClassifyAdapter
    """

    def build_sample_config(
        self,
        chunk: Any,              # ChunkAtomic
        model: Any,              # nn.Module (for model-aware sampling config)
        split: str,              # "train", "val", "test"
    ) -> SampleConfig:
        """Return sampling config for this chunk and split.

        Args:
            chunk: ChunkAtomic data container with graph structure
            model: Neural network model (allows task-aware tuning per model family)
            split: Data split ("train", "val", "test")

        Returns:
            SampleConfig specifying what to sample (edges, nodes, neighbors, etc.)
        """
        ...

    def compute_loss(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Tensor:
        """Compute task-specific loss from model output and batch.

        Args:
            model_output: Dictionary of tensors from model (e.g., 'pos_score', 'neg_score')
            batch: BatchData with inputs and labels

        Returns:
            Scalar loss tensor
        """
        ...

    def compute_metrics(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, float]:
        """Compute task-specific evaluation metrics.

        Args:
            model_output: Dictionary of tensors from model
            batch: BatchData with inputs and labels

        Returns:
            Dictionary of metric names to float values (e.g., {'auc': 0.92, 'ap': 0.88})
        """
        ...

    def format_output(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, Any]:
        """Format model output for saving/logging.

        Args:
            model_output: Dictionary of tensors from model
            batch: BatchData with inputs

        Returns:
            Dictionary with formatted predictions (for file output, inference APIs, etc.)
        """
        ...
