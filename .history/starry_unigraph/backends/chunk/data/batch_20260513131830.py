"""Chunk-line BatchData: Unified batch container for chunk pipeline.

Provides a simplified BatchData definition that works with chunk training,
bridging between PartitionData sampling and model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor


@dataclass
class BatchData:
    """Unified batch container for chunk pipeline training.

    Contains graph structure (MFG), node info, and task-specific fields.
    All samplers and models work with this unified interface.

    Attributes:
        mfg: Message Flow Graph (DGL Block or equivalent)
        node_ids: [N] All involved node IDs (global)
        pos_src: [M_pos] Source of positive edges (for link prediction)
        pos_dst: [M_pos] Destination of positive edges
        neg_src: [M_neg] Source of negative edges
        neg_dst: [M_neg] Destination of negative edges
        target_nodes: [M] Node IDs with labels (for node tasks)
        labels: [M] Labels or regression values
        timestamps: [E] Edge timestamps in this batch
        chunk_id: Chunk identifier (partition_id, local_chunk_id) or global_chunk_id
        local_node_mask: [N] Which nodes are local vs remote
        remote_manifest: Dict with remote data requests and manifests
    """

    mfg: Any  # DGL Block or PyTorch structure
    node_ids: Tensor  # [N] Global node IDs

    # Edge prediction fields
    pos_src: Optional[Tensor] = None  # [M_pos]
    pos_dst: Optional[Tensor] = None
    neg_src: Optional[Tensor] = None  # [M_neg]
    neg_dst: Optional[Tensor] = None
    labels: Optional[Tensor] = None  # [M] Edge labels

    # Node task fields
    target_nodes: Optional[Tensor] = None  # [M] Node IDs with labels

    # Timing info
    timestamps: Optional[Tensor] = None  # [E] Edge timestamps

    # Metadata
    chunk_id: Optional[Any] = None  # Chunk identifier
    local_node_mask: Optional[Tensor] = None  # [N] Local vs remote

    # Distributed / remote data
    remote_manifest: Optional[Dict[str, Any]] = None  # Remote data requests
