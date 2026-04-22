"""BatchData: Unified batch container for all graph modes and tasks.

Replaces task-specific batch types (DTDGBatch, CTDGBatch, etc.) with a single
schema that works across CTDG, DTDG, and Chunk modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor


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
