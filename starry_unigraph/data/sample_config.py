"""SampleConfig: Configuration for sampling passed to chunk.materialize().

Task adapters return SampleConfig to specify what to sample for each chunk.
This decouples task-specific sampling logic from graph mode implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import Tensor


@dataclass
class SampleConfig:
    """Configuration passed to chunk.materialize() to control sampling.

    Fields are task-specific and optional:
    - EdgePredict uses pos_src/pos_dst for edge pairs
    - NodeRegress/Classify use target_nodes/target_labels
    - All modes use num_neighbors/num_layers for GNN fanout
    """
    # EdgePredict fields (for temporal link prediction)
    pos_src: Optional[Tensor] = None     # [M] Source nodes of positive edges
    pos_dst: Optional[Tensor] = None     # [M] Destination nodes of positive edges

    # Negative sampling strategy
    neg_strategy: str = "none"           # "cache_aware", "random", "none"
    neg_ratio: int = 1                   # How many negatives per positive

    # NodeRegress/Classify fields
    target_nodes: Optional[Tensor] = None  # [M] Node IDs with labels
    target_labels: Optional[Tensor] = None # [M] Target values/labels

    # Common GNN sampling hyperparams
    num_neighbors: List[int] = field(default_factory=lambda: [20, 10])  # Per layer fanout
    num_layers: int = 2
    sample_type: str = "temporal"        # "temporal", "random", "importance", etc.

    # Additional task-specific params (for future extension)
    extra: Dict[str, Any] = field(default_factory=dict)
